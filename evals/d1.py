"""Liquid AI d1-3B (LFM2.5-VL-3B post-trained for decisions) as a pointwise reranker.

Every (query, passage) pair is one state and one question. The repository ships its runtime as
code; its own batch call packs a query's pairs into one pass with no padding, reading the tokens
they share (the query) once. Scores are the runtime's unrounded probabilities: no temperature and
no calibration are applied.

Fine-tuning trains a LoRA adapter on the language model with the 3B base frozen in bf16. Training
rows run as plain right-padded chains, which give the same answers as the packed pass and are
differentiable throughout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, inject_adapter_in_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch.nn import functional
from transformers import AutoModel

from evals.beir import Pool
from evals.decider import MODELS_DIR, PRESETS, Row
from evals.systems import Scored

# The runtime is imported from this reviewed, pinned commit.
CHECKPOINTS = {"3b": ("LiquidAI/d1-3B", "051bcc464b01b9f92942b364d9586b0ef5912432")}
_BASE = "3b"
# Every linear map of the language model: attention, the short convolutions' projections, the MLPs.
_LORA = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"],
}


def _question(preset: str) -> dict[str, Any]:
    """A Decider preset in the Decision Index schema d1 reads."""
    question = PRESETS[preset]
    criteria = getattr(question, "criteria", None)
    spec: dict[str, Any] = {
        "type": "score" if isinstance(criteria, list) else "noul",
        "instructions": question.instructions,
    }
    if criteria:
        spec["criteria"] = criteria
    return spec


class D1:
    """Spec: `d1:<model>:<preset>`; `<model>` is a `CHECKPOINTS` key or an adapter dir in evals/models."""

    def __init__(self, spec: str) -> None:
        _, model, preset = spec.split(":")
        if preset not in PRESETS:
            raise ValueError(f"unknown d1 preset {preset!r}; choose one of {sorted(PRESETS)}")
        local = MODELS_DIR / model
        if not local.is_dir() and model not in CHECKPOINTS:
            raise ValueError(f"unknown d1 model {model!r}")
        repo, revision = CHECKPOINTS[_BASE if local.is_dir() else model]
        self.name = spec
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        root = snapshot_download(repo, revision=revision)
        self.model = AutoModel.from_pretrained(root, trust_remote_code=True, dtype=dtype).to(self._device)
        self._engine = self.model.engine
        self._tok = self._engine.tokenizer
        self._lora = False
        if local.is_dir():
            self._attach_lora()
            set_peft_model_state_dict(self._lm, load_file(local / "adapter.safetensors"))
            self.model.eval()

        self._spec = _question(preset)
        # The question classes and the readout live in the runtime's own prompt module.
        api = sys.modules[type(self._engine).__mro__[1].__module__]
        self._question = api.as_question(self._spec)
        prompt = sys.modules[type(self._question).__module__]
        self._kind: str = self._spec["type"]
        self._forms: list[list[int]] = prompt.readout_ids(self._tok, self._question)

    @property
    def _lm(self) -> torch.nn.Module:
        return self.model.model.language_model

    def _attach_lora(self) -> None:
        """Inject the adapter in place, in fp32 over the bf16 base (PEFT casts around each layer)."""
        if self._lora:
            return
        inject_adapter_in_model(LoraConfig(**_LORA), self._lm)
        for name, param in self._lm.named_parameters():
            if "lora_" in name:
                param.data = param.data.float()
        self._lora = True

    def _ids(self, query: str, passage: str) -> list[int]:
        text = self._engine.render({"query": query, "passage": passage}, self._question)
        ids: list[int] = self._tok.encode(text, add_special_tokens=False)
        return ids

    def encode(self, query: str, passage: str, max_tokens: int | None = None) -> Row:
        """One full prompt as the runtime renders it, for training.

        The runtime itself never cuts a state; `max_tokens` cuts the passage from its end so a
        training row fits memory.
        """
        ids = self._ids(query, passage)
        truncated = max_tokens is not None and len(ids) > max_tokens
        while max_tokens is not None and len(ids) > max_tokens and passage:
            tokens = self._tok.encode(passage, add_special_tokens=False)
            passage = self._tok.decode(tokens[: max(0, len(tokens) - (len(ids) - max_tokens) - 4)])
            ids = self._ids(query, passage)
        return Row(ids, truncated)

    def margins(self, rows: list[Row]) -> torch.Tensor:
        """Yes-minus-no logit margin of a noul, each side its best form; its sigmoid is P(yes)."""
        width = max(len(r.ids) for r in rows)
        pad = self._tok.pad_token_id
        ids = torch.tensor([r.ids + [pad] * (width - len(r.ids)) for r in rows], device=self._device)
        hidden = self._lm(input_ids=ids).last_hidden_state
        last = hidden[torch.arange(len(rows), device=self._device), [len(r.ids) - 1 for r in rows]]
        yes, no = self._forms
        logits = functional.linear(last.float(), self.model.lm_head.weight[yes + no].float())
        return logits[:, : len(yes)].amax(dim=1) - logits[:, len(yes) :].amax(dim=1)

    def relevance(self, probabilities: list[float]) -> float:
        """A score in [0, 1]: P(yes) for a noul, the expected level over the top level for a score."""
        if self._kind == "noul":
            return probabilities[0]  # the runtime orders a noul (yes, no)
        return sum(i * p for i, p in enumerate(probabilities)) / (len(probabilities) - 1)

    def score(self, pool: Pool) -> Scored:
        requests = [({"query": pool.query, "passage": c.text}, [self._spec]) for c in pool.candidates]
        answers = self._engine.probabilities_batch(requests)
        # The runtime never cuts a state, and FiQA's longest is far inside the 32,768-token window.
        return Scored([self.relevance(a[0]) for a in answers], truncated=0)

    def param_groups(self, lr_head: float, lr_backbone: float) -> list[dict[str, Any]]:
        """Train a LoRA adapter; the base stays frozen. There is no separate head: `lr_head` is unused."""
        self._attach_lora()
        for name, param in self.model.named_parameters():
            param.requires_grad = "lora_" in name
        # use_reentrant=False lets checkpointing coexist with LoRA (inputs carry no grad).
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        lora = [p for p in self.model.parameters() if p.requires_grad]
        return [{"params": lora, "lr": lr_backbone, "weight_decay": 0.0}]

    def save(self, out: Path) -> None:
        """Write the adapter alone; `D1` loads it over the pinned base."""
        out.mkdir(parents=True, exist_ok=True)
        save_file(get_peft_model_state_dict(self._lm), out / "adapter.safetensors")
        (out / "adapter.json").write_text(json.dumps({"base": CHECKPOINTS[_BASE], "lora": _LORA}, indent=2))
