"""Decision 2.0 Kai (vLLM Semantic Router) as a pointwise reranker: one (query, passage) per prompt.

Kai's own runtime answers many questions about one state and refuses over-long input; reranking is
one question over many states, so this batches prompts across states and calls the packaged
`DecisionModel` directly, cutting an over-long passage instead of failing. Prompts are tokenized
the way the package's `encode` does it (prefix, each option, suffix, concatenated) and the scores
are the raw option logits at temperature 1, as the runtime reads them.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download

from evals.beir import Pool
from evals.decider import MODELS_DIR, PRESETS, Row
from evals.systems import Scored

# The repository ships its runtime as code; it is imported from this reviewed, pinned commit.
CHECKPOINTS = {"0.6b": ("vllm-sr/Decision-2.0-Kai-0.6B", "cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764")}
_CODE = "0.6b"


def _runtime() -> Any:
    """The package's vendored model module, its checkpoint loader and its directory."""
    repo, revision = CHECKPOINTS[_CODE]
    root = snapshot_download(repo, revision=revision)
    if root not in sys.path:
        sys.path.insert(0, root)
    from decision2._vendor.dev2model import decision_model  # ty: ignore[unresolved-import]
    from decision2.api import _without_remote_code_prompt  # ty: ignore[unresolved-import]

    # The tokenizer loads from tokenizer_config.json; the package's Transformers wrapper is not run.
    load = _without_remote_code_prompt(decision_model.DecisionModel.from_checkpoint)
    return decision_model, load, Path(root)


def _question(preset: str) -> dict[str, Any]:
    """A Decider preset as the training-row fields Kai's prompt is rendered from."""
    question = PRESETS[preset]
    criteria = getattr(question, "criteria", None)
    if isinstance(criteria, list):
        kind, options = "score", [(str(i), text) for i, text in enumerate(criteria)]
    else:
        defaults = {"false": "No", "true": "Yes"}
        kind, options = "noul", [(key, (criteria or defaults)[key]) for key in ("false", "true")]
    return {
        "task_type": kind,
        "instructions": question.instructions,
        "options": [{"key": key, "description": text} for key, text in options],
    }


class Kai:
    """Spec: `kai:<model>:<preset>`; `<model>` is a `CHECKPOINTS` key or a dir in evals/models."""

    def __init__(self, spec: str, max_batch_tokens: int = 24000) -> None:
        _, model, preset = spec.split(":")
        if preset not in PRESETS:
            raise ValueError(f"unknown kai preset {preset!r}; choose one of {sorted(PRESETS)}")
        self._runtime, load, package = _runtime()
        local = MODELS_DIR / model
        if not local.is_dir() and model not in CHECKPOINTS:
            raise ValueError(f"unknown kai model {model!r}")
        self.name = spec
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self._tok = load(local if local.is_dir() else package)
        self.model = self.model.float().to(self._device).eval()
        self._pad = self._tok.pad_token_id if self._tok.pad_token_id is not None else self._tok.eos_token_id
        self._max_batch_tokens = max_batch_tokens

        self._row = _question(preset)
        self._kind = self._row["task_type"]
        self._labels = [option["key"] for option in self._row["options"]]
        _, options, suffix = self._runtime.segments({**self._row, "state": ""})
        self._tail: list[int] = []
        ends = []
        for option in options:
            self._tail += self._tok.encode(option, add_special_tokens=False)
            ends.append(len(self._tail) - 1)
        self._tail += self._tok.encode(suffix, add_special_tokens=False)
        # Each option is read at its last token, counted here from the end of the prompt.
        self._option_offsets = [end - len(self._tail) for end in ends]
        config = json.loads((package / "config.json").read_text())
        self._max_tokens: int = config["max_input_tokens"]

    def _prefix(self, query: str, passage: str) -> list[int]:
        prefix, _, _ = self._runtime.segments({**self._row, "state": {"query": query, "passage": passage}})
        return self._tok.encode(prefix, add_special_tokens=False)

    def encode(self, query: str, passage: str, max_tokens: int | None = None) -> Row:
        """One full prompt; a passage that does not fit is cut from its end.

        `max_tokens` lowers the row cap below the checkpoint's window (training memory).
        """
        budget = min(self._max_tokens, max_tokens or self._max_tokens) - len(self._tail)
        ids = self._prefix(query, passage)
        truncated = len(ids) > budget
        while len(ids) > budget and passage:
            tokens = self._tok.encode(passage, add_special_tokens=False)
            passage = self._tok.decode(tokens[: max(0, len(tokens) - (len(ids) - budget) - 4)])
            ids = self._prefix(query, passage)
        return Row(ids + self._tail, truncated)

    def forward(self, rows: list[Row]) -> torch.Tensor:
        """Option logits, one row per prompt, in `self._labels` order."""
        width = -(-max(len(r.ids) for r in rows) // 8) * 8  # the package pads to a multiple of 8
        ids = torch.tensor([r.ids + [self._pad] * (width - len(r.ids)) for r in rows], device=self._device)
        mask = torch.tensor(
            [[1] * len(r.ids) + [0] * (width - len(r.ids)) for r in rows], device=self._device
        )
        lengths = torch.tensor([len(r.ids) for r in rows], device=self._device)
        positions = lengths[:, None] + torch.tensor(self._option_offsets, device=self._device)
        with torch.autocast(self._device, dtype=torch.bfloat16, enabled=self._device == "cuda"):
            return self.model(
                input_ids=ids,
                attention_mask=mask,
                candidate_positions=positions,
                candidate_mask=torch.ones_like(positions, dtype=torch.bool),
                query_positions=lengths - 1,
            )

    def margins(self, rows: list[Row]) -> torch.Tensor:
        """True-minus-false logit margin of a noul; its sigmoid is the score `relevance` returns."""
        logits = self.forward(rows)
        return logits[:, self._labels.index("true")] - logits[:, self._labels.index("false")]

    def relevance(self, logits: torch.Tensor) -> torch.Tensor:
        """A score in [0, 1]: P(true) for a noul, the expected level over the top level for a score."""
        logits = logits.double()
        if self._kind == "noul":
            true, false = self._labels.index("true"), self._labels.index("false")
            return torch.sigmoid(logits[:, true] - logits[:, false])
        levels = torch.arange(len(self._labels), device=logits.device)
        return (logits.softmax(dim=-1) * levels).sum(dim=-1) / (len(self._labels) - 1)

    def batches(self, rows: list[Row]) -> list[list[int]]:
        """Greedy packing over rows sorted by length, capped by padded tokens."""
        chunks: list[list[int]] = []
        current: list[int] = []
        for i in sorted(range(len(rows)), key=lambda i: len(rows[i].ids)):
            if current and (len(current) + 1) * len(rows[i].ids) > self._max_batch_tokens:
                chunks.append(current)
                current = []
            current.append(i)
        if current:
            chunks.append(current)
        return chunks

    @torch.inference_mode()
    def score(self, pool: Pool) -> Scored:
        rows = [self.encode(pool.query, c.text) for c in pool.candidates]
        scores = [0.0] * len(rows)
        for chunk in self.batches(rows):
            values = self.relevance(self.forward([rows[i] for i in chunk])).tolist()
            for i, value in zip(chunk, values, strict=True):
                scores[i] = value
        return Scored(scores, truncated=sum(r.truncated for r in rows))

    def param_groups(self, lr_head: float, lr_backbone: float) -> list[dict[str, Any]]:
        """Train the head and every backbone weight but the token embeddings (optimizer memory)."""
        backbone = self.model.backbone
        for name, param in backbone.named_parameters():
            param.requires_grad = "embed_tokens" not in name
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        body = [p for p in backbone.parameters() if p.requires_grad]
        return [
            {"params": list(self.model.head.parameters()), "lr": lr_head, "weight_decay": 0.01},
            {"params": body, "lr": lr_backbone, "weight_decay": 0.01},
        ]

    def save(self, out: Path) -> None:
        """Write a full checkpoint in the layout `DecisionModel.from_checkpoint` reads."""
        if out.exists():
            shutil.rmtree(out)
        self.model.backbone.gradient_checkpointing_disable()
        self.model.eval().save(out, self._tok)
