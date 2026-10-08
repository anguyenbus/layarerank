"""Mapika decider-12b (Gemma-4-12B-it read through `decider-ai`) as a pointwise reranker.

Every (query, passage) pair is one state and one question. Each prompt is built by the package
itself (chat layout) and read eagerly from the packaged `DecisionModel`, without the CUDA-graph
engine: the bf16 weights leave about 1.1 GB free on a 24 GB card. Rows are scored one at a time,
which reproduces `Decider.system_one` exactly; batching rows is no faster on that card and moves
bf16 scores by up to 0.07.

A noul is ranked by its yes-minus-no logit margin at temperature 1. The checkpoint serves noul at
temperature 0.05, which is the same order but saturates to exactly 0 or 1 and would tie candidates.
"""

from __future__ import annotations

from typing import Any

import torch
from decider.infer import Decider as Runtime
from decider.model import cap_logits, collate
from decider.systemone import render_state
from huggingface_hub import snapshot_download
from torch.nn import functional

from evals.beir import Pool
from evals.decider import PRESETS, Row
from evals.systems import Scored

CHECKPOINTS = {"12b": ("Mapika/decider-12b", "8ac1efa708b71b86ae33b01d2a8d7a3ddcb48e66")}  # v2
MAX_STATE_TOKENS = 32768  # the package's own request budget


def _question(preset: str) -> dict[str, Any]:
    """A Decider preset as a `system_one` question."""
    question = PRESETS[preset]
    criteria = getattr(question, "criteria", None)
    spec: dict[str, Any] = {
        "type": "score" if isinstance(criteria, list) else "noul",
        "instructions": question.instructions,
    }
    if criteria:
        spec["criteria"] = criteria
    return spec


class Mapika:
    """Spec: `mapika:<model>:<preset>`; `<model>` is a `CHECKPOINTS` key."""

    def __init__(self, spec: str) -> None:
        _, model, preset = spec.split(":")
        if preset not in PRESETS:
            raise ValueError(f"unknown mapika preset {preset!r}; choose one of {sorted(PRESETS)}")
        repo, revision = CHECKPOINTS[model]
        self.name = spec
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._runtime = Runtime(snapshot_download(repo, revision=revision), use_graphs=False)
        self.model = self._runtime.m
        self._pad: int = self.model.tok.pad_token_id

        self._question = _question(preset)
        self._kind: str = self._question["type"]
        self._temperature = float(self._runtime.T_by_type.get(self._kind, self._runtime.T))
        probe = self._item("", "")
        self._n_options: int = probe["nopts"][0]
        # Everything but the state is tokenized on its own, so a row's length minus this is its state.
        state = "Context:\n" + render_state({"query": "", "passage": ""})
        self._overhead = len(probe["ids"]) - len(self.model.tok.encode(state, add_special_tokens=False))

    def _item(self, query: str, passage: str) -> dict[str, Any]:
        state = {"query": query, "passage": passage}
        _, _, items = self._runtime._system_one_items(
            state, {"relevance": self._question}, max_state_tokens=MAX_STATE_TOKENS
        )
        (item,) = items
        return item

    def encode(self, query: str, passage: str) -> Row:
        """One full prompt; the package cuts a state over its budget from the end."""
        ids: list[int] = self._item(query, passage)["ids"]
        return Row(ids, len(ids) - self._overhead >= MAX_STATE_TOKENS)

    def forward(self, rows: list[Row]) -> torch.Tensor:
        """Soft-capped option logits at temperature 1, one row per prompt: (no, yes) or the levels.

        `DecisionModel.slot_logits` with `use_cache=False`: the key/value cache it would return costs
        0.4 GB per 1,024 tokens, more than the card has left for a long passage.
        """
        items = [
            {"ids": r.ids, "slots": [len(r.ids) - 1], "golds": [0], "nopts": [self._n_options]} for r in rows
        ]
        batch = collate(items, self._pad)
        hidden = self.model.lm.model(
            input_ids=batch["input_ids"].to(self._device),
            attention_mask=batch["attention_mask"].to(self._device),
            use_cache=False,
        ).last_hidden_state
        slots = hidden[batch["slot_batch"].to(self._device), batch["slot_idx"].to(self._device)]
        letters = self.model.lm.lm_head.weight[self.model.letters[: self._n_options]]
        return cap_logits(self.model, functional.linear(slots, letters).float())

    def relevance(self, logits: torch.Tensor) -> torch.Tensor:
        """A score in [0, 1]: sigmoid of the noul margin, or the expected level over the top level."""
        logits = logits.double()
        if self._kind == "noul":
            return torch.sigmoid(logits[:, 1] - logits[:, 0])
        levels = torch.arange(self._n_options, device=logits.device)
        return ((logits / self._temperature).softmax(dim=-1) * levels).sum(dim=-1) / (self._n_options - 1)

    @torch.inference_mode()
    def score(self, pool: Pool) -> Scored:
        rows = [self.encode(pool.query, c.text) for c in pool.candidates]
        scores = [self.relevance(self.forward([row])).item() for row in rows]
        return Scored(scores, truncated=sum(r.truncated for r in rows))
