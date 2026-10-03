"""Strands Decider as a pointwise reranker: every (query, passage) pair is one state, one question.

`strands_decider`'s own engine serves one state with many questions; reranking is the opposite
shape, so this batches full prompts across states and calls the model directly. Prompts are built
and tokenized the way the engine's plain batched path does it (state with special tokens, question
without, concatenated), and scores are read unrounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from strands_decider.infer import _option_token_index
from strands_decider.modeling import StrandsDeciderModel
from strands_decider.prompting import render_question, render_state
from strands_decider.schema import NoulQuestion, Question, ScoreQuestion

from evals.beir import Pool
from evals.systems import Scored

MODELS_DIR = Path(__file__).parent / "models"
CHECKPOINTS = {"2b": "StrandsAgents/strands-decider-2B-hobson-v19"}

_NOUL_CRITERIA = {
    "true": "The passage states facts that answer or directly help answer the query.",
    "false": "The passage is off-topic, only shares keywords, or lacks the needed facts.",
}
_SCORE_LEVELS = [
    "unrelated, or only shares keywords with the query",
    "on the query's topic but does not answer it",
    "partially answers the query",
    "fully answers the query",
]
# Decider renders a noul under "Decide whether the statement is true of the state.", so the
# statement presets phrase the instruction as a claim; `noul-question` keeps Laya's wording.
PRESETS: dict[str, Question] = {
    "noul-bare": NoulQuestion(instructions="`passage` answers `query`."),
    "noul-question": NoulQuestion(instructions="Does `passage` answer `query`?"),
    "noul-criteria": NoulQuestion(
        instructions="`passage` contains information that answers `query`.", criteria=_NOUL_CRITERIA
    ),
    "score": ScoreQuestion(instructions="How well does `passage` answer `query`?", criteria=_SCORE_LEVELS),
}


@dataclass(frozen=True)
class Row:
    ids: list[int]
    truncated: bool


class Decider:
    """Spec: `decider:<model>:<preset>`; `<model>` is a `CHECKPOINTS` key or a dir in evals/models."""

    def __init__(self, spec: str, max_batch_tokens: int = 12000) -> None:
        _, model, preset = spec.split(":")
        if preset not in PRESETS:
            raise ValueError(f"unknown decider preset {preset!r}; choose one of {sorted(PRESETS)}")
        local = MODELS_DIR / model
        checkpoint = str(local) if local.is_dir() else CHECKPOINTS[model]
        self.name = spec
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = StrandsDeciderModel.load(checkpoint).to(self._device).eval()
        if self._device == "cpu":
            self.model.torso.float()  # bf16 kernels are slower than fp32 on CPU
        self._tok = self.model.tokenizer
        self._max_batch_tokens = max_batch_tokens

        rendered = render_question(PRESETS[preset])
        encoded = self._tok(rendered.text, add_special_tokens=False, return_offsets_mapping=True)
        self._question_ids: list[int] = encoded["input_ids"]
        self._options = _option_token_index(encoded["offset_mapping"], rendered.option_spans, 0)
        self._labels = rendered.slot_labels
        self._kind = rendered.kind
        config = self.model.config
        self._temperature = float((config.temperature_by_kind or {}).get(self._kind, config.temperature))
        self._state_budget = config.max_length - len(self._question_ids)

    def encode(self, query: str, passage: str, max_tokens: int | None = None) -> Row:
        """One full prompt; an over-long state is cut from the end, as the engine cuts it.

        `max_tokens` lowers the row cap below the checkpoint's window (training memory).
        """
        budget = self._state_budget
        if max_tokens is not None:
            budget = min(budget, max_tokens - len(self._question_ids))
        state = self._tok(render_state({"query": query, "passage": passage}), add_special_tokens=True)
        ids: list[int] = state["input_ids"]
        return Row(ids[:budget] + self._question_ids, len(ids) > budget)

    def forward(self, rows: list[Row]) -> torch.Tensor:
        """Temperature-scaled log-probabilities over the question's options, one row per prompt."""
        width = max(len(r.ids) for r in rows)
        pad = self._tok.pad_token_id if self._tok.pad_token_id is not None else 0
        ids = torch.tensor([r.ids + [pad] * (width - len(r.ids)) for r in rows], device=self._device)
        mask = torch.tensor(
            [[1] * len(r.ids) + [0] * (width - len(r.ids)) for r in rows], device=self._device
        )
        first_question_token = [len(r.ids) - len(self._question_ids) for r in rows]
        opt_idx = torch.tensor([[base + i for i in self._options] for base in first_question_token])
        out: dict[str, Any] = self.model(
            input_ids=ids,
            attention_mask=mask,
            n_slots=torch.full((len(rows),), len(self._labels), device=self._device),
            temperature=self._temperature,
            opt_idx=opt_idx.to(self._device),
        )
        return out["log_probs"][:, : len(self._labels)]

    def margins(self, rows: list[Row]) -> torch.Tensor:
        """True-minus-false logit margin of a noul; its sigmoid is the score `relevance` returns."""
        log_probs = self.forward(rows)
        return log_probs[:, self._labels.index("true")] - log_probs[:, self._labels.index("false")]

    def relevance(self, log_probs: torch.Tensor) -> torch.Tensor:
        """A score in [0, 1]: P(true) for a noul, the expected level over the top level for a score."""
        log_probs = log_probs.double()
        if self._kind == "noul":
            true, false = self._labels.index("true"), self._labels.index("false")
            return torch.sigmoid(log_probs[:, true] - log_probs[:, false])
        levels = torch.tensor([int(label) for label in self._labels], device=log_probs.device)
        return (log_probs.exp() * levels).sum(dim=-1) / (len(self._labels) - 1)

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
