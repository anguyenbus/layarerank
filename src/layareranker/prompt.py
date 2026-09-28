"""Scoring presets: how a (query, passage) pair becomes a laya state and question, and back."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from .errors import ConfigurationError, ScoringError

QUESTION_ID = "relevance"
StateShape = Literal["dict", "text"]

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


@dataclass(frozen=True)
class Preset:
    """A state shape plus one fixed question. Its text is part of `score_version`."""

    name: str
    state_shape: StateShape
    question: dict[str, Any]

    @property
    def questions(self) -> dict[str, dict[str, Any]]:
        return {QUESTION_ID: self.question}

    def build_state(self, query: str, passage: str) -> dict[str, str] | str:
        # The query goes first: laya cuts states from the end, so the query always survives.
        if self.state_shape == "dict":
            return {"query": query, "passage": passage}
        return f"Query: {query}\nPassage: {passage}"

    def extract_score(self, result: dict[str, Any]) -> float:
        """Return a relevance score in [0, 1], or raise; never substitutes a default."""
        answer = result.get("answers", {}).get(QUESTION_ID)
        if not isinstance(answer, dict):
            raise ScoringError(f"model returned no '{QUESTION_ID}' answer")
        if self.question["type"] == "noul":
            value = _finite(answer.get("noul"))
        else:
            value = _finite(answer.get("score")) / (len(self.question["criteria"]) - 1)
        if not 0.0 <= value <= 1.0:
            raise ScoringError(f"model returned an out-of-range score: {value!r}")
        return value


def _finite(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ScoringError(f"model returned a non-numeric score: {value!r}")
    return float(value)


PRESETS: dict[str, Preset] = {
    preset.name: preset
    for preset in (
        Preset(
            "noul-dict",
            "dict",
            {
                "type": "noul",
                "instructions": "Does `passage` contain information that answers `query`?",
                "criteria": _NOUL_CRITERIA,
            },
        ),
        Preset(
            "noul-dict-bare",
            "dict",
            {"type": "noul", "instructions": "Does `passage` answer `query`?"},
        ),
        Preset(
            "noul-text",
            "text",
            {
                "type": "noul",
                "instructions": "Does the passage contain information that answers the query?",
                "criteria": _NOUL_CRITERIA,
            },
        ),
        Preset(
            "score-dict",
            "dict",
            {
                "type": "score",
                "instructions": "How well does `passage` answer `query`?",
                "criteria": _SCORE_LEVELS,
            },
        ),
    )
}
DEFAULT_PRESET = "noul-dict"


def get_preset(name: str) -> Preset:
    resolved = DEFAULT_PRESET if name == "default" else name
    if resolved not in PRESETS:
        raise ConfigurationError(f"unknown preset {name!r}; choose one of {sorted(PRESETS)} or 'default'")
    return PRESETS[resolved]
