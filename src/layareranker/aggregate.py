"""Fold unit scores back into passages and rank them deterministically."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .encoding import Unit


@dataclass(frozen=True)
class PassageScore:
    """The score for one input passage, with what the model could and could not see."""

    index: int
    score: float
    truncated: bool
    windows: int


def aggregate(units: Sequence[Unit], scores: Sequence[float], n_passages: int) -> list[PassageScore]:
    """Combine window scores per passage (max), in passage order."""
    if len(units) != len(scores):
        raise ValueError(f"{len(scores)} scores for {len(units)} units")
    best: list[float | None] = [None] * n_passages
    truncated = [False] * n_passages
    windows = [0] * n_passages
    for unit, score in zip(units, scores, strict=True):
        i = unit.passage_index
        current = best[i]
        best[i] = score if current is None else max(current, score)
        truncated[i] = truncated[i] or unit.truncated
        windows[i] += 1
    missing = [i for i, score in enumerate(best) if score is None]
    if missing:
        raise ValueError(f"passages {missing} have no scored units")
    return [
        PassageScore(index=i, score=score, truncated=truncated[i], windows=windows[i])
        for i, score in enumerate(best)
        if score is not None
    ]


def rank(
    scores: Sequence[PassageScore], *, top_k: int | None = None, threshold: float | None = None
) -> list[PassageScore]:
    """Sort by descending score; equal scores keep input order, so the caller's prior order breaks ties."""
    kept = [s for s in scores if threshold is None or s.score >= threshold]
    ordered = sorted(kept, key=lambda s: (-s.score, s.index))
    return ordered if top_k is None else ordered[:top_k]
