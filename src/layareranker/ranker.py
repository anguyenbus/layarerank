"""Public library API."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .aggregate import aggregate, rank
from .config import Settings
from .engine import Scorer
from .errors import InputError


@dataclass(frozen=True)
class RankedPassage:
    """One passage in the output. `index` is its position in the caller's input list."""

    index: int
    score: float
    truncated: bool
    windows: int
    text: str | None = None


@dataclass(frozen=True)
class RerankResult:
    """Ranked passages, best first. Scores are comparable only between equal `score_version`s."""

    results: list[RankedPassage]
    score_version: str
    degraded: bool


class LayaReranker:
    """Rerank passages for one or many queries; every pair is scored independently, in shared batches."""

    def __init__(self, settings: Settings | None = None, *, scorer: Scorer | None = None) -> None:
        self.scorer = scorer or Scorer.load(settings)

    @property
    def score_version(self) -> str:
        return self.scorer.score_version

    def rerank(
        self,
        query: str,
        passages: Sequence[str],
        *,
        top_k: int | None = None,
        threshold: float | None = None,
        return_text: bool = False,
    ) -> RerankResult:
        return self.rerank_many(
            [(query, passages)], top_k=top_k, threshold=threshold, return_text=return_text
        )[0]

    def rerank_many(
        self,
        requests: Sequence[tuple[str, Sequence[str]]],
        *,
        top_k: int | None = None,
        threshold: float | None = None,
        return_text: bool = False,
    ) -> list[RerankResult]:
        """Rerank several queries at once; all their passages share the same forward passes."""
        _validate_cutoffs(top_k, threshold)
        prepared = [self.scorer.prepare(query, passages) for query, passages in requests]
        scored = self.scorer.score_units([unit for units in prepared for unit in units])
        results, offset = [], 0
        for (_, passages), units in zip(requests, prepared, strict=True):
            scores = scored.scores[offset : offset + len(units)]
            offset += len(units)
            ranked = rank(aggregate(units, scores, len(passages)), top_k=top_k, threshold=threshold)
            results.append(
                RerankResult(
                    results=[
                        RankedPassage(
                            p.index,
                            p.score,
                            p.truncated,
                            p.windows,
                            passages[p.index] if return_text else None,
                        )
                        for p in ranked
                    ],
                    score_version=self.score_version,
                    degraded=scored.degraded,
                )
            )
        return results


def _validate_cutoffs(top_k: int | None, threshold: float | None) -> None:
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0):
        raise InputError("top_k must be a non-negative integer")
    if threshold is not None and (
        isinstance(threshold, bool) or not isinstance(threshold, int | float) or not math.isfinite(threshold)
    ):
        raise InputError("threshold must be a finite number")
