"""Ranking metrics against graded qrels, and a paired bootstrap for system differences."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def ndcg_at(ranked: Sequence[str], relevant: dict[str, int], k: int = 10) -> float:
    """NDCG@k with gain 2^rel - 1; the ideal ranking uses every judged document, as in BEIR."""
    dcg = sum((2 ** relevant.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def mrr_at(ranked: Sequence[str], relevant: dict[str, int], k: int = 10) -> float:
    return next((1.0 / (i + 1) for i, d in enumerate(ranked[:k]) if relevant.get(d, 0) > 0), 0.0)


def recall_at(ranked: Sequence[str], relevant: dict[str, int], k: int = 10) -> float:
    hits = sum(1 for d in ranked[:k] if relevant.get(d, 0) > 0)
    return hits / len(relevant) if relevant else 0.0


METRICS = {"ndcg@10": ndcg_at, "mrr@10": mrr_at, "recall@10": recall_at}


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], iterations: int = 5000, seed: int = 13
) -> tuple[float, float, float]:
    """Mean of (a - b) with a 95% percentile CI, resampling queries with replacement."""
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iterations))
    return sum(diffs) / n, means[int(0.025 * iterations)], means[int(0.975 * iterations) - 1]
