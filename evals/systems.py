"""Rerankers under evaluation. Each returns one score per candidate, in candidate order."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from evals.beir import Pool


@dataclass(frozen=True)
class Scored:
    scores: list[float]
    truncated: int


class System(Protocol):
    name: str

    def score(self, pool: Pool) -> Scored: ...


class Bm25:
    """The first-stage order itself: the no-rerank baseline."""

    name = "bm25"

    def score(self, pool: Pool) -> Scored:
        return Scored([c.bm25 for c in pool.candidates], truncated=0)


class Laya:
    """LayaReranker through the production engine. Spec: `laya:<model>:<preset>[:window_max]`."""

    def __init__(self, spec: str) -> None:
        from layareranker.config import Settings
        from layareranker.engine import Scorer

        _, model, preset, *rest = spec.split(":")
        long_doc = rest[0] if rest else "truncate"
        local = Path(__file__).parent / "models" / model  # fine-tuned checkpoints from evals.finetune
        if local.is_dir():
            model = str(local)
        self.name = spec
        self._scorer = Scorer.load(
            Settings(
                model=model, preset=preset, long_doc=long_doc, max_query_tokens=128, max_passage_chars=10**6
            )
        )

    def score(self, pool: Pool) -> Scored:
        result = self._scorer.score(pool.query, [c.text for c in pool.candidates])
        return Scored([p.score for p in result.passages], truncated=sum(p.truncated for p in result.passages))


class Bge:
    """Open cross-encoder baseline, BAAI/bge-reranker-v2-m3, truncating at 512 tokens like TEI."""

    name = "bge-reranker-v2-m3"

    def __init__(self) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)  # CUDA when available

    def score(self, pool: Pool) -> Scored:
        pairs = [(pool.query, c.text) for c in pool.candidates]
        return Scored([float(s) for s in self._model.predict(pairs, batch_size=16)], truncated=0)


def make_system(spec: str) -> System:
    if spec == "bm25":
        return Bm25()
    if spec == "bge":
        return Bge()
    if spec.startswith("laya:"):
        return Laya(spec)
    if spec.startswith("decider:"):
        from evals.decider import Decider

        return Decider(spec)
    if spec.startswith("kai:"):
        from evals.kai import Kai

        return Kai(spec)
    raise ValueError(
        f"unknown system {spec!r}; use bm25, bge, laya:<model>:<preset>[:window_max], "
        "decider:<model>:<preset> or kai:<model>:<preset>"
    )
