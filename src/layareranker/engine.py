"""The scoring engine: packs units into token-capped forward passes on one pinned laya agent."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import laya

from ._laya import AgentLike, LayaTokenizer, Tokenizer, load_agent
from .aggregate import PassageScore, aggregate
from .config import Settings
from .encoding import Budget, Unit, make_units, plan_budget, validate_request
from .errors import BackendFatal, ConfigurationError, ScoringError
from .prompt import Preset, get_preset

logger = logging.getLogger(__name__)

# RuntimeError messages after which the CUDA context cannot be trusted again.
_FATAL_CUDA_MARKERS = (
    "device-side assert",
    "illegal memory access",
    "unspecified launch failure",
    "cuda error: misaligned address",
)
_WARMUP_WORD = "evidence "


@dataclass(frozen=True)
class ScoredUnits:
    """Scores in unit order. `degraded` means laya retried at least one forward on CPU."""

    scores: list[float]
    degraded: bool


@dataclass(frozen=True)
class PassageScores:
    passages: list[PassageScore]
    degraded: bool


class Scorer:
    """Thread-safe scorer; forward passes never overlap, whichever thread calls in."""

    def __init__(self, agent: AgentLike, tokenizer: Tokenizer, preset: Preset, settings: Settings) -> None:
        self.agent = agent
        self.tokenizer = tokenizer
        self.preset = preset
        self.settings = settings
        self.budget: Budget = plan_budget(tokenizer, preset, settings)
        if settings.max_batch_tokens < self.budget.max_len:
            raise ConfigurationError(
                f"max_batch_tokens={settings.max_batch_tokens} is below one {self.budget.max_len}-token row"
            )
        self.score_version = self._score_version()
        self._forward_lock = threading.Lock()

    @classmethod
    def load(cls, settings: Settings | None = None) -> Scorer:
        settings = settings or Settings()
        preset = get_preset(settings.preset)
        agent = load_agent(settings)
        return cls(agent, LayaTokenizer(agent), preset, settings)

    def prepare(self, query: str, passages: Sequence[str]) -> list[Unit]:
        """Validate one request and tokenize it into units. Safe to call from any thread."""
        validate_request(query, passages, self.settings)
        return make_units(
            query,
            passages,
            preset=self.preset,
            tokenizer=self.tokenizer,
            budget=self.budget,
            settings=self.settings,
        )

    def score_units(self, units: Sequence[Unit]) -> ScoredUnits:
        """Score units in as few token-capped forward passes as possible; results keep unit order."""
        if not units:
            return ScoredUnits(scores=[], degraded=False)
        with self._forward_lock:
            fallbacks_before = self.agent.cpu_fallback_count
            scores: list[float | None] = [None] * len(units)
            for chunk in self._chunks(units):
                results = self._forward([units[i].state for i in chunk])
                for i, result in zip(chunk, results, strict=True):
                    scores[i] = self.preset.extract_score(result)
            degraded = self.agent.cpu_fallback_count > fallbacks_before
        if degraded:
            logger.warning("laya fell back to CPU during a forward pass; results are slow, not wrong")
        complete = [s for s in scores if s is not None]
        if len(complete) != len(units):
            raise ScoringError(f"{len(units) - len(complete)} units were never scored")
        return ScoredUnits(scores=complete, degraded=degraded)

    def score(self, query: str, passages: Sequence[str]) -> PassageScores:
        """Score one request end to end (prepare, forward, aggregate windows)."""
        units = self.prepare(query, passages)
        scored = self.score_units(units)
        return PassageScores(aggregate(units, scored.scores, len(passages)), scored.degraded)

    def warmup(self) -> bool:
        """Run the largest batch the settings allow. Returns False if it needed a CPU fallback."""
        long_passage = _WARMUP_WORD * (self.budget.room + 1)
        rows = min(self.settings.max_batch_rows, self.settings.max_batch_tokens // self.budget.max_len)
        units = self.prepare("warm-up query", [long_passage] * max(1, rows))
        return not self.score_units(units).degraded

    def _chunks(self, units: Sequence[Unit]) -> list[list[int]]:
        """Greedy packing over units sorted by length, capped by rows and by padded tokens."""
        order = sorted(range(len(units)), key=lambda i: units[i].row_tokens)
        chunks: list[list[int]] = []
        current: list[int] = []
        longest = 0
        for i in order:
            candidate_longest = max(longest, units[i].row_tokens)
            too_many_rows = len(current) + 1 > self.settings.max_batch_rows
            too_many_tokens = (len(current) + 1) * candidate_longest > self.settings.max_batch_tokens
            if current and (too_many_rows or too_many_tokens):
                chunks.append(current)
                current, candidate_longest = [], units[i].row_tokens
            current.append(i)
            longest = candidate_longest
        if current:
            chunks.append(current)
        return chunks

    def _forward(self, states: list[Any]) -> list[dict[str, Any]]:
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                results = self.agent.predict_batch(
                    states, self.preset.questions, batch_size=len(states), lang=None, sort_by_length=False
                )
        except RuntimeError as exc:
            if any(marker in str(exc).lower() for marker in _FATAL_CUDA_MARKERS):
                raise BackendFatal("the inference device is in an unrecoverable state") from exc
            raise
        finally:
            if captured.getvalue().strip():
                logger.warning("laya: %s", captured.getvalue().strip())
        if len(results) != len(states):
            raise ScoringError(f"model returned {len(results)} results for {len(states)} states")
        return results

    def _score_version(self) -> str:
        """Identity of everything that changes a score; equal versions mean comparable scores."""
        identity = {
            "laya": getattr(laya, "__version__", "unknown"),
            "model": self.settings.model,
            "revision": getattr(self.agent, "revision", None),
            "device": str(getattr(getattr(self.agent, "device", None), "type", None)),
            "dtype": str(getattr(self.agent, "dtype", None)),
            "preset": [self.preset.name, self.preset.state_shape, self.preset.question],
            "budget": [self.budget.max_len, self.budget.head_tokens, self.budget.room],
            "long_doc": [
                self.settings.long_doc,
                self.settings.window_stride_ratio,
                self.settings.max_windows_per_doc,
            ],
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
        return digest[:16]
