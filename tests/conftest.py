"""Hermetic fakes: a whitespace tokenizer and an agent that mimics laya's predict_batch contract."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest

from layareranker._laya import HeadLength
from layareranker.config import Settings
from layareranker.engine import Scorer
from layareranker.prompt import get_preset

RELEVANT = "RELEVANT"


class FakeTokenizer:
    """One token per whitespace-separated word of the serialized state."""

    def __init__(self, max_len: int = 512, head_tokens: int = 40, untruncated_head: int | None = None):
        self._max_len = max_len
        self._head = HeadLength(encoded=head_tokens, untruncated=untruncated_head or head_tokens)

    @property
    def max_len(self) -> int:
        return self._max_len

    def head_length(self, question: dict[str, Any]) -> HeadLength:
        return self._head

    def count_state(self, state: Any) -> int:
        text = state if isinstance(state, str) else json.dumps(state)
        return len(text.split())

    def split_text(self, text: str, window_tokens: int, stride_tokens: int) -> list[str]:
        words = text.split()
        if len(words) <= window_tokens:
            return [text]
        starts = range(0, max(1, len(words) - window_tokens + stride_tokens), stride_tokens)
        return [" ".join(words[s : s + window_tokens]) for s in starts]


class FakeAgent:
    """Scores 0.9 when a passage contains RELEVANT, else 0.1, through a batched call.

    Like laya, it only sees the first `visible_tokens` of a state (FakeTokenizer's default room).
    Hooks let tests break answers, raise errors, slow forwards down, and detect overlap.
    """

    def __init__(self, visible_tokens: int = 471) -> None:
        self.visible_tokens = visible_tokens
        self.cfg: dict[str, Any] = {"max_len": 512, "head_max_len": 192}
        self.cpu_fallback_count = 0
        self.calls: list[int] = []
        self.delay_s = 0.0
        self.error: BaseException | None = None
        self.fail_when: Callable[[list[Any]], BaseException | None] | None = None
        self.answer_override: Callable[[Any], Any] | None = None
        self.fallback_on_call = False
        self.overlapped = False
        self._active = 0
        self._guard = threading.Lock()

    def predict_batch(
        self,
        states: list[Any],
        questions: dict[str, Any],
        batch_size: int | None = None,
        lang: str | None = None,
        sort_by_length: bool = False,
    ) -> list[dict[str, Any]]:
        with self._guard:
            self._active += 1
            self.overlapped = self.overlapped or self._active > 1
        try:
            self.calls.append(len(states))
            if self.delay_s:
                time.sleep(self.delay_s)
            if self.error is not None:
                raise self.error
            if self.fail_when is not None and (exc := self.fail_when(states)) is not None:
                raise exc
            if self.fallback_on_call:
                self.cpu_fallback_count += 1
            (question_id,) = questions
            return [{"answers": {question_id: self._answer(state)}} for state in states]
        finally:
            with self._guard:
                self._active -= 1

    def _answer(self, state: Any) -> Any:
        if self.answer_override is not None:
            return self.answer_override(state)
        text = state if isinstance(state, str) else json.dumps(state)
        visible = " ".join(text.split()[: self.visible_tokens])
        return {"type": "noul", "noul": 0.9 if RELEVANT in visible else 0.1}


def make_scorer(
    agent: FakeAgent | None = None, tokenizer: FakeTokenizer | None = None, **settings: Any
) -> Scorer:
    return Scorer(
        agent or FakeAgent(), tokenizer or FakeTokenizer(), get_preset("default"), Settings(**settings)
    )


@pytest.fixture
def agent() -> FakeAgent:
    return FakeAgent()
