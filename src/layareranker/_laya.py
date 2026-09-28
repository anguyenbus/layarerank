"""The only module that touches laya internals.

Token accounting must match exactly how laya encodes a (question, state) row, so this mirrors
`laya.common.build_sequence` using laya's own helpers. It is pinned to laya==0.3.21 and covered
by real-checkpoint tests (`pytest -m model`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from laya.agent import Agent
from laya.common import build_sequence, encode_text, render_options, serialize_state
from laya.router import Router

from .config import Settings
from .errors import ConfigurationError

OPTION_TOKEN_CAP = 48  # build_sequence caps every option description at this many tokens


class AgentLike(Protocol):
    """The subset of `laya.agent.Agent` the engine relies on."""

    cfg: dict[str, Any]
    cpu_fallback_count: int

    def predict_batch(
        self,
        states: list[Any],
        questions: dict[str, Any],
        batch_size: int | None = None,
        lang: str | None = None,
        sort_by_length: bool = False,
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class HeadLength:
    """Tokens a question occupies before the state, as encoded and as it would be uncut."""

    encoded: int
    untruncated: int


class Tokenizer(Protocol):
    """Token accounting the engine needs; `LayaTokenizer` in production, a fake in tests."""

    @property
    def max_len(self) -> int: ...

    def head_length(self, question: dict[str, Any]) -> HeadLength: ...

    def count_state(self, state: Any) -> int: ...

    def split_text(self, text: str, window_tokens: int, stride_tokens: int) -> list[str]: ...


class LayaTokenizer:
    """Counts tokens exactly as a laya agent will when it encodes a row."""

    def __init__(self, agent: Any) -> None:
        self._tok = agent.tok
        self._max_len = int(agent.cfg.get("max_len", 512))
        self._head_max_len = int(agent.cfg.get("head_max_len", 192))

    @property
    def max_len(self) -> int:
        return self._max_len

    def _encode(self, text: str) -> list[int]:
        ids: list[int] = encode_text(
            self._tok, text.replace(self._tok.mask_token, " "), add_special_tokens=False
        )["input_ids"]
        return ids

    def head_length(self, question: dict[str, Any]) -> HeadLength:
        internal = Agent._to_internal(question)
        # An empty state leaves [CLS] head [SEP] options [SEP] + [SEP]; everything but the last token.
        seq, markers = build_sequence(
            self._tok, "", internal, self._max_len, self._head_max_len, state_ids=[]
        )
        options = render_options(internal)
        if len(markers) != len(options):
            raise ConfigurationError("question options do not fit the checkpoint's head budget")
        instruction = self._encode(f"{internal['t']} question: {internal['ins']}")
        option_tokens = sum(1 + min(OPTION_TOKEN_CAP, len(self._encode(" " + o))) for o in options)
        return HeadLength(encoded=len(seq) - 1, untruncated=1 + len(instruction) + 1 + option_tokens + 1)

    def count_state(self, state: Any) -> int:
        return len(self._encode(serialize_state(state)))

    def split_text(self, text: str, window_tokens: int, stride_tokens: int) -> list[str]:
        ids = self._encode(text)
        if len(ids) <= window_tokens:
            return [text]
        starts = range(0, max(1, len(ids) - window_tokens + stride_tokens), stride_tokens)
        return [self._tok.decode(ids[s : s + window_tokens]) for s in starts]


def load_agent(settings: Settings) -> Any:
    """Load exactly one pinned checkpoint through laya's Router (cache and digest verification).

    A `model` that names a local directory is loaded directly, e.g. a fine-tuned checkpoint.
    """
    if Path(settings.model).is_dir():
        return Agent(settings.model, device=settings.device)
    router = Router(device=settings.device, revision=settings.revision, max_loaded=1)
    return router.load(settings.model)
