"""Turn (query, passages) into scoring units with exact token accounting. No model calls."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from ._laya import Tokenizer
from .config import Settings
from .errors import ConfigurationError, InputError
from .prompt import Preset

# JSON escaping and window re-decoding can shift a window's size by a few tokens.
WINDOW_MARGIN_TOKENS = 8


@dataclass(frozen=True)
class Budget:
    """How much of a row the question takes and how many state tokens the model can see."""

    max_len: int
    head_tokens: int
    room: int

    def row_tokens(self, state_tokens: int) -> int:
        """Length of the encoded row: head, visible state, trailing separator."""
        return min(self.max_len, self.head_tokens + min(state_tokens, self.room) + 1)


@dataclass(frozen=True)
class Unit:
    """One (question, state) row. A passage becomes one unit, or several under `window_max`."""

    passage_index: int
    window: int
    state: Any
    state_tokens: int
    row_tokens: int
    truncated: bool


def plan_budget(tokenizer: Tokenizer, preset: Preset, settings: Settings) -> Budget:
    """Fail fast if the preset's instructions would be cut or leave too little room for passages."""
    head = tokenizer.head_length(preset.question)
    if head.encoded < head.untruncated:
        raise ConfigurationError(
            f"preset {preset.name!r} instructions are cut by the head budget "
            f"({head.untruncated} tokens needed, {head.encoded} kept); shorten them"
        )
    room = tokenizer.max_len - head.encoded - 1
    needed = settings.max_query_tokens + settings.min_passage_tokens
    if room < needed:
        raise ConfigurationError(
            f"preset {preset.name!r} leaves {room} state tokens; "
            f"max_query_tokens + min_passage_tokens needs {needed}"
        )
    return Budget(max_len=tokenizer.max_len, head_tokens=head.encoded, room=room)


def validate_request(query: str, passages: Sequence[str], settings: Settings) -> None:
    """Reject malformed or oversized requests before any tokenization."""
    if not isinstance(query, str) or not query.strip():
        raise InputError("query must be a non-empty string")
    if len(query) > settings.max_query_chars:
        raise InputError(f"query exceeds {settings.max_query_chars} characters")
    if isinstance(passages, str) or not isinstance(passages, Sequence):
        raise InputError("passages must be a list of strings")
    if len(passages) > settings.max_passages:
        raise InputError(f"at most {settings.max_passages} passages per request")
    for index, passage in enumerate(passages):
        if not isinstance(passage, str):
            raise InputError(f"passage {index} is not a string")
        if len(passage) > settings.max_passage_chars:
            raise InputError(f"passage {index} exceeds {settings.max_passage_chars} characters")


def make_units(
    query: str,
    passages: Sequence[str],
    *,
    preset: Preset,
    tokenizer: Tokenizer,
    budget: Budget,
    settings: Settings,
) -> list[Unit]:
    """Build every unit for one request. Units keep passage order; windows follow their passage."""
    query_tokens = tokenizer.count_state(preset.build_state(query, ""))
    if query_tokens > settings.max_query_tokens:
        raise InputError(f"query uses {query_tokens} tokens; the limit is {settings.max_query_tokens}")

    units: list[Unit] = []
    for index, passage in enumerate(passages):
        state = preset.build_state(query, passage)
        state_tokens = tokenizer.count_state(state)
        if state_tokens <= budget.room or settings.long_doc == "truncate":
            units.append(_unit(index, 0, state, state_tokens, budget))
        else:
            units.extend(
                _windows(
                    index,
                    query=query,
                    passage=passage,
                    query_tokens=query_tokens,
                    preset=preset,
                    tokenizer=tokenizer,
                    budget=budget,
                    settings=settings,
                )
            )
    return units


def _unit(index: int, window: int, state: Any, state_tokens: int, budget: Budget) -> Unit:
    return Unit(
        passage_index=index,
        window=window,
        state=state,
        state_tokens=state_tokens,
        row_tokens=budget.row_tokens(state_tokens),
        truncated=state_tokens > budget.room,
    )


def _windows(
    index: int,
    *,
    query: str,
    passage: str,
    query_tokens: int,
    preset: Preset,
    tokenizer: Tokenizer,
    budget: Budget,
    settings: Settings,
) -> list[Unit]:
    window_tokens = budget.room - query_tokens - WINDOW_MARGIN_TOKENS
    stride_tokens = max(1, int(window_tokens * settings.window_stride_ratio))
    pieces = tokenizer.split_text(passage, window_tokens, stride_tokens)
    capped = len(pieces) > settings.max_windows_per_doc
    units = []
    for window, piece in enumerate(pieces[: settings.max_windows_per_doc]):
        state = preset.build_state(query, piece)
        unit = _unit(index, window, state, tokenizer.count_state(state), budget)
        units.append(replace(unit, truncated=True) if capped else unit)
    return units
