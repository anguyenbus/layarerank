"""Scoring engine: batching, strict answers, error types, truncation and windows."""

from __future__ import annotations

import math

import pytest

from layareranker.aggregate import PassageScore, rank
from layareranker.errors import BackendFatal, ConfigurationError, InputError, ScoringError
from tests.conftest import RELEVANT, FakeAgent, FakeTokenizer, make_scorer


def test_batched_scores_match_passage_order_in_one_forward(agent: FakeAgent) -> None:
    passages = ["noise one", f"{RELEVANT} answer", "noise two", f"also {RELEVANT}"]
    result = make_scorer(agent).score("query", passages)

    assert [p.score for p in result.passages] == [0.1, 0.9, 0.1, 0.9]
    assert agent.calls == [4]
    assert not result.degraded


def test_forward_passes_respect_row_and_token_caps(agent: FakeAgent) -> None:
    scorer = make_scorer(agent, max_batch_rows=3, max_batch_tokens=1024)
    long_passage = "word " * 400  # rows of ~440 tokens: only two fit in 1024 tokens
    scorer.score("query", ["short"] * 5 + [long_passage] * 3)

    assert sum(agent.calls) == 8
    assert max(agent.calls) <= 3
    assert agent.calls[-2:] == [2, 1]


@pytest.mark.parametrize(
    "answer",
    [None, {"type": "noul"}, {"noul": math.nan}, {"noul": 1.5}, {"noul": "0.9"}, {"noul": True}],
)
def test_unusable_answers_raise_instead_of_defaulting(agent: FakeAgent, answer: object) -> None:
    agent.answer_override = lambda state: answer
    with pytest.raises(ScoringError):
        make_scorer(agent).score("query", ["passage"])


def test_fatal_cuda_errors_become_backend_fatal(agent: FakeAgent) -> None:
    agent.error = RuntimeError("CUDA error: device-side assert triggered")
    with pytest.raises(BackendFatal):
        make_scorer(agent).score("query", ["passage"])


def test_other_errors_keep_their_type(agent: FakeAgent) -> None:
    agent.error = RuntimeError("shape mismatch")
    with pytest.raises(RuntimeError, match="shape mismatch"):
        make_scorer(agent).score("query", ["passage"])


def test_cpu_fallback_marks_results_degraded(agent: FakeAgent) -> None:
    agent.fallback_on_call = True
    assert make_scorer(agent).score("query", ["passage"]).degraded


def test_passages_beyond_the_window_are_flagged_truncated(agent: FakeAgent) -> None:
    scorer = make_scorer(agent, tokenizer=FakeTokenizer(max_len=512, head_tokens=40))
    visible, hidden = "short passage", "word " * 500
    result = scorer.score("query", [visible, hidden])

    assert [p.truncated for p in result.passages] == [False, True]


def test_window_max_scores_the_best_window_of_a_long_passage(agent: FakeAgent) -> None:
    scorer = make_scorer(agent, long_doc="window_max")
    passage = "filler " * 800 + RELEVANT  # the relevant token is past the first window
    result = scorer.score("query", [passage])

    (scored,) = result.passages
    assert scored.score == 0.9
    assert scored.windows > 1
    assert not scored.truncated


def test_truncate_policy_scores_only_what_fits(agent: FakeAgent) -> None:
    result = make_scorer(agent).score("query", ["filler " * 800 + RELEVANT])

    assert result.passages[0].score == 0.1
    assert result.passages[0].truncated


def test_instructions_cut_by_the_head_budget_are_rejected() -> None:
    with pytest.raises(ConfigurationError, match="cut by the head budget"):
        make_scorer(tokenizer=FakeTokenizer(head_tokens=150, untruncated_head=190))


def test_budget_too_small_for_query_and_passage_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="state tokens"):
        make_scorer(tokenizer=FakeTokenizer(max_len=160, head_tokens=40))


@pytest.mark.parametrize(
    ("query", "passages"),
    [("", ["p"]), ("   ", ["p"]), ("q", "not a list"), ("q", ["ok", 3]), ("word " * 200, ["p"])],
)
def test_invalid_requests_raise_input_error(query: str, passages: object) -> None:
    with pytest.raises(InputError):
        make_scorer().score(query, passages)  # type: ignore[arg-type]


def test_rank_breaks_ties_by_input_order() -> None:
    scores = [PassageScore(i, s, False, 1) for i, s in enumerate([0.5, 0.9, 0.5, 0.9, 0.1])]

    assert [s.index for s in rank(scores)] == [1, 3, 0, 2, 4]
    assert [s.index for s in rank(scores, top_k=2)] == [1, 3]
    assert [s.index for s in rank(scores, threshold=0.5)] == [1, 3, 0, 2]
