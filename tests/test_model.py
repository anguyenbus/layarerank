"""Real-checkpoint tests. Run with `HF_HOME=<cache> pytest -m model`; they guard the laya pin."""

from __future__ import annotations

import pytest
from laya.agent import Agent
from laya.common import serialize_state

from layareranker.config import Settings
from layareranker.engine import Scorer
from layareranker.prompt import PRESETS, QUESTION_ID

pytestmark = pytest.mark.model

QUERY = "How long do I have to return an online order to ACME Shop?"
ANSWER = "ACME Shop accepts online returns within 30 days of delivery."
DISTRACTORS = [
    "The museum's east wing reopened with a sculpture garden and longer weekend hours.",
    "ACME Shop in-store purchases can be returned within 14 days of purchase.",
    'Section 8-1 "general deductions" \\ allows losses incurred in gaining assessable income.',
]


@pytest.fixture(scope="module")
def scorer() -> Scorer:
    return Scorer.load(Settings(device="cpu"))


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_no_preset_instruction_is_cut_by_the_head_budget(scorer: Scorer, name: str) -> None:
    head = scorer.tokenizer.head_length(PRESETS[name].question)
    assert head.encoded == head.untruncated


def test_row_length_and_truncation_match_laya_encoding(scorer: Scorer) -> None:
    internal = {QUESTION_ID: Agent._to_internal(scorer.preset.question)}
    for words in range(scorer.budget.room - 60, scorer.budget.room + 20, 3):
        unit = scorer.prepare(QUERY, [" ".join(["returns"] * words)])[0]
        (item,) = scorer.agent._encode_state(unit.state, [QUESTION_ID], internal)  # type: ignore[attr-defined]
        state_tokens = len(scorer.tokenizer._encode(serialize_state(unit.state)))  # type: ignore[attr-defined]
        assert len(item["ids"]) == unit.row_tokens
        assert (state_tokens > scorer.budget.room) == unit.truncated


def test_batched_scores_equal_one_at_a_time_scores(scorer: Scorer) -> None:
    passages = [ANSWER, *DISTRACTORS]
    batched = [p.score for p in scorer.score(QUERY, passages).passages]
    single = [scorer.score(QUERY, [p]).passages[0].score for p in passages]
    assert batched == pytest.approx(single, abs=1e-4)


def test_a_passage_scores_the_same_at_every_position(scorer: Scorer) -> None:
    scores = set()
    for position in range(len(DISTRACTORS) + 1):
        passages = list(DISTRACTORS)
        passages.insert(position, ANSWER)
        scores.add(round(scorer.score(QUERY, passages).passages[position].score, 4))
    assert len(scores) == 1
