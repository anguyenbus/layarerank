"""Library API: ranking output and cross-query batching."""

from __future__ import annotations

import pytest

from layareranker.errors import InputError
from layareranker.ranker import LayaReranker
from tests.conftest import RELEVANT, FakeAgent, make_scorer


@pytest.fixture
def reranker(agent: FakeAgent) -> LayaReranker:
    return LayaReranker(scorer=make_scorer(agent))


def test_rerank_orders_best_first_with_input_indices(reranker: LayaReranker) -> None:
    passages = ["noise", f"{RELEVANT} a", "more noise", f"{RELEVANT} b"]
    result = reranker.rerank("query", passages, top_k=3, return_text=True)

    assert [r.index for r in result.results] == [1, 3, 0]
    assert result.results[0].text == passages[1]
    assert result.score_version == reranker.score_version


def test_threshold_drops_low_scores(reranker: LayaReranker) -> None:
    result = reranker.rerank("query", ["noise", f"{RELEVANT}"], threshold=0.5)
    assert [r.index for r in result.results] == [1]


def test_rerank_many_shares_forward_passes_and_matches_single_calls(
    reranker: LayaReranker, agent: FakeAgent
) -> None:
    requests = [("q1", ["noise", f"{RELEVANT}"]), ("q2", [f"{RELEVANT}", "noise", "noise"])]
    together = reranker.rerank_many(requests)
    assert agent.calls == [5]

    separate = [reranker.rerank(q, p) for q, p in requests]
    assert together == separate


@pytest.mark.parametrize(("top_k", "threshold"), [(-1, None), (True, None), (None, float("nan"))])
def test_invalid_cutoffs_are_rejected(reranker: LayaReranker, top_k: object, threshold: object) -> None:
    with pytest.raises(InputError):
        reranker.rerank("query", ["p"], top_k=top_k, threshold=threshold)  # type: ignore[arg-type]
