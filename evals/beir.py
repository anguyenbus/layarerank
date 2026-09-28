"""Frozen BM25 candidate pools for BEIR test sets, so every reranker sees identical candidates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATASETS = ("fiqa",)


@dataclass(frozen=True)
class Candidate:
    doc_id: str
    text: str
    bm25: float


@dataclass(frozen=True)
class Pool:
    qid: str
    query: str
    candidates: list[Candidate]
    relevant: dict[str, int]  # every judged-relevant doc for the query, graded


def pool_path(dataset: str, k: int, split: str = "test") -> Path:
    name = dataset if split == "test" else f"{dataset}-{split}"
    return DATA_DIR / f"{name}.bm25-top{k}.jsonl"


def build_pools(dataset: str, k: int, split: str = "test") -> Path:
    """Retrieve BM25 top-k for every query in `split` with at least one relevant document."""
    import bm25s
    from datasets import load_dataset

    corpus = load_dataset(f"BeIR/{dataset}", "corpus", split="corpus")
    queries = {str(r["_id"]): r["text"] for r in load_dataset(f"BeIR/{dataset}", "queries", split="queries")}
    relevant: dict[str, dict[str, int]] = {}
    for row in load_dataset(f"BeIR/{dataset}-qrels", split=split):
        if row["score"] > 0:
            relevant.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])

    doc_ids = [str(i) for i in corpus["_id"]]
    texts = [f"{t}. {x}" if t else x for t, x in zip(corpus["title"], corpus["text"], strict=True)]
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(texts, stopwords="en", show_progress=False), show_progress=False)

    qids = sorted(q for q in relevant if q in queries)
    hits, scores = retriever.retrieve(
        bm25s.tokenize([queries[q] for q in qids], stopwords="en", show_progress=False),
        k=k,
        show_progress=False,
    )
    path = pool_path(dataset, k, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        for row, qid in enumerate(qids):
            candidates = [
                {"doc_id": doc_ids[i], "text": texts[i], "bm25": float(s)}
                for i, s in zip(hits[row], scores[row], strict=True)
            ]
            out.write(
                json.dumps(
                    {"qid": qid, "query": queries[qid], "candidates": candidates, "relevant": relevant[qid]}
                )
                + "\n"
            )
    return path


def load_pools(dataset: str, k: int, limit: int | None = None, split: str = "test") -> list[Pool]:
    """Load answerable pools (a judged-relevant doc is among the candidates).

    Unanswerable pools score zero for every reranker, so they only cost compute. `limit` takes a
    deterministic, id-hashed sample so subsets are reproducible.
    """
    path = pool_path(dataset, k, split)
    if not path.exists():
        build_pools(dataset, k, split)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [r for r in rows if any(c["doc_id"] in r["relevant"] for c in r["candidates"])]
    rows.sort(key=lambda r: hashlib.sha1(f"{dataset}:{r['qid']}".encode()).hexdigest())
    return [
        Pool(r["qid"], r["query"], [Candidate(**c) for c in r["candidates"]], r["relevant"])
        for r in rows[:limit]
    ]
