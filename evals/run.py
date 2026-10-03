"""Score frozen pools with one system; resumable, one JSON line per query.

python -m evals.run --dataset fiqa --system laya:english:noul-dict --queries 50 --k 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evals.beir import load_pools
from evals.systems import make_system

RESULTS_DIR = Path(__file__).parent / "results"


def result_dir(dataset: str, k: int, split: str = "test") -> Path:
    name = dataset if split == "test" else f"{dataset}-{split}"
    return RESULTS_DIR / f"{name}.top{k}"


def result_path(dataset: str, k: int, system: str, split: str = "test") -> Path:
    return result_dir(dataset, k, split) / f"{system.replace(':', '_')}.jsonl"


def run(dataset: str, system_spec: str, k: int, queries: int | None, split: str = "test") -> Path:
    pools = load_pools(dataset, k, queries, split)
    path = result_path(dataset, k, system_spec, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["qid"] for line in path.read_text().splitlines()} if path.exists() else set()
    todo = [p for p in pools if p.qid not in done]
    if not todo:
        return path
    system = make_system(system_spec)
    with path.open("a") as out:
        for n, pool in enumerate(todo, 1):
            start = time.perf_counter()
            scored = system.score(pool)
            elapsed = time.perf_counter() - start
            out.write(
                json.dumps(
                    {
                        "qid": pool.qid,
                        "doc_ids": [c.doc_id for c in pool.candidates],
                        "scores": scored.scores,
                        "truncated": scored.truncated,
                        "elapsed_s": elapsed,
                    }
                )
                + "\n"
            )
            out.flush()
            print(f"{system_spec} {dataset} {len(done) + n}/{len(pools)} {elapsed:.2f}s", flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--system", action="append", required=True)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--queries", type=int)
    parser.add_argument("--split", default="test", help="BEIR qrels split: test, validation or train")
    args = parser.parse_args()
    for spec in args.system:
        run(args.dataset, spec, args.k, args.queries, args.split)


if __name__ == "__main__":
    main()
