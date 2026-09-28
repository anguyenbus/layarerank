"""Compare systems on the same queries and print the Phase 1 go/no-go verdict.

python -m evals.report --dataset fiqa --k 30 --candidate laya:english:noul-dict
"""

from __future__ import annotations

import argparse
import json
from statistics import mean

from evals.beir import load_pools
from evals.metrics import METRICS, paired_bootstrap
from evals.run import RESULTS_DIR, result_path


def per_query(dataset: str, k: int, system: str) -> dict[str, dict[str, float]]:
    """Metric values per query for one system, from its stored scores (ties broken by BM25 order)."""
    relevant = {p.qid: p.relevant for p in load_pools(dataset, k)}
    rows = [json.loads(line) for line in result_path(dataset, k, system).read_text().splitlines()]
    out = {}
    for row in rows:
        order = sorted(range(len(row["scores"])), key=lambda i: (-row["scores"][i], i))
        ranked = [row["doc_ids"][i] for i in order]
        values = {name: fn(ranked, relevant[row["qid"]]) for name, fn in METRICS.items()}
        values["truncated_frac"] = row["truncated"] / len(row["scores"])
        values["ms_per_pair"] = 1000 * row["elapsed_s"] / len(row["scores"])
        scores = [row["scores"][i] for i in order]
        values["tie_at_10"] = float(len(scores) > 10 and scores[9] == scores[10])
        out[row["qid"]] = values
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--candidate", required=True, help="the system being gated")
    args = parser.parse_args()

    folder = RESULTS_DIR / f"{args.dataset}.top{args.k}"
    systems = sorted(p.stem for p in folder.glob("*.jsonl"))
    data = {s: per_query(args.dataset, args.k, s) for s in systems}
    shared = sorted(set.intersection(*(set(d) for d in data.values())))
    print(f"{args.dataset}: {len(shared)} queries scored by all of {systems}\n")
    columns = [*METRICS, "truncated_frac", "tie_at_10", "ms_per_pair"]
    print(f"{'system':32s}" + "".join(f"{c:>15s}" for c in columns))
    for s in systems:
        print(f"{s:32s}" + "".join(f"{mean(data[s][q][c] for q in shared):15.4f}" for c in columns))

    candidate = args.candidate.replace(":", "_")
    print()
    verdict = None
    for baseline in (b for b in systems if b != candidate):
        a = [data[candidate][q]["ndcg@10"] for q in shared]
        b = [data[baseline][q]["ndcg@10"] for q in shared]
        delta, low, high = paired_bootstrap(a, b)
        print(f"ndcg@10 {candidate} - {baseline}: {delta:+.4f}  95% CI [{low:+.4f}, {high:+.4f}]")
        if baseline == "bm25":
            verdict = "GO" if low > 0 else "NO-GO"
    if verdict:
        print(f"\nGate (beats BM25 on ndcg@10, CI above 0): {verdict}")


if __name__ == "__main__":
    main()
