# LayaReranker

Batched, token-aware document reranking on [Laya](https://huggingface.co/convaiinnovations/laya)
decision models. Every (query, passage) pair is scored independently in shared forward passes,
and any passage the model could not see in full is reported as `truncated`.

`LAYARERANKER_MODEL` takes a checkpoint name (`english`, `multilingual`, `typed-decisions`) or a local
checkpoint directory, such as one written by `evals.finetune`.

## Results (FiQA, 385 test queries, BM25 top-30)

| System | nDCG@10 | ms / pair (RTX 3090 Ti) |
|---|---|---|
| BM25 (no rerank) | 0.392 | — |
| Laya `english`, `noul-dict-bare` prompt | 0.422 | 9.8 |
| Laya fine-tuned on FiQA train (full model) | **0.580** | 9.8 |
| bge-reranker-v2-m3 | 0.578 | 12.6 |
| Laya fine-tuned + bge, 50/50 score blend | 0.600 | 22.4 |

Off the shelf, Laya's gain over BM25 is not significant. Fully fine-tuned, it ties bge and is ~22%
faster, but it is a FiQA specialist until tested on other domains. Details, confidence intervals and
analysis: [evals/reports/fiqa-2026-09-28.md](evals/reports/fiqa-2026-09-28.md).

## Development

```bash
uv sync --extra cpu --extra server      # CPU torch for local work and CI
uv run pytest                           # hermetic suite (fake agent)
HF_HOME=/path/to/hf uv run pytest -m model   # real-checkpoint tests
```

## Evaluation

Pass the extras on every `uv run` (a bare `uv run` re-syncs them away); use `--extra gpu` on a CUDA
machine, `--extra cpu` otherwise. Candidate pools are built from BEIR on
first use and cached in `evals/data/`; per-query scores are resumable and land in `evals/results/`.

```bash
# score systems: bm25, bge, or laya:<model>:<preset>[:window_max]
uv run --extra gpu --extra eval python -m evals.run --dataset fiqa --k 30 --system bm25 --system laya:english:noul-dict-bare
# compare every scored system and apply the gate (beats BM25 on nDCG@10, 95% CI above 0)
uv run --extra gpu --extra eval python -m evals.report --dataset fiqa --k 30 --candidate laya:english:noul-dict-bare
```

## Fine-tuning

`evals.finetune` trains a Laya checkpoint as a reranker on a BEIR train split (listwise loss over BM25
hard negatives), selects on the dev split, and writes the best checkpoint to `evals/models/<name>/`
(not in git). Score it with the spec `laya:<name>:<preset>`.

```bash
# full model, ~1 h on one RTX 3090 Ti
uv run --extra gpu --extra eval python -m evals.finetune --dataset fiqa --out ft-fiqa-full --epochs 3 --evals-per-epoch 4 --patience 4
uv run --extra gpu --extra eval python -m evals.run --dataset fiqa --k 30 --system laya:ft-fiqa-full:noul-dict-bare
```

Add `--head-only` to train only the decision head. It is faster, but gave a much smaller gain on FiQA
(0.448 nDCG@10).
