#!/usr/bin/env bash
# Phase 1 preset sweep: every system on the same deterministic query sample. Resumable.
set -euo pipefail
QUERIES="${QUERIES:-50}"
K="${K:-30}"
DATASETS="${DATASETS:-fiqa}"
SYSTEMS="${SYSTEMS:-bm25 laya:english:noul-dict laya:english:score-dict bge}"

for dataset in $DATASETS; do
  for system in $SYSTEMS; do
    uv run --extra cpu --extra eval python -m evals.run --dataset "$dataset" --k "$K" --queries "$QUERIES" \
      --system "$system"
  done
done
