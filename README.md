# LayaReranker

Batched, token-aware document reranking on [Laya](https://huggingface.co/convaiinnovations/laya)
decision models. Every (query, passage) pair is scored independently in shared forward passes,
and any passage the model could not see in full is reported as `truncated`.

## Development

```bash
uv sync --extra cpu --extra server      # CPU torch for local work and CI
uv run pytest                           # hermetic suite (fake agent)
HF_HOME=/path/to/hf uv run pytest -m model   # real-checkpoint tests
```
