"""Fine-tune a Laya checkpoint as a reranker on a BEIR train split, selecting on its dev split.

python -m evals.finetune --dataset fiqa --out ft-fiqa-head --head-only

Rows are encoded exactly as the engine encodes them at inference (same preset, same truncation),
and every query's candidates are scored together so the loss can be listwise: a softmax over the
candidates' true-minus-false logit margins, plus a smaller pointwise term. The checkpoint with the
best dev nDCG@10 is written to `evals/models/<out>/` in the layout `laya.Agent(path)` loads.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from laya.agent import Agent
from laya.common import collate_items
from safetensors.torch import load_file, save_file

from evals.beir import Pool, load_pools
from evals.metrics import ndcg_at
from layareranker.config import Settings
from layareranker.engine import Scorer
from layareranker.prompt import QUESTION_ID

MODELS_DIR = Path(__file__).parent / "models"
HEAD_PREFIXES = ("head.", "type_emb.", "scorer.")


def encode(agent: Any, scorer: Scorer, query: str, passage: str) -> dict[str, Any]:
    """One (question, state) row, tokenized by laya exactly as `predict_batch` would."""
    internal = {QUESTION_ID: Agent._to_internal(scorer.preset.question)}
    state = scorer.preset.build_state(query, passage)
    (item,) = agent._encode_state(state, [QUESTION_ID], internal)
    return item


def margins(
    model: torch.nn.Module, items: list[dict[str, Any]], pad_id: int, device: torch.device, head_only: bool
) -> torch.Tensor:
    """Logit margin z_true - z_false per row (monotone in the noul probability at any temperature)."""
    batch = collate_items([items], pad_id)
    args = [
        batch[k].to(device) for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
    ]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(*args, detach_encoder=head_only)
    return logits[:, 1] - logits[:, 0]


def evaluate(scorer: Scorer, pools: list[Pool]) -> float:
    scorer.agent.model.eval()
    scores = []
    for pool in pools:
        result = scorer.score(pool.query, [c.text for c in pool.candidates])
        order = sorted(range(len(pool.candidates)), key=lambda i: (-result.passages[i].score, i))
        scores.append(ndcg_at([pool.candidates[i].doc_id for i in order], pool.relevant))
    return mean(scores)


def save_checkpoint(out: Path, agent: Any, state: dict[str, torch.Tensor], meta: dict[str, Any]) -> None:
    """Write `state` in the base checkpoint's layout and dtypes, so `laya.Agent(out)` loads it."""
    src = Path(
        snapshot_download(
            "convaiinnovations/laya",
            revision=getattr(agent, "revision", None),
            allow_patterns=["encoder/*", "tokenizer/*", "rl_agent_config.json", "model.safetensors"],
        )
    )
    out.mkdir(parents=True, exist_ok=True)
    for part in ("encoder", "tokenizer"):
        shutil.copytree(src / part, out / part, dirs_exist_ok=True)
    shutil.copy(src / "rl_agent_config.json", out / "rl_agent_config.json")
    dtypes = {n: t.dtype for n, t in load_file(src / "model.safetensors").items()}
    save_file(
        {n: t.to(dtypes[n]).contiguous() for n, t in state.items() if n in dtypes}, out / "model.safetensors"
    )
    (out / "finetune.json").write_text(json.dumps(meta, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="fiqa")
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--preset", default="noul-dict-bare")
    parser.add_argument("--out", required=True, help="checkpoint name under evals/models/")
    parser.add_argument("--head-only", action="store_true", help="freeze the encoder")
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--negatives", type=int, default=7, help="hard negatives sampled per query per epoch")
    parser.add_argument("--queries-per-step", type=int, default=8)
    parser.add_argument("--micro-queries", type=int, default=4)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1e-5)
    parser.add_argument("--pointwise-weight", type=float, default=0.5)
    parser.add_argument("--evals-per-epoch", type=int, default=2)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def encode_pools(agent: Any, scorer: Scorer, pools: list[Pool]) -> list[dict[str, list[dict[str, Any]]]]:
    """Every candidate of every pool, encoded once and split into judged-relevant and the rest."""
    return [
        {
            "pos": [
                encode(agent, scorer, p.query, c.text)
                for c in p.candidates
                if p.relevant.get(c.doc_id, 0) > 0
            ],
            "neg": [
                encode(agent, scorer, p.query, c.text)
                for c in p.candidates
                if p.relevant.get(c.doc_id, 0) <= 0
            ],
        }
        for p in pools
    ]


def make_optimizer(
    model: torch.nn.Module, args: argparse.Namespace, total: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """AdamW over the head (and the encoder unless head-only), 5% warm-up then cosine decay."""
    for name, param in model.named_parameters():
        param.requires_grad = not args.head_only or name.startswith(HEAD_PREFIXES)
    head = [p for n, p in model.named_parameters() if n.startswith(HEAD_PREFIXES)]
    groups = [{"params": head, "lr": args.lr_head}]
    if not args.head_only:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        encoder = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
        groups.append({"params": encoder, "lr": args.lr_encoder})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    warmup = max(1, total // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))
    )
    return optimizer, schedule


def train_step(
    model: torch.nn.Module,
    encoded: list[dict[str, list[dict[str, Any]]]],
    queries: list[int],
    *,
    args: argparse.Namespace,
    pad_id: int,
    device: torch.device,
) -> float:
    """Accumulate gradients over `queries` (micro-batched by whole queries); returns the mean loss."""
    model.train()
    if args.head_only:
        model.encoder.eval()  # frozen: no dropout noise in the features the head learns from
    losses = []
    for start in range(0, len(queries), args.micro_queries):
        rows, spans, labels = [], [], []
        for qi in queries[start : start + args.micro_queries]:
            pos, neg = encoded[qi]["pos"], encoded[qi]["neg"]
            picked = pos + random.sample(neg, min(args.negatives, len(neg)))
            spans.append((len(rows), len(rows) + len(picked), len(pos)))
            labels += [1.0] * len(pos) + [0.0] * (len(picked) - len(pos))
            rows += picked
        s = margins(model, rows, pad_id, device, args.head_only).float()
        listwise = torch.stack(
            [torch.logsumexp(s[a : a + n], 0) - torch.logsumexp(s[a:b], 0) for a, b, n in spans]
        )
        pointwise = F.binary_cross_entropy_with_logits(s, torch.tensor(labels, device=device))
        loss = -listwise.mean() + args.pointwise_weight * pointwise
        (loss * len(spans) / len(queries)).backward()
        losses.append(loss.item())
    return mean(losses)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    settings = Settings(
        model="english", preset=args.preset, max_query_tokens=128, max_passage_chars=10**6, warmup=False
    )
    scorer = Scorer.load(settings)
    agent, model = scorer.agent, scorer.agent.model

    train = load_pools(args.dataset, args.k, split="train")
    dev = load_pools(args.dataset, args.k, split="validation")
    print(f"train {len(train)} queries | dev {len(dev)} queries", flush=True)
    t0 = time.time()
    encoded = encode_pools(agent, scorer, train)
    print(f"encoded train rows in {time.time() - t0:.0f}s", flush=True)

    steps_per_epoch = math.ceil(len(encoded) / args.queries_per_step)
    total = int(steps_per_epoch * args.epochs)
    optimizer, schedule = make_optimizer(model, args, total)
    eval_every = max(1, steps_per_epoch // args.evals_per_epoch)

    best, best_step, stale = evaluate(scorer, dev), 0, 0
    best_state = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}
    print(f"step 0 | dev ndcg@10 {best:.4f} (untrained)", flush=True)
    history = [{"step": 0, "epoch": 0.0, "dev_ndcg@10": best}]
    step, t0 = 0, time.time()
    order: list[int] = []
    while step < total and stale < args.patience:
        if not order:
            order = random.sample(range(len(encoded)), len(encoded))
        queries, order = order[: args.queries_per_step], order[args.queries_per_step :]
        loss = train_step(
            model, encoded, queries, args=args, pad_id=agent.tok.pad_token_id, device=agent.device
        )
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % 25 == 0:
            lr = schedule.get_last_lr()[0]
            print(
                f"step {step}/{total} | loss {loss:.4f} | lr {lr:.2e} | {time.time() - t0:.0f}s", flush=True
            )
        if step % eval_every == 0 or step == total:
            score = evaluate(scorer, dev)
            history.append({"step": step, "epoch": round(step / steps_per_epoch, 2), "dev_ndcg@10": score})
            if score > best:
                best, best_step, stale = score, step, 0
                best_state = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}
            else:
                stale += 1
            epoch = step / steps_per_epoch
            print(
                f"step {step} (epoch {epoch:.2f}) | dev ndcg@10 {score:.4f} | best {best:.4f} @ {best_step}",
                flush=True,
            )

    out = MODELS_DIR / args.out
    meta = {
        "args": vars(args),
        "best_step": best_step,
        "best_dev_ndcg@10": best,
        "steps_per_epoch": steps_per_epoch,
        "history": history,
    }
    save_checkpoint(out, agent, best_state, meta)
    print(f"saved best checkpoint (step {best_step}, dev ndcg@10 {best:.4f}) to {out}", flush=True)


if __name__ == "__main__":
    main()
