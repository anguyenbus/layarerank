"""Fine-tune a decision model (Strands Decider, Kai) as a reranker on a BEIR train split.

python -m evals.finetune_decider --dataset fiqa --out decider-ft-fiqa --preset noul-question
python -m evals.finetune_decider --family kai --lr-backbone 2e-5 --out kai-ft-fiqa --preset noul-question

The same recipe as `evals.finetune` (BM25 hard negatives, a listwise softmax over each query's
true-minus-false margins plus a smaller pointwise term, best dev nDCG@10 kept), applied to what
each family trains: Decider's LoRA adapter and pointer head with the 2B base frozen, or all of Kai
but its token embeddings. The best checkpoint is written to `evals/models/<out>/` in the layout the
family loads; score it with the spec `<family>:<out>:<preset>`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F

from evals.beir import Pool, load_pools
from evals.decider import MODELS_DIR, Decider, Row
from evals.kai import Kai
from evals.metrics import ndcg_at

# family -> (system class, default starting checkpoint)
FAMILIES: dict[str, tuple[type[Decider] | type[Kai], str]] = {
    "decider": (Decider, "2b"),
    "kai": (Kai, "0.6b"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="fiqa")
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--family", default="decider", choices=sorted(FAMILIES))
    parser.add_argument("--model", help="starting checkpoint, as in the system spec (default: the family's)")
    parser.add_argument("--preset", required=True, help="a noul preset from evals.decider")
    parser.add_argument("--out", required=True, help="checkpoint name under evals/models/")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--negatives", type=int, default=7, help="hard negatives sampled per query per epoch")
    parser.add_argument("--queries-per-step", type=int, default=8)
    parser.add_argument("--micro-queries", type=int, default=1)
    parser.add_argument("--max-row-tokens", type=int, default=512, help="training rows only")
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument(
        "--lr-backbone", "--lr-lora", type=float, default=1e-4, help="Decider's LoRA adapter, Kai's backbone"
    )
    parser.add_argument("--pointwise-weight", type=float, default=0.5)
    parser.add_argument("--evals-per-epoch", type=int, default=3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--dev-queries", type=int, help="select on a sample of the dev queries")
    parser.add_argument("--max-steps", type=int, help="cap the run (timing probe)")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    args.model = args.model or FAMILIES[args.family][1]
    return args


def evaluate(decider: Decider | Kai, pools: list[Pool]) -> float:
    decider.model.eval()
    scores = []
    for pool in pools:
        scored = decider.score(pool).scores
        order = sorted(range(len(pool.candidates)), key=lambda i: (-scored[i], i))
        scores.append(ndcg_at([pool.candidates[i].doc_id for i in order], pool.relevant))
    return mean(scores)


def encode_pools(decider: Decider | Kai, pools: list[Pool], max_tokens: int) -> list[dict[str, list[Row]]]:
    """Every candidate of every pool, encoded once and split into judged-relevant and the rest."""
    encoded = []
    for p in pools:
        rows = [
            (decider.encode(p.query, c.text, max_tokens), p.relevant.get(c.doc_id, 0) > 0)
            for c in p.candidates
        ]
        encoded.append({"pos": [r for r, rel in rows if rel], "neg": [r for r, rel in rows if not rel]})
    return encoded


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def make_optimizer(
    decider: Decider | Kai, args: argparse.Namespace, total: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """AdamW over the family's trainable parameters, 5% warm-up then cosine decay."""
    optimizer = torch.optim.AdamW(decider.param_groups(args.lr_head, args.lr_backbone))
    warmup = max(1, total // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))
    )
    return optimizer, schedule


def train_step(
    decider: Decider | Kai, encoded: list[dict[str, list[Row]]], queries: list[int], args: argparse.Namespace
) -> float:
    """Accumulate gradients over `queries` (micro-batched by whole queries); returns the mean loss."""
    decider.model.train()
    losses = []
    for start in range(0, len(queries), args.micro_queries):
        rows: list[Row] = []
        spans, labels = [], []
        for qi in queries[start : start + args.micro_queries]:
            pos, neg = encoded[qi]["pos"], encoded[qi]["neg"]
            picked = pos + random.sample(neg, min(args.negatives, len(neg)))
            spans.append((len(rows), len(rows) + len(picked), len(pos)))
            labels += [1.0] * len(pos) + [0.0] * (len(picked) - len(pos))
            rows += picked
        s = decider.margins(rows).float()
        listwise = torch.stack(
            [torch.logsumexp(s[a : a + n], 0) - torch.logsumexp(s[a:b], 0) for a, b, n in spans]
        )
        pointwise = F.binary_cross_entropy_with_logits(s, torch.tensor(labels, device=s.device))
        loss = -listwise.mean() + args.pointwise_weight * pointwise
        (loss * len(spans) / len(queries)).backward()
        losses.append(loss.item())
    return mean(losses)


def save_checkpoint(
    decider: Decider | Kai, state: dict[str, torch.Tensor], name: str, meta: dict[str, Any]
) -> None:
    """Restore the best trainable weights and write them in the layout the family loads."""
    decider.model.load_state_dict(state, strict=False)
    out = MODELS_DIR / name
    decider.save(out)
    (out / "finetune.json").write_text(json.dumps(meta, indent=2))
    print(f"saved best checkpoint (step {meta['best_step']}) to {out}", flush=True)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    decider = FAMILIES[args.family][0](f"{args.family}:{args.model}:{args.preset}")
    model = decider.model

    train = load_pools(args.dataset, args.k, split="train")
    dev = load_pools(args.dataset, args.k, args.dev_queries, split="validation")
    print(f"train {len(train)} queries | dev {len(dev)} queries", flush=True)
    t0 = time.time()
    encoded = encode_pools(decider, train, args.max_row_tokens)
    cut = sum(r.truncated for e in encoded for r in e["pos"] + e["neg"])
    print(f"encoded train rows in {time.time() - t0:.0f}s ({cut} cut at {args.max_row_tokens})", flush=True)

    steps_per_epoch = math.ceil(len(encoded) / args.queries_per_step)
    total = min(int(steps_per_epoch * args.epochs), args.max_steps or 10**9)
    optimizer, schedule = make_optimizer(decider, args, total)
    print(
        f"trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True
    )
    eval_every = max(1, steps_per_epoch // args.evals_per_epoch)

    best, best_step, stale = evaluate(decider, dev), 0, 0
    best_state = trainable_state(model)
    print(f"step 0 | dev ndcg@10 {best:.4f} (untrained)", flush=True)
    history = [{"step": 0, "epoch": 0.0, "dev_ndcg@10": best}]
    step, t0 = 0, time.time()
    order: list[int] = []
    while step < total and stale < args.patience:
        if not order:
            order = random.sample(range(len(encoded)), len(encoded))
        queries, order = order[: args.queries_per_step], order[args.queries_per_step :]
        loss = train_step(decider, encoded, queries, args)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % args.log_every == 0:
            lr, peak = schedule.get_last_lr()[0], torch.cuda.max_memory_allocated() >> 20
            elapsed = time.time() - t0
            print(
                f"step {step}/{total} | loss {loss:.4f} | lr {lr:.2e} | {elapsed:.0f}s | {peak} MiB",
                flush=True,
            )
        if step % eval_every == 0 or step == total:
            score = evaluate(decider, dev)
            history.append({"step": step, "epoch": round(step / steps_per_epoch, 2), "dev_ndcg@10": score})
            if score > best:
                best, best_step, stale = score, step, 0
                best_state = trainable_state(model)
            else:
                stale += 1
            epoch = step / steps_per_epoch
            print(
                f"step {step} (epoch {epoch:.2f}) | dev ndcg@10 {score:.4f} | best {best:.4f} @ {best_step}",
                flush=True,
            )

    meta = {
        "args": vars(args),
        "best_step": best_step,
        "best_dev_ndcg@10": best,
        "steps_per_epoch": steps_per_epoch,
        "history": history,
    }
    save_checkpoint(decider, best_state, args.out, meta)


if __name__ == "__main__":
    main()
