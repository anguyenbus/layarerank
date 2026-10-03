# How Strands Decider performs as a reranker on FiQA

*2026-10-03 · BEIR FiQA-2018 test · BM25 top-30 · 385 queries · strands-decider 0.1.0 · transformers 5.17.0 · torch 2.14.0+cu126 · RTX 3060 12 GB*

This report tests [Strands Decider 2B](https://strandsagents.com/blog/introducing-strands-decider/) as a
replacement for Laya in the reranker. It uses the same candidates, queries, metrics and gate as the Laya
report ([fiqa-2026-09-28.md](fiqa-2026-09-28.md)), so every number here is directly comparable with the
numbers there. The Laya and bge rows below are the stored per-query scores from that report, not re-runs.

## Summary

- **Fine-tuned, Decider is the best single reranker measured on FiQA.** One epoch of training on FiQA's
  train split (checkpoint selected at 0.66 epochs) gives **0.627 nDCG@10**. That is **+0.050 [+0.029,
  +0.068] over bge-reranker-v2-m3** (0.578) and **+0.048 [+0.030, +0.065] over fully fine-tuned Laya**
  (0.580). MRR@10 rises from about 0.74 for both of those to **0.811**.
- **Off the shelf, Decider does not pass the gate.** With its best prompt it reaches 0.400 nDCG@10 against
  0.392 for BM25: +0.008 [−0.027, +0.042], which is **NO-GO**. It is slightly behind untrained Laya
  (0.422).
- **It is about an order of magnitude slower.** Decider takes **157 ms per pair** on the RTX 3060 used
  here, 4.7 s for a 30-candidate query. Laya took 9.8 ms and bge 12.6 ms per pair, measured on a faster
  RTX 3090 Ti, so the true gap is somewhat smaller than 16×, but it is not close (§4.5).
- **A cascade recovers most of the gain at a fraction of the cost.** Letting bge rank all 30 candidates
  and Decider reorder only bge's top 5 gives **0.618 nDCG@10**, which is +0.040 [+0.024, +0.056] over bge
  alone and not significantly below Decider on all 30, with one sixth of the Decider calls (§6).
- **Blending no longer helps.** Mixing fine-tuned Decider with bge gives 0.634, +0.006 [−0.002, +0.015]
  over Decider alone. Laya needed bge to reach 0.600; Decider alone is already past that.
- **Fine-tuning trains 17.9M parameters, not the whole model.** Only the LoRA adapter and the pointer
  head train; the 2B base stays frozen. The fine-tuned checkpoint is 88 MB. Training took about 4.3 hours
  on the RTX 3060, against 60 minutes for Laya on the 3090 Ti.
- **Caveat: this is a FiQA specialist from one training run.** bge was not fine-tuned, and no other
  domain was tested (§8).
- **Recommendation:** where quality matters more than latency, use fine-tuned Decider as a second stage
  on the top 5 from bge or fine-tuned Laya. Where latency matters, fine-tuned Laya stays the choice.

## 1. Setup

| | |
|---|---|
| Dataset, candidates, queries, metrics, significance, gate | Identical to the Laya report (§1 there): frozen BM25 top-30, 385 answerable test queries, nDCG@10 / MRR@10 / Recall@10, paired bootstrap with 5,000 resamples, gate = beats BM25 on nDCG@10 with the 95% CI above 0. |
| Pools re-verified | Rebuilt on this machine: BM25 scores 0.3915 nDCG@10 and all 385 candidate lists match the stored ones. |
| Model | `StrandsAgents/strands-decider-2B-hobson-v19`: `Qwen/Qwen3.5-2B-Base` (about 2B parameters, frozen) + a rank-16 LoRA adapter (16.8M) + a pointer head (1.1M). Loaded in bf16, 3.7 GB on the GPU. |
| Scoring | Pointwise: one prompt per (query, passage) pair, state `{"query", "passage"}`, one `noul` question. The score is P(true), read unrounded. |
| Harness | `evals/decider.py`, system spec `decider:<model>:<preset>`. It batches whole prompts across passages; the package's own engine serves one state at a time. Scored one row at a time, it reproduces the package engine's probabilities exactly; batched scores differ by up to about 0.005 from bf16 padding effects. |
| Context window | 4,096 tokens. The longest test prompt is 2,197 tokens (mean 332), so **no passage is truncated**. Laya's 512-token rows truncated 9.1% of passages. |
| Hardware | One RTX 3060 12 GB. The Laya report used an RTX 3090 Ti, so timings are not like for like. |

Names used below: **untrained** = `decider:2b:noul-question`; **fine-tuned** =
`decider:decider-ft-fiqa:noul-question`. Both use the same prompt.

**Ceiling.** A perfect reranker of these candidates would reach **0.747 nDCG@10**:

| | nDCG@10 | Share of the BM25→perfect gap closed |
|---|---|---|
| bm25 | 0.392 | 0% |
| Decider untrained | 0.400 | 2% |
| Laya untrained (best prompt) | 0.422 | 9% |
| bge-reranker-v2-m3 | 0.578 | 52% |
| Laya full-FT | 0.580 | 53% |
| Laya full-FT + bge blend | 0.600 | 59% |
| **Decider fine-tuned** | **0.627** | **66%** |
| perfect reranker | 0.747 | 100% |

## 2. Headline results (385 test queries)

| System | nDCG@10 | MRR@10 | Recall@10 | ms / pair | Δ nDCG@10 vs BM25 [95% CI] | Δ nDCG@10 vs bge [95% CI] |
|---|---|---|---|---|---|---|
| Decider fine-tuned + bge blend (α = 0.7–0.8)¹ | 0.6336 | 0.8200 | 0.6560 | ≈ 170² | +0.242 [+0.213, +0.272] | +0.056 [+0.039, +0.072] |
| **Decider fine-tuned** | **0.6273** | **0.8114** | **0.6478** | 157 (3060) | **+0.236 [+0.205, +0.267]** | **+0.050 [+0.029, +0.068]** |
| bge → Decider fine-tuned on top 5¹ | 0.6181 | — | — | ≈ 39² | +0.227 | +0.040 [+0.024, +0.056] |
| Laya full-FT + bge blend | 0.6002 | 0.7738 | 0.6448 | ≈ 22.4 (3090 Ti) | +0.209 [+0.181, +0.239] | +0.022 [+0.010, +0.035] |
| Laya full-FT | 0.5796 | 0.7374 | 0.6321 | 9.8 (3090 Ti) | +0.188 [+0.159, +0.220] | +0.002 [−0.016, +0.020] (tie) |
| bge-reranker-v2-m3 | 0.5778 | 0.7408 | 0.6327 | 12.6 (3090 Ti) | +0.186 [+0.161, +0.214] | — |
| Laya untrained (`noul-dict-bare`) | 0.4219 | 0.5052 | 0.5481 | 9.8 (3090 Ti) | +0.031 [−0.004, +0.064] | −0.156 [−0.182, −0.132] |
| Decider untrained (`noul-question`) | 0.3996 | 0.4837 | 0.5283 | 157 (3060) | +0.008 [−0.027, +0.042] | −0.178 |
| bm25 (no rerank) | 0.3915 | 0.4838 | 0.4974 | — | — | −0.186 [−0.214, −0.161] |

¹ Computed offline from the stored per-query scores (§6). The blend uses min-max normalised scores with
the Decider weight α chosen by 2-fold cross-validation; its MRR@10 and Recall@10 are at α = 0.75.

² Mixed hardware: Decider's 157 ms per pair on the 3060 plus bge's 12.6 ms on the 3090 Ti. The cascade
figure is the average per candidate when Decider scores 5 of 30. Treat both as rough.

Fine-tuned Decider against each other system, query by query:

| Fine-tuned Decider minus … | Δ nDCG@10 [95% CI] | Wins / ties / losses |
|---|---|---|
| bm25 | +0.236 [+0.205, +0.267] (passes the gate) | 270 / 75 / 40 |
| Decider untrained | +0.228 [+0.199, +0.256] | 268 / 87 / 30 |
| bge-reranker-v2-m3 | +0.050 [+0.029, +0.068] | 141 / 171 / 73 |
| Laya full-FT | +0.048 [+0.030, +0.065] | 143 / 179 / 63 |

## 3. Off-the-shelf Decider configurations

Prompts were compared on the **dev** split (281 queries) and only the winner was scored on test. This
avoids the selection optimism noted in the Laya report, where the best of 8 presets was picked on the
test queries.

| Preset | Dev nDCG@10 | Dev MRR@10 | Dev Recall@10 | Δ nDCG@10 vs BM25 on dev [95% CI] | Test nDCG@10 |
|---|---|---|---|---|---|
| **noul-question** | **0.4116** | **0.4882** | 0.5361 | −0.000 [−0.041, +0.041] | 0.3996 |
| noul-criteria | 0.4057 | 0.4731 | **0.5438** | −0.006 [−0.046, +0.033] | not run |
| noul-bare | 0.3916 | 0.4694 | 0.5163 | −0.020 [−0.063, +0.022] | not run |
| bm25 | 0.4120 | — | — | — | 0.3915 |

For reference, on the same dev queries untrained Laya scores 0.423 and bge 0.612.

What the presets ask the model (the state is `{"query", "passage"}` in all three):

| Preset | Instruction | Criteria |
|---|---|---|
| `noul-question` | "Does \`passage\` answer \`query\`?" (Laya's `noul-dict-bare` wording) | Decider's defaults |
| `noul-bare` | "\`passage\` answers \`query\`." (a statement, matching Decider's own header) | Decider's defaults |
| `noul-criteria` | "\`passage\` contains information that answers \`query\`." | The true/false criteria from Laya's `noul-dict` |

### What the sweep shows

1. **No prompt makes untrained Decider a reranker.** All three intervals against BM25 straddle zero, and
   the three prompts are within noise of each other.
2. **This matches the model card.** It says `noul` "transfers poorly" to yes/no tasks unlike the
   training mix, and that the model reads the question less than the state. Passage relevance is not
   among its training tasks.
3. **Not run:** a 4-level `score` preset was defined but skipped once fine-tuning was decided. A listwise
   `choice` prompt (passages as options) was not attempted: 30 passages do not fit one 4,096-token
   prompt, and scores from separate groups are not comparable.

## 4. How Decider behaves

### 4.1 Separating relevant from non-relevant passages

Average, over queries, of how often a relevant candidate outscores a non-relevant one (per-query AUC
over the 30 candidates):

| System | AUC |
|---|---|
| bm25 | 0.770 |
| Decider untrained | 0.814 |
| Laya untrained | 0.834 |
| bge-reranker-v2-m3 | 0.929 |
| Laya full-FT | 0.933 |
| **Decider fine-tuned** | **0.956** |

Untrained Decider separates better than BM25 (0.814 vs 0.770) but turns none of that into nDCG. The
fine-tuned model's advantage over bge and Laya shows most at the top of the list: MRR@10 is 0.811 against
about 0.74, while Recall@10 moves only from 0.633 to 0.648.

### 4.2 Hard and easy queries

Queries grouped by how well BM25 ranks them (nDCG@10). As in the Laya report, the grouping favours BM25
in the top group, so bge is the control.

| Query group | n | bm25 | Decider untrained | Laya untrained | bge | Laya full-FT | **Decider fine-tuned** |
|---|---|---|---|---|---|---|---|
| BM25 misses (nDCG = 0) | 75 | 0.000 | 0.316 | 0.339 | 0.410 | 0.470 | **0.537** |
| BM25 middling (0 < nDCG < 0.5) | 169 | 0.279 | 0.324 | 0.345 | 0.479 | 0.480 | **0.530** |
| BM25 already good (nDCG ≥ 0.5) | 141 | 0.734 | 0.534 | 0.559 | 0.785 | 0.757 | **0.792** |

- **Untrained Decider has the same failure as untrained Laya:** it recovers about 0.32 where BM25 fails
  and loses 0.20 on queries BM25 already handles.
- **Fine-tuned Decider leads in every group.** The lead is largest on queries BM25 misses entirely (0.537
  against 0.470 for Laya and 0.410 for bge) and smallest on the easy group, where it is level with bge.

### 4.3 Score distribution and thresholds

The `noul` score is a probability in [0, 1].

| Model | Median score, relevant | Median score, non-relevant | ≥ 0.2: relevant kept | ≥ 0.2: non-relevant kept | ≥ 0.5: relevant kept | ≥ 0.5: non-relevant kept |
|---|---|---|---|---|---|---|
| **Decider fine-tuned** | **0.883** | **0.002** | 90% | **10%** | **76%** | **5%** |
| Laya full-FT | 0.749 | 0.043 | 91% | 21% | 72% | 7% |
| Decider untrained | 0.708 | 0.425 | 100% | 93% | 83% | 39% |

Untrained Decider is useless as a filter: at 0.2 it passes 93% of non-relevant candidates. Fine-tuned, it
is a sharper filter than fine-tuned Laya: at 0.2 it keeps the same share of relevant passages and half
as many non-relevant ones. The temperature was not re-fitted after fine-tuning, so the probabilities are
not calibrated; choose any threshold on held-out data.

### 4.4 Long passages

Decider sees every passage in full, while Laya and bge cut at 512 tokens (12% of Decider's prompts are
longer than that). That is not where the gain comes from. On the 49 queries where Laya truncated nothing,
fine-tuned Decider leads fine-tuned Laya by +0.060 [+0.005, +0.115]; on the 336 queries with at least one
truncated candidate the lead is +0.046 [+0.027, +0.065]. The lead is no larger where truncation occurred,
which agrees with the Laya report's finding that truncation does not limit quality on FiQA.

Training rows were capped at 512 tokens to fit the 12 GB card (§7), so the model was trained on truncated
long passages and scored on full ones.

### 4.5 Speed and reproducibility

- **Speed:** 157 ms per pair, 4.7 s per 30-candidate query (median 4.6 s), the same before and after
  fine-tuning. Peak GPU memory while scoring is about 5 GB.
- **Not like for like.** Laya's 9.8 ms and bge's 12.6 ms were measured on an RTX 3090 Ti; neither was
  re-timed on the 3060. The 16× and 12× ratios therefore overstate the gap by the difference between the
  two cards, which was not measured.
- **Reference kernels.** transformers reports that it falls back to slow reference implementations
  because `flash-linear-attention` and `causal_conv1d` are not installed. Installing
  `flash-linear-attention` gave no measurable speed-up in a short test and shifted scores slightly, so it
  was removed. `causal_conv1d` was not tried.
- **No ties.** Scores are read unrounded: all 30 candidates have distinct scores on every query.

## 5. Decider blended with BM25

Scores are min-max normalised within each query's 30 candidates and combined as
`α · decider + (1 − α) · bm25`, from the stored scores.

| α (Decider weight) | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nDCG@10, **fine-tuned** | 0.391 | 0.434 | 0.478 | 0.519 | 0.560 | 0.600 | 0.613 | 0.620 | 0.623 | 0.625 | **0.627** |
| nDCG@10, untrained | 0.391 | 0.411 | 0.428 | 0.442 | 0.455 | 0.466 | **0.467** | 0.464 | 0.445 | 0.423 | 0.400 |

| Blend | CV α | nDCG@10 | vs BM25 | vs the same Decider alone |
|---|---|---|---|---|
| untrained + BM25 | 0.7 / 0.6 | 0.4626 | +0.071 [+0.048, +0.094] | +0.063 [+0.043, +0.084] |
| fine-tuned + BM25 | 1.0 / 1.0 | 0.6273 | +0.236 [+0.205, +0.267] | 0 (BM25 gets no weight) |

The pattern is the same as for Laya: blending with BM25 rescues the untrained model (0.463, which passes
the gate, about the same as untrained Laya + BM25 at 0.473) and adds nothing once the model is
fine-tuned.

## 6. Decider combined with bge and Laya

Simulated from the stored per-query scores, as in the Laya report. In a cascade the first model ranks all
30 candidates and fine-tuned Decider reorders only the top N; the rest keep the first model's order.

| Approach | nDCG@10 | Decider pairs per query | Δ vs first stage alone [95% CI] | Δ vs Decider on all 30 [95% CI] |
|---|---|---|---|---|
| **Decider fine-tuned alone** | **0.627** | 30 | — | — |
| bge → Decider top 3 | 0.608 | 3 | +0.030 [+0.017, +0.044] | −0.019 [−0.035, −0.003] |
| **bge → Decider top 5** | **0.618** | 5 | +0.040 [+0.024, +0.056] | −0.009 [−0.021, +0.003] |
| bge → Decider top 10 | 0.619 | 10 | +0.042 [+0.024, +0.058] | −0.008 [−0.017, +0.001] |
| bge → Decider top 20 | 0.627 | 20 | +0.050 [+0.030, +0.067] | −0.000 [−0.005, +0.004] |
| Laya full-FT → Decider top 5 | 0.613 | 5 | +0.033 [+0.019, +0.047] | −0.015 [−0.027, −0.002] |
| Laya full-FT → Decider top 10 | 0.618 | 10 | +0.039 [+0.023, +0.054] | −0.009 [−0.019, +0.001] |
| Laya full-FT → Decider top 20 | 0.631 | 20 | +0.052 [+0.034, +0.069] | +0.004 [+0.001, +0.007] |
| BM25 → Decider top 10 | 0.508 | 10 | +0.117 [+0.096, +0.137] | −0.119 [−0.148, −0.092] |
| Score blend, Decider + bge (CV α 0.7 / 0.8) | 0.634 | 30 | +0.056 [+0.039, +0.072] | +0.006 [−0.002, +0.015] |
| Score blend, Decider + Laya full-FT (CV α 0.8 / 0.8) | 0.633 | 30 | +0.053 [+0.037, +0.070] | +0.006 [−0.000, +0.012] |
| Score blend, untrained Decider + bge (CV α 0.1 / 0.2) | 0.576 | 30 | −0.002 [−0.012, +0.007] | — |

- **A cheap first stage plus Decider on the top 5 keeps most of the gain.** bge → Decider top 5 reaches
  0.618 with 5 Decider calls per query, about 0.8 s on the 3060 against 4.7 s for all 30. It is not
  significantly below Decider on all 30.
- **The first stage must be a good reranker.** With BM25 order as the first stage, Decider on the top 10
  reaches only 0.508, because the relevant passages are often below rank 10.
- **Blends add nothing significant.** Decider gets 70–80% of the weight and the gain over Decider alone
  is within noise. An equal-weight blend of all three models scores 0.628, the same as Decider alone.
- **Untrained Decider only harms bge**, the same as untrained Laya did.

## 7. Fine-tuning

### 7.1 Setup

The run uses `evals/finetune_decider.py`, which applies the Laya recipe (`evals/finetune.py`) to the
parts of Decider that train.

| | Laya full-FT (earlier report) | Decider fine-tune (this report) |
|---|---|---|
| Command | `python -m evals.finetune --dataset fiqa --out ft-fiqa-full --epochs 3 --evals-per-epoch 4 --patience 4` | `python -m evals.finetune_decider --dataset fiqa --preset noul-question --out decider-ft-fiqa --epochs 1 --evals-per-epoch 6 --dev-queries 100 --patience 3` |
| Trained parameters | All 421M | **17.9M**: the existing LoRA adapter (rank 16) and the pointer head. The 2B base is frozen. |
| Learning rate | encoder 1e-5, head 1e-4 | LoRA 1e-4, head 1e-4 |
| Row length in training | 512 tokens | 512 tokens (12% of training rows cut); inference uses the full window |
| Micro-batch | 4 queries | 1 query, with gradient checkpointing, to fit 12 GB (peak 9.6 GB) |
| Epoch cap / dev checks / patience | 3 / every ¼ epoch / 4 | 1 / every ⅙ epoch / 3 |
| Dev set for selection | all 281 dev queries | a fixed 100-query sample (a full check costs 23 minutes) |
| Speed | 2.4 s per step (3090 Ti) | 40 s per step (3060) |
| Selected checkpoint | epoch 1.74 (step 672) | epoch 0.66 (step 256) |
| Wall time | ~60 min | ~5 h 10 min, of which ~4.3 h training and ~1 h dev checks |
| Output | 808 MB | 88 MB (`evals/models/decider-ft-fiqa/`, not in git) |

Shared by both runs: FiQA train qrels on each train query's BM25 top-30 (3,085 queries); positives are
the judged-relevant candidates and negatives are the other top-30 candidates; 8 queries per step, each
with all its positives plus 7 negatives resampled every epoch; the loss is a listwise softmax over each
query's true-minus-false margins plus 0.5 × binary cross-entropy; AdamW with 5% warm-up, cosine decay and
gradient clipping at 1.0; the test split is scored once, afterwards.

Two differences from the upstream Decider recipe: the true/false option order is fixed, as at inference
(upstream shuffles it to keep the head generic), and there is no KL term holding the model to its
original behaviour. Both choices make the checkpoint a specialist.

### 7.2 Dev curve (100-query dev sample)

| Epoch | 0 | 0.17 | 0.33 | 0.50 | **0.66** | 0.83 | 0.99 | 1.00 |
|---|---|---|---|---|---|---|---|---|
| Dev nDCG@10 | 0.3767 | 0.5985 | 0.6040 | 0.6311 | **0.6344** | 0.6284 | 0.6327 | 0.6292 |

- **Most of the gain arrived in the first sixth of an epoch** (64 steps, 43 minutes): 0.377 → 0.599. Laya
  showed the same shape.
- **The curve was flat from half an epoch.** The last four checks span 0.628–0.634, which is within the
  noise of a 100-query sample, so the choice of step 256 over its neighbours is not meaningful.
- **A second epoch was not run.** Patience was never exhausted; the run ended at the epoch cap.

### 7.3 Full dev and test results

| | Dev (281 queries) | Test (385 queries) |
|---|---|---|
| bm25 | 0.4120 | 0.3915 |
| Decider untrained | 0.4116 | 0.3996 |
| bge-reranker-v2-m3 | 0.6117¹ | 0.5778 |
| Laya full-FT | 0.6139¹ | 0.5796 |
| **Decider fine-tuned** | **0.6622** | **0.6273** |
| Decider fine-tuned minus untrained [95% CI] | +0.251 [+0.220, +0.282] | +0.228 [+0.199, +0.256] |

¹ From the Laya report; per-query dev scores for bge and Laya are not stored, so there is no interval
against them on dev.

The lead over bge and Laya is about 0.05 on both splits, and dev runs about 0.03 easier than test for
every system, as before. There is no sign of overfitting to the dev sample.

## 8. Caveats

- **Latency was measured on different hardware.** Decider ran on an RTX 3060, Laya and bge on an RTX
  3090 Ti. Re-time bge and Laya on the 3060, or Decider on a 3090 Ti, before quoting a speed ratio.
- **bge was not fine-tuned.** Both Laya and Decider saw 3,085 FiQA training queries; bge saw none. A
  fine-tuned bge is still the open comparison, as in the Laya report.
- **Specialist, one dataset.** Nothing here measures the fine-tuned checkpoint outside FiQA, or what it
  lost on Decider's other tasks. The training choices in §7.1 make some loss likely.
- **One training run.** One seed, one learning rate, one epoch. The size of the lead over bge and Laya
  (about 0.05) is well outside the bootstrap interval, but run-to-run variance was not measured.
- **Selection on a 100-query sample.** The checkpoint was chosen on a noisy sample. The full-dev and test
  scores agree, so this did not inflate the result, but a different check could as easily have won.
- **Off-the-shelf coverage is thin.** Three `noul` prompts on dev and one on test. The `score` preset and
  a listwise `choice` prompt were not measured.
- **Simulated combinations.** The cascades and blends in §6 are computed from stored scores, and their
  per-query cost mixes timings from two GPUs.
- **Noisy labels.** As in the Laya report, unjudged passages count as non-relevant in training and in the
  metrics.
- **The checkpoint is not backed up.** `evals/models/decider-ft-fiqa/` is git-ignored and sits on
  container storage that does not survive a recycle or destroy.

## 9. Recommendations

1. **Where quality matters more than latency, add fine-tuned Decider as a second stage on the top 5**
   from bge or fine-tuned Laya. Expect about +0.04 nDCG@10 over the first stage for 5 Decider calls per
   query. This needs a cascade option in the reranker; it is not a `layareranker` feature today.
2. **Where latency matters, keep fine-tuned Laya.** It is 0.048 nDCG@10 behind and more than ten times
   faster per pair.
3. **Do not use Decider off the shelf for reranking.** It fails the BM25 gate with every prompt tried.
4. **Before adopting it, measure it on the target hardware and on a second dataset** (e.g. SciFact or
   NFCorpus), against bge and fine-tuned Laya.
5. **Copy the checkpoint off the instance.** It is 88 MB and took about 5 hours to produce.
6. **Possible next steps:** fine-tune bge on the same data for a like-for-like comparison; try the
   `causal_conv1d` kernel and a larger GPU for speed; repeat the fine-tune with a second seed; port
   `evals/decider.py` into `src/` behind the existing `Scorer` interface if Decider is adopted.

## Reproduce

```bash
uv sync --extra gpu --extra eval
# off-the-shelf prompts on dev, then the winner on test
for p in noul-bare noul-question noul-criteria; do
  uv run --extra gpu --extra eval python -m evals.run --dataset fiqa --k 30 --split validation --system "decider:2b:$p"
done
uv run --extra gpu --extra eval python -m evals.run --dataset fiqa --k 30 --system decider:2b:noul-question
# fine-tune (about 5 h on one RTX 3060), then score the checkpoint on dev and test
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run --extra gpu --extra eval python -m evals.finetune_decider \
  --dataset fiqa --preset noul-question --out decider-ft-fiqa --epochs 1 --evals-per-epoch 6 --dev-queries 100 --patience 3
for split in validation test; do
  uv run --extra gpu --extra eval python -m evals.run --dataset fiqa --k 30 --split $split --system decider:decider-ft-fiqa:noul-question
done
uv run --extra gpu --extra eval python -m evals.report --dataset fiqa --k 30 --candidate decider:decider-ft-fiqa:noul-question
```

- The base weights (`Qwen/Qwen3.5-2B-Base`, 4.5 GB) and the Decider checkpoint (88 MB) download from the
  Hugging Face Hub into `HF_HOME` on first use.
- Per-query scores are in `evals/results/fiqa.top30/` (test) and `evals/results/fiqa-validation.top30/`
  (dev). The Laya, bge and BM25 files there are the ones from the Laya report.
- The dev curve and run settings are in `evals/models/decider-ft-fiqa/finetune.json` (not in git).
- The ceiling, AUC, threshold, query-group, blend and cascade analyses (§1, §4–6) were computed with
  one-off scripts over those files. They are not part of `evals/report.py`.
