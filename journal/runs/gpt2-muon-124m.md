# gpt2-muon (124M) — Muon-split + WSD on 10B FineWeb-Edu

**Run date:** 2026-06-28 · **Hardware:** 8×A100-80GB SXM4 (cloud), ~2.1h wall-clock · **Config:** `configs/gpt2-muon.yaml` (commit 451b592)

A faithful GPT-2 (124M) trained from scratch on 10B FineWeb-Edu tokens with the **Muon-split optimizer + warmup-stable-decay (WSD)** schedule, vs the standard AdamW+cosine reference. The model is byte-identical to `gpt2.yaml`; only the training recipe differs.

## Results

| metric | value |
|---|---|
| val_loss (FineWeb-Edu val shard) | **2.989** |
| HellaSwag acc_norm | **0.3238** |
| HellaSwag acc | 0.2991 |
| LAMBADA accuracy | 0.2826 |
| LAMBADA perplexity | 27.81 |

Eval set: 10,042 HellaSwag examples, 5,153 LAMBADA. HellaSwag uses Karpathy's exact method (acc = argmin of summed completion loss; acc_norm = argmin of length-normalized loss — confirmed identical in `src/harness/eval/hellaswag.py`), so our numbers are directly comparable to his.

### val_loss trajectory (WSD decay begins ~step 11,444)
| tokens | step | val_loss |
|---|---|---|
| 2.6B | 5,000 | 3.464 |
| 5.2B | 10,000 | 3.341 |
| 7.9B | 15,000 | 3.123 |
| 10B | 19,072 (final) | **2.989** |

The decay phase (last 40%, LR annealed to 0) drives the steepest drop — ~0.35 of the total descent lands here.

## Comparison to references (same data, val shard, gpt2 tokenizer, 124M)

| | val_loss | HellaSwag | recipe |
|---|---|---|---|
| GPT-2 124M (OpenAI) | ~3.29 \* | 0.294 | — |
| build-nanogpt / llm.c 124M | **3.28** | ~0.299–0.305 † | AdamW + cosine |
| **gpt2-muon (ours)** | **2.989** | **acc 0.299 / acc_norm 0.324** | Muon-split + WSD |

\* GPT-2 checkpoint scored on the FineWeb-Edu val set (home-field caveat below — not a clean LM-quality win).
† llm.c discussion #481 reports 29.9 (metric unlabeled; read as acc_norm); build-nanogpt video ~0.305. GPT-3 Small is 33.7 but trained on 300B tokens (apples-to-oranges per Karpathy).

Sources: [llm.c #481](https://github.com/karpathy/llm.c/discussions/481) · [build-nanogpt](https://github.com/karpathy/build-nanogpt) · [openai-community/gpt2](https://huggingface.co/openai-community/gpt2)

## How to state the finding (defensible)

> On 10B FineWeb-Edu tokens, a from-scratch GPT-2 (124M) with a Muon-split optimizer + warmup-stable-decay schedule reaches **val_loss 2.989** and **HellaSwag acc_norm 0.324**, evaluated with Karpathy's exact HellaSwag method — improving on the AdamW+cosine reference (build-nanogpt/llm.c: ~3.28 / ~0.30) and the GPT-2 124M baseline (0.294) on the same data and metric.

## Caveats (keep these attached)
1. **Recipe-vs-recipe, not Muon in isolation.** Muon + WSD + tuned weight-decay/LRs all moved together. The **WSD decay-to-zero** plausibly accounts for a real slice of the val-loss gap (Karpathy decays cosine to 10% of max, not 0). To attribute Muon's share, run **Muon+WSD vs AdamW+WSD** with the schedule held fixed.
2. **Single seed.** The ~0.02–0.03 HellaSwag edge over the reproduction wants a second seed to harden.
3. **Do not claim "beats GPT-2 on val_loss."** That's home-field — we trained on FineWeb-Edu, GPT-2 didn't. Lead with HellaSwag (neutral benchmark) for the GPT-2 comparison.
4. **Reference numbers are from a quick web dig, not re-measured.** llm.c #481 does not label acc vs acc_norm; read as acc_norm (its headline metric + the standard HellaSwag metric).

## Open follow-ups
- [ ] Second-seed rerun — harden the HellaSwag edge.
- [ ] Muon+WSD vs AdamW+WSD ablation — isolate Muon from the schedule.
- [ ] Carry Muon onto the modern **skyai** stack; scale toward skyai-xl.

## Artifacts
- Checkpoints: `checkpoints/gpt2-muon/` — `best.pt` = final/lowest-val_loss (step 19,072), plus step_10k/15k/19072.
- Logs: `logs/gpt2-muon/` — `run.log` (full training curve), `eval.log` (HellaSwag/LAMBADA).
- Lean local proxy that motivated the cloud run (786M tokens): muon **3.351** vs AdamW **3.823**.
