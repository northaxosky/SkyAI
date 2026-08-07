# skyai (modern stack, gpt2 scale) - architecture change on 10B FineWeb-Edu

**Run date:** 2026-08-05 | **Hardware:** 4xH100-80GB SXM5 (Lambda, us-south-2), 2h33m wall-clock | **Config:** `configs/skyai.yaml` (commit `b4713f4`) | **Cost:** ~$35

The modern decoder stack - RMSNorm + post-embedding norm, RoPE, SwiGLU, grouped-query
attention, QK-norm, untied embeddings, logit soft-capping - trained at **gpt2's depth,
width, context, tokenizer, data, token budget, and schedule shape**. This is rung 3 of the
ladder: `gpt2` (AdamW+cosine) -> `gpt2-muon` (recipe change) -> **`skyai` (architecture change)**.

FlashAttention-3 was built from source on Hopper and was active for the whole run
(`sm_90` gate confirmed in the launch log); gpt2-muon used PyTorch SDPA.

## Results

All numbers below were **re-measured locally on the released checkpoints**, not read from
the training log - see "Measurement protocol" for why that matters.

| metric | **skyai** | gpt2-muon | gap |
|---|---|---|---|
| val_loss (full 100M-token FineWeb-Edu val shard) | **2.9548** | 2.9653 | **+0.0104** |
| HellaSwag acc_norm | 0.3256 | 0.3238 | +0.0018 (n.s.) |
| HellaSwag acc | 0.3008 | 0.2991 | +0.0017 (n.s.) |
| LAMBADA perplexity | **26.25** | 27.81 | **-5.6%** |
| LAMBADA accuracy | 0.2793 | 0.2826 | -0.0033 (n.s.) |

Eval sets: 10,042 HellaSwag examples, 5,153 LAMBADA, 97,656 val sequences of 1024 tokens.
HellaSwag uses Karpathy's exact method (acc = argmin of summed completion loss; acc_norm =
argmin of length-normalized loss). LAMBADA scores at a hardcoded `block_size=1024`.

**Every likelihood metric favors skyai; every accuracy metric is null.** val_loss and
LAMBADA perplexity both move, while HellaSwag acc/acc_norm and LAMBADA accuracy all sit
within +/-0.55 SE. Read that as better-calibrated probabilities without a measurable change
in task accuracy at this scale.

### val_loss is precisely measured

Paired per-sequence comparison over the full val shard (97,656 sequences):

- mean paired gap **+0.010436**
- moving-block bootstrap 95% CI **[+0.010218, +0.010664]** (block = 512 sequences, 190 blocks)
- skyai lower on **65.4%** of sequences
- **190 of 190 disjoint blocks favor skyai** (per-block range +0.0071 ... +0.0151)

All of that measures the difference between *these two checkpoints on this data*, which is
not the same thing as run-to-run uncertainty. See caveat 1.

### val_loss trajectory (WSD decay begins step 11,444)

| tokens | step | skyai | gpt2-muon |
|---|---|---|---|
| 2.6B | 5,000 | 3.4416 | 3.464 |
| 5.2B | 10,000 | 3.3259 | 3.341 |
| 7.9B | 15,000 | 3.1185 | 3.123 |
| 10B | 19,072 (final) | 2.9810 | 2.989 |

*(In-run protocol, shape only. See below.)* The lead shows up by step 5,000 and stays roughly
flat from there, so it never opens up during decay. As with gpt2-muon, the decay phase is
where the steepest descent happens.

## Measurement protocol (and a correction)

**The in-run val_loss numbers are not comparable between the two runs.** `eval.val_steps`
counts *micro-batches*, not tokens, so tokens scored = `val_steps x batch_size x block_size
x world_size`:

| run | B | world | val tokens scored |
|---|---|---|---|
| gpt2-muon | 64 | 8 | 10,485,760 |
| skyai | 32 | 4 | 2,621,440 |

skyai's val set was a 4x smaller prefix of gpt2-muon's. Re-scoring both checkpoints on
identical windows (the script reproduces each run's logged number on its own window, which
validates the measurement path):

| window | gpt2-muon | skyai | gap |
|---|---|---|---|
| 2,560 seq (skyai's in-run) | 2.9913 | 2.9810 | +0.0103 |
| 10,240 seq (gpt2-muon's in-run) | 2.9892 | 2.9787 | +0.0105 |
| **97,656 seq (full shard)** | **2.9653** | **2.9548** | **+0.0104** |
| *as originally logged* | 2.9891 | 2.9810 | +0.0082 |

The asymmetry biased **against** skyai, so the honest gap is +0.0104 rather than the +0.0082
the raw logs imply. Always quote the full-shard figure with its protocol attached.

Absolute val_loss is protocol-sensitive enough to matter: the same skyai checkpoint reads
2.9810 on 2.6M tokens and 2.9548 on 100M. That makes comparisons to outside numbers like
"3.28" looser than they look.

## Scale is matched; parameters are not

| | gpt2-muon | skyai |
|---|---|---|
| total params | 124,475,904 | 151,584,768 (**+21.8%**) |
| **non-embedding params** | 85,056,000 | **74,317,824 (-12.6%)** |
| **forward FLOPs/token** | 266.0 M | **244.8 M (-8.0%)** |

Matched: depth (12), width (768), heads (12), context (1024), tokenizer (gpt2 BPE), dataset,
token budget (10B), batch (524,288 tokens/step), and schedule shape.

**Never call this "parameter-matched."** Every one of those +27.1M parameters is untied
output embedding, costing zero FLOPs, and the whole -21.2 MFLOP/token comes from GQA. The
SwiGLU MLP is FLOP-identical to the 4x GELU MLP at 9,437,184 FLOPs/token/layer, since the 8/3
scaling lands exactly. FLOPs/token is the fairest single axis here, and it favors skyai.

## How to state the finding (defensible)

> On 10B FineWeb-Edu tokens with the GPT-2 BPE tokenizer, matched depth/width/context, an
> identical 524,288-token batch and warmup-stable-decay schedule, and a shared Muon-split
> optimizer family, the modern stack (RMSNorm, RoPE, SwiGLU, GQA, QK-norm, untied
> embeddings, logit soft-cap) reaches **val_loss 2.9548 vs 2.9653** for the faithful GPT-2
> architecture - a **0.0104-nat advantage** on an identical 100M-token validation shard
> (paired 95% CI [+0.0102, +0.0107]; lower on 190/190 disjoint blocks) - and **5.6% lower
> LAMBADA perplexity**, at **~8% fewer FLOPs per token**. On accuracy-based benchmarks the
> two are statistically indistinguishable (HellaSwag acc_norm 0.3256 vs 0.3238, paired
> McNemar p = 0.56; LAMBADA accuracy 0.2793 vs 0.2826). Both are **single-seed** runs, and
> the architecture change was bundled with an LR retune, an init-policy change, and removal
> of gradient clipping - so this characterizes **the modern stack as a bundle**, not any
> component in isolation.

## Caveats (keep these attached)

1. **Single seed per arm - this is the load-bearing caveat.** The +0.0104 gap is measured to
   +/-0.0002 *on this validation set*, but that is measurement precision, not run-to-run
   uncertainty. There is **zero measured seed variance in this project**. Community estimates
   for seed-to-seed final val_loss sigma at 124M/10B are roughly **0.01-0.03 nats** - the same
   order as, or larger than, the observed gap. **The result is precisely measured but not
   replicated.** Do not state it as established until seeds exist.
2. **Architecture + recipe bundle, not architecture alone.** Three things moved with the
   architecture and were not held fixed: `init_policy` (gpt2 -> sky-ai), `grad_clip`
   (1.0 -> null), and the learning rates (`matrix_lr` 0.015 -> 0.02, `embedding_lr`
   0.006 -> 0.3). Only *part* of the LR gap is forced - `embedding_lr: 0.3` provably diverges
   on tied-`wte` + LayerNorm gpt2 - but the specific values 0.006/0.015 were never validated
   at the full 10B horizon.
3. **Tuning effort was asymmetric, and the direction is unresolved.** All local LR search
   (`logs/cmp-*`, 120 to 1500-step proxies) went to the baseline. skyai got none. But skyai's
   canonical values carry a lot of external community tuning for exactly that architecture
   family. gpt2-muon's LRs were also tuned at 1/13th of the final token budget, then
   extrapolated. "The baseline is undertuned at full scale" remains a live criticism.
4. **Not parameter-matched.** +21.8% total params, -12.6% non-embedding, -8.0% FLOPs/token.
   Compare on FLOPs/token or non-embedding parameters; never claim matched parameters.
5. **Accuracy benchmarks are null.** HellaSwag and LAMBADA accuracy show no significant
   difference (all within +/-0.55 SE; McNemar p = 0.56 on acc_norm with 848 discordant pairs
   splitting 433/415). The minimum detectable HellaSwag acc_norm difference at 80% power is
   ~0.013, roughly 7x the observed gap. A val_loss win with null accuracy benchmarks is a
   **weak generalization claim**.
6. **A known ceiling limits skyai's attention.** `qk_sharpen = 1.2` came from nanochat, which
   uses head_dim 128. skyai uses 64. QK-norm pins the norms of q and k, so that multiplier
   fixes the largest attention logit the model can make: `1.44 * sqrt(head_dim)` = **11.52**.
   A confident previous-token head over ~1000 keys needs a gap of about 11.50. Measured on the
   checkpoints, gpt2 has 5/144 heads with top-1 above 0.8 (sharpest 0.996, attending offset 1
   for 95.9% of queries). **skyai has 0/144**, max 0.534. Sharp positional circuits sit outside
   what skyai can represent. This handicaps skyai, and it is the leading explanation for the
   small architecture gain.
7. **Minor protocol asymmetries, quantified.** skyai trained and evaluated with FA3 on H100;
   gpt2-muon with SDPA on A100 (measured effect on HellaSwag ~ 2 examples, 0.0002). Shard
   tail-handling gave skyai ~0.26% more unique tokens (9.888B vs 9.862B). The 47 padded vocab
   classes in gpt2's loss contribute exactly 0.000000 nats. None are material.
8. **A latent optimizer bug, symmetric across both runs.** `optimizer.py` applied Muon's
   second-moment bias correction as `1 - beta2` instead of `1 - beta2**t`. That made the
   effective `matrix_lr` **3.16x** the configured value once the EMA warmed up. Both runs share
   the code, so the comparison still holds. But the learning rates recorded here are not the
   ones actually applied. **Fixed after these runs.** The fixed code now runs the canonical
   modern-stack LRs as the reference implementations intend them.
9. **The `gpt2` rung was never run in-house at full scale.** The "+0.291 from the recipe
   change" is measured against a *literature* reference (3.28, build-nanogpt/llm.c), while
   "+0.0104 from the architecture change" is a true in-house head-to-head on an identical
   protocol. Do not present those two deltas as equivalent measurements.
10. **Do not claim a win over OpenAI GPT-2 on val_loss** - home-field advantage, since these
    models trained on FineWeb-Edu and GPT-2 did not.

## Where the gains actually came from

| rung | val_loss | delta | measured how |
|---|---|---|---|
| gpt2, AdamW + cosine | ~3.28 | - | literature reference (not run in-house) |
| gpt2-muon (**recipe**) | 2.989 | +0.291 | vs literature |
| skyai (**architecture**) | 2.9548 | +0.0104 | in-house head-to-head, matched protocol |

Subject to caveat 9, the recipe change was worth roughly **an order of magnitude more** than
the architecture change at this scale. That's the honest headline: at 124M parameters and 10B
tokens, the optimizer and schedule dominated. Most of the modern stack (GQA, RoPE, untied
embeddings) exists for inference efficiency, long context, and scale, and this configuration
exercises none of those.

## Open follow-ups

Ranked by value per unit cost. The local 4090 is free; cloud is self-funded.

- [ ] **Seed variance at proxy scale - the single highest-value experiment.** 3 seeds x
      both architectures at 1500 steps / 786M tokens (~12h unattended on the 4090). Converts
      caveat 1 from an unbounded weakness into a measured error bar, for $0.
- [ ] **`qk_sharpen` 1.2 -> 1.427 ablation** (restores nanochat's logit ceiling at head_dim
      64). Directly tests caveat 6. Blocked on plumbing `qk_sharpen`/`use_qk_norm` through
      `GPTConfig` -> `ModelConfig` -> `schema.py`; they are currently constructor defaults, so
      the ablation would need a source edit and would break `--override` reproducibility.
- [ ] **`grad_clip` and `init_policy` ablations at proxy scale** (~4h each) - removes the two
      cleanest unforced confounds from caveat 2.
- [ ] **Baseline LR sensitivity at a longer horizon** (~6h) - answers the strongest remaining
      attack (caveat 3): were gpt2-muon's LRs, tuned at 1500 steps, mistuned at 19,073?
- [x] **Fix `eval.val_steps` to be token-denominated.** Added `eval.val_tokens`, pinned to
      10,485,760 in `gpt2.yaml`; all three configs now score an identical val budget at any
      batch/world geometry.
- [x] **Fix the Muon bias correction** to `1 - beta2**t`. Note this changes the effective
      learning rate of any future muon-split run by ~3.16x relative to these two.
- [ ] Add ARC-easy / PIQA / WinoGrande via the existing `harness/eval/` scaffold - three
      agreeing benchmarks are far more persuasive than one null one.
- [ ] Second seed of each arm at full 10B scale (~$35 each) - only after the proxy-scale
      seed study says whether it is needed.

## Artifacts

- Checkpoints: `checkpoints/skyai/` - `best.pt` (= `step_00019072.pt`, final = lowest
  val_loss), plus steps 5k/10k/15k. All verified finite, 151,584,768 params, SHA256-matched
  against the box before it was terminated.
- Logs: `logs/skyai/` - `run.log`, `train_console.log` (full 19,073-step curve, FA3-active
  confirmation), `fa3build.log`.
- Offline W&B run: `wandb/offline-run-20260805_174722-z5eg9wci` (syncable with `wandb sync`).
- Data: `data/edu_fineweb10B`, 100 shards, gpt2 BPE. Byte-identical to the gpt2-muon run -
  verified by SHA256 on both the train and val shards
  (`val` = `8e06151653328dbbd1a225bf0ab3ea902c561564c76d9fc2dc6278be8f754c0f`).
- FlashAttention-3: built from source for `sm_90a`, restricted to head_dim 64 / bf16 /
  causal. Measured **1.03x** on the attention op in isolation and ~0% end-to-end, because
  attention is ~1% of FLOPs at 1024 context - FA3's benefit scales with sequence length.
