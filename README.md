# SkyAI

A from-scratch language model project: build two decoder LMs by hand — a faithful
**gpt2 (124M)** and a modern stack, **skyai**, trained head-to-head at the same
scale on the same tokens — sharing one training harness. Every layer, every
optimizer step, and every tokenization decision is implemented and explained
rather than imported.

> **🤗 Both models are trained and published.** On 10B FineWeb-Edu tokens at matched scale,
> the modern stack edges the faithful GPT-2 architecture on likelihood — **val_loss 2.9548 vs
> 2.9653**, **LAMBADA perplexity 26.25 vs 27.81** — at **~8% fewer FLOPs/token**, and is
> statistically tied on accuracy benchmarks. Single-seed, and the architecture moved as a
> bundle; the [run journal](./journal/runs/skyai-124m.md) carries the full caveats.
>
> [`muteptr/skyai-modern-xs`](https://huggingface.co/muteptr/skyai-modern-xs) ·
> [`muteptr/gpt2-muon-124m`](https://huggingface.co/muteptr/gpt2-muon-124m)

## Why this repo exists

Calling an API is one thing. Building the thing the API calls is another. The
point of this project is the second kind of understanding: writing the attention
mechanism by hand, watching gradients flow, debugging shape mismatches,
profiling kernel launches, and training a model from random weights to
coherent(ish) English.

The companion [`journal/`](./journal/) directory captures the reasoning behind
each module. Read those alongside the code to follow the thinking, not just the
result.

## gpt2 — a faithful GPT-2 (124M)

[`src/gpt/`](./src/gpt/) is GPT-2 small, reproduced clean-room: LayerNorm, learned
positional embeddings, GELU MLP, standard multi-head attention, tied input/output
embeddings, biases throughout. It is verified at every level:

- **Exactly 124,439,808 parameters at the canonical `vocab_size=50257`** — GPT-2-small to the
  parameter, asserted in [`tests/gpt/test_model.py`](./tests/gpt/test_model.py). Training pads
  the vocab to 50,304 for tensor-core alignment, so the published checkpoint is 124,475,904.
- **`from_pretrained` matches OpenAI bit-for-bit** — loading the released `gpt2`
  weights (the four Conv1D transposes and all) reproduces HuggingFace's logits to
  floating-point tolerance, with identical argmax predictions.
- **A golden short-run test** ([`tests/test_golden_gpt2.py`](./tests/test_golden_gpt2.py))
  pins its training numerics against accidental drift.

Recipe ([`configs/gpt2.yaml`](./configs/gpt2.yaml)): standard nanoGPT — AdamW
(weight decay 0.1 on 2D params, 0 elsewhere), cosine LR with linear warmup,
gradient clipping at 1.0, gradient accumulation to a 0.5M-token effective batch,
bf16 mixed precision via autocast, Flash Attention through PyTorch SDPA, fused
AdamW, `torch.compile`, and weight tying. Trained on FineWeb-Edu.

### Result: trained with the Muon recipe

The trained gpt2 run swapped the AdamW+cosine baseline for a **Muon-split optimizer
+ warmup-stable-decay** schedule ([`configs/gpt2-muon.yaml`](./configs/gpt2-muon.yaml)):
Newton-Schulz orthogonalized momentum on the 2D matrices, AdamW on the embeddings /
norms / biases, LR annealed to zero over the final 40%. The canonical Muon learning
rates diverge on the faithful arch (tied embeddings + LayerNorm + biases), so they were
re-tuned down. On 8×A100, ~2.1h over 10B FineWeb-Edu tokens:

| metric | **gpt2-muon** | GPT-2 124M \* | nanoGPT / llm.c (AdamW) |
|---|---|---|---|
| val_loss (FineWeb-Edu val shard) | **2.9653** | ~3.29 | 3.28 |
| HellaSwag (acc_norm) | **0.3238** | 0.294 | ~0.30 † |
| LAMBADA (ppl) | **27.81** | ~35 ‡ | — |

\* GPT-2 scored on FineWeb-Edu, which it never trained on — **home-field advantage, not an
LM-quality win**. The neutral comparison is HellaSwag.
† llm.c does not label whether it reports `acc` or `acc_norm`; the columns may not be
strictly like-for-like.
‡ Community `lm-evaluation-harness` figure, not the GPT-2 paper.

This improves on the published AdamW+cosine reference (build-nanogpt / llm.c) at matched
data and scale — but measured against *literature*, not an in-house baseline, so it is a
looser comparison than the skyai head-to-head below. Single-seed, recipe-level; the
[model card](https://huggingface.co/muteptr/gpt2-muon-124m) and
[run journal](./journal/runs/gpt2-muon-124m.md) carry the full caveats.

**🤗 [huggingface.co/muteptr/gpt2-muon-124m](https://huggingface.co/muteptr/gpt2-muon-124m)** —
load with `AutoModelForCausalLM.from_pretrained("muteptr/gpt2-muon-124m")`.

## skyai — the modern stack, at gpt2 scale

[`src/skyai/`](./src/skyai/) rebuilds the model around the current decoder-LM stack:

- RMSNorm (plus a post-embedding norm) in place of LayerNorm
- Rotary position embeddings (RoPE) instead of learned positions
- SwiGLU feed-forward instead of the GELU MLP
- Grouped-query attention (GQA)
- QK-normalization with q/k sharpening
- Untied input and output embeddings
- Logit soft-capping
- Internal vocab padding kept separate from the logical vocab

Training recipe: the Muon optimizer for the transformer matrices and AdamW for the
embeddings and head, a warmup-stable-decay LR schedule, and width- and batch-scaled
per-group learning rates.

### Result: the three-way comparison

[`configs/skyai.yaml`](./configs/skyai.yaml) holds gpt2's depth, width, context,
tokenizer, data, token budget, and schedule shape fixed and changes the architecture,
completing a ladder where each rung moves one coherent stack change:

| run | architecture | recipe | val_loss † | HellaSwag acc_norm |
|---|---|---|---|---|
| `gpt2` | faithful GPT-2 | AdamW + cosine | ~3.28 \* | ~0.30 \* |
| `gpt2-muon` | faithful GPT-2 | **Muon-split + WSD** | 2.9653 | 0.3238 |
| **`skyai`** | **modern stack** | Muon-split + WSD | **2.9548** | 0.3256 |

† Both trained models re-scored on the identical full 100M-token FineWeb-Edu val shard.
\* Literature reference (build-nanogpt / llm.c), not run in-house — a looser comparison.

On 10B FineWeb-Edu tokens at matched scale, the modern stack reaches **val_loss 2.9548 vs
2.9653** — a **0.0104-nat advantage** (paired 95% CI [+0.0102, +0.0107]; lower on 190 of 190
disjoint val blocks) — and **5.6% lower LAMBADA perplexity** (26.25 vs 27.81), at **~8%
fewer FLOPs per token**. On accuracy-based benchmarks the two are **statistically
indistinguishable** (HellaSwag acc_norm 0.3256 vs 0.3238, paired McNemar p = 0.56; LAMBADA
accuracy 0.2793 vs 0.2826). Every likelihood metric favours the modern stack; every accuracy
metric is null.

Two things this result is **not**. It is **single-seed on both arms**, and seed-to-seed
variance at this scale is plausibly the same size as the gap — precisely measured, not
replicated. And the rung is a **bundle**: `init_policy`, `grad_clip`, and the per-group
learning rates all moved with the architecture, so this measures the modern stack as a
package, not any component in isolation. The recipe change (rung 2) was roughly an order of
magnitude larger than the architecture change.

### A measurement bug, and the correction

The first version of this comparison was wrong. `eval.val_steps` counted *micro-batches*, so
the validation budget silently scaled with `batch_size × world_size`: gpt2-muon (B=64, 8 GPUs)
scored 10.5M val tokens while skyai (B=32, 4 GPUs) scored only 2.6M — a 4× smaller prefix of
the same shard. Two numbers that looked comparable were not.

Re-scoring both checkpoints on identical windows resolves it, and the harness reproduces each
run's originally-logged number on its own window, which validates the measurement path:

| window | gpt2-muon | skyai | gap |
|---|---|---|---|
| 2,560 seq (skyai's in-run) | 2.9913 | 2.9810 | +0.0103 |
| 10,240 seq (gpt2-muon's in-run) | 2.9892 | 2.9787 | +0.0105 |
| **97,656 seq (full shard)** | **2.9653** | **2.9548** | **+0.0104** |
| *as originally logged* | 2.9891 | 2.9810 | +0.0082 |

The asymmetry had been biasing *against* skyai, and the gap is protocol-stable at +0.0104
across every window. The harness now takes a token-denominated `eval.val_tokens`
([`loop.py`](./src/harness/training/loop.py)), pinned identically across all three configs, so
val_loss can no longer depend on batch or world geometry. The re-scoring is reproducible with
[`scripts/compare_val_loss.py`](./scripts/compare_val_loss.py).

On parameters: untied embeddings put skyai at ~151.6M against gpt2's 124.5M, but the
compute-bearing **non-embedding** params run *lower* — ~74.3M vs ~85.1M — because GQA
shrinks the attention projections while the SwiGLU MLP is deliberately 8/3-scaled to
stay FLOP-matched to the GELU one. The entire total-param gap is the untied output table,
which costs no FLOPs. Compare on FLOPs/token, never on total parameters.

Full numbers, protocol, and the complete caveat list:
[`journal/runs/skyai-124m.md`](./journal/runs/skyai-124m.md).

### FlashAttention-3, and why it bought nothing here

[`src/skyai/flash.py`](./src/skyai/flash.py) dispatches to FlashAttention-3 when the GPU
reports compute capability 9, and falls back to PyTorch SDPA otherwise — so the fast path
is Hopper-only and cannot run on the 4090 used for local development. FA3 was built from
source for `sm_90a` on the H100 box and was active for the entire skyai run.

Then it was measured, and the honest answer is that it did not matter: **1.03× on the
attention op in isolation, and ~0% end-to-end.** At 1024 context with head_dim 64, attention
is roughly 1% of total FLOPs — 0.89 ms of a 76 ms step — so the MLP and the output head
dominate completely. FA3's advantage scales with sequence length, and this configuration is
too short to see it. The kernel is correct (forward matches SDPA to bf16 tolerance, backward
gradients finite under GQA, and an instrumented count confirms it fires once per layer); it
simply is not the bottleneck at this scale.

The weights are public. The architecture is not a built-in `transformers` class, so the
repo ships its own modeling code and loads with `trust_remote_code=True`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("muteptr/skyai-modern-xs", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("muteptr/skyai-modern-xs")
```

**🤗 [huggingface.co/muteptr/skyai-modern-xs](https://huggingface.co/muteptr/skyai-modern-xs)** —
exported by [`scripts/export_skyai_hf.py`](./scripts/export_skyai_hf.py), which verifies the
port reproduces the harness logits exactly (max |Δ| = 0.0) before publishing.

[`configs/skyai-xl.yaml`](./configs/skyai-xl.yaml) (48 layers, 1536 hidden, 32 heads,
8 KV heads, 2048 context, cl100k) stages a ~1.5B run for 8×H100 and is parked.

## The harness

[`src/harness/`](./src/harness/) is the part that is model-agnostic, and it is what makes the
comparison possible at all. One training loop drives both families through a `build_model`
factory; everything downstream depends only on the model's
`forward(idx, targets) -> (logits, loss)` contract, so adding an architecture means adding a
package under `src/`, not touching the loop.

- **Configs** — pydantic-validated YAML with `extra="forbid"` (a typo is a hard error),
  single inheritance via `extends:`, and `--override a.b.c=value` for any field.
- **Training** — DDP via `torchrun`, gradient accumulation derived from a fixed token budget,
  bf16 autocast, `torch.compile`, AdamW and a hand-written Muon (Newton–Schulz
  orthogonalization), cosine and warmup-stable-decay schedules.
- **Robustness** — crash-safe checkpoint/resume including RNG and dataloader position,
  non-finite-gradient detection, OOM diagnostics that dump batch geometry and VRAM.
- **Evaluation** — HellaSwag and LAMBADA implemented directly, sharded across ranks.
- **Tooling** — `doctor` (env + config preflight), an ablation runner for config sweeps, a
  region profiler, and golden-fixture tests that pin exact losses and a parameter checksum so
  a numerics change on refactor breaks loudly.

## Project layout

```
src/harness/    training framework: cli/, config/, data/, eval/, training/ + checkpoint.py, generate.py, sample.py, ablation.py, log.py, wandb_logger.py
src/gpt/        the faithful gpt2 (124M) model: model.py, attention.py, mlp.py, block.py, init.py
src/skyai/      the modern model (RMSNorm, RoPE, SwiGLU, GQA): model.py, attention.py, mlp.py, block.py, layers.py, init.py, flash.py
notebooks/      prereq explorations and post-train sanity checks
journal/        module-by-module learning notes
tests/          shape + gradient unit tests, end-to-end smoke, golden numerics fixtures (per family)
scripts/        shard_text.py (data prep), export_hf.py / export_skyai_hf.py (HuggingFace export), hf_skyai/ (the trust_remote_code modeling code shipped with the published model), compare_val_loss.py (matched-protocol scoring), hellaswag.py (standalone eval), train_reference.py (the original Karpathy monolith, a reference)
configs/        gpt2.yaml is the shared recipe anchor; gpt2-muon.yaml and skyai.yaml are the comparison runs; skyai-xl.yaml is the parked scale-up; smoke.yaml is a tiny run
data/           token shards (gitignored)
checkpoints/    saved models (gitignored)
```

## Hardware

NVIDIA RTX 4090 (24GB VRAM), 64GB RAM, WSL Ubuntu on Windows for local
development. Cloud runs are 8×A100 (gpt2-muon) and 4×H100 (skyai) on Lambda;
skyai-xl is staged for 8×H100. `bf16`, Flash Attention, and `torch.compile` are all
in play; `torch.compile` requires Linux (Triton has no Windows wheels).

## Setup

```bash
uv sync                 # install dependencies (resolves the CUDA 12.8 torch wheel)
uv run pytest           # fast suite
uv run skyai doctor     # one-shot env + project sanity check
```

`uv run pytest` asserts CUDA + a 4090 + bf16 and fails anywhere else **by design**. Off that
box, deselect the hardware gate rather than weakening it:

```bash
uv run pytest --ignore=tests/test_environment.py
```

### Data

Training needs GPT-2 BPE token shards in `data/edu_fineweb10B` (~19GB, 100 shards). Either
tokenize FineWeb-Edu yourself (~45–60 min, CPU-bound):

```bash
uv run python scripts/shard_text.py
```

or download the identical pre-tokenized shards (~10 min). Karpathy's preprocessing is
deterministic, so these are byte-identical — the runs here were verified by SHA256 against
locally generated shards:

```bash
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='jfzhang/edu_fineweb10B_tokens_npy_files', repo_type='dataset',
                  local_dir='data/edu_fineweb10B', allow_patterns=['*.npy'], max_workers=16)
"
```

Set `HF_TOKEN` in `.env` (copy `.env.example`) for uncapped download bandwidth. The mirror is
an individual account and could disappear; `scripts/shard_text.py` is the durable path.

## Using the harness

Six subcommands cover the lifecycle — `version`, `train`, `eval`, `sample`,
`doctor`, `ablate`:

```bash
uv run skyai train --config configs/gpt2.yaml             # train gpt2 from scratch
uv run skyai train --config configs/gpt2.yaml --resume    # resume from latest checkpoint
uv run skyai eval --config configs/gpt2.yaml --checkpoint checkpoints/gpt2/best.pt
uv run skyai sample --checkpoint checkpoints/gpt2/best.pt --prompt "Hello"
uv run skyai doctor --config configs/gpt2.yaml            # env + config sanity
uv run skyai ablate --spec ablations/softcap.yaml --output-dir runs/softcap --dry-run
```

Configs are YAML, validated through pydantic, with single-inheritance via
`extends:`. Any field can be overridden on the CLI:

```bash
uv run skyai train --config configs/gpt2.yaml --override schedule.max_steps=1000
```

The default configs are sized for 80GB cloud GPUs. On a single 24GB card, shrink the
micro-batch — the harness recomputes gradient accumulation to preserve the 524,288-token
effective batch, so the optimization math is unchanged:

```bash
uv run skyai train --config configs/gpt2.yaml  --override data.batch_size=8
uv run skyai train --config configs/skyai.yaml --override data.batch_size=8
```

Multi-GPU is just `torchrun`:

```bash
uv run torchrun --standalone --nproc_per_node=8 -m harness.cli.main train --config configs/gpt2.yaml
```

Each family pins its numerics with a golden-fixture short-run test
(`tests/test_golden.py`, `tests/test_golden_gpt2.py`) so an accidental numerical
change on refactor breaks loudly.

## Reproducing the ladder

```bash
uv run python scripts/shard_text.py                    # data, once (~19GB)

# rung 2 — 8×A100, ~2.1h, ~$24
uv run torchrun --standalone --nproc_per_node=8 -m harness.cli.main train --config configs/gpt2-muon.yaml
# rung 3 — 4×H100, ~2h33m, ~$35
uv run torchrun --standalone --nproc_per_node=4 -m harness.cli.main train --config configs/skyai.yaml

uv run python scripts/compare_val_loss.py              # matched-protocol paired re-score
uv run skyai eval --config configs/skyai.yaml --checkpoint checkpoints/skyai/best.pt
```

## What was hard

The parts that took real work, and what they taught:

- **Muon's learning rates are not portable across architectures.** The canonical modern-stack
  values (`embedding_lr: 0.3`) provably diverge on the faithful gpt2 — tied `wte` plus
  LayerNorm plus biases cannot absorb them — so the baseline needed its own retune. That is an
  architecture-*caused* constraint, not a free knob, which is why the two configs legitimately
  differ on LR while still forming a controlled comparison.
- **A constant inherited from a wider model silently capped the architecture.** skyai's q/k
  sharpening (1.2) came from nanochat, which uses head_dim 128; skyai uses 64. Because QK-norm
  pins ‖q‖ = ‖k‖, that multiplier *fixes* the maximum attention logit — at 11.52 here, while a
  confident previous-token head over ~1000 keys needs a gap of ~11.50. Measured on the trained
  checkpoints: gpt2 has 5/144 heads with top-1 attention > 0.8 (sharpest 0.996, attending
  offset 1 for 95.9% of queries); **skyai has 0/144**. Sharp positional circuits are outside
  its representable set — a dimensional-analysis bug, not a coding one.
- **Knowing which bugs invalidate a result and which don't.** An audit found Muon's
  second-moment debias used `1 - beta2` instead of `1 - beta2**t`, inflating the effective
  learning rate 3.16× once the EMA warmed up. Real bug, now fixed — but *symmetric* across
  both runs, so the comparison between them still holds. The val-budget bug above was the
  opposite case: asymmetric, and therefore disqualifying until corrected.
- **Measuring the thing you claim.** FA3 was integrated, verified correct, and then measured
  to be worth ~0% at this sequence length. Reporting that is more useful than not measuring it.

## References

- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU): Karpathy's video
- [build-nanogpt](https://github.com/karpathy/build-nanogpt): companion repo
- [nanochat](https://github.com/karpathy/nanochat): modern speedrun harness
- [llm.c](https://github.com/karpathy/llm.c): pure C/CUDA reference
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): the original transformer paper
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): GPT-2 paper
- [RoFormer (RoPE)](https://arxiv.org/abs/2104.09864), [GLU Variants (SwiGLU)](https://arxiv.org/abs/2002.05202), [GQA](https://arxiv.org/abs/2305.13245), [Muon](https://kellerjordan.github.io/posts/muon/): modern architecture and optimizer
