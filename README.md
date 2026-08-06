# SkyAI

A from-scratch language model project: build two decoder LMs by hand — a faithful
**gpt2 (124M)** and a modern stack, **skyai**, trained head-to-head at the same
scale on the same tokens — sharing one training harness. Every layer, every
optimizer step, and every tokenization decision is implemented and explained
rather than imported.

> **🤗 The faithful gpt2 is trained and published →** [MuteBuster/gpt2-muon-124m](https://huggingface.co/MuteBuster/gpt2-muon-124m) — val_loss 2.99, HellaSwag acc_norm 0.324, beating the AdamW reference.
>
> **Both models are now trained head-to-head on 10B FineWeb-Edu tokens.** The modern stack
> edges the faithful architecture on likelihood (val_loss 2.9548 vs 2.9653, LAMBADA ppl
> 26.25 vs 27.81) at ~8% fewer FLOPs/token, and ties on accuracy benchmarks. Single-seed;
> see [`.agents/runs/skyai-124m.md`](./.agents/runs/skyai-124m.md) for the full caveats.

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

- **Exactly 124,439,808 parameters** — the canonical GPT-2-small count, to the parameter.
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

The headline cloud run swapped the AdamW+cosine baseline for a **Muon-split optimizer
+ warmup-stable-decay** schedule ([`configs/gpt2-muon.yaml`](./configs/gpt2-muon.yaml)):
Newton-Schulz orthogonalized momentum on the 2D matrices, AdamW on the embeddings /
norms / biases, LR annealed to zero over the final 40%. The canonical Muon learning
rates diverge on the faithful arch (tied embeddings + LayerNorm + biases), so they were
re-tuned down. On 8×A100, ~2.1h over 10B FineWeb-Edu tokens:

| metric | **gpt2-muon** | GPT-2 124M | nanoGPT / llm.c (AdamW) |
|---|---|---|---|
| val_loss (FineWeb-Edu) | **2.99** | ~3.29 | 3.28 |
| HellaSwag (acc_norm) | **0.324** | 0.294 | ~0.30 |
| LAMBADA (ppl) | **27.8** | ~35 | — |

The Muon recipe beats the AdamW+cosine reference on val_loss and HellaSwag at matched
data and scale — a recipe-level, single-seed result (the
[model card](https://huggingface.co/MuteBuster/gpt2-muon-124m) carries the full caveats).
The weights are public:

**🤗 [huggingface.co/MuteBuster/gpt2-muon-124m](https://huggingface.co/MuteBuster/gpt2-muon-124m)** —
load with `AutoModelForCausalLM.from_pretrained("MuteBuster/gpt2-muon-124m")`.

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

On parameters: untied embeddings put skyai at ~151.6M against gpt2's 124.5M, but the
compute-bearing **non-embedding** params run *lower* — ~74.3M vs ~85.1M — because GQA
shrinks the attention projections while the SwiGLU MLP is deliberately 8/3-scaled to
stay FLOP-matched to the GELU one. The entire total-param gap is the untied output table,
which costs no FLOPs. Compare on FLOPs/token, never on total parameters.

Full numbers, protocol, and the complete caveat list:
[`.agents/runs/skyai-124m.md`](./.agents/runs/skyai-124m.md).

[`configs/skyai-xl.yaml`](./configs/skyai-xl.yaml) (48 layers, 1536 hidden, 32 heads,
8 KV heads, 2048 context, cl100k) stages a ~1.5B run for 8×H100 and is parked.

## Project layout

```
src/harness/    training framework: cli/, config/, data/, eval/, training/ + checkpoint.py, generate.py, sample.py, ablation.py, log.py, wandb_logger.py
src/gpt/        the faithful gpt2 (124M) model: model.py, attention.py, mlp.py, block.py, init.py
src/skyai/      the modern model (RMSNorm, RoPE, SwiGLU, GQA): model.py, attention.py, mlp.py, block.py, layers.py, init.py, flash.py
notebooks/      prereq explorations and post-train sanity checks
journal/        module-by-module learning notes
tests/          shape + gradient unit tests, end-to-end smoke, golden numerics fixtures (per family)
scripts/        shard_text.py (data prep) + train_reference.py (the original Karpathy monolith, a reference)
configs/        gpt2.yaml is the shared recipe anchor; gpt2-muon.yaml and skyai.yaml are the comparison runs; skyai-xl.yaml is the parked scale-up; smoke.yaml is a tiny run
data/           token shards (gitignored)
checkpoints/    saved models (gitignored)
```

## Hardware

NVIDIA RTX 4090 (24GB VRAM), 64GB RAM, WSL Ubuntu on Windows for local
development. Cloud runs are 8×A100 / 8×H100 on Lambda. `bf16`, Flash Attention,
and `torch.compile` are all in play; `torch.compile` requires Linux (Triton has no
Windows wheels).

## Setup

```bash
uv sync                 # install dependencies (resolves the CUDA 12.8 torch wheel)
uv run pytest           # full test suite (asserts CUDA + 4090 + bf16; fails on other hardware by design)
uv run skyai doctor     # one-shot env + project sanity check
```

The test suite's environment assertions will fail anywhere that isn't a 4090
box. That's intentional, not a bug to relax.

## Using the harness

Six subcommands cover the lifecycle — `version`, `train`, `eval`, `sample`,
`doctor`, `ablate`:

```bash
uv run skyai train --config configs/gpt2.yaml             # train gpt2 from scratch
uv run skyai train --config configs/gpt2.yaml --resume    # resume from latest checkpoint
uv run skyai eval --config configs/gpt2.yaml --checkpoint checkpoints/gpt2/best.pt
uv run skyai sample --checkpoint checkpoints/gpt2/best.pt --prompt "Hello"
uv run skyai doctor --config configs/gpt2.yaml            # env + config sanity
```

Configs are YAML, validated through pydantic, with single-inheritance via
`extends:`. Any field can be overridden on the CLI:

```bash
uv run skyai train --config configs/gpt2.yaml --override schedule.max_steps=1000
```

Multi-GPU is just `torchrun`:

```bash
torchrun --standalone --nproc_per_node=8 -m harness.cli.main train --config configs/gpt2.yaml
```

Each family pins its numerics with a golden-fixture short-run test
(`tests/test_golden.py`, `tests/test_golden_gpt2.py`) so an accidental numerical
change on refactor breaks loudly.

## References

- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU): Karpathy's video
- [build-nanogpt](https://github.com/karpathy/build-nanogpt): companion repo
- [nanochat](https://github.com/karpathy/nanochat): modern speedrun harness
- [llm.c](https://github.com/karpathy/llm.c): pure C/CUDA reference
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): the original transformer paper
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): GPT-2 paper
- [RoFormer (RoPE)](https://arxiv.org/abs/2104.09864), [GLU Variants (SwiGLU)](https://arxiv.org/abs/2002.05202), [GQA](https://arxiv.org/abs/2305.13245), [Muon](https://kellerjordan.github.io/posts/muon/): modern architecture and optimizer
