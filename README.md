# SkyAI

Two decoder language models built by hand: a faithful **gpt2 (124M)** and a modern stack,
**skyai**. Both train head-to-head at the same scale on the same tokens, through one shared
harness. Every layer, every optimizer step, and every tokenizer decision is written here,
not imported.

> **🤗 Both models are trained and published.** On 10B FineWeb-Edu tokens at matched scale, the
> modern stack wins on likelihood: **val_loss 2.9548 vs 2.9653** and **LAMBADA perplexity
> 26.25 vs 27.81**, at **~8% fewer FLOPs/token**. On accuracy benchmarks the two are tied.
> Both runs are single-seed, and the architecture changed as a bundle. The
> [run journal](./journal/runs/skyai-124m.md) has the full caveats.
>
> Models: [`muteptr/skyai-modern-xs`](https://huggingface.co/muteptr/skyai-modern-xs) and
> [`muteptr/gpt2-muon-124m`](https://huggingface.co/muteptr/gpt2-muon-124m)

## Why this repo exists

Calling an API is one thing. Building the thing the API calls is another. This project is
about the second kind of understanding: writing attention by hand, watching gradients flow,
debugging shape mismatches, profiling kernel launches, and training a model from random
weights to coherent(ish) English.

The [`journal/`](./journal/) directory has the reasoning behind each module. Read it next to
the code to follow the thinking, not just the result.

## gpt2: a faithful GPT-2 (124M)

[`src/gpt/`](./src/gpt/) is GPT-2 small, rebuilt clean-room. LayerNorm, learned positional
embeddings, GELU MLP, standard multi-head attention, tied input/output embeddings, and
biases throughout. Three things pin it down:

- **124,439,808 parameters** at the canonical `vocab_size=50257`, which is GPT-2-small to the
  parameter. Asserted in [`tests/gpt/test_model.py`](./tests/gpt/test_model.py). Training pads
  the vocab to 50,304 for tensor-core alignment, so the published checkpoint has 124,475,904.
- **`from_pretrained` matches OpenAI.** Loading the released `gpt2` weights (the four Conv1D
  transposes included) reproduces HuggingFace's logits to floating-point tolerance, with
  identical argmax predictions.
- **A golden short-run test** ([`tests/test_golden_gpt2.py`](./tests/test_golden_gpt2.py))
  pins the training numerics, so accidental drift breaks the build.

Recipe ([`configs/gpt2.yaml`](./configs/gpt2.yaml)): standard nanoGPT. AdamW with weight
decay 0.1 on 2D parameters and 0 elsewhere, cosine LR with linear warmup, gradient clipping
at 1.0, gradient accumulation to a 0.5M-token batch, bf16 autocast, Flash Attention through
PyTorch SDPA, fused AdamW, `torch.compile`, and weight tying. Trained on FineWeb-Edu.

### Result: trained with the Muon recipe

This run replaced AdamW+cosine with a **Muon-split optimizer and a warmup-stable-decay
schedule** ([`configs/gpt2-muon.yaml`](./configs/gpt2-muon.yaml)). Muon applies
Newton-Schulz orthogonalized momentum to the 2D matrices; AdamW handles the embeddings,
norms, and biases. The LR anneals to zero over the final 40%.

The canonical Muon learning rates diverge on this architecture, because it ties the
embeddings and uses LayerNorm and biases. They had to be re-tuned down. On 8×A100, about 2.1h over 10B FineWeb-Edu tokens:

| metric | **gpt2-muon** | GPT-2 124M | nanoGPT / llm.c (AdamW) |
|---|---|---|---|
| val_loss (FineWeb-Edu val shard) | **2.9653** | ~3.29 | 3.28 |
| HellaSwag (acc_norm) | **0.3238** | 0.294 | ~0.30 |
| LAMBADA (ppl) | **27.81** | ~35 | n/a |

Three notes on that table:

- GPT-2 was scored on FineWeb-Edu, which it never trained on. That is home-field advantage,
  not a win on language-model quality. Use HellaSwag for the neutral comparison.
- llm.c does not say whether it reports `acc` or `acc_norm`, so those columns may not be
  strictly like-for-like.
- The GPT-2 LAMBADA figure comes from the community `lm-evaluation-harness`, not the paper.

This improves on the published AdamW+cosine reference at matched data and scale, but it's
measured against literature rather than an in-house baseline, which makes it a looser
comparison than the skyai head-to-head below. Single-seed, and recipe-level rather than
attributable to Muon alone. The
[model card](https://huggingface.co/muteptr/gpt2-muon-124m) and the
[run journal](./journal/runs/gpt2-muon-124m.md) carry the full caveats.

Load it with
`AutoModelForCausalLM.from_pretrained("muteptr/gpt2-muon-124m")`.

## skyai: the modern stack, at gpt2 scale

[`src/skyai/`](./src/skyai/) rebuilds the model around the current decoder-LM stack:

- RMSNorm, plus a post-embedding norm, in place of LayerNorm
- Rotary position embeddings (RoPE) instead of learned positions
- SwiGLU feed-forward instead of the GELU MLP
- Grouped-query attention (GQA)
- QK-normalization with q/k sharpening
- Untied input and output embeddings
- Logit soft-capping
- Internal vocab padding kept separate from the logical vocab

Training recipe: Muon for the transformer matrices, AdamW for the embeddings and head, a
warmup-stable-decay schedule, and per-group learning rates scaled by width and batch size.

### Result: the three-way comparison

[`configs/skyai.yaml`](./configs/skyai.yaml) holds gpt2's depth, width, context, tokenizer,
data, token budget, and schedule shape fixed, and changes the architecture. That completes a
ladder where each rung moves one coherent thing:

| run | architecture | recipe | val_loss | HellaSwag acc_norm |
|---|---|---|---|---|
| `gpt2` | faithful GPT-2 | AdamW + cosine | ~3.28 | ~0.30 |
| `gpt2-muon` | faithful GPT-2 | **Muon-split + WSD** | 2.9653 | 0.3238 |
| **`skyai`** | **modern stack** | Muon-split + WSD | **2.9548** | 0.3256 |

Both trained models were re-scored on the same full 100M-token FineWeb-Edu val shard. The
`gpt2` row is a literature reference (build-nanogpt / llm.c) rather than an in-house run, so
treat it as the loosest number in the table.

The modern stack gains **0.0104 nats** of val_loss (paired 95% CI [+0.0102, +0.0107], lower
on 190 of 190 disjoint val blocks) and **5.6% lower LAMBADA perplexity**, at **~8% fewer
FLOPs per token**. On accuracy benchmarks the two are **statistically indistinguishable**:
HellaSwag acc_norm 0.3256 vs 0.3238 (paired McNemar p = 0.56) and LAMBADA accuracy 0.2793 vs
0.2826.

That split is the interesting part: every likelihood metric moves, every accuracy metric
sits still.

Two caveats do most of the work here.

**Single seed on both arms.** Seed-to-seed variance at this scale is plausibly as large as
the gap itself, so the number is measured precisely but never replicated.

**The rung is a bundle.** `init_policy`, `grad_clip`, and the per-group learning rates all
moved along with the architecture, so this measures the modern stack as a package rather than
any one piece of it. For scale, the recipe change in rung 2 was worth roughly ten times more
than the architecture change here.

### A measurement bug, and the correction

The first version of this comparison was wrong. `eval.val_steps` counted *micro-batches*
rather than tokens, so the validation budget quietly scaled with `batch_size × world_size`:
gpt2-muon (B=64, 8 GPUs) scored 10.5M val tokens while skyai (B=32, 4 GPUs) got 2.6M, a 4×
smaller prefix of the same shard. Two numbers that looked comparable weren't.

Re-scoring both checkpoints on identical windows fixes it. The harness reproduces each run's
originally-logged number on its own window, which validates the measurement path:

| window | gpt2-muon | skyai | gap |
|---|---|---|---|
| 2,560 seq (skyai's in-run) | 2.9913 | 2.9810 | +0.0103 |
| 10,240 seq (gpt2-muon's in-run) | 2.9892 | 2.9787 | +0.0105 |
| **97,656 seq (full shard)** | **2.9653** | **2.9548** | **+0.0104** |
| *as originally logged* | 2.9891 | 2.9810 | +0.0082 |

The bias ran *against* skyai, and once corrected the gap holds at +0.0104 across every
window. The harness now takes a token-denominated `eval.val_tokens`
([`loop.py`](./src/harness/training/loop.py)), pinned across all three configs, so val_loss
can't drift with batch or world geometry again.
[`scripts/compare_val_loss.py`](./scripts/compare_val_loss.py) reproduces the re-scoring.

### On parameter counts

Untied embeddings put skyai at ~151.6M parameters against gpt2's 124.5M, which looks bad
until you count the compute-bearing ones: ~74.3M vs ~85.1M, in skyai's favor. GQA shrinks the
attention projections, and the SwiGLU MLP is 8/3-scaled to stay FLOP-matched to the GELU one.
The entire total-parameter gap is the untied output table, which costs no FLOPs at all.

So compare these two on FLOPs/token, never on total parameters.

Full numbers, protocol, and the complete caveat list:
[`journal/runs/skyai-124m.md`](./journal/runs/skyai-124m.md).

### FlashAttention-3, and why it bought nothing here

[`src/skyai/flash.py`](./src/skyai/flash.py) dispatches to FlashAttention-3 when the GPU
reports compute capability 9, and falls back to PyTorch SDPA otherwise. The fast path is
Hopper-only, so it cannot run on the 4090 used for local work. FA3 was built from source for
`sm_90a` on the H100 box and was active for the whole skyai run.

Then it got measured, and it barely registered: **1.03× on the attention op alone, ~0%
end-to-end.** At 1024 context with head_dim 64, attention is about 1% of total FLOPs, 0.89 ms
out of a 76 ms step, so the MLP and the output head dominate everything. FA3's advantage grows
with sequence length, and this configuration is far too short to show it.

The kernel itself is fine. Forward matches SDPA to bf16 tolerance, backward gradients stay
finite under GQA, and an instrumented count confirms it fires once per layer. It just isn't
the bottleneck here.

### Using the published model

skyai is not a built-in `transformers` architecture, so the repo ships its own modeling code
and loads with `trust_remote_code=True`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("muteptr/skyai-modern-xs", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("muteptr/skyai-modern-xs")
```

[`scripts/export_skyai_hf.py`](./scripts/export_skyai_hf.py) does the export. It checks that
the ported model reproduces the harness logits exactly (max |Δ| = 0.0) before
it publishes anything.

[`configs/skyai-xl.yaml`](./configs/skyai-xl.yaml) stages a ~1.5B run for 8×H100: 48 layers,
1536 hidden, 32 heads, 8 KV heads, 2048 context, cl100k. It is parked, not run.

## The harness

[`src/harness/`](./src/harness/) is the model-agnostic half, and it's what makes the
comparison possible at all. One training loop drives both families through a `build_model`
factory, and everything downstream depends only on the model's
`forward(idx, targets) -> (logits, loss)` contract. Adding an architecture means adding a
package under `src/`, not touching the loop.

- **Configs.** Pydantic-validated YAML with `extra="forbid"`, so a typo is a hard error.
  Single inheritance through `extends:`, and `--override a.b.c=value` for any field.
- **Training.** DDP through `torchrun`, gradient accumulation derived from a fixed token
  budget, bf16 autocast, `torch.compile`, AdamW, a hand-written Muon, and cosine plus
  warmup-stable-decay schedules.
- **Robustness.** Crash-safe checkpoint and resume, including RNG and dataloader position.
  Non-finite-gradient detection. OOM diagnostics that dump batch geometry and VRAM.
- **Evaluation.** HellaSwag and LAMBADA written directly, sharded across ranks.
- **Tooling.** `doctor` for env and config preflight, an ablation runner for config sweeps, a
  region profiler, and golden-fixture tests that pin exact losses and a parameter checksum.

## Project layout

```
src/harness/    training framework: cli/, config/, data/, eval/, training/ + checkpoint.py, generate.py, sample.py, ablation.py, log.py, wandb_logger.py
src/gpt/        the faithful gpt2 (124M): model.py, attention.py, mlp.py, block.py, init.py
src/skyai/      the modern model: model.py, attention.py, mlp.py, block.py, layers.py, init.py, flash.py
notebooks/      prereq explorations and post-train sanity checks
journal/        module notes, plus runs/ for the training-run write-ups
tests/          shape + gradient unit tests, end-to-end smoke, golden numerics fixtures
scripts/        shard_text.py (data prep), export_hf.py + export_skyai_hf.py (HuggingFace export), hf_skyai/ (modeling code shipped with the published model), compare_val_loss.py (matched-protocol scoring), hellaswag.py, train_reference.py (Karpathy's monolith, kept as a reference)
configs/        gpt2.yaml is the anchor; gpt2-muon.yaml and skyai.yaml are the comparison runs; skyai-xl.yaml is the parked scale-up; smoke.yaml is a tiny run
data/           token shards (gitignored)
checkpoints/    saved models (gitignored)
```

## Hardware

Local development is one NVIDIA RTX 4090 (24GB), 64GB RAM, on WSL Ubuntu. Cloud runs were
8×A100 for gpt2-muon and 4×H100 for skyai, both on Lambda. skyai-xl is staged for 8×H100.
bf16, Flash Attention, and `torch.compile` are all in use. `torch.compile` needs Linux,
because Triton has no Windows wheels.

## Setup

```bash
uv sync                 # install dependencies (resolves the CUDA 12.8 torch wheel)
uv run pytest           # fast suite
uv run skyai doctor     # env + project sanity check
```

`uv run pytest` asserts CUDA, a 4090, and bf16. It fails anywhere else **by design**. Off
that box, skip the hardware gate instead of weakening it:

```bash
uv run pytest --ignore=tests/test_environment.py
```

### Data

Training needs GPT-2 BPE token shards in `data/edu_fineweb10B`, about 19GB across 100 shards.
Either tokenize FineWeb-Edu yourself, which takes 45 to 60 minutes and is CPU-bound:

```bash
uv run python scripts/shard_text.py
```

Or download the same shards in about 10 minutes. Karpathy's preprocessing is deterministic,
so these are byte-identical; the runs here were verified by SHA256 against locally generated
shards:

```bash
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='jfzhang/edu_fineweb10B_tokens_npy_files', repo_type='dataset',
                  local_dir='data/edu_fineweb10B', allow_patterns=['*.npy'], max_workers=16)
"
```

Set `HF_TOKEN` in `.env` (copy `.env.example`) for uncapped download bandwidth. That mirror is
an individual account and could disappear, so `scripts/shard_text.py` is the durable path.

## Using the harness

Six subcommands cover the lifecycle: `version`, `train`, `eval`, `sample`, `doctor`, and
`ablate`.

```bash
uv run skyai train --config configs/gpt2.yaml             # train gpt2 from scratch
uv run skyai train --config configs/gpt2.yaml --resume    # resume from latest checkpoint
uv run skyai eval --config configs/gpt2.yaml --checkpoint checkpoints/gpt2/best.pt
uv run skyai sample --checkpoint checkpoints/gpt2/best.pt --prompt "Hello"
uv run skyai doctor --config configs/gpt2.yaml            # env + config sanity
uv run skyai ablate --spec ablations/softcap.yaml --output-dir runs/softcap --dry-run
```

Configs are YAML, validated through pydantic, with single inheritance via `extends:`. Any
field can be overridden on the command line:

```bash
uv run skyai train --config configs/gpt2.yaml --override schedule.max_steps=1000
```

The default configs are sized for 80GB cloud GPUs. On a single 24GB card, shrink the
micro-batch. The harness recomputes gradient accumulation to keep the 524,288-token effective
batch, so the optimization math does not change:

```bash
uv run skyai train --config configs/gpt2.yaml  --override data.batch_size=8
uv run skyai train --config configs/skyai.yaml --override data.batch_size=8
```

Multi-GPU is just `torchrun`:

```bash
uv run torchrun --standalone --nproc_per_node=8 -m harness.cli.main train --config configs/gpt2.yaml
```

Each family pins its numerics with a golden-fixture short-run test (`tests/test_golden.py`
and `tests/test_golden_gpt2.py`), so a numerical change during a refactor breaks loudly.

## Reproducing the ladder

```bash
uv run python scripts/shard_text.py                    # data, once (~19GB)

# rung 2: 8×A100, ~2.1h, ~$24
uv run torchrun --standalone --nproc_per_node=8 -m harness.cli.main train --config configs/gpt2-muon.yaml
# rung 3: 4×H100, ~2h33m, ~$35
uv run torchrun --standalone --nproc_per_node=4 -m harness.cli.main train --config configs/skyai.yaml

uv run python scripts/compare_val_loss.py              # matched-protocol paired re-score
uv run skyai eval --config configs/skyai.yaml --checkpoint checkpoints/skyai/best.pt
```

## What was hard

- **Muon's learning rates do not port across architectures.** The canonical modern-stack
  values (`embedding_lr: 0.3`) diverge on the faithful gpt2, because tied `wte` plus LayerNorm
  plus biases cannot absorb them. The baseline needed its own retune. That is a constraint
  caused by the architecture, not a free knob, which is why the two configs can differ on LR
  and still form a controlled comparison.
- **A constant inherited from a wider model silently capped the architecture.** skyai's q/k
  sharpening (1.2) came from nanochat, which uses head_dim 128. skyai uses 64. QK-norm pins
  the norms of q and k, so that multiplier *fixes* the largest attention logit the model can
  produce: 11.52 here, while a confident previous-token head over ~1000 keys needs about
  11.50. Measured on the trained checkpoints, gpt2 has 5 of 144 heads with top-1 attention
  above 0.8 (the sharpest is 0.996, attending to offset 1 for 95.9% of queries). **skyai has
  0 of 144.** Sharp positional circuits are outside its representable set. That is a
  dimensional-analysis bug, not a coding one.
- **Knowing which bugs invalidate a result, and which do not.** An audit found that Muon's
  second-moment debias used `1 - beta2` instead of `1 - beta2**t`, which inflated the
  effective learning rate 3.16× once the EMA warmed up. Real bug, now fixed. But it was
  *symmetric* across both runs, so the comparison between them still holds. The val-budget
  bug above was the opposite case: asymmetric, and disqualifying until corrected.
- **Measuring the thing you claim.** FA3 was integrated, verified, and then measured to be
  worth about 0% at this sequence length. Reporting that is more useful than not measuring it.

## References

- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU): Karpathy's video
- [build-nanogpt](https://github.com/karpathy/build-nanogpt): companion repo
- [nanochat](https://github.com/karpathy/nanochat): modern speedrun harness
- [llm.c](https://github.com/karpathy/llm.c): pure C/CUDA reference
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): the original transformer paper
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): GPT-2 paper
- [RoFormer (RoPE)](https://arxiv.org/abs/2104.09864), [GLU Variants (SwiGLU)](https://arxiv.org/abs/2002.05202), [GQA](https://arxiv.org/abs/2305.13245), [Muon](https://kellerjordan.github.io/posts/muon/): modern architecture and optimizer
