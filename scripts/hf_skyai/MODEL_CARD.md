---
license: mit
language:
  - en
library_name: transformers
pipeline_tag: text-generation
tags:
  - gpt
  - causal-lm
  - from-scratch
  - rope
  - swiglu
  - gqa
  - muon
datasets:
  - HuggingFaceFW/fineweb-edu
---

# skyai-modern-xs

A from-scratch modern decoder LM (~152M parameters) trained on 10B FineWeb-Edu tokens,
built as the third rung of a controlled architecture comparison. Every layer, the
optimizer, and the training harness are hand-written — nothing is imported from an
existing model implementation.

**Repository:** https://github.com/northaxosky/sky-ai · **Baseline it is compared against:**
[muteptr/gpt2-muon-124m](https://huggingface.co/muteptr/gpt2-muon-124m)

## What this is

Three models were trained on the **identical token stream** (gpt2 BPE, FineWeb-Edu,
10B tokens, 524,288-token batches, warmup-stable-decay schedule), each rung changing one
coherent thing:

| rung | architecture | recipe |
|---|---|---|
| 1 | faithful GPT-2 | AdamW + cosine |
| 2 | faithful GPT-2 | Muon-split + WSD |
| **3 (this model)** | **modern stack** | Muon-split + WSD |

The modern stack is RMSNorm (plus a post-embedding norm), RoPE, SwiGLU, grouped-query
attention (12 query / 3 KV heads), QK-normalization with q/k sharpening, untied
input/output embeddings, and final-logit soft-capping.

## Results

Both trained models were re-scored on the **identical** full 100M-token FineWeb-Edu
validation shard (the in-training numbers were not comparable — see "Measurement" below).

| metric | **skyai-modern-xs** | gpt2-muon-124m | verdict |
|---|---|---|---|
| val_loss (FineWeb-Edu) | **2.9548** | 2.9653 | **+0.0104** |
| LAMBADA perplexity | **26.25** | 27.81 | **−5.6%** |
| HellaSwag acc_norm | 0.3256 | 0.3238 | not significant (p = 0.56) |
| HellaSwag acc | 0.3008 | 0.2991 | not significant |
| LAMBADA accuracy | 0.2793 | 0.2826 | not significant |
| forward FLOPs/token | **244.8 M** | 266.0 M | **−8.0%** |

The val_loss gap is precisely measured: paired 95% CI **[+0.0102, +0.0107]**, lower on
65.4% of 97,656 sequences, and favoured in **190 of 190** disjoint blocks of the shard.

**The pattern is the finding.** Every likelihood-based metric favours the modern stack;
every accuracy-based metric is statistically null. It produces better-calibrated
probabilities at ~8% fewer FLOPs per token, without measurably changing task accuracy at
this scale.

## Caveats — please read before citing these numbers

1. **Single seed per arm.** The gap is measured to ±0.0002 *on this validation set*, but
   that is measurement precision, not run-to-run uncertainty. Seed-to-seed variance at this
   scale is commonly 0.01–0.03 nats — the same order as the effect. **Precisely measured,
   not replicated.**
2. **This is a bundle, not an isolated architecture change.** The initialization policy,
   gradient clipping (1.0 → off), and per-group learning rates moved together with the
   architecture. Only part of the LR difference is forced by the architecture.
3. **Not parameter-matched.** +21.8% total parameters (entirely the untied output
   embedding, which costs no FLOPs), −12.6% non-embedding parameters. Compare on
   FLOPs/token or non-embedding parameters, never on totals.
4. **Accuracy benchmarks are null.** A val_loss win with null downstream accuracy is a
   weak generalisation claim.
5. **Tuning effort was asymmetric** and the direction is unresolved: local LR search went
   to the baseline, while this model used canonical modern-stack values that encode
   substantial external tuning for exactly this architecture family.
6. **A known ceiling limits this model's attention.** The q/k sharpening constant (1.2) was
   inherited from a reference implementation with head_dim 128; this model uses head_dim 64.
   Because QK-norm pins ‖q‖ = ‖k‖, that fixes the maximum attention logit at 11.52 — right
   at the threshold a confident previous-token head requires. Measured on the checkpoints,
   the GPT-2 baseline has 5/144 heads with top-1 attention > 0.8; this model has **0/144**.
   Sharp positional circuits are effectively outside its representable set.
7. **Do not read this as beating OpenAI's GPT-2.** Both models here trained on FineWeb-Edu;
   GPT-2 did not. That would be home-field advantage.

Full protocol, statistics, and the complete audit are in the
[project repository](https://github.com/northaxosky/sky-ai).

## Measurement

The in-training validation numbers of the two runs were **not comparable**: the harness
counted validation *micro-batches* rather than tokens, so the baseline (batch 64, 8 GPUs)
scored 10.49M tokens while this model (batch 32, 4 GPUs) scored 2.62M. Re-scoring both
checkpoints on identical windows found the asymmetry had biased **against** this model —
the honest gap is +0.0104, not the +0.0082 the raw logs implied. All numbers above use the
matched protocol. (Absolute val_loss is protocol-sensitive: this checkpoint reads 2.9810 on
2.6M tokens and 2.9548 on 100M.)

## Usage

The architecture is not a built-in `transformers` class, so the repo ships its own modeling
code and requires `trust_remote_code=True`.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "muteptr/skyai-modern-xs", trust_remote_code=True
).eval()
tokenizer = AutoTokenizer.from_pretrained("muteptr/skyai-modern-xs")

ids = tokenizer("Hello, I am a language model,", return_tensors="pt").input_ids
out = model.generate(ids, max_new_tokens=40, do_sample=True, top_k=50, temperature=0.8)
print(tokenizer.decode(out[0]))
```

The exported model reproduces the training-harness logits **exactly** (max |Δ| = 0.0,
100% argmax agreement). Note there is no KV cache: generation recomputes the prefix each
step, which is fine at this scale but not optimized for throughput.

## Model details

| | |
|---|---|
| Parameters | 151,584,768 total / 74,317,824 non-embedding |
| Layers / width / heads | 12 / 768 / 12 query, 3 KV (GQA) |
| Context length | 1024 |
| Tokenizer | GPT-2 BPE (50,257; padded to 50,304 internally) |
| Normalization | RMSNorm (parameter-free), pre-norm, plus post-embedding norm |
| Positional encoding | RoPE (θ = 100,000) |
| Feed-forward | SwiGLU, 8/3-scaled (hidden 2048) — FLOP-matched to a 4× GELU MLP |
| Attention | QK-norm + 1.2 sharpening; FlashAttention-3 in training, SDPA here |
| Output | Untied embeddings, logit soft-cap at 15.0 |
| Optimizer | Muon (Newton–Schulz orthogonalized momentum) on 2-D matrices, AdamW elsewhere |
| Schedule | Warmup-stable-decay, 715 warmup steps, decay to zero over the final 40% |
| Precision | bf16 mixed precision |
| Training | 19,073 steps × 524,288 tokens = 10B tokens, 4×H100, 2h33m |

## Intended use and limitations

This is a **research artifact for studying architecture and optimizer choices at small
scale**, not a product. At ~152M parameters trained on 10B tokens it is far below the
capability of any modern assistant model. It is a base model with no instruction tuning,
no alignment, and no safety filtering, trained on web text (FineWeb-Edu) — it will produce
factually wrong, biased, or otherwise objectionable output. It is best used for research
into training dynamics and architecture comparison, not for deployment.

## Citation

```bibtex
@software{skyai2026,
  author = {Kuzey Gok},
  title  = {SkyAI: a controlled comparison of GPT-2 and a modern decoder stack},
  year   = {2026},
  url    = {https://github.com/northaxosky/sky-ai}
}
```
