"""HuggingFace modeling code for the skyai modern decoder stack.

A faithful port of src/skyai (RMSNorm, RoPE, SwiGLU, GQA, QK-norm, untied embeddings,
logit soft-capping). Module names mirror the training code so checkpoints transfer
without key remapping. Self-contained: nothing here may import from this project.

Attention uses SDPA for portability; the training runs used FlashAttention-3 on Hopper,
which is numerically equivalent up to reduction order.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

from .configuration_skyai import SkyAIConfig


class RMSNorm(nn.Module):
    """Parameter-free RMS normalization."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.dim,))


class Linear(nn.Linear):
    """Linear that casts weights to the input dtype, so autocast never upcasts activations."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(dtype=x.dtype)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)


class ResidualProjection(Linear):
    """Marker subclass for projections feeding the residual stream."""


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    y1 = x1 * cos + x2 * sin
    y2 = -x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=-1).to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_kv_head: int, qk_sharpen: float) -> None:
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embd // n_head
        self.qk_sharpen = qk_sharpen

        self.c_q = Linear(n_embd, n_head * self.head_dim, bias=False)
        self.c_k = Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.c_proj = ResidualProjection(n_embd, n_embd, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = self.q_norm(q) * self.qk_sharpen
        k = self.k_norm(k) * self.qk_sharpen

        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True, enable_gqa=True
        ).transpose(1, 2)
        return self.c_proj(y.contiguous().view(B, T, -1))


class MLP(nn.Module):
    """SwiGLU, 8/3-scaled so parameters match a 4x GELU MLP."""

    def __init__(self, n_embd: int, hidden_multiple: int, align: int = 256) -> None:
        super().__init__()
        hidden = int(2 * hidden_multiple * n_embd / 3)
        hidden = ((hidden + align - 1) // align) * align
        self.gate_proj = Linear(n_embd, hidden, bias=False)
        self.up_proj = Linear(n_embd, hidden, bias=False)
        self.down_proj = ResidualProjection(hidden, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config: SkyAIConfig) -> None:
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(
            config.n_embd, config.n_head, config.n_kv_head, config.qk_sharpen
        )
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config.n_embd, config.hidden_multiple)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), cos, sin)
        return x + self.mlp(self.ln_2(x))


class _Transformer(nn.Module):
    def __init__(self, config: SkyAIConfig) -> None:
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size_padded, config.n_embd)
        self.embed_norm = RMSNorm(config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)


class SkyAIPreTrainedModel(PreTrainedModel):
    config_class = SkyAIConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = False
    _no_split_modules = ["Block"]


class SkyAIForCausalLM(SkyAIPreTrainedModel, GenerationMixin):
    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, config: SkyAIConfig) -> None:
        super().__init__(config)
        self.transformer = _Transformer(config)
        self.lm_head = Linear(config.n_embd, config.vocab_size_padded, bias=False)

        head_dim = config.head_dim
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        angles = torch.outer(torch.arange(config.block_size).float(), inv_freq)
        # Persistent, unlike the training code: from_pretrained does not restore
        # non-persistent buffers, which would leave the RoPE tables uninitialised.
        self.register_buffer("cos", angles.cos()[None, :, None, :], persistent=True)
        self.register_buffer("sin", angles.sin()[None, :, None, :], persistent=True)

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.transformer.wte

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.transformer.wte = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutput:
        T = input_ids.size(1)
        if self.config.block_size < T:
            raise ValueError(f"sequence length {T} exceeds block_size {self.config.block_size}")

        x = self.transformer.embed_norm(self.transformer.wte(input_ids))
        cos, sin = self.cos[:, :T], self.sin[:, :T]
        for block in self.transformer.h:
            x = block(x, cos, sin)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)[..., : self.config.vocab_size].float()
        if self.config.logit_softcap is not None:
            cap = self.config.logit_softcap
            logits = cap * torch.tanh(logits / cap)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1)
            )
        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, **kwargs) -> dict:
        # No KV cache: the model recomputes the prefix each step, so keep it inside block_size.
        return {"input_ids": input_ids[:, -self.config.block_size :]}
