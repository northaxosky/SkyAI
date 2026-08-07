"""HuggingFace config for the skyai modern decoder stack.

Self-contained by design: Hub repos loaded with trust_remote_code cannot import from
this project, so nothing here may reference src/skyai.
"""

from __future__ import annotations

from transformers import PretrainedConfig


class SkyAIConfig(PretrainedConfig):
    model_type = "skyai"

    def __init__(
        self,
        vocab_size: int = 50257,
        vocab_pad_multiple: int = 128,
        n_layer: int = 12,
        n_head: int = 12,
        n_kv_head: int = 3,
        n_embd: int = 768,
        hidden_multiple: int = 4,
        block_size: int = 1024,
        rope_theta: float = 100_000.0,
        qk_sharpen: float = 1.2,
        logit_softcap: float | None = 15.0,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.vocab_pad_multiple = vocab_pad_multiple
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.hidden_multiple = hidden_multiple
        self.block_size = block_size
        self.rope_theta = rope_theta
        self.qk_sharpen = qk_sharpen
        self.logit_softcap = logit_softcap
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def vocab_size_padded(self) -> int:
        m = self.vocab_pad_multiple
        return ((self.vocab_size + m - 1) // m) * m

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    # transformers introspects these generic names for generation and resizing.
    @property
    def hidden_size(self) -> int:
        return self.n_embd

    @property
    def num_attention_heads(self) -> int:
        return self.n_head

    @property
    def num_hidden_layers(self) -> int:
        return self.n_layer

    @property
    def max_position_embeddings(self) -> int:
        return self.block_size
