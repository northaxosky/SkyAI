from __future__ import annotations
from dataclasses import dataclass
from typing import cast
import math
import torch, torch.nn as nn
from torch.nn import functional as F
import sys
import time
import inspect

# ===============================

@dataclass
class GPTConfig:
    block_size: int = 1024      # Maximum Sequence Length
    vocab_size: int = 50257     # Number of Tokens: 50k BPE merges + tokens
    n_layer: int    = 12        # Number of Layers
    n_head: int     = 12        # Number of Heads
    n_embed: int    = 768       # Embedding Dimension


class CausalSelfAttention(nn.Module):
    config: GPTConfig
    c_attn: nn.Linear
    c_proj: nn.Linear
    
    n_head: int
    n_embed: int
    bias: torch.Tensor

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embed % config.n_head == 0

        # Key, Query, Value projections for all heads in a batch
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        
        # Output Projection and regularization
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # pyright: ignore
        self.n_head = config.n_head
        self.n_embed = config.n_embed

        # Mask/Bias following the OpenAI/HF naming
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get the batch size, sequence length, & embedding dimensionality
        B, T, C = x.size()

        # Calculate query, key, & values for all heads in batch and move head forward
        # nh: # of heads, hs: head size, C: number of channels (nh * ns)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Attention: materialize the large (T, T) matrix for all the queries and keys
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.c_proj(y)
        return y
    
class MLP(nn.Module):
    config: GPTConfig
    c_fc: nn.Linear
    gelu: nn.GELU
    c_proj: nn.Linear

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # pyright: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    config: GPTConfig
    ln_1: nn.LayerNorm
    attn: CausalSelfAttention
    ln_2: nn.LayerNorm
    mlp: MLP

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    config: GPTConfig
    transformer: nn.ModuleDict
    lm_head: nn.Linear

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Follow Hugging Face schema so we can load it easily
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed), 
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # Weight Sharing Scheme
        self.transformer.wte.weight = self.lm_head.weight # pyright: ignore

        # Initialize parameters
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        wte    = cast(nn.Embedding,  self.transformer.wte)
        wpe    = cast(nn.Embedding,  self.transformer.wpe)
        blocks = cast(nn.ModuleList, self.transformer.h)
        ln_f   = cast(nn.LayerNorm,  self.transformer.ln_f)

        # idx is of shape (B, T)
        _, T = idx.size()
        assert T <= self.config.block_size, f'Cannot forward sequence of length{T}, block size invalid'

        # Forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = wpe(pos)
        tok_emb = wte(idx)
        x = tok_emb + pos_emb

        # Forward the blocks of the transformer
        for block in blocks:
            x = block(x)

        # Forward the final layernorm and the classifier
        x = ln_f(x)
        logits = self.lm_head(x) 
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: str) -> GPT:
        """ Loads pre-trained GPT-2 model weights from Hugging Face """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f'Loading weights from pre-trained GPT: {model_type}')

        # n_layer, n_head, n_embed are determined from model type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embed=768),   # 124M Parameters
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embed=1024),  # 350M Parameters
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embed=1280),  # 774M Parameters
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embed=1600),  # 1558M Parameters
        }[model_type]

        config_args['vocab_size'] = 50257 # Always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # Always 1024 for GPT model checkpoints

        # Create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # Discard

        # Initialize a hugging face/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Copy while ensuring all of the parameters are aligned
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # Transpose the weights (OpenAI checkpoints with "Conv1D" module)
        assert len(sd_keys_hf) == len(sd_keys), f'Mismatched Keys: {len(sd_keys_hf)} != {len(sd_keys)}'
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] ==sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Vanilla copy of the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # Start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Create optim groups where 2D parameters will be weight delayed
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f'# Decated Parameters Tensors: {len(decay_params)}, with {num_decay_params:,} parameters')
        print(f'# Non-decayed Parameter Tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters')

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# ============================================
import tiktoken

class DataLoaderLite:
    def __init__(self, B: int, T: int):
        self.B = B
        self.T = T

        # Load tokens from disk and store in memory
        with open('../../data/shakespeare.txt', 'r') as file:
            text = file.read()
        enc = tiktoken.get_encoding('gpt2')
        self.tokens = torch.tensor(enc.encode(text))

        print(f'Loaded {len(self.tokens)} tokens')
        print(f'1 Epoch = {len(self.tokens) // (B * T)} batches')

        self.current_position = 0
    
    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + (B * T) + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        # Advance the position in the tensor
        self.current_position += B * T
        
        # If loading the next batch would be out of bounds: reset
        if self.current_position + (B * T) + 1 > len(self.tokens):
            self.current_position = 0
        return x, y

# ============================================
# Attempt to auto-detect the device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f'Using device: {device}')

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# Train loader
train_loader = DataLoaderLite(B=16, T=1024)
torch.set_float32_matmul_precision('high')

# model = GPT.from_pretrained('gpt2-xl')
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
if sys.platform == 'linux':
    model = torch.compile(model)
    print('Using torch-compiled GPT')

# Cosine decay learning rate
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
    # Linear warmup for warmup_iters steps, min learning rate when past max steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    
    # Use cosine decay down to minimum learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8) # pyright: ignore

for step in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()

    # Use bfloat16 for optimal speed/precision
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)

    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f'Step {step + 1}, LR: {lr:.4f}  Loss: {loss.item()}, Norm: {norm:.4f}, dT: {dt:.2f}ms, Tok/sec: {tokens_sec:.2f}')    

sys.exit(0)
# ============================================
# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, I am a language model,")
# tokens = torch.tensor(tokens, dtype=torch.long)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
# x = tokens.to(device)

# # Generation
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

# while x.size(1) < max_length:
#     # Forward the model to get the logits
#     with torch.no_grad():
#         logits = model(x)

#         # Take the logits at the last position & get the probabilities
#         logits = logits[:, -1, :]
#         probs = F.softmax(logits, dim=-1)

#         # do top-k sampling of 50 & select a token 
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
#         ix = torch.multinomial(topk_probs, 1)

#         # Gather the corresponding indices & append to the sequence
#         xcol = torch.gather(topk_indices, -1, ix)
#         x = torch.cat((x, xcol), dim=1)

# # Print the generated text
# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(f'> {decoded}')