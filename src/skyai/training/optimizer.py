"""Optimizer constrution with GPT-2 weight-decay policy"""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn

def build_optimizer(
        model: nn.Module,
        *,
        learning_rate: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        device_type: str = "cuda",
) -> torch.optim.Optimizer:
    """Build AdamW with GPT-2's parameter group weight decay policy"""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("Model has no trainable parameters")
    
    decay_params = [p for p in params if p.dim() >= 2]
    nodecay_params = [p for p in params if p.dim() < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"

    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, fused=use_fused)