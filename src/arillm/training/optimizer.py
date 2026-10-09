"""Optimizer and learning rate scheduler builders for Ari-LLM."""

import math
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """Configures AdamW with weight decay applied only to 2D matrix weights (not norms/biases)."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Biases and 1D normalization weights do not receive weight decay
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps)
    return optimizer


def create_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Cosine learning rate scheduler with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay phase
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)
