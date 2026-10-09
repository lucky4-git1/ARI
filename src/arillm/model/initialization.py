"""Parameter initialization routines for Ari-LLM."""

import math
import torch
import torch.nn as nn
from arillm.model.config import ModelConfig


def init_weights(module: nn.Module, config: ModelConfig):
    """Initializes weights from scratch using normal distribution scaled by depth for residual layers."""
    std = config.init_std
    if isinstance(module, nn.Linear):
        # Truncated or normal initialization
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
    elif hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
        # RMSNorm or other normalization weights
        if module.weight.dim() == 1:
            torch.nn.init.ones_(module.weight)


def apply_residual_scaling(model: nn.Module, config: ModelConfig):
    """Scales residual projection weights by 1 / sqrt(2 * num_layers) to stabilize deep Transformer training."""
    scale = 1.0 / math.sqrt(2 * config.num_layers)
    for name, param in model.named_parameters():
        if "out_proj.weight" in name or "down_proj.weight" in name:
            with torch.no_grad():
                param.mul_(scale)
