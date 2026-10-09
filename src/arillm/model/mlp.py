"""Feed-Forward Networks and SwiGLU MLP for Ari-LLM."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from arillm.model.config import ModelConfig


class SwiGLUMLP(nn.Module):
    """SwiGLU Multi-Layer Perceptron.
    
    SwiGLU combines Swish (SiLU) gating with linear transformation:
        SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
    Shown to deliver improved perplexity and gradient flow in modern LLMs.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate and up linear projections followed by SiLU gating
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        activated = gate * up
        out = self.down_proj(activated)
        return self.dropout(out)
