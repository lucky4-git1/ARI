"""Sparse Mixture-of-Experts (MoE) Architecture for Ari-LLM.

Provides top-K gated sparse expert routing to scale total parameter capacity
while maintaining low active parameter count and high token throughput on CPU.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from arillm.model.config import ModelConfig
from arillm.model.mlp import SwiGLUMLP


class SparseMoEBlock(nn.Module):
    """Top-K Gated Sparse Mixture-of-Experts MLP Layer.
    
    Routes each token dynamically to the top-K experts based on learned
    gating probabilities. Incorporates auxiliary load-balancing loss during
    training to prevent expert collapse.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.aux_loss_coeff = config.moe_aux_loss_coeff

        # Gating router: maps token embeddings to expert logits
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        # Expert pool: SwiGLU feed-forward networks
        self.experts = nn.ModuleList([
            SwiGLUMLP(config) for _ in range(self.num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through Sparse MoE block.
        
        Args:
            x: Tensor of shape [batch, seq_len, hidden_dim]
            
        Returns:
            out: Tensor of shape [batch, seq_len, hidden_dim]
            aux_loss: Load balancing auxiliary loss scalar
        """
        batch_size, seq_len, hidden_dim = x.shape
        flat_x = x.view(-1, hidden_dim)  # [num_tokens, hidden_dim]
        num_tokens = flat_x.size(0)

        # 1. Compute gating logits
        router_logits = self.gate(flat_x)  # [num_tokens, num_experts]

        # 2. Compute probabilities for load balancing loss
        routing_probs = F.softmax(router_logits, dim=-1)

        # 3. Select Top-K experts
        topk_logits, topk_indices = torch.topk(
            router_logits, self.num_experts_per_tok, dim=-1
        )  # [num_tokens, num_experts_per_tok]

        # Normalize weights over selected experts
        topk_weights = F.softmax(topk_logits, dim=-1)  # [num_tokens, num_experts_per_tok]

        # 4. Compute auxiliary load balancing loss (Switch Transformer / GShard formulation)
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.training and self.aux_loss_coeff > 0.0:
            # Average probability routed to each expert: P_e = (1 / N) * sum(P_i,e)
            density_p = torch.mean(routing_probs, dim=0)  # [num_experts]

            # Fraction of tokens dispatched as top-1 to each expert: f_e = (1 / N) * sum(top_1 == e)
            top1_indices = topk_indices[:, 0]
            mask_dispatched = F.one_hot(top1_indices, num_classes=self.num_experts).float()
            fraction_f = torch.mean(mask_dispatched, dim=0)  # [num_experts]

            # Auxiliary loss encourages uniform dispatch: num_experts * sum(f_e * P_e)
            aux_loss = self.aux_loss_coeff * self.num_experts * torch.sum(density_p * fraction_f)

        # 5. Dispatch tokens to active experts
        flat_out = torch.zeros_like(flat_x)

        for expert_idx in range(self.num_experts):
            for k in range(self.num_experts_per_tok):
                # Mask tokens where this expert was selected at position k
                expert_mask = (topk_indices[:, k] == expert_idx)
                if expert_mask.any():
                    selected_inputs = flat_x[expert_mask]
                    expert_output = self.experts[expert_idx](selected_inputs)
                    weights = topk_weights[expert_mask, k].unsqueeze(-1)
                    flat_out[expert_mask] += expert_output * weights

        out = flat_out.view(batch_size, seq_len, hidden_dim)
        return out, aux_loss
