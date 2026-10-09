"""Rotary Positional Embeddings (RoPE) and Attention mechanisms for LAYA-LLM."""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from layallm.model.config import ModelConfig


def precompute_rope_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precomputes frequency tensor for complex exponential representation of RoPE."""
    # dim must be even (head_dim)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # [end, dim // 2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex tensor [end, dim // 2]
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary position embeddings to query and key tensors."""
    # xq: [batch, seq_len, num_heads, head_dim]
    # xk: [batch, seq_len, num_kv_heads, head_dim]
    # freqs_cis: [seq_len, head_dim // 2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Broadcast freqs_cis to [1, seq_len, 1, head_dim // 2]
    freqs_cis = freqs_cis.view(1, xq.shape[1], 1, xq.shape[-1] // 2)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats key/value heads for Grouped Query Attention (GQA).
    
    x: [batch, seq_len, num_kv_heads, head_dim]
    returns: [batch, seq_len, num_heads, head_dim]
    """
    if n_rep == 1:
        return x
    batch, seq_len, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(batch, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch, seq_len, n_kv_heads * n_rep, head_dim)
    )


class CausalSelfAttention(nn.Module):
    """Causal Multi-Head / Grouped-Query Attention with Rotary Embeddings and optional KV Caching."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_rep = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.hidden_dim = config.hidden_dim
        
        self.q_proj = nn.Linear(config.hidden_dim, config.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape

        # Projections
        xq = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        xk = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # KV Cache handling (for fast autoregressive CPU decoding)
        if kv_cache is not None:
            past_k, past_v = kv_cache
            xk = torch.cat([past_k, xk], dim=1)
            xv = torch.cat([past_v, xv], dim=1)
        
        current_cache = (xk, xv) if use_cache else None

        # Repeat KV for GQA
        xk_rep = repeat_kv(xk, self.num_rep)
        xv_rep = repeat_kv(xv, self.num_rep)

        # Transpose for scaled dot product attention: [batch, num_heads, seq_len, head_dim]
        q = xq.transpose(1, 2)
        k = xk_rep.transpose(1, 2)
        v = xv_rep.transpose(1, 2)

        # Fast Scaled Dot-Product Attention (PyTorch F.scaled_dot_product_attention)
        # Handles causal masking automatically if is_causal is True and attention_mask is None
        total_seq_len = k.shape[2]
        is_causal = (seq_len > 1 and total_seq_len == seq_len and attention_mask is None)

        if hasattr(F, "scaled_dot_product_attention"):
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            # Fallback manual implementation
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if is_causal:
                mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1)
                scores = scores + mask
            elif attention_mask is not None:
                scores = scores + attention_mask
            attn_weights = F.softmax(scores, dim=-1)
            output = torch.matmul(self.dropout(attn_weights), v)

        # Reshape output back to [batch, seq_len, hidden_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(output)
        output = self.dropout(output)

        return output, current_cache
