"""LAYA-LLM Decoder-only Transformer Architecture implemented from scratch."""

from typing import Optional, Tuple, Dict, Any, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from layallm.model.config import ModelConfig
from layallm.model.normalization import RMSNorm
from layallm.model.attention import CausalSelfAttention, precompute_rope_freqs_cis
from layallm.model.mlp import SwiGLUMLP
from layallm.model.initialization import init_weights, apply_residual_scaling


class TransformerBlock(nn.Module):
    """Transformer decoder block with Pre-RMSNorm, Causal GQA Self-Attention, and SwiGLU MLP."""

    def __init__(self, layer_id: int, config: ModelConfig):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        
        self.attn_norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)
        self.attention = CausalSelfAttention(config)
        
        self.mlp_norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm Self-Attention with residual
        normed_attn = self.attn_norm(x)
        attn_out, new_kv_cache = self.attention(
            normed_attn,
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )
        x = x + attn_out

        # Pre-norm MLP with residual
        normed_mlp = self.mlp_norm(x)
        mlp_out = self.mlp(normed_mlp)
        x = x + mlp_out

        return x, new_kv_cache


class LayaTransformer(nn.Module):
    """Decoder-only Transformer Language Model (Ari-LLM / LAYA-LLM)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding table
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Decoder layers
        self.layers = nn.ModuleList([
            TransformerBlock(i, config) for i in range(config.num_layers)
        ])

        # Final RMS Normalization
        self.norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)

        # Language Model Head (LM Head)
        if config.tie_word_embeddings:
            self.lm_head = None  # Re-uses embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Precompute RoPE freqs for max_seq_len
        freqs_cis = precompute_rope_freqs_cis(
            dim=config.head_dim,
            end=config.max_seq_len * 2,
            theta=config.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Initialize all weights randomly from scratch
        self.apply(lambda m: init_weights(m, config))
        apply_residual_scaling(self, config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        if self.config.tie_word_embeddings:
            return self.embed_tokens
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Forward pass.
        
        Args:
            input_ids: Tensor of shape [batch, seq_len] with integer token IDs.
            targets: Optional ground-truth next-token labels of shape [batch, seq_len].
            kv_caches: Optional list of cached (k, v) tensors for each layer.
            use_cache: Whether to return updated KV caches for decoding.
            start_pos: Starting position index for rotary embeddings (used in streaming generation).
            
        Returns:
            logits: Tensor of shape [batch, seq_len, vocab_size]
            loss: Cross-entropy scalar loss (if targets is provided, else None)
            new_kv_caches: List of updated KV caches if use_cache is True, else None
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Lookup embeddings: [batch, seq_len, hidden_dim]
        h = self.embed_tokens(input_ids)

        # Slice precomputed RoPE frequencies for the current sequence slice
        end_pos = start_pos + seq_len
        freqs_cis = self.freqs_cis[start_pos:end_pos].to(device)

        new_kv_caches = [] if use_cache else None

        # Pass through Transformer blocks
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            h, updated_cache = layer(
                h,
                freqs_cis=freqs_cis,
                attention_mask=None,
                kv_cache=layer_cache,
                use_cache=use_cache,
            )
            if use_cache:
                new_kv_caches.append(updated_cache)

        # Final normalization
        h = self.norm(h)

        # Compute output logits
        if self.config.tie_word_embeddings:
            # Tied weights: projection using transposed embedding matrix
            logits = F.linear(h, self.embed_tokens.weight)
        else:
            logits = self.lm_head(h)

        # Compute cross-entropy loss if targets are provided
        loss = None
        if targets is not None:
            # Shift tokens: predict targets[:, :] from logits[:, :]
            # Targets can use -100 for ignored tokens (e.g., prompt masking in SFT)
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss, new_kv_caches

    def count_parameters(self) -> Dict[str, int]:
        """Calculates total and trainable parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Calculate tied params deduplicated
        non_embedding_params = sum(
            p.numel() for n, p in self.named_parameters() if "embed_tokens" not in n
        )
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_embedding_parameters": non_embedding_params,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressive text generation with KV caching and sampling."""
        self.eval()
        device = input_ids.device
        curr_ids = input_ids.clone()
        
        batch_size, prompt_len = curr_ids.shape
        kv_caches = None
        
        # First pass: process prompt
        logits, _, kv_caches = self(curr_ids, use_cache=use_cache, start_pos=0)
        
        for step in range(max_new_tokens):
            if use_cache:
                # Next token prediction using only the last logit
                next_token_logits = logits[:, -1, :]
            else:
                # Fallback without cache: pass full context
                logits, _, _ = self(curr_ids, use_cache=False, start_pos=0)
                next_token_logits = logits[:, -1, :]

            # Temperature scaling
            if temperature > 0:
                scaled_logits = next_token_logits / temperature
                
                # Top-K filtering
                if top_k > 0:
                    top_k_val = min(top_k, scaled_logits.size(-1))
                    indices_to_remove = scaled_logits < torch.topk(scaled_logits, top_k_val)[0][..., -1, None]
                    scaled_logits[indices_to_remove] = float("-inf")
                
                # Top-P (nucleus) filtering
                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift indices right to keep first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    scaled_logits[indices_to_remove] = float("-inf")

                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            curr_ids = torch.cat([curr_ids, next_token], dim=1)

            # Check EOS token
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            if use_cache:
                # Forward only the newly sampled single token with updated KV cache
                start_pos = prompt_len + step
                logits, _, kv_caches = self(
                    next_token,
                    kv_caches=kv_caches,
                    use_cache=True,
                    start_pos=start_pos,
                )

        return curr_ids
