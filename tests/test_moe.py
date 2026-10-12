"""Unit tests for Sparse Mixture-of-Experts (MoE) Architecture."""

import pytest
import torch
from arillm.model.config import ModelConfig
from arillm.model.moe import SparseMoEBlock
from arillm.model.transformer import AriTransformer


def test_sparse_moe_block_shape_and_routing():
    batch, seq_len, dim = 2, 8, 64
    config = ModelConfig(
        hidden_dim=dim,
        intermediate_dim=128,
        num_heads=4,
        num_kv_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
    )
    moe = SparseMoEBlock(config)
    x = torch.randn(batch, seq_len, dim)

    # Eval mode
    moe.eval()
    out, aux_loss = moe(x)
    assert out.shape == (batch, seq_len, dim)
    assert aux_loss == 0.0

    # Train mode (aux loss should be > 0 and finite)
    moe.train()
    out_train, aux_loss_train = moe(x)
    assert out_train.shape == (batch, seq_len, dim)
    assert aux_loss_train > 0.0
    assert torch.isfinite(aux_loss_train)


def test_transformer_with_moe():
    config = ModelConfig(
        vocab_size=128,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_dim=128,
        max_seq_len=64,
        num_experts=4,
        num_experts_per_tok=2,
        moe_aux_loss_coeff=0.01,
    )
    model = AriTransformer(config)
    counts = model.count_parameters()
    
    # Active parameters must be strictly less than total parameters
    assert counts["active_parameters"] < counts["total_parameters"]
    assert counts["total_parameters"] > 0

    input_ids = torch.randint(0, 128, (2, 16))
    targets = torch.randint(0, 128, (2, 16))

    # Forward pass
    logits, loss, _ = model(input_ids, targets=targets)
    assert logits.shape == (2, 16, 128)
    assert loss is not None
    assert torch.isfinite(loss)

    # Backward pass checks gradients reach expert and router weights
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert torch.isfinite(param.grad).all(), f"Gradient has non-finite values for {name}"


def test_moe_backward_compatibility():
    """Verify standard dense config still functions identically with default num_experts=1."""
    config = ModelConfig(
        vocab_size=128,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_dim=128,
        max_seq_len=64,
        num_experts=1,
    )
    model = AriTransformer(config)
    counts = model.count_parameters()
    assert counts["active_parameters"] == counts["total_parameters"]

    input_ids = torch.randint(0, 128, (2, 8))
    targets = torch.randint(0, 128, (2, 8))
    logits, loss, _ = model(input_ids, targets=targets)
    assert logits.shape == (2, 8, 128)
    assert loss is not None
