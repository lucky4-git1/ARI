"""Unit tests for LAYA-LLM Architecture components."""

import pytest
import torch
from layallm.model.config import ModelConfig, CONFIG_PRESETS
from layallm.model.normalization import RMSNorm
from layallm.model.attention import CausalSelfAttention, precompute_rope_freqs_cis
from layallm.model.mlp import SwiGLUMLP
from layallm.model.transformer import LayaTransformer


def test_model_config_presets():
    for name, config in CONFIG_PRESETS.items():
        assert config.vocab_size > 0
        assert config.hidden_dim % config.num_heads == 0
        assert config.num_heads % config.num_kv_heads == 0
        assert config.head_dim == config.hidden_dim // config.num_heads


def test_rmsnorm():
    batch, seq_len, dim = 2, 8, 64
    x = torch.randn(batch, seq_len, dim)
    norm = RMSNorm(dim)
    out = norm(x)
    assert out.shape == x.shape
    # Check that variance is approx 1
    rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)


def test_swiglu_mlp():
    config = ModelConfig(hidden_dim=64, intermediate_dim=128, num_heads=4, num_kv_heads=4)
    mlp = SwiGLUMLP(config)
    x = torch.randn(2, 8, 64)
    out = mlp(x)
    assert out.shape == (2, 8, 64)


def test_causal_self_attention():
    config = ModelConfig(hidden_dim=64, num_heads=4, num_kv_heads=2, max_seq_len=32)
    attn = CausalSelfAttention(config)
    freqs = precompute_rope_freqs_cis(config.head_dim, 32)
    x = torch.randn(2, 8, 64)
    out, cache = attn(x, freqs_cis=freqs[:8])
    assert out.shape == (2, 8, 64)
    assert cache is None

    # Test with cache
    out_cached, new_cache = attn(x, freqs_cis=freqs[:8], use_cache=True)
    assert out_cached.shape == (2, 8, 64)
    assert new_cache is not None
    k, v = new_cache
    assert k.shape == (2, 8, 2, 16)  # batch, seq, kv_heads, head_dim


def test_transformer_forward_backward():
    config = ModelConfig(
        vocab_size=256,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_dim=128,
        max_seq_len=64,
    )
    model = LayaTransformer(config)
    
    # Check parameter count
    counts = model.count_parameters()
    assert counts["total_parameters"] > 0
    
    # Input batch
    input_ids = torch.randint(0, 256, (2, 16))
    targets = torch.randint(0, 256, (2, 16))

    logits, loss, _ = model(input_ids, targets=targets)
    assert logits.shape == (2, 16, 256)
    assert loss is not None
    assert torch.isfinite(loss)

    # Backpropagation check
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient is None for {name}"
            assert torch.isfinite(param.grad).all(), f"Gradient has non-finite values for {name}"


def test_causality():
    """Verify that earlier token representations are not affected by future tokens."""
    config = ModelConfig(vocab_size=128, hidden_dim=64, num_layers=2, num_heads=4, num_kv_heads=4)
    model = LayaTransformer(config)
    model.eval()

    input1 = torch.tensor([[5, 10, 15, 20]])
    input2 = torch.tensor([[5, 10, 15, 99]])  # 4th token changed

    with torch.no_grad():
        logits1, _, _ = model(input1)
        logits2, _, _ = model(input2)

    # Logits for first 3 tokens must be EXACTLY identical
    assert torch.allclose(logits1[:, :3, :], logits2[:, :3, :], atol=1e-5), (
        "Causality leak detected! Future token affected earlier token representations."
    )


def test_generation():
    config = ModelConfig(vocab_size=128, hidden_dim=64, num_layers=2, num_heads=4, num_kv_heads=4)
    model = LayaTransformer(config)
    prompt = torch.tensor([[10, 20]])
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.7)
    assert generated.shape == (1, 12)
    assert (generated >= 0).all() and (generated < 128).all()
