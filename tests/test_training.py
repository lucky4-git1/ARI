"""Unit tests for Tokenizer, Training loop, Checkpoint restoration, and CPU Inference."""

import os
import tempfile
import pytest
import torch
from torch.utils.data import DataLoader

from arillm.model.config import ModelConfig
from arillm.model.transformer import AriTransformer
from arillm.tokenizer.tokenizer import train_bpe_tokenizer, AriTokenizer
from arillm.data.dataset import CausalLMDataset
from arillm.training.optimizer import create_optimizer, create_cosine_scheduler
from arillm.training.checkpoint import save_checkpoint, load_checkpoint
from arillm.training.trainer import Trainer
from arillm.inference.generate import CPUInferenceEngine


def test_tokenizer_training_and_encoding():
    # Create temporary corpus
    corpus = (
        "def add(a, b):\n    return a + b\n\n"
        "def multiply(x, y):\n    return x * y\n\n"
        "The quick brown fox jumps over the lazy dog. 12345! Unicode: café, ñ, 你好.\n"
    )
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as f:
        f.write(corpus)
        corpus_path = f.name

    tok_file = corpus_path + ".json"
    try:
        # Train BPE
        train_bpe_tokenizer([corpus_path], vocab_size=256, min_frequency=1, save_path=tok_file)
        assert os.path.exists(tok_file)

        # Load wrapper
        tok = AriTokenizer.load(tok_file)
        assert tok.vocab_size > 0

        # Encode and decode
        test_str = "def add(a, b): return a + b"
        ids = tok.encode(test_str)
        assert len(ids) > 0
        decoded = tok.decode(ids)
        assert decoded == test_str

    finally:
        if os.path.exists(corpus_path): os.remove(corpus_path)
        if os.path.exists(tok_file): os.remove(tok_file)


def test_checkpoint_atomic_save_load():
    config = ModelConfig(vocab_size=128, hidden_dim=64, num_layers=2, num_heads=4, num_kv_heads=4)
    model = AriTransformer(config)
    optimizer = create_optimizer(model)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "test_ckpt.pt")
        save_checkpoint(
            save_path=ckpt_path,
            model=model,
            optimizer=optimizer,
            step=42,
            loss=1.234,
            config=config,
            extra_metadata={"exp": "test"},
        )
        assert os.path.exists(ckpt_path)

        # Restore in fresh model
        loaded = load_checkpoint(ckpt_path, device="cpu", load_optimizer=True, optimizer=optimizer)
        assert loaded["step"] == 42
        assert loaded["loss"] == 1.234
        assert loaded["extra_metadata"]["exp"] == "test"
        
        # Verify weight parity
        for p1, p2 in zip(model.parameters(), loaded["model"].parameters()):
            assert torch.allclose(p1, p2)


def test_tiny_training_loop_overfitting():
    """Verify that training decreases loss monotonically on a tiny repeating pattern."""
    vocab_size = 64
    seq_len = 16
    config = ModelConfig(
        vocab_size=vocab_size,
        hidden_dim=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        intermediate_dim=64,
        max_seq_len=32,
    )
    model = AriTransformer(config)

    # Repeat sequence: [0, 1, 2, ..., 15] 20 times
    tokens = list(range(16)) * 20
    dataset = CausalLMDataset(tokens, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            learning_rate=1e-3,
            max_steps=30,
            warmup_steps=5,
            checkpoint_dir=tmp_dir,
            save_every_steps=15,
            eval_every_steps=15,
            experiment_id="overfit_test",
        )
        result = trainer.train()

        assert result["total_steps"] == 30
        assert result["final_loss"] < result["initial_loss"], "Training loss did not decrease!"
        assert result["best_val_loss"] < result["initial_loss"]
