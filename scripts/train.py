"""Pretraining script for LAYA-LLM."""

import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split

from layallm.model.config import ModelConfig
from layallm.model.transformer import LayaTransformer
from layallm.tokenizer.tokenizer import LayaTokenizer
from layallm.data.dataset import CausalLMDataset
from layallm.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Pretrain LAYA-LLM from scratch")
    parser.add_argument("--corpus", type=str, default="data/raw/pretrain_corpus.txt")
    parser.add_argument("--tokenizer", type=str, default="artifacts/tokenizers/laya_tokenizer.json")
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--experiment_id", type=str, default="exp_pretrain_v1")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = LayaTokenizer.load(args.tokenizer)
    print(f"Loaded tokenizer from {args.tokenizer} (Vocab size: {tokenizer.vocab_size})")

    # Read and encode text
    with open(args.corpus, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Tokenizing pretraining corpus ({len(text)} chars)...")
    token_ids = tokenizer.encode(text)
    print(f"Total corpus tokens: {len(token_ids):,}")

    # Create dataset
    full_dataset = CausalLMDataset(token_ids, seq_len=args.seq_len)
    val_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Initialize model from scratch
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_heads // 2 if args.num_heads >= 4 else args.num_heads,
        max_seq_len=args.seq_len,
        tie_word_embeddings=True,
    )
    model = LayaTransformer(config)
    counts = model.count_parameters()
    print(f"Instantiated original LayaTransformer with {counts['total_parameters']:,} parameters (Weights randomly initialized).")

    # Run training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        warmup_steps=15,
        max_steps=args.max_steps,
        checkpoint_dir="artifacts/checkpoints",
        save_every_steps=50,
        eval_every_steps=50,
        experiment_id=args.experiment_id,
        device="cpu",
    )

    results = trainer.train()
    print("Training finished with results:", results)


if __name__ == "__main__":
    main()
