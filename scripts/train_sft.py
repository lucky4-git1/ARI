"""Supervised Fine-Tuning (SFT) script for instruction following."""

import argparse
import torch
from torch.utils.data import DataLoader

from layallm.tokenizer.tokenizer import LayaTokenizer
from layallm.training.checkpoint import load_checkpoint, save_checkpoint
from layallm.data.dataset import SFTDataset, collate_sft
from layallm.training.optimizer import create_optimizer, create_cosine_scheduler


def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning (SFT) on instruction pairs")
    parser.add_argument("--base_checkpoint", type=str, default="artifacts/checkpoints/exp_pretrain_v1_best.pt")
    parser.add_argument("--tokenizer", type=str, default="artifacts/tokenizers/laya_tokenizer.json")
    parser.add_argument("--sft_data", type=str, default="data/processed/sft_data.jsonl")
    parser.add_argument("--max_steps", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_checkpoint", type=str, default="artifacts/checkpoints/exp_sft_v1.pt")
    args = parser.parse_args()

    tokenizer = LayaTokenizer.load(args.tokenizer)
    loaded = load_checkpoint(args.base_checkpoint, device=args.device)
    model = loaded["model"].to(args.device)
    print(f"Loaded base pretrained model from {args.base_checkpoint}")

    # Build dataset
    dataset = SFTDataset(args.sft_data, tokenizer=tokenizer, max_seq_len=256)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_sft)
    print(f"Loaded {len(dataset)} SFT examples")

    optimizer = create_optimizer(model, learning_rate=args.lr, weight_decay=0.01)
    scheduler = create_cosine_scheduler(optimizer, warmup_steps=5, total_steps=args.max_steps)

    model.train()
    data_iter = iter(loader)
    print(f"Starting SFT fine-tuning for {args.max_steps} steps...")

    for step in range(1, args.max_steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)

        x = x.to(args.device)
        y = y.to(args.device)

        optimizer.zero_grad()
        _, loss, _ = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 10 == 0 or step == 1:
            print(f"SFT Step {step:3d}/{args.max_steps:3d} | Loss: {loss.item():.4f}")

    # Save fine-tuned checkpoint
    save_checkpoint(
        save_path=args.output_checkpoint,
        model=model,
        optimizer=optimizer,
        step=step,
        loss=loss.item(),
        config=model.config,
        extra_metadata={"stage": "SFT"},
    )
    print(f"SFT finished! Saved checkpoint to {args.output_checkpoint}")


if __name__ == "__main__":
    main()
