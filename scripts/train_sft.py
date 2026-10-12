"""Supervised Fine-Tuning (SFT) script for instruction following."""

import argparse
import torch
from torch.utils.data import DataLoader

from arillm.tokenizer.tokenizer import AriTokenizer
from arillm.training.checkpoint import load_checkpoint, save_checkpoint
from arillm.data.dataset import SFTDataset, collate_sft
from arillm.training.optimizer import create_optimizer, create_cosine_scheduler


def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning (SFT) on instruction pairs")
    parser.add_argument("--base_checkpoint", type=str, default="artifacts/checkpoints/exp_kaggle_sft_best.pt")
    parser.add_argument("--tokenizer", type=str, default="artifacts/tokenizers/ari_tokenizer_v2.json")
    parser.add_argument("--sft_data", type=str, default="data/processed/sft_paraphrase_quillbot.jsonl")
    parser.add_argument("--max_steps", type=int, default=120)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_checkpoint", type=str, default="artifacts/checkpoints/exp_kaggle_sft_best.pt")
    args = parser.parse_args()

    tokenizer = AriTokenizer.load(args.tokenizer)
    loaded = load_checkpoint(args.base_checkpoint, device=args.device)
    model = loaded["model"].to(args.device)
    # Update model max_seq_len if requested
    if hasattr(model, "config") and model.config.max_seq_len < args.max_seq_len:
        model.config.max_seq_len = args.max_seq_len

    counts = model.count_parameters()
    print(f"Loaded model from {args.base_checkpoint}: {counts['total_parameters']:,} params")
    print(f"Context window set to: {args.max_seq_len} tokens")

    # Build dataset
    dataset = SFTDataset(args.sft_data, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_sft)
    print(f"Loaded {len(dataset)} SFT examples")

    optimizer = create_optimizer(model, learning_rate=args.lr, weight_decay=0.01)
    scheduler = create_cosine_scheduler(optimizer, warmup_steps=10, total_steps=args.max_steps)

    model.train()
    data_iter = iter(loader)
    print(f"Starting QuillBot Paraphrase + Knowledge SFT for {args.max_steps} steps (grad_accum={args.grad_accum})...")

    optimizer.zero_grad()
    accum_loss = 0.0

    for step in range(1, args.max_steps + 1):
        for _ in range(args.grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x = x.to(args.device)
            y = y.to(args.device)

            _, loss, _ = model(x, targets=y)
            loss_scaled = loss / args.grad_accum
            loss_scaled.backward()
            accum_loss += loss.item() / args.grad_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % 10 == 0 or step == 1:
            print(f"SFT Step {step:3d}/{args.max_steps:3d} | Loss: {accum_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        accum_loss = 0.0

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
