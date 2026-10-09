"""Text generation and demonstration script using trained checkpoint."""

import argparse
import torch
from arillm.tokenizer.tokenizer import AriTokenizer
from arillm.training.checkpoint import load_checkpoint
from arillm.inference.generate import CPUInferenceEngine


def main():
    parser = argparse.ArgumentParser(description="Generate text with trained Ari-LLM checkpoint")
    parser.add_argument("--checkpoint", type=str, default="artifacts/checkpoints/exp_pretrain_v1_best.pt")
    parser.add_argument("--tokenizer", type=str, default="artifacts/tokenizers/ari_tokenizer.json")
    parser.add_argument("--prompt", type=str, default="def binary_search(arr, target):")
    parser.add_argument("--max_tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    loaded = load_checkpoint(args.checkpoint, device="cpu")
    model = loaded["model"]
    step = loaded["step"]
    print(f"Loaded checkpoint saved at step {step} with loss {loaded['loss']:.4f}")

    tokenizer = AriTokenizer.load(args.tokenizer)
    engine = CPUInferenceEngine(model, tokenizer, num_threads=args.threads)

    print(f"\n--- PROMPT ---\n{args.prompt}")
    print(f"\n--- GENERATING ({args.max_tokens} tokens, temp={args.temperature}) ---")
    output = engine.generate(
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print("\n--- OUTPUT ---")
    print(output)
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()
