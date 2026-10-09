"""CPU Inference Benchmark script for Ari-LLM."""

import argparse
import json
import torch

from arillm.tokenizer.tokenizer import AriTokenizer
from arillm.training.checkpoint import load_checkpoint
from arillm.inference.generate import CPUInferenceEngine


def main():
    parser = argparse.ArgumentParser(description="Benchmark CPU inference performance")
    parser.add_argument("--checkpoint", type=str, default="artifacts/checkpoints/exp_pretrain_v1_best.pt")
    parser.add_argument("--tokenizer", type=str, default="artifacts/tokenizers/ari_tokenizer.json")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=40)
    args = parser.parse_args()

    loaded = load_checkpoint(args.checkpoint, device="cpu")
    model = loaded["model"]
    tokenizer = AriTokenizer.load(args.tokenizer)
    
    engine = CPUInferenceEngine(model, tokenizer, num_threads=args.threads)
    params = model.count_parameters()["total_parameters"]

    print("Running CPU inference benchmark...")
    results = engine.benchmark_throughput(
        prompt="def quicksort(arr):",
        num_tokens_to_generate=args.tokens,
        num_runs=3,
    )

    print("\n================ CPU BENCHMARK REPORT ================")
    print(f"Model Parameters:       {params:,}")
    print(f"Hardware Threads:       {results['num_threads']}")
    print(f"Prompt Length:          {results['prompt_length']} tokens")
    print(f"Generated Tokens:       {results['tokens_generated']} tokens")
    print(f"Time to First Token:    {results['ttft_ms']:.2f} ms")
    print(f"Decode Throughput:      {results['tokens_per_second']:.2f} tokens/second")
    print("======================================================")

    # Save benchmark json
    with open("artifacts/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
