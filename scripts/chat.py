"""Interactive and terminal-based REPL chat interface for Ari-LLM."""

import sys
import argparse
import torch

from arillm.tokenizer.tokenizer import AriTokenizer
from arillm.training.checkpoint import load_checkpoint
from arillm.inference.generate import CPUInferenceEngine


def start_chat(checkpoint_path: str, tokenizer_path: str, temperature: float = 0.7):
    print("=" * 60)
    print("         Ari-LLM / Ari-LLM Interactive Console")
    print("=" * 60)
    print(f"Loading checkpoint: {checkpoint_path}")
    loaded = load_checkpoint(checkpoint_path, device="cpu")
    model = loaded["model"]
    tokenizer = AriTokenizer.load(tokenizer_path)
    engine = CPUInferenceEngine(model, tokenizer, num_threads=4)
    params = model.count_parameters()["total_parameters"]
    print(f"Model online: {params:,} parameters (CPU 4 threads)")
    print("Type your prompt or question below. Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            user_input = input("User > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting Ari-LLM. Goodbye!")
                break

            print("\nAri-LLM > ", end="", flush=True)
            output = engine.generate(
                prompt=user_input,
                max_new_tokens=45,
                temperature=temperature,
            )
            # Print completion beyond prompt
            if output.startswith(user_input):
                completion = output[len(user_input):].strip()
            else:
                completion = output.strip()
            print(completion + "\n")

        except KeyboardInterrupt:
            print("\nInterrupted. Exiting...")
            break
        except Exception as e:
            print(f"\n[Error during generation: {e}]\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ari-LLM Interactive Console")
    parser.add_argument("--checkpoint", type=str, default="artifacts/checkpoints/exp_pretrain_v1_best.pt")
    parser.add_argument("--tokenizer", type=str, default="artifacts/tokenizers/ari_tokenizer.json")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    start_chat(args.checkpoint, args.tokenizer, args.temperature)
