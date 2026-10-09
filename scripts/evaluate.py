"""Evaluation suite: Perplexity, general text checks, and coding completion."""

import math
import torch
from arillm.model.transformer import AriTransformer
from arillm.tokenizer.tokenizer import AriTokenizer
from arillm.training.checkpoint import load_checkpoint
from arillm.inference.generate import CPUInferenceEngine


def evaluate_perplexity(model: AriTransformer, tokenizer: AriTokenizer, text: str) -> float:
    """Computes token-level perplexity on a held-out text passage."""
    model.eval()
    ids = tokenizer.encode(text)
    if len(ids) < 2:
        return float("nan")

    x = torch.tensor([ids[:-1]], dtype=torch.long)
    y = torch.tensor([ids[1:]], dtype=torch.long)

    with torch.no_grad():
        _, loss, _ = model(x, targets=y)
    
    return math.exp(loss.item()) if loss is not None else float("nan")


def run_code_completion_eval(engine: CPUInferenceEngine):
    test_cases = [
        {"prompt": "def add(a, b):", "expected_keyword": "return"},
        {"prompt": "def is_even(n):", "expected_keyword": "return"},
    ]
    print("\n--- Code Completion Qualitative Evaluation ---")
    for case in test_cases:
        out = engine.generate(case["prompt"], max_new_tokens=25, temperature=0.2)
        print(f"Prompt: {case['prompt']}")
        print(f"Completion:\n{out.strip()}\n")


if __name__ == "__main__":
    loaded = load_checkpoint("artifacts/checkpoints/exp_pretrain_v1_best.pt", device="cpu")
    model = loaded["model"]
    tokenizer = AriTokenizer.load("artifacts/tokenizers/ari_tokenizer.json")
    
    test_text = "A Transformer is a deep learning architecture based on multi-head attention."
    ppl = evaluate_perplexity(model, tokenizer, test_text)
    print(f"Held-out Perplexity: {ppl:.2f}")

    engine = CPUInferenceEngine(model, tokenizer)
    run_code_completion_eval(engine)
