"""Trains general language + code BPE tokenizer with 4096 vocabulary size."""

import os
from layallm.tokenizer.tokenizer import train_bpe_tokenizer, LayaTokenizer

def train_main_tokenizer():
    corpus_file = "data/processed/combined_language_code_corpus.txt"
    save_file = "artifacts/tokenizers/laya_tokenizer_v2.json"

    print(f"Training unified BPE tokenizer on {corpus_file} (12.4 MB)...")
    train_bpe_tokenizer(
        files=[corpus_file],
        vocab_size=4096,
        min_frequency=3,
        save_path=save_file,
    )
    
    tokenizer = LayaTokenizer.load(save_file)
    print(f"Tokenizer trained successfully! Final vocab size: {tokenizer.vocab_size}")

    # Verification on both prose and code
    samples = [
        "The quick brown fox jumps over the lazy dog.",
        "In artificial intelligence, deep neural networks learn representations.",
        "def binary_search(arr, target):",
        "SELECT id, name FROM users WHERE age > 21;",
    ]

    print("\n--- Verifying Tokenizer Reconstruction ---")
    for s in samples:
        ids = tokenizer.encode(s)
        rec = tokenizer.decode(ids)
        assert rec == s, f"Mismatch: '{s}' vs '{rec}'"
        print(f"PASS: '{s}' -> {len(ids)} tokens")

    print("\nTokenizer v2 ready and verified!")

if __name__ == "__main__":
    train_main_tokenizer()
