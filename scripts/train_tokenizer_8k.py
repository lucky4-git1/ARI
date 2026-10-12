"""Trains high-capacity BPE tokenizer with 8,192 vocabulary size for RewriteBot."""

import os
from arillm.tokenizer.tokenizer import train_bpe_tokenizer, AriTokenizer

def train_8k_tokenizer():
    corpus_file = "data/processed/combined_language_code_corpus.txt"
    save_file = "artifacts/tokenizers/ari_tokenizer_8k.json"

    print(f"Training high-capacity 8,192 BPE tokenizer on {corpus_file}...")
    train_bpe_tokenizer(
        files=[corpus_file],
        vocab_size=8192,
        min_frequency=2,
        save_path=save_file,
    )
    
    tokenizer = AriTokenizer.load(save_file)
    print(f"Tokenizer trained successfully! Final vocab size: {tokenizer.vocab_size}")

    samples = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence fundamentally transforms linguistic representations.",
        "Because the storm was severe, the flight was canceled immediately.",
        "Maintaining nutritious eating habits fosters physical well-being.",
    ]

    print("\n--- Verifying 8k Tokenizer Reconstruction ---")
    for s in samples:
        ids = tokenizer.encode(s)
        rec = tokenizer.decode(ids)
        assert rec == s, f"Mismatch: '{s}' vs '{rec}'"
        print(f"PASS ({len(ids)} tokens): {s}")

    print("\nTokenizer 8k ready and verified!")

if __name__ == "__main__":
    train_8k_tokenizer()
