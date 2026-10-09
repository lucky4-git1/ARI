"""Train BPE Tokenizer from scratch on project corpus."""

import os
from layallm.tokenizer.tokenizer import train_bpe_tokenizer, LayaTokenizer

def run():
    corpus_file = "data/raw/pretrain_corpus.txt"
    save_file = "artifacts/tokenizers/laya_tokenizer.json"

    print(f"Training BPE tokenizer on {corpus_file}...")
    # Using vocab size 2048 for compact model and fast CPU inference
    train_bpe_tokenizer(
        files=[corpus_file],
        vocab_size=2048,
        min_frequency=2,
        save_path=save_file,
    )
    
    tokenizer = LayaTokenizer.load(save_file)
    print(f"Tokenizer trained successfully! Final vocab size: {tokenizer.vocab_size}")

    # Verification
    sample_text = "def quicksort(arr): return arr"
    ids = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(ids)
    print(f"Encode test: '{sample_text}' -> {ids}")
    print(f"Decode test: '{decoded}'")
    assert decoded == sample_text, "Tokenizer decode did not match original text!"
    print("Tokenizer verification PASSED!")

if __name__ == "__main__":
    run()
