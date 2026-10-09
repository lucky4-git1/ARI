"""Builds unified general language and coding corpus."""

import os

def build_combined():
    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/combined_language_code_corpus.txt"
    
    print("Combining Wikitext-2, TinyShakespeare, multi-language code, and algorithms...")
    sources = [
        "data/raw/wikitext2.txt",            # ~10.7 MB of Wikipedia English prose
        "data/raw/tinyshakespeare.txt",      # ~1.1 MB of literature/dialogue
        "data/raw/multilang_code.txt",       # Python, JS, C, SQL algorithms
        "data/raw/pretrain_corpus.txt",      # Transformer theory & algorithms
    ]

    total_bytes = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for src in sources:
            if os.path.exists(src):
                size = os.path.getsize(src)
                print(f"Adding {src} ({size:,} bytes)...")
                with open(src, "r", encoding="utf-8", errors="ignore") as in_f:
                    for line in in_f:
                        stripped = line.strip()
                        if stripped:
                            out_f.write(stripped + "\n")
                total_bytes += size

    print(f"\nFinal Unified Corpus: {out_path} ({os.path.getsize(out_path):,} bytes)")

if __name__ == "__main__":
    build_combined()
