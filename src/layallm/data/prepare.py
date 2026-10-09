"""Data preparation, exact deduplication, and corpus building."""

import hashlib
import os
from typing import List, Set


def deduplicate_lines(input_paths: List[str], output_path: str) -> int:
    """Exact deduplication of documents/paragraphs based on SHA-256 hashes."""
    seen_hashes: Set[str] = set()
    total_written = 0

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for p in input_paths:
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8", errors="ignore") as in_f:
                for line in in_f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    line_hash = hashlib.sha256(stripped.encode("utf-8")).hexdigest()
                    if line_hash not in seen_hashes:
                        seen_hashes.add(line_hash)
                        out_f.write(line)
                        total_written += 1

    return total_written
