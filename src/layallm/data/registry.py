"""Dataset registry, metadata, and licensing manifest."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DatasetEntry:
    name: str
    source_url_or_origin: str
    license: str
    intended_stage: str  # "Stage A: Pretraining", "Stage B: Coding", "Stage C: SFT"
    language: str
    approximate_size: str
    status: str
    deduplication_status: str
    known_limitations: str


DATASET_REGISTRY: Dict[str, DatasetEntry] = {
    "tiny_smoke_corpus": DatasetEntry(
        name="tiny_smoke_corpus",
        source_url_or_origin="Synthetically generated seed corpus of algorithms and explanations",
        license="MIT / Public Domain",
        intended_stage="Stage A & Stage B Verification",
        language="en, python, javascript",
        approximate_size="~200 KB",
        status="ready",
        deduplication_status="exact deduplicated",
        known_limitations="Very small; strictly for pipeline testing and overfitting verification",
    ),
    "wikitext_103_sample": DatasetEntry(
        name="wikitext_sample",
        source_url_or_origin="Creative Commons Attribution-ShareAlike 3.0 Unported (Wikipedia articles)",
        license="CC BY-SA 3.0",
        intended_stage="Stage A: Pretraining",
        language="en",
        approximate_size="~2 MB",
        status="ready",
        deduplication_status="exact deduplicated",
        known_limitations="General factual prose; no code",
    ),
    "permissive_code_corpus": DatasetEntry(
        name="permissive_code_corpus",
        source_url_or_origin="Permissively licensed algorithms (MIT/Apache-2.0 / The Algorithms Project / standard libs)",
        license="MIT / Apache-2.0",
        intended_stage="Stage B: Coding specialization",
        language="python, javascript, c",
        approximate_size="~1.5 MB",
        status="ready",
        deduplication_status="exact deduplicated",
        known_limitations="Algorithmic code; lacks large-scale system patterns",
    ),
    "instruction_alpaca_tiny": DatasetEntry(
        name="instruction_alpaca_tiny",
        source_url_or_origin="Permissively licensed curated instruction-tuning pairs",
        license="Apache-2.0 / CC0",
        intended_stage="Stage C: SFT",
        language="en",
        approximate_size="~500 KB",
        status="ready",
        deduplication_status="deduplicated",
        known_limitations="Small curated set for SFT pipeline verification",
    ),
}
