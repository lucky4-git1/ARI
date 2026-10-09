"""Data package exports."""

from arillm.data.registry import DATASET_REGISTRY, DatasetEntry
from arillm.data.dataset import CausalLMDataset, SFTDataset, collate_sft
from arillm.data.prepare import deduplicate_lines

__all__ = [
    "DATASET_REGISTRY",
    "DatasetEntry",
    "CausalLMDataset",
    "SFTDataset",
    "collate_sft",
    "deduplicate_lines",
]
