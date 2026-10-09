"""Model configuration dataclass for Ari-LLM."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json
import yaml


@dataclass
class ModelConfig:
    # Architecture dimensions
    vocab_size: int = 8192
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    num_kv_heads: Optional[int] = None  # None = MHA (equal to num_heads), or smaller for GQA
    intermediate_dim: Optional[int] = None  # Feed-forward hidden dimension (default 8/3 * hidden_dim for SwiGLU)
    max_seq_len: int = 512
    
    # Normalization & Regularization
    norm_eps: float = 1e-5
    dropout: float = 0.0
    
    # RoPE settings
    rope_theta: float = 10000.0
    
    # Weight tying
    tie_word_embeddings: bool = True
    
    # Initialization
    init_std: float = 0.02
    
    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.intermediate_dim is None:
            # SwiGLU standard dimension: multiple of 64 or 256 close to 8/3 * hidden_dim
            raw_dim = int(2 * (4 * self.hidden_dim) / 3)
            # Make multiple of 64 for efficient hardware memory alignment
            self.intermediate_dim = ((raw_dim + 63) // 64) * 64
        
        assert self.hidden_dim % self.num_heads == 0, (
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
        )
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        )

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "intermediate_dim": self.intermediate_dim,
            "max_seq_len": self.max_seq_len,
            "norm_eps": self.norm_eps,
            "dropout": self.dropout,
            "rope_theta": self.rope_theta,
            "tie_word_embeddings": self.tie_word_embeddings,
            "init_std": self.init_std,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(**data)

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# Standard predefined configurations
CONFIG_PRESETS: Dict[str, ModelConfig] = {
    # 1. Smoke-test model (~1.5M - 2.5M params) for fast CI, unit testing, and pipeline verification
    "smoke": ModelConfig(
        vocab_size=4096,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        num_kv_heads=4,
        intermediate_dim=384,
        max_seq_len=256,
        tie_word_embeddings=True,
    ),
    # 2. Tiny research model (~12M - 18M params) for fast local training and validation
    "tiny": ModelConfig(
        vocab_size=8192,
        hidden_dim=256,
        num_layers=8,
        num_heads=8,
        num_kv_heads=4,  # GQA
        intermediate_dim=682,
        max_seq_len=512,
        tie_word_embeddings=True,
    ),
    # 3. First serious model (~50M - 75M params)
    "small": ModelConfig(
        vocab_size=16384,
        hidden_dim=512,
        num_layers=12,
        num_heads=8,
        num_kv_heads=4,  # GQA
        intermediate_dim=1365,
        max_seq_len=1024,
        tie_word_embeddings=True,
    ),
    # 4. Scaled experiment (~150M params)
    "medium": ModelConfig(
        vocab_size=32768,
        hidden_dim=768,
        num_layers=16,
        num_heads=12,
        num_kv_heads=4,  # GQA
        intermediate_dim=2048,
        max_seq_len=2048,
        tie_word_embeddings=True,
    ),
}
