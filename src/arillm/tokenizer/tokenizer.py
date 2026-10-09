"""BPE Tokenizer trainer and wrapper for Ari-LLM."""

import os
from typing import List, Optional, Union
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing


# Standard special tokens for Ari-LLM
SPECIAL_TOKENS = [
    "<pad>",     # ID 0: Padding token
    "<unk>",     # ID 1: Unknown token
    "<bos>",     # ID 2: Beginning of sequence
    "<eos>",     # ID 3: End of sequence
    "<user>",    # ID 4: User message turn boundary
    "<assistant>", # ID 5: Assistant message turn boundary
    "<system>",  # ID 6: System message turn boundary
    "<code>",    # ID 7: Code snippet boundary
    "</code>",   # ID 8: Code snippet end boundary
]


def train_bpe_tokenizer(
    files: List[str],
    vocab_size: int = 8192,
    min_frequency: int = 2,
    save_path: Optional[str] = None,
) -> Tokenizer:
    """Trains a Byte-Level BPE tokenizer from scratch on specified text/code files."""
    # Instantiate byte-level BPE model
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train(files=files, trainer=trainer)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        tokenizer.save(save_path)

    return tokenizer


class AriTokenizer:
    """Wrapper around trained HuggingFace Tokenizer with convenient helpers."""

    def __init__(self, tokenizer_file: Optional[str] = None):
        if tokenizer_file is not None and os.path.exists(tokenizer_file):
            self.tokenizer = Tokenizer.from_file(tokenizer_file)
        else:
            self.tokenizer = None

        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

    @classmethod
    def load(cls, path: str) -> "AriTokenizer":
        wrapper = cls()
        wrapper.tokenizer = Tokenizer.from_file(path)
        # Update special token IDs
        for name, id_val in [("<pad>", 0), ("<unk>", 1), ("<bos>", 2), ("<eos>", 3)]:
            tid = wrapper.tokenizer.token_to_id(name)
            if name == "<pad>": wrapper.pad_token_id = tid if tid is not None else 0
            elif name == "<unk>": wrapper.unk_token_id = tid if tid is not None else 1
            elif name == "<bos>": wrapper.bos_token_id = tid if tid is not None else 2
            elif name == "<eos>": wrapper.eos_token_id = tid if tid is not None else 3
        return wrapper

    def save(self, path: str):
        assert self.tokenizer is not None, "Tokenizer has not been trained or loaded."
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.tokenizer.save(path)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size() if self.tokenizer else 0

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        assert self.tokenizer is not None
        enc = self.tokenizer.encode(text)
        ids = enc.ids
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        assert self.tokenizer is not None
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
