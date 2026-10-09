"""PyTorch Dataset implementations for pretraining and SFT."""

import json
from typing import List, Optional, Tuple
import torch
from torch.utils.data import Dataset
from arillm.tokenizer.tokenizer import AriTokenizer


class CausalLMDataset(Dataset):
    """Autoregressive language modeling dataset from tokenized integer sequence.
    
    Splits continuous stream of tokens into chunks of `seq_len` tokens with next-token prediction targets.
    """

    def __init__(self, token_ids: List[int], seq_len: int = 256):
        self.seq_len = seq_len
        # Cut into chunks of length (seq_len + 1) so input is [:seq_len] and target is [1:seq_len+1]
        self.num_samples = len(token_ids) // (seq_len + 1)
        self.total_tokens = self.num_samples * (seq_len + 1)
        self.tokens = torch.tensor(token_ids[:self.total_tokens], dtype=torch.long)

    def __len__(self) -> int:
        return max(0, self.num_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * (self.seq_len + 1)
        chunk = self.tokens[start : start + self.seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class SFTDataset(Dataset):
    """Supervised Fine-Tuning (SFT) dataset with prompt masking.
    
    Formats prompt and target response as:
    `<bos><user>\n{prompt}\n<assistant>\n{response}<eos>`
    Masks prompt tokens with -100 so loss is computed exclusively on the assistant response.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: AriTokenizer,
        max_seq_len: int = 512,
    ):
        self.examples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                prompt = item.get("prompt") or item.get("instruction", "")
                response = item.get("response") or item.get("output", "")
                
                # Encode parts
                # System / User prefix
                prompt_text = f"<user>\n{prompt}\n<assistant>\n"
                response_text = f"{response}<eos>"
                
                prompt_ids = tokenizer.encode(prompt_text)
                response_ids = tokenizer.encode(response_text)
                
                # Combine
                all_ids = prompt_ids + response_ids
                if len(all_ids) > max_seq_len:
                    all_ids = all_ids[:max_seq_len]
                
                # Labels: mask prompt tokens with -100
                prompt_len = min(len(prompt_ids), len(all_ids))
                labels = [-100] * prompt_len + all_ids[prompt_len:]
                
                # Input is tokens[:-1], target is labels[1:]
                if len(all_ids) > 1:
                    x = torch.tensor(all_ids[:-1], dtype=torch.long)
                    y = torch.tensor(labels[1:], dtype=torch.long)
                    self.examples.append((x, y))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.examples[idx]


def collate_sft(batch, pad_token_id: int = 0):
    """Pads variable-length SFT sequences dynamically within a batch."""
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    
    batch_x = []
    batch_y = []
    for x, y in zip(xs, ys):
        pad_len = max_len - len(x)
        if pad_len > 0:
            padded_x = torch.cat([x, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            padded_y = torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)])
        else:
            padded_x = x
            padded_y = y
        batch_x.append(padded_x)
        batch_y.append(padded_y)
        
    return torch.stack(batch_x), torch.stack(batch_y)
