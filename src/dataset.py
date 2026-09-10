"""Character-level tokenizer and batching helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CharTokenizer:
    chars: list[str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(chars=sorted(list(set(text))))

    def __post_init__(self) -> None:
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def split_data(data: torch.Tensor, train_ratio: float = 0.9) -> tuple[torch.Tensor, torch.Tensor]:
    split_idx = int(train_ratio * len(data))
    return data[:split_idx], data[split_idx:]


def get_batch(
    split: str,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
