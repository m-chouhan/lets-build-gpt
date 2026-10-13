"""Raw data: get the text, split it, and serve random batches."""

import urllib.request
from pathlib import Path

import torch


def load_text(path: Path, url: str) -> str:
    """Read the dataset, downloading it first if it isn't on disk."""
    if not path.exists():
        print(f"downloading {url} -> {path}")
        urllib.request.urlretrieve(url, path)
    return path.read_text(encoding="utf-8")


def split(data: torch.Tensor, train_frac: float) -> tuple[torch.Tensor, torch.Tensor]:
    """First train_frac of the tokens for training, the rest for validation.

    Why hold some back: https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets
    """
    n = int(train_frac * len(data))
    return data[:n], data[n:]


def get_batch(data: torch.Tensor, batch_size: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick batch_size random chunks. y is x shifted one to the right.

        text "hello!", block_size 4  ->  x = "hell", y = "ello"
    """
    starts = torch.randint(len(data) - block_size, (batch_size,))
    xs, ys = [], []
    for i in starts:
        xs.append(data[i : i + block_size])  # block_size chars starting at i
        ys.append(data[i + 1 : i + block_size + 1])  # the same chars, shifted one right
    x = torch.stack(xs)  # list of B rows of T -> one (B, T) tensor
    y = torch.stack(ys)
    return x, y
