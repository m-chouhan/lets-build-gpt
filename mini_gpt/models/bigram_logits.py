"""Level 1. Same 65x65 table as BigramCounts, but it holds logits learned by gradient descent.

After training, logits ≈ log(counts): the same answer, found by trial and error.
Karpathy's version: https://www.youtube.com/watch?v=kCc8FmEb1nY
"""

import torch

from .base import LanguageModel


class BigramLogits(LanguageModel):
    fit_method = "gradient_descent"

    def __init__(self, vocab_size: int):
        # all zeros = every next token equally likely
        # requires_grad=True: track how the loss depends on these numbers, so we can step them
        self.table = torch.zeros(vocab_size, vocab_size, requires_grad=True)

    def parameters(self) -> list[torch.Tensor]:
        return [self.table]

    def logits(self, idx: torch.Tensor) -> torch.Tensor:
        return self.table[idx]  # row idx = scores for what comes after idx
