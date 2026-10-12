"""Level 0. Count which token follows which; counts / row total = probabilities.

A.k.a. a first-order Markov chain: https://en.wikipedia.org/wiki/Markov_chain
Counting is the maximum likelihood estimate: https://en.wikipedia.org/wiki/Maximum_likelihood_estimation
Karpathy builds this in makemore part 1: https://www.youtube.com/watch?v=PaCmpygFfXo
"""

import torch

from .base import LanguageModel


class BigramCounts(LanguageModel):
    fit_method = "counting"
    smoothing = 1.0  # pretend every pair was seen once: https://en.wikipedia.org/wiki/Additive_smoothing

    def __init__(self, vocab_size: int, cfg):
        super().__init__(vocab_size, cfg)
        self.counts = torch.zeros(vocab_size, vocab_size)  # counts[a, b] = times b followed a
        self.log_probs = self._to_log_probs()

    def count_pairs(self, data: torch.Tensor):
        current, following = data[:-1], data[1:]
        # counts[current, following] += 1 for every pair at once
        # (plain `+=` would count repeated pairs only once; accumulate=True counts them all)
        self.counts.index_put_((current, following), torch.ones(len(current)), accumulate=True)
        self.log_probs = self._to_log_probs()

    def _to_log_probs(self) -> torch.Tensor:
        smoothed = self.counts + self.smoothing
        probs = smoothed / smoothed.sum(dim=1, keepdim=True)  # each row sums to 1
        return probs.log()  # log turns probabilities into logits: softmax(log(p)) == p

    def logits(self, idx: torch.Tensor) -> torch.Tensor:
        return self.log_probs[idx]
