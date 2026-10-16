"""Level 0. Count which token follows which; counts / row total = probabilities.

A.k.a. a first-order Markov chain: https://en.wikipedia.org/wiki/Markov_chain
Counting is the maximum likelihood estimate: https://en.wikipedia.org/wiki/Maximum_likelihood_estimation
Karpathy builds this in makemore part 1: https://www.youtube.com/watch?v=PaCmpygFfXo
"""

from collections import Counter

import torch

from .base import LanguageModel


class BigramCounts(LanguageModel):
    train_method = "counting"
    smoothing = 1.0  # pretend every pair was seen once: https://en.wikipedia.org/wiki/Additive_smoothing

    def __init__(self, vocab_size: int, cfg):
        super().__init__(vocab_size, cfg)
        self.counts = torch.zeros(vocab_size, vocab_size)  # counts[a, b] = times b followed a
        self.log_probs = self._to_log_probs()

    def count_pairs(self, data: torch.Tensor):
        # line the text up against itself shifted by one: "First" -> (F,i) (i,r) (r,s) (s,t)
        current, following = data[:-1].tolist(), data[1:].tolist()

        # count each distinct pair in a plain dict: {(t, h): 2100, (h, e): 1500, ...}
        # https://docs.python.org/3/library/collections.html#collections.Counter
        pair_counts = Counter(zip(current, following))

        # copy into the table (~4k distinct pairs, so this loop is quick)
        for (a, b), n in pair_counts.items():
            self.counts[a, b] += n
        self.log_probs = self._to_log_probs()

    def _to_log_probs(self) -> torch.Tensor:
        smoothed = self.counts + self.smoothing
        probs = smoothed / smoothed.sum(dim=1, keepdim=True)  # each row sums to 1
        return probs.log()  # log turns probabilities into logits: softmax(log(p)) == p

    def logits(self, idx: torch.Tensor) -> torch.Tensor:
        return self.log_probs[idx]
