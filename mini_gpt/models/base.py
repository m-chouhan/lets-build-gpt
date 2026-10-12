"""What every language model shares. A model only has to say how it turns ids into logits."""

import torch
import torch.nn.functional as F


class LanguageModel:
    fit_method: str  # key into fitting.FIT_METHODS

    def logits(self, idx: torch.Tensor) -> torch.Tensor:
        """(B, T) token ids -> (B, T, vocab_size) scores for the next token."""
        raise NotImplementedError

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.logits(idx)
        if targets is None:
            return logits, None

        # cross-entropy = average surprise at the real next token:
        # https://en.wikipedia.org/wiki/Cross-entropy
        # it wants (N, C) and (N,), so flatten batch and time into one dim
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            last = self.logits(idx)[:, -1, :]  # (B, C) scores after the last token
            probs = F.softmax(last, dim=-1)  # https://en.wikipedia.org/wiki/Softmax_function
            next_id = torch.multinomial(probs, num_samples=1)  # weighted dice roll
            idx = torch.cat([idx, next_id], dim=1)
        return idx
