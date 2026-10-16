"""One scorecard for every model: average loss on random batches."""

import torch

from data import get_batch


@torch.no_grad()
def estimate_loss(model, data, cfg) -> float:
    """A single batch is noisy, so average over cfg.eval_iters of them."""
    losses = []
    for _ in range(cfg.eval_iters):
        x, y = get_batch(data, cfg.batch_size, cfg.block_size)
        _, loss = model.forward(x, y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


def report(model, train_data, val_data, cfg) -> str:
    """training_loss: on text the model learns from. validation_loss: on held-out text it never learns from."""
    training_loss = estimate_loss(model, train_data, cfg)
    validation_loss = estimate_loss(model, val_data, cfg)
    return f"training_loss {training_loss:.4f}  validation_loss {validation_loss:.4f}"
