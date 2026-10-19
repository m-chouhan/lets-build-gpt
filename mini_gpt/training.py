"""How a model's numbers get set from data. Each model names its method via `train_method`.

Counting is closed-form: one exact pass over the data.
Gradient descent is iterative: guess, measure the loss, step downhill, repeat.
"""

import torch

from data import get_batch
from evaluate import report


def train_by_counting(model, train_data, val_data, cfg):
    model.count_pairs(train_data)


def train_by_gradient_descent(model, train_data, val_data, cfg):
    # https://en.wikipedia.org/wiki/Gradient_descent
    # AdamW = gradient descent with per-parameter step sizes and momentum:
    # https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for step in range(cfg.max_steps):
        if step % cfg.eval_interval == 0:
            print(f"step {step:5d}: {report(model, train_data, val_data, cfg)}")

        x, y = get_batch(train_data, cfg.batch_size, cfg.block_size)
        loss = model.loss(x, y)  # predict and measure; changes nothing
        optimizer.zero_grad()  # gradients add up by default, so clear last step's
        loss.backward()  # gradient for every parameter: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
        optimizer.step()  # the only line that changes the parameters


TRAIN_METHODS = {
    "counting": train_by_counting,
    "gradient_descent": train_by_gradient_descent,
}


def train(model, train_data, val_data, cfg):
    """Train the model the way it says it learns (model.train_method)."""
    TRAIN_METHODS[model.train_method](model, train_data, val_data, cfg)
