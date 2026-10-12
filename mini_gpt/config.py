"""All the knobs in one place. Tune here, not inside the other files."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # data
    data_url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_path: Path = Path(__file__).parent.parent / "input.txt"
    train_frac: float = 0.9  # 90% train, 10% validation

    # batching
    batch_size: int = 32  # B: chunks processed in parallel
    block_size: int = 8  # T: characters per chunk (max context)

    # gradient descent (ignored by counting)
    lr: float = 1e-1  # learning rate: size of each step
    max_steps: int = 3000
    eval_interval: int = 500

    # evaluation
    eval_iters: int = 50  # batches averaged per loss estimate

    # generation
    gen_tokens: int = 500

    seed: int = 1337  # same random numbers every run
