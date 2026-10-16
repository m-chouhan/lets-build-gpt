"""The pipeline. Same for every model; pick one by name.

Run: uv run python mini_gpt/train.py --model BigramCounts
Other text: uv run python mini_gpt/train.py --data_path datasets/names.txt
Any knob in config.py works as a flag, e.g. --seed 42 --lr 0.01
"""

import torch

from config import Config
from data import split
from evaluate import report
from training import train
from models import MODELS
from tokenizer import CharTokenizer


def sample(model, tok, n: int) -> str:
    start = torch.zeros((1, 1), dtype=torch.long)  # token 0 is '\n'
    return tok.decode(model.generate(start, n)[0].tolist())


def set_seed(seed: int | None):
    """Same seed => same "random" numbers => same output every run.
    https://pytorch.org/docs/stable/notes/randomness.html
    """
    if seed is None:
        torch.seed()  # fresh random seed from the OS
    else:
        torch.manual_seed(seed)
    print(f"seed: {torch.initial_seed()}")  # rerun with --seed <this> to reproduce


def main():
    cfg = Config.from_cli(model=list(MODELS))
    set_seed(cfg.seed)

    # 1. get data
    text = cfg.data_path.read_text(encoding="utf-8")
    print(f"dataset: {cfg.data_path.name}, {len(text):,} characters")

    # 2. build encoder/decoder
    tok = CharTokenizer(text)
    print(f"vocab size: {tok.vocab_size}")

    # 3. make training set
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    train_data, val_data = split(data, cfg.train_frac)

    # 4. make model
    model_class = MODELS[cfg.model]  # look up the class by name, e.g. "BigramCounts" -> BigramCounts
    model = model_class(tok.vocab_size, cfg)  # initialize it
    print(f"model: {cfg.model}  (trained by {model.train_method})")
    print("\n--- BEFORE training ---")
    print(sample(model, tok, 200))

    # 5. train
    print()
    train(model, train_data, val_data, cfg)
    print(f"final: {report(model, train_data, val_data, cfg)}")

    # 6. generate
    print("\n--- AFTER training ---")
    print(sample(model, tok, cfg.gen_tokens))


if __name__ == "__main__":
    main()
