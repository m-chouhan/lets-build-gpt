"""The pipeline. Same for every model; pick one by name.

Run: uv run python mini_gpt/train.py --model BigramCounts
"""

import argparse

import torch

from config import Config
from data import load_text, split
from evaluate import estimate_loss
from fitting import FIT_METHODS
from models import MODELS
from tokenizer import CharTokenizer


def sample(model, tok, n: int) -> str:
    start = torch.zeros((1, 1), dtype=torch.long)  # token 0 is '\n'
    return tok.decode(model.generate(start, n)[0].tolist())


def main():
    cfg = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default=cfg.model)
    cfg.model = parser.parse_args().model
    torch.manual_seed(cfg.seed)

    # 1. get data
    text = load_text(cfg.data_path, cfg.data_url)
    print(f"dataset: {len(text):,} characters")

    # 2. build encoder/decoder
    tok = CharTokenizer(text)
    print(f"vocab size: {tok.vocab_size}")

    # 3. make training set
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    train_data, val_data = split(data, cfg.train_frac)

    # 4. make model
    model = MODELS[cfg.model](tok.vocab_size)
    print(f"model: {cfg.model}  (fit by {model.fit_method})")
    print("\n--- BEFORE fitting ---")
    print(sample(model, tok, 200))

    # 5. fit
    print()
    FIT_METHODS[model.fit_method](model, train_data, val_data, cfg)
    print(f"final: train {estimate_loss(model, train_data, cfg):.4f}  val {estimate_loss(model, val_data, cfg):.4f}")

    # 6. generate
    print("\n--- AFTER fitting ---")
    print(sample(model, tok, cfg.gen_tokens))


if __name__ == "__main__":
    main()
