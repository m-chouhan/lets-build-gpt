"""Data for the interactive version of the blog posts. Saved as JSON in blog/data/.

Run: uv run python blog/export_data.py
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "mini_gpt"))  # reuse the real model code
from config import Config
from data import split
from models import BigramCounts
from tokenizer import CharTokenizer

DATA = Path(__file__).parent / "data"


def bigram_counts():
    cfg = Config()
    text = cfg.data_path.read_text(encoding="utf-8")
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text))
    train_data, _ = split(data, cfg.train_frac)

    model = BigramCounts(tok.vocab_size, cfg)
    model.count_pairs(train_data)

    # raw counts, so the page applies the same +smoothing and gets the same probabilities
    out = {
        "vocab": tok.itos,
        "smoothing": model.smoothing,
        "counts": model.counts.int().tolist(),
    }
    DATA.mkdir(exist_ok=True)
    (DATA / "bigram_counts.json").write_text(json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    bigram_counts()
