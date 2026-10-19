"""Data for the interactive version of the blog posts. Saved as JSON in blog/data/.

Run: uv run python blog/export_data.py
"""

import json
import random
import re
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


def heldout_snippets(n=60, context=45):
    """Lines from the locked-away 10%, cut off mid-word, for the "afte_" guessing game."""
    cfg = Config()
    text = cfg.data_path.read_text(encoding="utf-8")
    tok = CharTokenizer(text)
    _, val_data = split(torch.tensor(tok.encode(text)), cfg.train_frac)
    lines = [line for line in tok.decode(val_data.tolist()).split("\n") if len(line) >= 30]

    rng = random.Random(0)
    snippets = []
    while len(snippets) < n:
        line = rng.choice(lines)
        # cut between two lowercase letters, so the blank is in the middle of a word
        cuts = [k for k in range(20, len(line)) if line[k - 1].isalpha() and line[k].islower()]
        if not cuts:
            continue
        k = rng.choice(cuts)
        before = line[max(0, k - context):k]
        if k > context:
            before = before[before.find(" ") + 1:]  # start on a whole word
        rest_of_word = re.match(r"[a-z]*", line[k:]).group()
        snippet = {"before": before, "answer": line[k], "rest": rest_of_word[1:]}
        if snippet not in snippets:
            snippets.append(snippet)

    (DATA / "heldout_snippets.json").write_text(json.dumps(snippets, indent=1))


if __name__ == "__main__":
    bigram_counts()
    heldout_snippets()
