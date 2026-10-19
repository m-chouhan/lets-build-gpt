# Let's build GPT, brick by brick

A language model built from scratch, one idea at a time, by someone who isn't comfortable with neural networks yet. I'm following Andrej Karpathy's ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) and writing a blog post for every step.

| Post | Model | Validation loss |
| --- | --- | --- |
| [1. Just Count: The Smallest Model That Writes](blog/01-just-count.md) | `BigramCounts`: a 65×65 table of pair counts | 2.46 |

Random guessing scores 4.17. Every post adds a row.

## Run it

Needs [uv](https://docs.astral.sh/uv/).

```
uv run python mini_gpt/train.py --model BigramCounts
```

## What's where

| Path | What |
| --- | --- |
| `mini_gpt/` | The code: load data → tokenizer → split → train → evaluate → generate. [Notes](mini_gpt/README.md) |
| `blog/` | The posts, plus `make_charts.py` for their images (`uv run --group blog python blog/make_charts.py`) |
| `datasets/` | Training text ([sources](datasets/README.md)) |
