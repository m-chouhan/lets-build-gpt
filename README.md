# Let's build GPT, brick by brick

A language model built from scratch, one idea at a time, by someone who isn't comfortable with neural networks yet. I'm following Andrej Karpathy's ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) and writing a blog post for every step on [mchouhan.co.in](https://mchouhan.co.in).

| Post | Model | Validation loss |
| --- | --- | --- |
| [Part 01: The Dumbest Model That Writes](https://mchouhan.co.in/blogs/lets-build-gpt-from-scratch-01/) | `BigramCounts`: a 65×65 table of pair counts | 2.46 |

Random guessing scores 4.17. Every post adds a row.

## Run it

Needs [uv](https://docs.astral.sh/uv/).

```
uv run python mini_gpt/train.py --model BigramCounts
```

## What's where

| Path | What |
| --- | --- |
| `mini_gpt/` | The code: load data → tokenizer → split → train → evaluate → generate. |
| `blog/` | Drafts of the posts, plus `make_charts.py` for their images (`uv run --group blog python blog/make_charts.py`) and `export_data.py` for the interactive pages' data (`blog/data/`) |
| `datasets/` | Training text ([sources](datasets/README.md)) |
