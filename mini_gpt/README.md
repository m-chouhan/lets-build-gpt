# mini_gpt: Building a Language Model, Brick by Brick

> Raw notes while following Andrej Karpathy's ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY).
> Written by someone who is *not* comfortable with neural networks yet. The goal is intuition, not math.
> These notes grow as the conversation continues. Later they'll become a blog post.

---

## 1. What are we trying to do?

Build the **smallest thing that learns to write** and generate *something*, even gibberish, from Shakespeare's text.

Not a good model. Just the **outline of training**: data in, model learns, text out. Every later piece of the video (self-attention, transformers) slots into this same outline.

The model we use is a **bigram** model: *predict the next character by looking at only the current character.*

---

## 2. The architecture: Lego bricks

```
 ┌──────────┐   ┌───────────┐   ┌──────────────┐   ┌───────────┐   ┌───────────┐   ┌──────────┐   ┌──────────┐
 │  data    │ → │ tokenizer │ → │ training set │ → │  model    │ → │   train   │ → │ evaluate │ → │ generate │
 │ read a   │   │ text ↔ ids│   │ train / val  │   │ (by name) │   │ (model's  │   │ training │   │   text   │
 │ .txt     │   │           │   │ + batches    │   │           │   │  method)  │   │ / val    │   │          │
 └──────────┘   └───────────┘   └──────────────┘   └───────────┘   └───────────┘   └──────────┘   └──────────┘
```

| File | Brick | Job |
|---|---|---|
| `config.py` | knobs | Every knob in one place; each one is also a CLI flag (`--seed 42`) |
| `data.py` | data | `split` (90/10), `get_batch` (random chunks). Text files live in `datasets/` |
| `tokenizer.py` | tokenizer | `CharTokenizer`: 65 characters ↔ ids 0–64 |
| `models/base.py` | model contract | `LanguageModel`: shared `loss` (average surprise) and `generate`. A model only defines `logits(idx)` |
| `models/bigram_counts.py` | level 0 model | `BigramCounts`: a table of counts |
| `models/__init__.py` | registry | `MODELS = {name: class}`, so a model is picked by name |
| `training.py` | training methods | `train_by_counting` (more methods arrive in later posts) |
| `evaluate.py` | scorecard | `estimate_loss`, the same measure for every model |
| `train.py` | pipeline | Wires everything together. **Never changes between levels** |

**Run a level:**
```
uv run python mini_gpt/train.py --model BigramCounts
uv run python mini_gpt/train.py --seed 42      # different (but repeatable) samples
uv run python mini_gpt/train.py --seed false   # new random seed every run
```

**Design principles**
- **The model name says what the model *is* (its structure). The training method says how it *learns*.** These are separate bricks.
- Each model declares its training method in one line: `train_method = "counting"`.
- Leveling up = add a model file and register it. `train.py` stays the same.
- Dependencies point one way: `data`, `tokenizer` and `models` import nothing from each other. `evaluate` and `training` may use `data`. `train.py` uses everything.
- Code should explain its intent through names. Comments are short and link out to read more.

### The level ladder

| Level | Model | Training method | New idea |
|---|---|---|---|
| 0 | `BigramCounts` | counting (closed-form) | next-token probabilities, sampling, loss as a scorecard |
| 1 | `BigramLogits` *(next post)* | gradient descent (iterative) | logits, backward, optimizer |
| 2 | `BigramEmbedding` *(planned)* | gradient descent | `nn.Module`, embeddings (Karpathy's version) |
| 3 | `AttentionHead` *(planned)* | gradient descent | looking back more than one token |
| 4+ | … → `GPT` *(planned)* | gradient descent | the transformer |

From level 1 onward, the training method never changes. Only the model gets smarter.

---

## 3. The data, in shapes

The whole text becomes one long list of ids (about 1.1M numbers). Training doesn't use it all at once. It grabs **random chunks**:

- **block_size (T)** = how many characters per chunk (the context length). We use 8.
- **batch_size (B)** = how many chunks we process in parallel. We use 32.
- **vocab_size (C)** = 65 possible characters.

For each chunk, `x` is the input and `y` is **the same chunk shifted one to the right**, i.e. the answer at every position:

```
text:  h  e  l  l  o  !
x:     h  e  l  l  o        ← "given this…"
y:     e  l  l  o  !        ← "…the next char should be this"
```

One chunk of 8 gives 8 training examples at once.

---

## 4. The bigram idea: first, the simple way (just count)

Take a tiny alphabet `{h, e, l, o}` and the training text `hello`. Every pair of neighbours is an example:

```
h→e   e→l   l→l   l→o
```

Tally them into a table. Each row is the current letter, and each column is the letter that came next:

```
              next letter
             h   e   l   o
current  h [ 0   1   0   0 ]
         e [ 0   0   1   0 ]
         l [ 0   0   1   1 ]   ← after 'l': 'l' once, 'o' once
         o [ 0   0   0   0 ]
```

To generate, look up the row for the current letter and roll a die weighted by that row:

```
start 'h' → always 'e' → always 'l' → 50% 'l' / 50% 'o' 🎲
```

**That's a complete bigram model.** For Shakespeare, it's a 65×65 table.

---

## 5. What to expect when it runs

Actual run (seed 1337):

| Model | Start | Final training / validation loss | Why |
|---|---|---|---|
| `BigramCounts` | — | 2.47 / 2.46 | The exact answer, found in one pass |

It stalls around 2.5. One character of context is a hard ceiling, and that's the motivation for the rest of the video.

Output goes from random symbol soup to word-shaped gibberish with speaker names like `ROMEO:`.

**training_loss vs validation_loss**: training loss is measured on text the model learns from, validation loss on the held-out 10% it never sees. If validation stays close to training, the model is learning general patterns rather than memorizing.

---

## 6. My doubts along the way (Q&A)

**Q: What even is a "model"?**
A function with adjustable numbers inside. Ids go in, predictions come out. "Training" means adjusting those numbers until the predictions are good. For the bigram model, the "numbers inside" are just one 65×65 table.

**Q: Why break everything out of one file?**
So each step (load → tokenize → batch → train → evaluate → generate) is a brick I can open, tune or replace on its own as I learn more, without the rest breaking.

**Q: Can we start with a "stupid" approach: walk the text, count which character follows each character, and turn those counts into probabilities?**
Yes, and it isn't stupid. It's exactly the counting method in section 4, and it's how Karpathy's *makemore* series starts.
```python
counts = torch.zeros(65, 65)
for a, b in zip(data[:-1], data[1:]):        # every (current → next) pair
    counts[a, b] += 1
probs = (counts + 1) / (counts + 1).sum(dim=1, keepdim=True)   # each row sums to 1
```
- `+ 1` (smoothing): a pair never seen would otherwise get probability 0, and a model that says "impossible" and is then wrong gets infinite loss.
- It hits a ceiling, a loss of about 2.45–2.5, because it sees only one character.
- It stops working beyond one character of context (65⁸ rows). That's why later posts switch to learning the table instead.

**Q: Isn't counting also "training" a model?**
Yes. Counting is the most primitive way to set a model's numbers from data. That's why it became level 0 of the same pipeline instead of a side experiment.

**Q: Why does each model declare a `train_method`?**
Counting is *closed-form* (one exact pass). Later models learn *iteratively* (step by step), and forcing both through one loop would make it harder to understand. So each model declares its own `train_method`, and `train.py` just calls it.

**Q: Why "train" and not "fit"?**
It was called "fit" at first. "Fitting a model to data" is the statistics / scikit-learn term (`.fit()`), and it avoided a clash with PyTorch's `optimizer`. But the GPT world (Karpathy, nanoGPT, PyTorch tutorials) says **train**, and a beginner reading the blog already knows that word. So it's now `training.py` with `train_by_counting`. The lesson: counting is training too.

**Q: Why does `BigramCounts` return `log(probs)` as its logits?**
So every model plugs into the same `loss` and `generate`. softmax(log(p)) = p, so nothing changes for sampling, and the loss numbers are directly comparable across levels.

**Q: Why `model.loss` and not `model.forward` like Karpathy?**
It was `forward`, returning `(logits, loss)`. But every caller threw the logits away, and our models aren't `nn.Module`s, so `forward` gave us none of PyTorch's `model(x)` magic. It was just a name that hid what the method does: measure average surprise. `forward` (and the "forward pass") comes back when the neural models do.

**Q: How does `count_pairs` fill the table, and why not a one-liner?**
It first counts pairs in a plain Python `Counter` (`{(t, h): 2100, ...}`), then copies the ~4k distinct pairs into the table. An earlier version did it in one PyTorch line, `counts.index_put_((current, following), torch.ones(len(current)), accumulate=True)`: about 5× faster (0.01 s vs 0.05 s on 1M pairs), but hard to read, and it needed `accumulate=True` because plain `counts[current, following] += 1` silently counts each repeated pair only once. A plain `for` loop over all 1M pairs is the most obvious version but takes ~1.8 s. `Counter` is the middle ground: nearly as clear as the loop and still effectively instant.

**Q: Why does every run print the exact same text?**
Because of the fixed seed (`seed = 1337` in `config.py`). Counting always gives the same table, and generating text picks each next character at random (`torch.multinomial`). A fixed seed makes PyTorch return the same "random" numbers every time, so it makes the same picks. That's on purpose: if the output changes, it's because you changed something, not because of luck. Use `--seed 42` to get a different repeatable run, or `--seed false` to get a new random one. The run prints the seed it used, so you can repeat any run you like. The loss stays about the same (~2.47) whatever the seed; only the samples change. ([PyTorch randomness notes](https://pytorch.org/docs/stable/notes/randomness.html))

---

## 7. Status / TODOs

- [x] Rename `bigram/` → `mini_gpt/`; split model vs training method
- [x] Rename fit → train (`fitting.py` → `training.py`)
- [x] Level 0: `BigramCounts` (val ≈ 2.46)
- [ ] Level 1: `BigramLogits`, gradient descent (next post)
- [ ] Level 2: `BigramEmbedding` as an `nn.Module` (Karpathy's version)
- [ ] Level 3: self-attention

## 8. Learnings log

- A bigram model is just a lookup table of "what usually comes next."
- Loss = average surprise at the real next character. Lower is better; 4.17 is random guessing.
- Counting is training too: the simplest way to set a model's numbers from data.
- Name things by what they *are*. Keep "what it is" (model) separate from "how it learns" (training method).
