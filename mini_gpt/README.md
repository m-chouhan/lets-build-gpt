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
 ┌──────────┐   ┌───────────┐   ┌──────────────┐   ┌───────────┐   ┌──────────┐   ┌──────────┐
 │  data    │ → │ tokenizer │ → │ training set │ → │  model    │ → │   fit    │ → │ generate │
 │ download │   │ text ↔ ids│   │ train / val  │   │ (by name) │   │ (model's │   │   text   │
 │ + read   │   │           │   │ + batches    │   │           │   │  method) │   │          │
 └──────────┘   └───────────┘   └──────────────┘   └───────────┘   └──────────┘   └──────────┘
```

| File | Brick | Job |
|---|---|---|
| `config.py` | knobs | Every knob in one place; each one is also a CLI flag (`--lr 0.01`) |
| `data.py` | data | `load_text` (download if missing), `split` (90/10), `get_batch` (random chunks) |
| `tokenizer.py` | tokenizer | `CharTokenizer`: 65 characters ↔ ids 0–64 |
| `models/base.py` | model contract | `LanguageModel`: shared `forward` (logits + loss) and `generate`. A model only defines `logits(idx)` |
| `models/bigram_counts.py` | level 0 model | `BigramCounts`: a table of counts |
| `models/bigram_logits.py` | level 1 model | `BigramLogits`: a table of learned logits |
| `models/__init__.py` | registry | `MODELS = {name: class}`, so a model is picked by name |
| `fitting.py` | fit methods | `fit_by_counting`, `fit_by_gradient_descent` |
| `evaluate.py` | scorecard | `estimate_loss`, the same measure for every model |
| `train.py` | pipeline | Wires everything together. **Never changes between levels** |

**Run a level:**
```
uv run python mini_gpt/train.py --model BigramCounts
uv run python mini_gpt/train.py --model BigramLogits
uv run python mini_gpt/train.py --seed 42      # different (but repeatable) samples
uv run python mini_gpt/train.py --seed false   # new random seed every run
```

**Design principles**
- **The model name says what the model *is* (its structure). The fit method says how it *learns*.** These are separate bricks.
- Each model declares its fit method in one line: `fit_method = "counting"`.
- Leveling up = add a model file and register it. `train.py` stays the same.
- Dependencies point one way: `data`, `tokenizer` and `models` import nothing from each other. `evaluate` and `fitting` may use `data`. `train.py` uses everything.
- Code should explain its intent through names. Comments are short and link out to read more.

### The level ladder

| Level | Model | Fit method | New idea |
|---|---|---|---|
| 0 | `BigramCounts` | counting (closed-form) | next-token probabilities, sampling, loss as a scorecard |
| 1 | `BigramLogits` | gradient descent (iterative) | logits, backward, optimizer |
| 2 | `BigramEmbedding` *(planned)* | gradient descent | `nn.Module`, embeddings (Karpathy's version) |
| 3 | `AttentionHead` *(planned)* | gradient descent | looking back more than one token |
| 4+ | … → `GPT` *(planned)* | gradient descent | the transformer |

From level 1 onward, the fit method never changes. Only the model gets smarter.

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

## 5. The same table, the neural way (trial and error)

Pretend we can't count. Start with a table of **all zeros** and **nudge** it until it's good:

```
 ┌──────────────────────────────────────────────────────────┐
 │  table row 'l'   (raw numbers = LOGITS = "scores")       │
 │      h     e     l     o                                 │
 │   [ 0.0   0.0   0.0   0.0 ]                              │
 └───────────────────────┬──────────────────────────────────┘
                         │ softmax: turn scores into %
                         ▼
                 [ 25%  25%  25%  25% ]    ← model's guess
                         │
      truth: after 'l' came 'o'
                         │ LOSS = how surprised was it?
                         ▼
          gave 'o' only 25%  →  loss = -ln(0.25) = 1.39
          (100% → loss 0,  tiny % → huge loss)
                         │
                         │ BACKWARD: for each score, which way
                         │ would have made the loss smaller?
                         ▼
                 [  ↓     ↓     ↓     ↑  ]
                         │
                         │ OPTIMIZER STEP: nudge each score
                         │ a little in its arrow's direction
                         ▼
   new row 'l':  [ -0.1  -0.1  -0.1  +0.3 ]  → 'o' now ~33%
```

Repeat thousands of times on random pairs. Row `l` drifts to about 50/50 `l`/`o`, **the same answer counting gave.**

### Vocabulary cheat sheet

| Word | Plain meaning |
|---|---|
| **parameters / weights** | The adjustable numbers inside the model. Here, the 65×65 = 4,225 table entries. |
| **logits** | Raw scores in the table. They can be any number, including negative ones. |
| **softmax** | Turns a row of scores into percentages that add up to 100. |
| **loss** (cross-entropy) | How surprised the model was by the real next character. Lower is better. |
| **gradient** | For each parameter, an up/down arrow (and how strongly) that would lower the loss. |
| **backward** | Computes the gradient for every parameter. |
| **optimizer** (AdamW) | Moves every parameter a little along its arrow. |
| **learning rate** (`lr`) | How big each nudge is. |

### The training loop: 4 lines that never change

```python
logits, loss = model.forward(xb, yb)   # forward: predict, measure surprise
optimizer.zero_grad()                  # wipe last step's arrows (they pile up otherwise)
loss.backward()                        # backward: for each knob, which way lowers the loss?
optimizer.step()                       # turn each knob a little that way
```

They live in `fit_by_gradient_descent` in `fitting.py`, the one place the loop is written down.

---

## 6. Why bother with the neural way if counting works?

Counting only works because the model looks back **one** character. To look at the previous 8, the table would need 65⁸ rows (about 300 trillion), which is impossible to count.

The score → surprise → arrows → nudge loop doesn't care *what* produces the scores. Today it's a lookup table. Later it's a transformer. **The training loop stays the same; only the box that makes the scores changes.** That's why we learn it on the simplest possible box first.

---

## 7. What to expect when it runs

Actual run (seed 1337):

| Model | Start | Final train / val | Why |
|---|---|---|---|
| `BigramCounts` | — | 2.47 / 2.46 | The exact answer, found in one pass |
| `BigramLogits` | 4.17 | 2.50 / 2.50 | All-zero table = uniform guess = −ln(1/65). It then learns its way to about the counting answer. (Karpathy's random init starts near 4.8.) |

Both stall around 2.5. One character of context is a hard ceiling, and that's the motivation for the rest of the video.

Output goes from random symbol soup to word-shaped gibberish with speaker names like `ROMEO:`.

**train vs val loss**: train loss is measured on text the model learns from, val loss on the 10% it never sees. If val stays close to train, the model is learning general patterns rather than memorizing.

---

## 8. My doubts along the way (Q&A)

**Q: What even is a "model"?**
A function with adjustable numbers inside. Ids go in, predictions come out. "Training" means adjusting those numbers until the predictions are good. For the bigram model, the "numbers inside" are just one 65×65 table.

**Q: Why break everything out of one file?**
So each step (download → tokenize → batch → train → generate) is a brick I can open, tune or replace on its own as I learn more, without the rest breaking.

**Q: What is the optimizer? Is AdamW just a known algorithm for training?**
Yes, it's a ready-made recipe for *how to nudge* the weights. The plain version is "step downhill by a fixed amount." AdamW is smarter: it adapts the step size per weight and carries momentum, so it usually needs fewer steps. You pick it and mostly forget it.

**Q: `max_steps` = how many times we run the optimizer?**
Yes. Each step is one batch → one round of nudges.

**Q: Why print the loss every `eval_interval` steps?**
To see whether the model is getting better or worse over time. The **val** loss is the honest number, since that text is never trained on. Each printed loss is averaged over several batches because a single batch is noisy.

**Q: Doesn't `model.forward` compute the next set of weights from the current ones?**
❌ No, and this was my main confusion. `forward` only *uses* the current weights to make predictions and measure the loss. It never changes them. The weights change only in `optimizer.step()`.

**Q: Then what is `loss.backward()` doing?**
After `forward`, we know *how wrong* we are (e.g. 4.17), but not *which way* to turn each of the 4,225 knobs. `backward()` answers, for every knob: *"if I nudged you up a tiny bit, would the loss go up or down, and by how much?"* That's the gradient. PyTorch works it out by replaying the forward computation in reverse (the chain rule). Example: if the batch had `h` followed by `e`, the arrow on table[`h`,`e`] says "turn up" and the rest of row `h` says "turn down."

**Q: Why `zero_grad()`?**
PyTorch *adds* new gradients onto old ones by default. We want fresh arrows each step, so we wipe them first.

**Q: The logits / loss / optimizer part goes over my head. Is there a simpler approach?**
Yes: **counting** (section 4). It produces the same bigram table with no logits, loss or optimizer. The neural version is "counting, learned slowly by trial and error." It's worth learning because it's the only version that scales to real models (section 6).

**Q: Can a library abstract away the model computation?**
PyTorch already hides the hard parts: `F.cross_entropy` (softmax + surprise), `loss.backward()` (the arrows) and `AdamW` (the nudging). Higher-level libraries like Lightning hide the loop too, but the loop is the thing I'm trying to understand, so I'm not using them. Instead, the loop lives in one clearly named function, `fit_by_gradient_descent`.

**Q: Can we start with a "stupid" approach: walk the text, count which character follows each character, and turn those counts into probabilities?**
Yes, and it isn't stupid. It's exactly the counting method in section 4, and it's how Karpathy's *makemore* series starts. For a bigram it gives essentially the answer the neural version slowly learns its way to.
```python
counts = torch.zeros(65, 65)
for a, b in zip(data[:-1], data[1:]):        # every (current → next) pair
    counts[a, b] += 1
probs = (counts + 1) / (counts + 1).sum(dim=1, keepdim=True)   # each row sums to 1
```
- `+ 1` (smoothing): a pair never seen would otherwise get probability 0, and a model that says "impossible" and is then wrong gets infinite loss.
- It hits the same ceiling, a loss of about 2.45–2.5, because it also sees only one character. If the neural version reaches about the same number, it has learned the counting table.
- It stops working beyond one character of context (65⁸ rows). That's why the neural version exists.

**Q: Isn't counting also "training" a model?**
Yes. Counting is the most primitive way to set a model's numbers from data. That's why it became level 0 of the same pipeline instead of a side experiment.

**Q: Should counting go through the same batch loop as gradient descent?**
No. Forcing it in would make the loop harder to understand just to make the code look uniform. Counting is *closed-form* (one exact pass), and gradient descent is *iterative* (step by step). So each model declares its own `fit_method`, and `train.py` just calls it.

**Q: Why the names `BigramCounts` / `BigramLogits`?**
Plain-English names (`tally`, `nudge`) watered the idea down, and they described *how* the model learns, which isn't the model's job here. Both models have the **same structure**, a 65×65 bigram table. The only difference is what the table holds: counts or logits. After training, logits ≈ log(counts). So the name differs by exactly the word that is the lesson. The technical terms (Markov chain, maximum likelihood estimate) go in the docstrings instead.

**Q: Why "fit" and not "trainer"?**
What this brick really does is *estimate the model's numbers from data*. "Fitting a model to data" is the standard term in statistics and ML (scikit-learn's `.fit()`). It doesn't clash with PyTorch names, unlike `optimizer`, which already means AdamW. It also leaves room for the real lesson: `fit_by_counting` (closed-form) vs `fit_by_gradient_descent` (iterative).

**Q: Why does `BigramCounts` return `log(probs)` as its logits?**
So every model plugs into the same `forward`, loss and `generate`. softmax(log(p)) = p, so nothing changes for sampling, and the loss numbers are directly comparable across levels.

**Q: Why `index_put_(..., accumulate=True)` instead of `counts[a, b] += 1`?**
With a whole tensor of pairs at once, plain `+=` counts each repeated pair only once. `accumulate=True` adds every occurrence. A Python `for` loop over 1M pairs would also be correct, just slow.

**Q: Why does every run print the exact same text?**
Because of the fixed seed (`seed = 1337` in `config.py`). Counting always gives the same table, and generating text picks each next character at random (`torch.multinomial`). A fixed seed makes PyTorch return the same "random" numbers every time, so it makes the same picks. That's on purpose: if the output changes, it's because you changed something, not because of luck. Use `--seed 42` to get a different repeatable run, or `--seed false` to get a new random one. The run prints the seed it used, so you can repeat any run you like. The loss stays about the same (~2.47) whatever the seed; only the samples change. ([PyTorch randomness notes](https://pytorch.org/docs/stable/notes/randomness.html))

---

## 9. Status / TODOs

- [x] Rename `bigram/` → `mini_gpt/`; split model vs fit method
- [x] Level 0: `BigramCounts` (val ≈ 2.46)
- [x] Level 1: `BigramLogits` (val ≈ 2.50)
- [ ] Level 2: `BigramEmbedding` as an `nn.Module` (Karpathy's version)
- [ ] Level 3: self-attention

## 10. Learnings log

- A bigram model is just a lookup table of "what usually comes next."
- Training = predict → measure surprise → compute arrows → nudge. The same 4 lines will train GPT.
- `forward` doesn't learn; `optimizer.step()` does.
- For a bigram, counting pairs and gradient-descent training arrive at essentially the same table.
- Counting is training too: closed-form vs iterative is the real difference between level 0 and level 1.
- Name things by what they *are*. Keep "what it is" (model) separate from "how it learns" (fit method).
