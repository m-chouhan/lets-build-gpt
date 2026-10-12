# %% [markdown]
# # Let's build GPT — following along
# Karpathy: https://www.youtube.com/watch?v=kCc8FmEb1nY
#
# Each `# %%` below is a runnable cell. Hit shift+enter to run one.

# %%
import torch

# %% read the dataset
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("length of dataset in characters:", len(text))

# %%
print(text[:300])

# %% the vocabulary
dictionary = sorted(set(text))
vocab_size = len(dictionary)
print("".join(dictionary))
print("vocab size:", vocab_size)

# %% tokenizer: character level
stoi = {ch: i for i, ch in enumerate(dictionary)}
itos = {i: ch for i, ch in enumerate(dictionary)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

print(encode("hii there, im mahendra"))
print(decode(encode("hii there, im mahendra")))

# %% encode the whole dataset into a tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:100])

# %% train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(len(train_data), len(val_data))

# %% [markdown]
# ## Next up: block_size / batching (~14:00 in the video)
block_size = 8
train_data_block = train_data[:block_size+1]
print(train_data_block)
# %%
