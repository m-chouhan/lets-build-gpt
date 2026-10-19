"""Charts for the blog posts. Saved as PNGs in blog/images/.

Run: uv run --with matplotlib python blog/make_charts.py
"""

import io
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "mini_gpt"))  # reuse the real model code
from config import Config
from data import split
from evaluate import estimate_loss
from models import BigramCounts
from tokenizer import CharTokenizer

IMAGES = Path(__file__).parent / "images"

# colors: one blue for every bar, quiet greys for text and grid
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # light -> dark

# one row per level; each new post appends a row
SCOREBOARD = [
    ("random guess", math.log(65)),  # every char equally likely: -ln(1/65)
    ("L0 · BigramCounts", 2.4599),
]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, length=0)


def scoreboard_chart():
    names = [name for name, _ in SCOREBOARD]
    losses = [loss for _, loss in SCOREBOARD]

    fig, ax = plt.subplots(figsize=(8, 1 + 0.6 * len(SCOREBOARD)), facecolor=SURFACE)
    style(ax)
    ax.barh(names, losses, color=BLUE, height=0.5)
    ax.invert_yaxis()  # first level on top
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 4.6)

    for y, loss in enumerate(losses):
        ax.text(loss + 0.05, y, f"{loss:.2f}", va="center", color=TEXT)

    ax.set_title("Validation loss (lower is better)", loc="left", color=TEXT, fontsize=12)
    ax.set_xlabel("average surprise per character (nats)", color=TEXT_MUTED)
    for label in ax.get_yticklabels():
        label.set_color(TEXT)

    fig.tight_layout()
    fig.savefig(IMAGES / "scoreboard.png", dpi=200)


def bigram_heatmap():
    """The trained model itself: P(next | current) for space + a–z."""
    cfg = Config()
    text = cfg.data_path.read_text(encoding="utf-8")
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    train_data, _ = split(data, cfg.train_frac)

    model = BigramCounts(tok.vocab_size, cfg)
    model.count_pairs(train_data)
    probs = model.log_probs.exp()  # back from logits to percentages

    # 65 chars is too many to read, so zoom in on space and lowercase letters
    chars = " abcdefghijklmnopqrstuvwxyz"
    ids = tok.encode(chars)
    zoom = probs[ids][:, ids]
    labels = ["␣" if c == " " else c for c in chars]

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=SURFACE)
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    image = ax.imshow(zoom, cmap=cmap, vmin=0, vmax=0.5)  # clip at 50% so mid values stay visible

    ax.set_xticks(range(len(chars)), labels)
    ax.set_yticks(range(len(chars)), labels)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(colors=TEXT, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("next character", color=TEXT_MUTED)
    ax.set_ylabel("current character", color=TEXT_MUTED)

    bar = fig.colorbar(image, ax=ax, shrink=0.7, ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5])
    bar.ax.set_yticklabels(["0%", "10%", "20%", "30%", "40%", "≥50%"], color=TEXT_MUTED)
    bar.outline.set_visible(False)

    fig.suptitle("The whole model: one row = one die to roll", x=0.06, ha="left", color=TEXT, fontsize=13)
    fig.tight_layout()
    fig.savefig(IMAGES / "bigram_heatmap.png", dpi=200)


def poll_chart(polls=20):
    """Validation loss estimated from random samples ("polls") vs the exact average over every pair."""
    cfg = Config()
    text = cfg.data_path.read_text(encoding="utf-8")
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    train_data, val_data = split(data, cfg.train_frac)

    model = BigramCounts(tok.vocab_size, cfg)
    model.count_pairs(train_data)

    estimates = []
    for seed in range(polls):
        torch.manual_seed(seed)  # a different random sample each time
        estimates.append(estimate_loss(model, val_data, cfg))

    # exact: score every neighbouring pair in the validation text
    surprises = -model.log_probs[val_data[:-1], val_data[1:]]
    exact = surprises.mean().item()

    fig, ax = plt.subplots(figsize=(8, 2.4), facecolor=SURFACE)
    style(ax)
    ax.scatter(estimates, [0] * len(estimates), s=60, color=BLUE, alpha=0.6, edgecolors=SURFACE, linewidths=1.5)
    ax.plot([exact, exact], [-0.35, 0.35], color=TEXT, linewidth=1.5)  # short line so it doesn't cross the labels
    ax.text(exact, 0.5, f"exact, every pair: {exact:.3f}", ha="center", color=TEXT, fontsize=10)
    ax.text(min(estimates), -0.6, f"each dot: one poll of 12,800 random characters ({polls} polls)", ha="left", color=TEXT_MUTED, fontsize=10)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xlim(2.40, 2.55)
    ax.set_title("Validation loss: a poll is close, but not exact", loc="left", color=TEXT, fontsize=12)

    fig.tight_layout()
    fig.savefig(IMAGES / "loss_polls.png", dpi=200)


# made-up probabilities, just to show the loop: (text so far, next-word guesses, word picked)
PREDICTION_STEPS = [
    ("The cat sat on the", [("mat", 0.31), ("floor", 0.12), ("sofa", 0.08), ("roof", 0.05), ("bed", 0.04)], "mat"),
    ("The cat sat on the mat", [(".", 0.38), ("and", 0.21), (",", 0.12), ("while", 0.05), ("all", 0.04)], "and"),
    ("The cat sat on the mat and", [("purred", 0.24), ("looked", 0.15), ("slept", 0.12), ("stared", 0.07), ("waited", 0.05)], "purred"),
]


def prediction_frame(text, guesses, picked, show_pick, question="what comes next?", unit="word") -> Image.Image:
    """One frame: the text so far on top, the model's next-word (or next-character) guesses as bars below."""
    fig = plt.figure(figsize=(8, 4.2), facecolor=SURFACE)

    fig.text(0.06, 0.88, text, fontsize=17, color=TEXT, family="serif")
    if show_pick:
        fig.text(0.06, 0.80, f"picked: \"{picked}\"  → add it and ask again", fontsize=11, color=BLUE)
    else:
        fig.text(0.06, 0.80, question, fontsize=11, color=TEXT_MUTED)

    ax = fig.add_axes([0.2, 0.08, 0.7, 0.62])
    style(ax)
    words = [word for word, _ in guesses]
    probs = [prob for _, prob in guesses]
    colors = [BLUE if (show_pick and word == picked) or not show_pick else "#cde2fb" for word in words]
    ax.barh(words, probs, color=colors, height=0.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.45)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    for label in ax.get_yticklabels():
        label.set_color(TEXT)
        label.set_fontsize(12)
    for y, prob in enumerate(probs):
        ax.text(prob + 0.008, y, f"{prob:.0%}", va="center", color=TEXT_MUTED)
    ax.text(0.45, len(words) - 0.3, f"… and every other {unit}", ha="right", color=TEXT_MUTED, fontsize=9)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, facecolor=SURFACE)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def prediction_animation():
    """GIF of the core LLM loop: predict next word -> pick one -> append -> repeat."""
    frames, durations = [], []
    for text, guesses, picked in PREDICTION_STEPS:
        frames.append(prediction_frame(text, guesses, picked, show_pick=False))
        durations.append(1600)
        frames.append(prediction_frame(text, guesses, picked, show_pick=True))
        durations.append(1600)
    durations[-1] = 3000  # linger on the last frame before looping
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#saving
    frames[0].save(IMAGES / "llm_prediction.gif", save_all=True, append_images=frames[1:], duration=durations, loop=0)


# (starting text, characters to append): the picks are dice rolls, not always the top bar
CHAR_PREDICTION = ("hear me sp", "ear")


def char_prediction_animation():
    """Same loop as above, one character at a time, with real probabilities from the counted table."""
    cfg = Config()
    text = cfg.data_path.read_text(encoding="utf-8")
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    train_data, _ = split(data, cfg.train_frac)

    model = BigramCounts(tok.vocab_size, cfg)
    model.count_pairs(train_data)
    probs = model.log_probs.exp()

    so_far, picks = CHAR_PREDICTION
    frames, durations = [], []
    for picked in picks:
        last = so_far[-1]
        top_probs, top_ids = probs[tok.stoi[last]].topk(5)
        guesses = []
        for prob, i in zip(top_probs.tolist(), top_ids.tolist()):
            char = tok.itos[i]
            guesses.append(("␣" if char == " " else char, prob))  # show space as a visible symbol

        question = f"what comes after \"{last}\"?"
        frames.append(prediction_frame(so_far, guesses, picked, False, question, unit="character"))
        durations.append(1600)
        frames.append(prediction_frame(so_far, guesses, picked, True, question, unit="character"))
        durations.append(1600)
        so_far += picked
    durations[-1] = 3000
    frames[0].save(IMAGES / "char_prediction.gif", save_all=True, append_images=frames[1:], duration=durations, loop=0)


# one brick per pipeline step, bottom of the stack first: (name, what it does in post 1, lego color)
BRICKS = [
    ("load data", "reads shakespeare.txt", "#c91a09"),
    ("tokenizer", "text \u2194 ids:  \"hi\" \u2192 [46, 47]", "#fe8a18"),
    ("split", "90% for learning, 10% locked away", "#f2cd37"),
    ("train", "count every pair \u2192 65\u00d765 table", "#0055bf"),
    ("evaluate", "score on training + the locked-away 10%", "#237841"),
    ("generate", "roll dice \u2192 new text", "#81007b"),
]


def shade(hex_color, factor):
    """Darker (factor < 1) or lighter (factor > 1) version of a color, for the brick's 3D faces."""
    rgb = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    if factor < 1:
        rgb = [c * factor for c in rgb]
    else:
        rgb = [c + (1 - c) * (factor - 1) for c in rgb]
    return tuple(rgb)


def lego_bricks():
    """The pipeline as a stack of Lego bricks: each step sits on the one below.

    The train brick is pulled halfway out, as if we're about to swap it.
    """
    from matplotlib.patches import Ellipse, Polygon, Rectangle

    fig, ax = plt.subplots(figsize=(10, 6.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_axis_off()
    ax.set_aspect("equal")
    width, height = 4.0, 1.1  # front face of a 2x4 brick
    depth_x, depth_y = 1.0, 0.55  # how far the top/side faces recede
    pulled_out = 2.0  # how far the train brick sticks out to the right
    radius, stud_height = 0.3, 0.22
    label_x = width + depth_x + pulled_out + 0.35  # one column of notes, clear of the pulled-out brick

    # draw bottom to top, so each brick hides the studs of the one below it
    for level, (name, detail, color) in enumerate(BRICKS):
        x, y = (pulled_out if name == "train" else 0), level * height
        front = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
        side = [(x + width, y), (x + width + depth_x, y + depth_y), (x + width + depth_x, y + height + depth_y), (x + width, y + height)]
        top = [(x, y + height), (x + width, y + height), (x + width + depth_x, y + height + depth_y), (x + depth_x, y + height + depth_y)]
        edge = shade(color, 0.6)
        for face, tone in [(side, 0.75), (top, 1.25), (front, 1.0)]:
            ax.add_patch(Polygon(face, closed=True, fc=shade(color, tone), ec=edge, lw=0.8))

        for row in (0.75, 0.3):  # back row first so the front row overlaps it
            for col in range(4):
                cx = x + 0.5 + col * 1.0 + row * depth_x
                cy = y + height + row * depth_y
                ax.add_patch(Ellipse((cx, cy), 2 * radius, radius, fc=shade(color, 0.75), ec=edge, lw=0.8))
                ax.add_patch(Rectangle((cx - radius, cy), 2 * radius, stud_height, fc=shade(color, 0.75), ec="none"))
                ax.add_patch(Ellipse((cx, cy + stud_height), 2 * radius, radius, fc=shade(color, 1.25), ec=edge, lw=0.8))

        ink = TEXT if color in ("#fe8a18", "#f2cd37") else "white"
        ax.text(x + width / 2, y + height / 2, f"{level + 1}. {name}", ha="center", va="center", color=ink, fontsize=13, fontweight="bold")
        ax.text(label_x, y + height / 2, detail, ha="left", va="center", color=TEXT_MUTED, fontsize=11)

    top_y = len(BRICKS) * height
    ax.set_xlim(-0.3, label_x + 5.0)
    ax.set_ylim(-0.2, top_y + 0.95)
    fig.tight_layout()
    fig.savefig(IMAGES / "lego_bricks.png", dpi=200)


if __name__ == "__main__":
    IMAGES.mkdir(exist_ok=True)
    lego_bricks()
    scoreboard_chart()
    bigram_heatmap()
    poll_chart()
    prediction_animation()
    char_prediction_animation()
    print(f"saved charts to {IMAGES}")
