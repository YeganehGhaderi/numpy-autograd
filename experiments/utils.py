"""Helpers shared by the experiments: data loading, mini-batches, evaluation."""

from pathlib import Path

import numpy as np

from minigrad import Tensor, no_grad

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

FASHION_CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def load_fashion_mnist():
    """Return (x_train, y_train, x_test, y_test) as NumPy arrays.

    Images are flattened to 784 values in [0, 1] (float32); labels are int64.
    torchvision is used only to download the files once; afterwards a cached
    .npz file is read and torchvision is not needed.
    """
    cache = DATA_DIR / "fashion_mnist.npz"
    if not cache.exists():
        from torchvision.datasets import FashionMNIST

        DATA_DIR.mkdir(exist_ok=True)
        train = FashionMNIST(DATA_DIR, train=True, download=True)
        test = FashionMNIST(DATA_DIR, train=False, download=True)
        np.savez_compressed(
            cache,
            x_train=train.data.numpy(), y_train=train.targets.numpy(),
            x_test=test.data.numpy(), y_test=test.targets.numpy(),
        )
    d = np.load(cache)

    def prep(x):
        return (x.reshape(len(x), -1) / 255.0).astype(np.float32)

    return prep(d["x_train"]), d["y_train"].astype(np.int64), prep(d["x_test"]), d["y_test"].astype(np.int64)


def iterate_minibatches(x, y, batch_size, rng, shuffle=True):
    """Yield (x_batch, y_batch) pairs that cover the whole dataset once (one epoch)."""
    order = rng.permutation(len(x)) if shuffle else np.arange(len(x))
    for start in range(0, len(x), batch_size):
        idx = order[start:start + batch_size]
        yield x[idx], y[idx]


def accuracy(model, x, y, batch_size=1000):
    """Fraction of correctly classified examples (multi-class logits)."""
    model.eval()
    correct = 0
    with no_grad():
        for start in range(0, len(x), batch_size):
            logits = model(Tensor(x[start:start + batch_size]))
            correct += (logits.data.argmax(axis=1) == y[start:start + batch_size]).sum()
    model.train()
    return correct / len(x)


# ---- Shared by the MLP experiments (day 3) ----------------------------------

def build_mlp(dropout=0.2, rng=None):
    """784 -> 256 -> ReLU -> Dropout -> 128 -> ReLU -> 10, in minigrad."""
    from minigrad import nn

    rng = rng if rng is not None else np.random.default_rng(0)
    layers = [nn.Linear(784, 256, rng=rng), nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout, rng=rng))
    layers += [nn.Linear(256, 128, rng=rng), nn.ReLU(), nn.Linear(128, 10, rng=rng)]
    return nn.Sequential(*layers)


def train(model, opt, data, epochs, batch_size=128, seed=0, log=True):
    """Train a minigrad model; return a history dict with per-epoch numbers."""
    from minigrad import nn
    import time

    x_train, y_train, x_test, y_test = data
    rng = np.random.default_rng(seed)
    loss_fn = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "test_acc": [], "epoch_seconds": []}
    for epoch in range(1, epochs + 1):
        model.train()
        start = time.perf_counter()
        losses = []
        for xb, yb in iterate_minibatches(x_train, y_train, batch_size, rng):
            opt.zero_grad()
            loss = loss_fn(model(Tensor(xb)), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        history["epoch_seconds"].append(time.perf_counter() - start)
        history["train_loss"].append(float(np.mean(losses)))
        history["train_acc"].append(float(accuracy(model, x_train, y_train)))
        history["test_acc"].append(float(accuracy(model, x_test, y_test)))
        if log:
            print(f"  epoch {epoch:2d}  loss {history['train_loss'][-1]:.4f}  "
                  f"train {history['train_acc'][-1]:.4f}  test {history['test_acc'][-1]:.4f}  "
                  f"({history['epoch_seconds'][-1]:.1f} s)")
    return history


# ---- Plot styling -------------------------------------------------------------
# Colour-blind-checked categorical colours (blue, orange, aqua) on a light surface.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"


def style_axes(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(True, color="#e6e5e1", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")
