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
