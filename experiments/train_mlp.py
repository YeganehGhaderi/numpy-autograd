"""Train the MLP on Fashion-MNIST with minigrad: optimizer and dropout comparisons.

Four runs, all from the same initial weights and the same batch order:
  1. Adam            (lr 1e-3),              dropout 0.2   <- the main model
  2. Adam            (lr 1e-3),              no dropout
  3. SGD             (lr 1e-2),              dropout 0.2
  4. SGD + momentum  (lr 1e-2, momentum 0.9), dropout 0.2

Run from the project root (about 5-10 minutes on a laptop):
    python experiments/train_mlp.py
Quick check that everything runs (a few seconds, numbers are meaningless):
    python experiments/train_mlp.py --quick
Outputs: results/mlp_results.json, results/optimizers.png, results/dropout.png
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from minigrad import optim  # noqa: E402
from utils import (COLORS, RESULTS_DIR, SURFACE, build_mlp, load_fashion_mnist,  # noqa: E402
                   style_axes, train)

SEED = 0

RUNS = {
    "Adam":               dict(dropout=0.2, make_opt=lambda p: optim.Adam(p, lr=1e-3)),
    "Adam (no dropout)":  dict(dropout=0.0, make_opt=lambda p: optim.Adam(p, lr=1e-3)),
    "SGD":                dict(dropout=0.2, make_opt=lambda p: optim.SGD(p, lr=1e-2)),
    "SGD + momentum":     dict(dropout=0.2, make_opt=lambda p: optim.SGD(p, lr=1e-2, momentum=0.9)),
}


def plot_optimizers(results, path):
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    for color, name in zip(COLORS, ["SGD", "SGD + momentum", "Adam"]):
        losses = results[name]["train_loss"]
        epochs = np.arange(1, len(losses) + 1)
        ax.plot(epochs, losses, color=color, linewidth=2, label=name)
        ax.annotate(name, (epochs[-1], losses[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=color, fontsize=9)
    style_axes(ax, "Training loss by optimizer (same model, same initial weights)",
               "Epoch", "Training loss")
    ax.legend(frameon=False)
    ax.set_xlim(right=len(epochs) * 1.18)  # room for the line-end labels
    ax.set_xticks(epochs[::max(1, len(epochs) // 8)])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_dropout(results, path):
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    for color, name, label in [(COLORS[0], "Adam", "dropout 0.2"),
                               (COLORS[1], "Adam (no dropout)", "no dropout")]:
        r = results[name]
        epochs = np.arange(1, len(r["test_acc"]) + 1)
        ax.plot(epochs, np.array(r["train_acc"]) * 100, color=color, linewidth=2,
                linestyle="--", label=f"{label}, train")
        ax.plot(epochs, np.array(r["test_acc"]) * 100, color=color, linewidth=2,
                label=f"{label}, test")
    style_axes(ax, "Train vs test accuracy, with and without dropout", "Epoch", "Accuracy (%)")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--quick", action="store_true", help="tiny smoke run")
    args = parser.parse_args()

    data = load_fashion_mnist()
    epochs = args.epochs
    if args.quick:
        x_train, y_train, x_test, y_test = data
        data = (x_train[:3000], y_train[:3000], x_test[:1000], y_test[:1000])
        epochs = 2

    results = {}
    for name, cfg in RUNS.items():
        print(f"\n{name}")
        model = build_mlp(dropout=cfg["dropout"], rng=np.random.default_rng(SEED))
        opt = cfg["make_opt"](model.parameters())
        results[name] = train(model, opt, data, epochs, batch_size=128, seed=SEED)

    if args.quick:
        print("\nQuick run finished; nothing saved.")
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "mlp_results.json").write_text(json.dumps(results, indent=2))
    plot_optimizers(results, RESULTS_DIR / "optimizers.png")
    plot_dropout(results, RESULTS_DIR / "dropout.png")

    print("\n| Run | Final train acc | Final test acc | Best test acc | s / epoch |")
    print("|---|---|---|---|---|")
    for name, r in results.items():
        print(f"| {name} | {r['train_acc'][-1]:.4f} | {r['test_acc'][-1]:.4f} | "
              f"{max(r['test_acc']):.4f} | {np.mean(r['epoch_seconds']):.1f} |")
    print(f"\nSaved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
