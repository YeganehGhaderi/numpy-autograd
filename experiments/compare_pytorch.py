"""Compare minigrad with PyTorch on the same MLP.

Experiment 1 — lockstep curves:
    Both frameworks start from identical weights, use dropout 0, Adam, and see
    exactly the same batches in the same order. If the engine is correct, the
    two loss curves lie on top of each other (differences only from float32
    rounding).

Experiment 2 — accuracy and speed:
    PyTorch trains the main model (dropout 0.2, Adam, 15 epochs) and is compared
    with minigrad's run from train_mlp.py. Run that script first.

Run from the project root:
    python experiments/train_mlp.py        # first
    python experiments/compare_pytorch.py
Outputs: results/lockstep.png, results/comparison.json
"""

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from minigrad import Tensor, nn, optim  # noqa: E402
from utils import (COLORS, RESULTS_DIR, SURFACE, build_mlp, iterate_minibatches,  # noqa: E402
                   load_fashion_mnist, style_axes)

SEED = 0


def torch_copy_of(model, dropout):
    """A PyTorch MLP holding exactly the same weights as the minigrad `model`."""
    linears = [layer for layer in model.layers if isinstance(layer, nn.Linear)]
    layers = [torch.nn.Linear(784, 256), torch.nn.ReLU()]
    if dropout > 0:
        layers.append(torch.nn.Dropout(dropout))
    layers += [torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10)]
    net = torch.nn.Sequential(*layers)
    torch_linears = [layer for layer in net if isinstance(layer, torch.nn.Linear)]
    with torch.no_grad():
        for mine, ref in zip(linears, torch_linears):
            ref.weight.copy_(torch.from_numpy(mine.weight.data.T.copy()))  # torch stores (out, in)
            ref.bias.copy_(torch.from_numpy(mine.bias.data.copy()))
    return net


def lockstep(data, steps):
    x_train, y_train, _, _ = data
    mine = build_mlp(dropout=0.0, rng=np.random.default_rng(SEED))
    ref = torch_copy_of(mine, dropout=0.0)
    opt_mine = optim.Adam(mine.parameters(), lr=1e-3)
    opt_ref = torch.optim.Adam(ref.parameters(), lr=1e-3)
    loss_mine_fn = nn.CrossEntropyLoss()
    loss_ref_fn = torch.nn.CrossEntropyLoss()

    rng = np.random.default_rng(SEED)
    losses_mine, losses_ref = [], []
    while len(losses_mine) < steps:
        for xb, yb in iterate_minibatches(x_train, y_train, 128, rng):
            if len(losses_mine) == steps:
                break
            opt_mine.zero_grad()
            loss = loss_mine_fn(mine(Tensor(xb)), yb)
            loss.backward()
            opt_mine.step()
            losses_mine.append(loss.item())

            opt_ref.zero_grad()
            loss = loss_ref_fn(ref(torch.from_numpy(xb)), torch.from_numpy(yb))
            loss.backward()
            opt_ref.step()
            losses_ref.append(loss.item())
    return np.array(losses_mine), np.array(losses_ref)


def plot_lockstep(losses_mine, losses_ref, path):
    steps = np.arange(1, len(losses_mine) + 1)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 5.6), facecolor=SURFACE,
                                      gridspec_kw={"height_ratios": [3, 1.4]}, sharex=True)
    top.plot(steps, losses_ref, color=COLORS[1], linewidth=3.5, label="PyTorch")
    top.plot(steps, losses_mine, color=COLORS[0], linewidth=1.5, label="minigrad")
    style_axes(top, "Training loss: minigrad vs PyTorch from identical weights", "", "Training loss")
    top.legend(frameon=False)

    diff = np.abs(losses_mine - losses_ref)
    bottom.plot(steps, np.maximum(diff, 1e-12), color="#52514e", linewidth=1.2)
    bottom.set_yscale("log")
    style_axes(bottom, "Absolute difference (float32 rounding)", "Training step", "|Δ loss|")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_torch(data, epochs, dropout=0.2):
    x_train, y_train, x_test, y_test = data
    torch.manual_seed(SEED)
    net = torch_copy_of(build_mlp(dropout=dropout, rng=np.random.default_rng(SEED)), dropout)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(SEED)
    seconds = []
    for epoch in range(1, epochs + 1):
        net.train()
        start = time.perf_counter()
        for xb, yb in iterate_minibatches(x_train, y_train, 128, rng):
            opt.zero_grad()
            loss = loss_fn(net(torch.from_numpy(xb)), torch.from_numpy(yb))
            loss.backward()
            opt.step()
        seconds.append(time.perf_counter() - start)
        net.eval()
        with torch.no_grad():
            test_acc = (net(torch.from_numpy(x_test)).argmax(1).numpy() == y_test).mean()
        print(f"  epoch {epoch:2d}  test {test_acc:.4f}  ({seconds[-1]:.1f} s)")
    return float(test_acc), float(np.mean(seconds))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--quick", action="store_true", help="tiny smoke run")
    args = parser.parse_args()

    data = load_fashion_mnist()
    steps, epochs = args.steps, args.epochs
    if args.quick:
        data = tuple(a[:3000] for a in data)
        steps, epochs = 20, 1

    print("Experiment 1: lockstep training")
    losses_mine, losses_ref = lockstep(data, steps)
    max_diff = float(np.abs(losses_mine - losses_ref).max())
    print(f"  max |loss difference| over {steps} steps: {max_diff:.2e}")

    print("\nExperiment 2: PyTorch training (dropout 0.2, Adam)")
    torch_acc, torch_sec = train_torch(data, epochs)

    if args.quick:
        print("\nQuick run finished; nothing saved.")
        return

    plot_lockstep(losses_mine, losses_ref, RESULTS_DIR / "lockstep.png")
    mine = json.loads((RESULTS_DIR / "mlp_results.json").read_text())["Adam"]
    mine_acc, mine_sec = mine["test_acc"][-1], float(np.mean(mine["epoch_seconds"]))
    comparison = {
        "lockstep_steps": steps,
        "lockstep_max_abs_loss_diff": max_diff,
        "minigrad": {"test_acc": mine_acc, "seconds_per_epoch": mine_sec},
        "pytorch": {"test_acc": torch_acc, "seconds_per_epoch": torch_sec},
        "speed_ratio": mine_sec / torch_sec,
    }
    (RESULTS_DIR / "comparison.json").write_text(json.dumps(comparison, indent=2))

    print("\n| Framework | Test accuracy | s / epoch |")
    print("|---|---|---|")
    print(f"| minigrad | {mine_acc:.4f} | {mine_sec:.1f} |")
    print(f"| PyTorch  | {torch_acc:.4f} | {torch_sec:.1f} |")
    print(f"\nPyTorch is {mine_sec / torch_sec:.1f}x faster per epoch.")
    print(f"Saved {RESULTS_DIR / 'lockstep.png'} and comparison.json")


if __name__ == "__main__":
    main()
