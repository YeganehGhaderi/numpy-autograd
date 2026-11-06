"""Compare minigrad with PyTorch on the same MLP.

Experiment 1 — lockstep curves:
    Both frameworks start from identical weights, use dropout 0, Adam, and see
    exactly the same batches in the same order. This is run twice:
      * float64 — the correctness test: the curves must agree to ~1e-10.
      * float32 — what real training uses: the curves start identical, then
        drift slowly, because Adam amplifies the tiny rounding differences
        between two libraries that sum numbers in a different order.

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


def torch_copy_of(model, dropout, dtype=np.float32):
    """A PyTorch MLP holding exactly the same weights as the minigrad `model`."""
    linears = [layer for layer in model.layers if isinstance(layer, nn.Linear)]
    layers = [torch.nn.Linear(784, 256), torch.nn.ReLU()]
    if dropout > 0:
        layers.append(torch.nn.Dropout(dropout))
    layers += [torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10)]
    net = torch.nn.Sequential(*layers)
    torch_linears = [layer for layer in net if isinstance(layer, torch.nn.Linear)]
    if dtype == np.float64:
        net = net.double()  # float64 weights; done before copying so nothing is rounded
    with torch.no_grad():
        for mine, ref in zip(linears, torch_linears):
            ref.weight.copy_(torch.from_numpy(mine.weight.data.T.copy()))  # torch stores (out, in)
            ref.bias.copy_(torch.from_numpy(mine.bias.data.copy()))
    return net


def lockstep(data, steps, dtype):
    x_train, y_train, _, _ = data
    x_train = x_train.astype(dtype)
    mine = build_mlp(dropout=0.0, rng=np.random.default_rng(SEED), dtype=dtype)
    ref = torch_copy_of(mine, dropout=0.0, dtype=dtype)
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


def plot_lockstep(runs, path):
    """runs: {"float64": (mine, ref), "float32": (mine, ref)}"""
    mine64, ref64 = runs["float64"]
    steps = np.arange(1, len(mine64) + 1)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 5.8), facecolor=SURFACE,
                                      gridspec_kw={"height_ratios": [3, 1.6]}, sharex=True)
    top.plot(steps, ref64, color=COLORS[1], linewidth=3.5, label="PyTorch")
    top.plot(steps, mine64, color=COLORS[0], linewidth=1.5, label="minigrad")
    style_axes(top, "Training loss: minigrad vs PyTorch from identical weights",
               "", "Training loss")
    top.legend(frameon=False)

    for color, name in [(COLORS[0], "float64"), (COLORS[2], "float32")]:
        mine, ref = runs[name]
        diff = np.maximum(np.abs(mine - ref), 1e-16)
        bottom.plot(steps, diff, color=color, linewidth=1.2, label=name)
    bottom.set_yscale("log")
    style_axes(bottom, "Absolute difference between the two curves", "Training step", "|Δ loss|")
    bottom.legend(frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 0.98))
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
    runs, stats = {}, {}
    for name, dtype in [("float64", np.float64), ("float32", np.float32)]:
        mine, ref = lockstep(data, steps, dtype)
        diff = np.abs(mine - ref)
        runs[name] = (mine, ref)
        stats[name] = {"max_diff_first_50_steps": float(diff[:50].max()),
                       "max_diff_all_steps": float(diff.max())}
        print(f"  {name}: max |loss difference| first 50 steps {diff[:50].max():.1e}, "
              f"all {steps} steps {diff.max():.1e}")

    print("\nExperiment 2: PyTorch training (dropout 0.2, Adam)")
    torch_acc, torch_sec = train_torch(data, epochs)

    if args.quick:
        print("\nQuick run finished; nothing saved.")
        return

    plot_lockstep(runs, RESULTS_DIR / "lockstep.png")
    mine = json.loads((RESULTS_DIR / "mlp_results.json").read_text())["Adam"]
    mine_acc, mine_sec = mine["test_acc"][-1], float(np.mean(mine["epoch_seconds"]))
    comparison = {
        "lockstep_steps": steps,
        "lockstep_loss_difference": stats,
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
