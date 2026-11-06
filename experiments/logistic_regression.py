"""Logistic regression built with minigrad.

Part 1 — binary:      breast-cancer dataset (30 features, 2 classes),
                      compared with scikit-learn's LogisticRegression.
Part 2 — multi-class: softmax regression on Fashion-MNIST (784 pixels, 10 classes),
                      the baseline that the MLP must beat.

Run from the project root:
    python experiments/logistic_regression.py
Results are printed and written to results/logistic_regression.md.
"""

import time

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from minigrad import Tensor, nn, optim
from utils import RESULTS_DIR, accuracy, iterate_minibatches, load_fashion_mnist

SEED = 0


def binary_logistic_regression():
    x, y = load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=SEED)
    # Put every feature on the same scale (mean 0, std 1), using train statistics only.
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    # Logistic regression = one Linear layer with a single output + sigmoid.
    model = nn.Linear(30, 1, rng=np.random.default_rng(SEED))
    loss_fn = nn.BCEWithLogitsLoss()
    opt = optim.SGD(model.parameters(), lr=0.1)

    for step in range(1000):  # full-batch gradient descent: the dataset is tiny
        opt.zero_grad()
        loss = loss_fn(model(Tensor(x_train)), y_train)
        loss.backward()
        opt.step()
        if step % 200 == 0:
            print(f"  step {step:4d}  loss {loss.item():.4f}")

    # sigmoid(z) > 0.5  <=>  z > 0
    ours = ((model(Tensor(x_test)).data[:, 0] > 0) == y_test).mean()
    sk = LogisticRegression(max_iter=1000).fit(x_train, y_train).score(x_test, y_test)
    print(f"  test accuracy: minigrad {ours:.4f} | scikit-learn {sk:.4f}")
    return ours, sk


def train_softmax_regression(x_train, y_train, x_test, y_test, epochs=5, batch_size=128, lr=1e-3):
    rng = np.random.default_rng(SEED)
    n_features, n_classes = x_train.shape[1], int(y_train.max()) + 1
    model = nn.Linear(n_features, n_classes, rng=rng)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        losses = []
        for xb, yb in iterate_minibatches(x_train, y_train, batch_size, rng):
            opt.zero_grad()
            loss = loss_fn(model(Tensor(xb)), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        test_acc = accuracy(model, x_test, y_test)
        print(f"  epoch {epoch}  train loss {np.mean(losses):.4f}  test acc {test_acc:.4f}")
    seconds = time.perf_counter() - start
    return accuracy(model, x_train, y_train), test_acc, seconds


def main():
    print("Part 1: binary logistic regression (breast cancer)")
    ours, sk = binary_logistic_regression()

    print("\nPart 2: softmax regression (Fashion-MNIST)")
    train_acc, test_acc, seconds = train_softmax_regression(*load_fashion_mnist())

    RESULTS_DIR.mkdir(exist_ok=True)
    report = (
        "# Logistic regression results\n\n"
        "Produced by `python experiments/logistic_regression.py` (seed 0).\n\n"
        "## Binary: breast cancer (80/20 split)\n\n"
        "| Model | Test accuracy |\n|---|---|\n"
        f"| minigrad (SGD, 1000 full-batch steps) | {ours:.4f} |\n"
        f"| scikit-learn LogisticRegression | {sk:.4f} |\n\n"
        "## Multi-class: softmax regression on Fashion-MNIST\n\n"
        "| Train accuracy | Test accuracy | Training time (5 epochs) |\n|---|---|---|\n"
        f"| {train_acc:.4f} | {test_acc:.4f} | {seconds:.1f} s |\n"
    )
    (RESULTS_DIR / "logistic_regression.md").write_text(report, encoding="utf-8")
    print(f"\nSaved {RESULTS_DIR / 'logistic_regression.md'}")


if __name__ == "__main__":
    main()
