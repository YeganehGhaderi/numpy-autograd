"""Tests for the optimizers."""

import numpy as np
import pytest

from minigrad import Tensor, nn, optim

CONFIGS = {
    "sgd": (lambda p: optim.SGD(p, lr=0.1), "SGD", {"lr": 0.1}),
    "momentum": (lambda p: optim.SGD(p, lr=0.1, momentum=0.9), "SGD", {"lr": 0.1, "momentum": 0.9}),
    "adam": (lambda p: optim.Adam(p, lr=0.01), "Adam", {"lr": 0.01}),
}


@pytest.mark.parametrize("name", list(CONFIGS))
def test_optimizer_matches_torch(name):
    """Same start, same 5 gradients -> same parameters as torch.optim."""
    torch = pytest.importorskip("torch")
    make_mine, torch_name, torch_kwargs = CONFIGS[name]
    rng = np.random.default_rng(0)
    start = [rng.normal(size=(3, 4)), rng.normal(size=(4,))]
    grads = [[rng.normal(size=a.shape) for a in start] for _ in range(5)]

    mine = [Tensor(a.copy(), requires_grad=True) for a in start]
    ref = [torch.tensor(a.copy(), requires_grad=True) for a in start]
    opt_mine = make_mine(mine)
    opt_ref = getattr(torch.optim, torch_name)(ref, **torch_kwargs)

    for step_grads in grads:
        for p, g in zip(mine, step_grads):
            p.grad = g.copy()
        for p, g in zip(ref, step_grads):
            p.grad = torch.tensor(g.copy())
        opt_mine.step()
        opt_ref.step()

    for m, r in zip(mine, ref):
        np.testing.assert_allclose(m.data, r.detach().numpy(), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_optimizer_minimises_a_quadratic(name):
    """Without PyTorch: every optimizer should walk to the minimum of (x - 3)^2."""
    x = Tensor(np.zeros(2), requires_grad=True)
    opt = CONFIGS[name][0]([x])
    if name == "adam":
        opt.lr = 0.1
    for _ in range(500):
        opt.zero_grad()
        ((x - 3.0) ** 2).sum().backward()
        opt.step()
    np.testing.assert_allclose(x.data, [3.0, 3.0], atol=1e-2)


def test_zero_grad_clears_gradients():
    x = Tensor(np.ones(3), requires_grad=True)
    opt = optim.SGD([x], lr=0.1)
    (x * 2).sum().backward()
    opt.zero_grad()
    assert x.grad is None


def test_mlp_learns_xor():
    """End-to-end: a small MLP must fit XOR, which no linear model can."""
    rng = np.random.default_rng(0)
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0])
    model = nn.Sequential(nn.Linear(2, 16, rng=rng), nn.ReLU(), nn.Linear(16, 2, rng=rng))
    opt = optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(300):
        opt.zero_grad()
        loss = loss_fn(model(Tensor(x)), y)
        loss.backward()
        opt.step()
    predictions = model(Tensor(x)).data.argmax(axis=1)
    np.testing.assert_array_equal(predictions, y)
    assert loss.item() < 0.05
