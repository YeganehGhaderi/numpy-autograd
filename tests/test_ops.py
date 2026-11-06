"""Compare every operation's gradient with PyTorch on the same inputs."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from minigrad import Tensor, cross_entropy, cross_entropy_composed  # noqa: E402

rng = np.random.default_rng(0)


def check(fn_mine, fn_torch, *arrays):
    """Run the same function in minigrad and PyTorch; compare outputs and grads."""
    arrays = [np.array(a, dtype=np.float64) for a in arrays]
    mine = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    ref = [torch.tensor(a, requires_grad=True) for a in arrays]

    out_mine = fn_mine(*mine)
    out_ref = fn_torch(*ref)
    np.testing.assert_allclose(out_mine.data, out_ref.detach().numpy(), rtol=1e-6, atol=1e-8)

    out_mine.backward()
    out_ref.backward()
    for m, r in zip(mine, ref):
        np.testing.assert_allclose(m.grad, r.grad.numpy(), rtol=1e-6, atol=1e-8)


# ---- arithmetic with broadcasting ------------------------------------------
BROADCAST_SHAPES = [
    ((3, 4), (3, 4)),  # same shape
    ((3, 4), (4,)),    # bias-like: row vector added to every row
    ((4, 1), (1, 5)),  # both sides stretched
    ((2, 3), ()),      # tensor and a scalar tensor
]


@pytest.mark.parametrize("shape_a, shape_b", BROADCAST_SHAPES)
@pytest.mark.parametrize(
    "op",
    [
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a / (b * b + 1),  # keep the denominator away from 0
    ],
    ids=["add", "sub", "mul", "div"],
)
def test_binary_ops_broadcast(op, shape_a, shape_b):
    a, b = rng.normal(size=shape_a), rng.normal(size=shape_b)
    f = lambda x, y: op(x, y).sum()  # noqa: E731
    check(f, f, a, b)


def test_scalar_on_both_sides():
    f = lambda x: (2.0 * x - 1.0 + 3.0 / (x * x + 1) - x / 4.0).sum()  # noqa: E731
    check(f, f, rng.normal(size=(3, 3)))


# ---- matmul, reductions, shapes --------------------------------------------
def test_matmul():
    f = lambda a, b: (a @ b).sum()  # noqa: E731
    check(f, f, rng.normal(size=(4, 3)), rng.normal(size=(3, 5)))


def test_linear_relu_broadcast():
    f = lambda a, b, c: (a @ b + c).relu().sum()  # noqa: E731
    check(f, f, rng.normal(size=(4, 3)), rng.normal(size=(3, 5)), rng.normal(size=(5,)))


@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_sum_and_mean(axis, keepdims):
    w = rng.normal(size=(3, 4) if keepdims or axis is None else ((4,) if axis == 0 else (3,)))

    def mine(a):
        return (a.sum(axis=axis, keepdims=keepdims) * 2 + a.mean(axis=axis, keepdims=keepdims)) * Tensor(w)

    def ref(a):
        dims = {} if axis is None else {"dim": axis, "keepdim": keepdims}
        return (a.sum(**dims) * 2 + a.mean(**dims)) * torch.tensor(w)

    check(lambda a: mine(a).sum(), lambda a: ref(a).sum(), rng.normal(size=(3, 4)))


def test_reshape_and_transpose():
    w = rng.normal(size=(6, 2))
    check(
        lambda a: (a.reshape(2, 6).T * Tensor(w)).sum(),
        lambda a: (a.reshape(2, 6).T * torch.tensor(w)).sum(),
        rng.normal(size=(3, 4)),
    )


def test_indexing_with_repeated_rows():
    idx = np.array([0, 2, 2, 1])
    check(
        lambda a: (a[idx] ** 2).sum(),
        lambda a: (a[torch.tensor(idx)] ** 2).sum(),
        rng.normal(size=(3, 4)),
    )


# ---- element-wise functions ------------------------------------------------
@pytest.mark.parametrize(
    "name, mine, ref",
    [
        ("exp", lambda a: a.exp(), lambda a: a.exp()),
        ("log", lambda a: (a * a + 1).log(), lambda a: (a * a + 1).log()),
        ("pow", lambda a: a**3, lambda a: a**3),
        ("relu", lambda a: a.relu(), lambda a: a.relu()),
        ("sigmoid", lambda a: a.sigmoid(), lambda a: a.sigmoid()),
        ("tanh", lambda a: a.tanh(), lambda a: a.tanh()),
    ],
)
def test_elementwise(name, mine, ref):
    x = rng.normal(size=(4, 5))
    w = rng.normal(size=(4, 5))  # random weights so each element gets a different upstream grad
    check(lambda a: (mine(a) * Tensor(w)).sum(), lambda a: (ref(a) * torch.tensor(w)).sum(), x)


def test_node_used_twice():
    # x appears three times; its gradient must be the sum of all contributions: 2x + 1
    f = lambda x: (x * x + x).sum()  # noqa: E731
    check(f, f, rng.normal(size=(3,)))


# ---- softmax family --------------------------------------------------------
def test_softmax_and_log_softmax():
    w = rng.normal(size=(4, 6))
    check(
        lambda a: (a.softmax(axis=1) * Tensor(w) + a.log_softmax(axis=1) * Tensor(w)).sum(),
        lambda a: (a.softmax(dim=1) * torch.tensor(w) + a.log_softmax(dim=1) * torch.tensor(w)).sum(),
        rng.normal(size=(4, 6)),
    )


def test_log_softmax_is_stable_for_huge_logits():
    x = np.array([[1000.0, 0.0, -1000.0]])
    check(lambda a: a.log_softmax(axis=1)[0, 1], lambda a: a.log_softmax(dim=1)[0, 1], x)


@pytest.mark.parametrize("loss_fn", [cross_entropy, cross_entropy_composed], ids=["fused", "composed"])
def test_cross_entropy(loss_fn):
    targets = np.array([2, 0, 4, 4, 1])
    check(
        lambda z: loss_fn(z, targets),
        lambda z: F.cross_entropy(z, torch.tensor(targets)),
        rng.normal(size=(5, 6)),
    )
