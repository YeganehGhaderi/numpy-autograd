"""Check the engine against numerical derivatives (finite differences).

This uses only NumPy: it trusts nothing but the definition of the derivative,
    df/dx ≈ (f(x + h) - f(x - h)) / (2h)
"""

import numpy as np
import pytest

from minigrad import Tensor, cross_entropy, cross_entropy_composed

H = 1e-6
rng = np.random.default_rng(42)


def numerical_grad(f, arrays, i):
    """Numerical gradient of the scalar f(*arrays) with respect to arrays[i]."""
    x = arrays[i]
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        original = x[idx]
        x[idx] = original + H
        f_plus = f(*[Tensor(a) for a in arrays]).item()
        x[idx] = original - H
        f_minus = f(*[Tensor(a) for a in arrays]).item()
        x[idx] = original
        grad[idx] = (f_plus - f_minus) / (2 * H)
    return grad


def gradcheck(f, *arrays):
    arrays = [np.array(a, dtype=np.float64) for a in arrays]
    tensors = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    f(*tensors).backward()
    for i, t in enumerate(tensors):
        expected = numerical_grad(f, arrays, i)
        np.testing.assert_allclose(t.grad, expected, rtol=1e-5, atol=1e-7)


CASES = {
    "add_mul_broadcast": (lambda a, b: (a * b + a).sum(), [(3, 4), (4,)]),
    "division": (lambda a, b: (a / b).sum(), [(2, 3), (2, 3)]),
    "two_layer_mlp": (
        lambda x, w1, w2: ((x @ w1).tanh() @ w2).sigmoid().mean(),
        [(4, 3), (3, 5), (5, 2)],
    ),
    "exp_log_pow": (lambda a: ((a * a + 1).log() + (a * 0.5).exp() + a**3).sum(), [(3, 3)]),
    "sum_axis_keepdims": (lambda a: (a.sum(axis=0, keepdims=True) * a).sum(), [(3, 4)]),
    "mean_axis": (lambda a: (a.mean(axis=1) ** 2).sum(), [(3, 4)]),
    "reshape_transpose": (lambda a: (a.reshape(2, 6).T ** 2).sum(), [(3, 4)]),
    "softmax": (lambda a, w: (a.softmax(axis=1) * w).sum(), [(3, 5), (3, 5)]),
    "log_softmax": (lambda a, w: (a.log_softmax(axis=1) * w).sum(), [(3, 5), (3, 5)]),
}


@pytest.mark.parametrize("name", list(CASES))
def test_gradcheck(name):
    f, shapes = CASES[name]
    arrays = [rng.normal(size=s) for s in shapes]
    gradcheck(f, *arrays)


@pytest.mark.parametrize("loss_fn", [cross_entropy, cross_entropy_composed])
def test_cross_entropy_gradcheck(loss_fn):
    targets = np.array([0, 3, 1, 3])
    gradcheck(lambda z: loss_fn(z, targets), rng.normal(size=(4, 5)))


def test_relu_gradcheck_away_from_kink():
    # ReLU has no derivative at exactly 0, so keep inputs away from it.
    x = rng.normal(size=(4, 4))
    x[np.abs(x) < 0.1] = 0.5
    gradcheck(lambda a: (a.relu() * a).sum(), x)
