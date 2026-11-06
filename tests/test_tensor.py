"""Basic behaviour of Tensor that does not need PyTorch."""

import numpy as np

from minigrad import Tensor, no_grad


def test_hand_computed_example():
    # The example from the lesson: f = x*w + b, L = (f - y)^2
    x, y = Tensor(2.0), Tensor(10.0)
    w = Tensor(3.0, requires_grad=True)
    b = Tensor(1.0, requires_grad=True)
    loss = (x * w + b - y) ** 2
    loss.backward()
    assert loss.item() == 9.0
    assert w.grad == -12.0
    assert b.grad == -6.0


def test_integer_input_becomes_float32():
    assert Tensor([1, 2, 3]).dtype == np.float32


def test_dtype_is_preserved():
    assert Tensor(np.ones(3, dtype=np.float64)).dtype == np.float64
    x = Tensor(np.ones(3, dtype=np.float32))
    # Python numbers must not silently upcast float32 to float64.
    assert (x * 2.0 + 1).dtype == np.float32


def test_no_grad_builds_no_graph():
    w = Tensor(np.ones((2, 2)), requires_grad=True)
    with no_grad():
        out = (w @ w).sum()
    assert out.requires_grad is False
    assert out._prev == ()


def test_grad_accumulates_over_two_backward_calls():
    x = Tensor(3.0, requires_grad=True)
    (x * 2).backward()
    (x * 2).backward()
    assert x.grad == 4.0
    x.zero_grad()
    assert x.grad is None


def test_numpy_array_on_the_left():
    x = Tensor(np.ones(3), requires_grad=True)
    out = (np.array([1.0, 2.0, 3.0]) * x).sum()
    out.backward()
    np.testing.assert_allclose(x.grad, [1.0, 2.0, 3.0])
