"""Tests for layers, dropout and losses."""

import numpy as np
import pytest

from minigrad import Tensor, binary_cross_entropy_with_logits, nn

rng = np.random.default_rng(1)


# ---- Module bookkeeping ------------------------------------------------------
def test_parameters_are_found_recursively():
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(0.5), nn.Linear(8, 3))
    params = model.parameters()
    assert len(params) == 4  # two weights + two biases
    assert sum(p.size for p in params) == 4 * 8 + 8 + 8 * 3 + 3


def test_train_and_eval_reach_every_layer():
    model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
    model.eval()
    assert not model[1].training
    model.train()
    assert model[1].training


def test_he_initialisation_scale():
    layer = nn.Linear(1000, 500, rng=np.random.default_rng(0))
    assert layer.weight.dtype == np.float32
    assert abs(layer.weight.data.std() - np.sqrt(2 / 1000)) < 1e-3
    assert np.all(layer.bias.data == 0)


# ---- Dropout -----------------------------------------------------------------
def test_dropout_is_identity_in_eval_mode():
    drop = nn.Dropout(0.5).eval()
    x = Tensor(rng.normal(size=(10, 10)))
    assert drop(x) is x


def test_dropout_keeps_the_mean_in_train_mode():
    drop = nn.Dropout(0.3, rng=np.random.default_rng(0))
    x = Tensor(np.ones((1000, 1000)))
    out = drop(x).data
    assert abs(out.mean() - 1.0) < 0.01               # expectation preserved
    assert abs((out == 0).mean() - 0.3) < 0.01        # about p of the values dropped
    assert np.allclose(out[out != 0], 1 / 0.7)        # survivors scaled by 1/(1-p)


def test_dropout_gradient_is_the_mask():
    drop = nn.Dropout(0.5, rng=np.random.default_rng(0))
    x = Tensor(rng.normal(size=(5, 6)), requires_grad=True)
    drop(x).sum().backward()
    np.testing.assert_array_equal(x.grad, drop.mask)


# ---- Comparisons with PyTorch ---------------------------------------------------
def test_linear_matches_torch():
    torch = pytest.importorskip("torch")
    mine = nn.Linear(4, 3, dtype=np.float64)
    ref = torch.nn.Linear(4, 3).double()
    with torch.no_grad():  # copy our weights into torch; torch stores W transposed
        ref.weight.copy_(torch.tensor(mine.weight.data.T))
        ref.bias.copy_(torch.tensor(mine.bias.data))

    x_np = rng.normal(size=(5, 4))
    x_mine = Tensor(x_np, requires_grad=True)
    x_ref = torch.tensor(x_np, requires_grad=True)

    out_mine, out_ref = mine(x_mine), ref(x_ref)
    np.testing.assert_allclose(out_mine.data, out_ref.detach().numpy(), rtol=1e-10)

    (out_mine**2).sum().backward()
    (out_ref**2).sum().backward()
    np.testing.assert_allclose(mine.weight.grad, ref.weight.grad.numpy().T, rtol=1e-10)
    np.testing.assert_allclose(mine.bias.grad, ref.bias.grad.numpy(), rtol=1e-10)
    np.testing.assert_allclose(x_mine.grad, x_ref.grad.numpy(), rtol=1e-10)


def test_mlp_with_cross_entropy_matches_torch():
    torch = pytest.importorskip("torch")
    sizes = [(6, 16), (16, 8), (8, 4)]
    mine = nn.Sequential(nn.Linear(*sizes[0], dtype=np.float64), nn.ReLU(),
                         nn.Linear(*sizes[1], dtype=np.float64), nn.Tanh(),
                         nn.Linear(*sizes[2], dtype=np.float64))
    ref = torch.nn.Sequential(torch.nn.Linear(*sizes[0]), torch.nn.ReLU(),
                              torch.nn.Linear(*sizes[1]), torch.nn.Tanh(),
                              torch.nn.Linear(*sizes[2])).double()
    mine_linears = [mine[0], mine[2], mine[4]]
    ref_linears = [ref[0], ref[2], ref[4]]
    with torch.no_grad():
        for m, r in zip(mine_linears, ref_linears):
            r.weight.copy_(torch.tensor(m.weight.data.T))
            r.bias.copy_(torch.tensor(m.bias.data))

    x = rng.normal(size=(10, 6))
    y = rng.integers(0, 4, size=10)
    loss_mine = nn.CrossEntropyLoss()(mine(Tensor(x)), y)
    loss_ref = torch.nn.CrossEntropyLoss()(ref(torch.tensor(x)), torch.tensor(y))
    np.testing.assert_allclose(loss_mine.item(), loss_ref.item(), rtol=1e-10)

    loss_mine.backward()
    loss_ref.backward()
    for m, r in zip(mine_linears, ref_linears):
        np.testing.assert_allclose(m.weight.grad, r.weight.grad.numpy().T, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(m.bias.grad, r.bias.grad.numpy(), rtol=1e-8, atol=1e-12)


def test_bce_with_logits_matches_torch():
    torch = pytest.importorskip("torch")
    z = np.array([-800.0, -3.0, -0.5, 0.0, 0.7, 4.0, 900.0])  # includes extreme values
    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    z_mine = Tensor(z, requires_grad=True)
    z_ref = torch.tensor(z, requires_grad=True)
    loss_mine = binary_cross_entropy_with_logits(z_mine, y)
    loss_ref = torch.nn.functional.binary_cross_entropy_with_logits(z_ref, torch.tensor(y))
    np.testing.assert_allclose(loss_mine.item(), loss_ref.item(), rtol=1e-10)
    loss_mine.backward()
    loss_ref.backward()
    np.testing.assert_allclose(z_mine.grad, z_ref.grad.numpy(), rtol=1e-10, atol=1e-15)


def test_bce_with_logits_gradcheck():
    # Finite differences, no PyTorch needed.
    z = rng.normal(size=6) * 3
    y = (rng.random(6) > 0.5).astype(float)
    t = Tensor(z.copy(), requires_grad=True)
    binary_cross_entropy_with_logits(t, y).backward()
    h = 1e-6
    numeric = np.zeros_like(z)
    for i in range(z.size):
        zp, zm = z.copy(), z.copy()
        zp[i] += h
        zm[i] -= h
        numeric[i] = (binary_cross_entropy_with_logits(Tensor(zp), y).item()
                      - binary_cross_entropy_with_logits(Tensor(zm), y).item()) / (2 * h)
    np.testing.assert_allclose(t.grad, numeric, rtol=1e-5, atol=1e-8)
