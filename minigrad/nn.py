"""Neural-network building blocks on top of Tensor.

Read in this order:
  1. Module      — the base class: finds parameters, switches train/eval
  2. Linear      — y = x @ W + b, the layer that holds the learnable numbers
  3. Activations — ReLU, Sigmoid, Tanh
  4. Dropout     — randomly switches neurons off while training
  5. Sequential  — chains layers into a network
  6. Losses      — CrossEntropyLoss, BCEWithLogitsLoss
"""

import numpy as np

from .tensor import Tensor, binary_cross_entropy_with_logits, cross_entropy


# =============================================================================
# 1. Module
# =============================================================================
class Module:
    """Base class for every layer and model.

    A subclass stores its learnable Tensors and sub-layers as attributes;
    `parameters()` finds them automatically, so models never list them by hand.
    """

    training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def _children(self):
        """Every attribute that is a Module (or a list of Modules)."""
        for value in vars(self).values():
            if isinstance(value, Module):
                yield value
            elif isinstance(value, (list, tuple)):
                yield from (v for v in value if isinstance(v, Module))

    def parameters(self):
        params = [v for v in vars(self).values() if isinstance(v, Tensor) and v.requires_grad]
        for child in self._children():
            params.extend(child.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def train(self):
        self.training = True
        for child in self._children():
            child.train()
        return self

    def eval(self):
        self.training = False
        for child in self._children():
            child.eval()
        return self


# =============================================================================
# 2. Linear
# =============================================================================
class Linear(Module):
    """y = x @ W + b, with W of shape (in_features, out_features).

    (PyTorch stores the transpose, (out, in), and computes x @ W.T + b.)
    """

    def __init__(self, in_features, out_features, bias=True, rng=None, dtype=np.float32):
        rng = rng if rng is not None else np.random.default_rng()
        # He initialisation: variance 2/in keeps the signal from shrinking or
        # exploding as it passes through many ReLU layers.
        std = np.sqrt(2.0 / in_features)
        self.weight = Tensor(rng.normal(0.0, std, (in_features, out_features)).astype(dtype),
                             requires_grad=True)
        self.bias = Tensor(np.zeros(out_features, dtype=dtype), requires_grad=True) if bias else None

    def forward(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias  # (N, out) + (out,): broadcasting, handled by _unbroadcast
        return out

    def __repr__(self):
        n_in, n_out = self.weight.shape
        return f"Linear({n_in}, {n_out})"


# =============================================================================
# 3. Activations
# =============================================================================
class ReLU(Module):
    def forward(self, x):
        return x.relu()

    def __repr__(self):
        return "ReLU()"


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()

    def __repr__(self):
        return "Sigmoid()"


class Tanh(Module):
    def forward(self, x):
        return x.tanh()

    def __repr__(self):
        return "Tanh()"


# =============================================================================
# 4. Dropout
# =============================================================================
class Dropout(Module):
    """Inverted dropout: zero each value with probability p while training,
    and scale the survivors by 1/(1-p) so the expected value is unchanged.
    In eval mode it does nothing.
    """

    def __init__(self, p=0.5, rng=None):
        if not 0.0 <= p < 1.0:
            raise ValueError("p must be in [0, 1)")
        self.p = p
        self.rng = rng if rng is not None else np.random.default_rng()
        self.mask = None  # the last mask used; kept so tests can inspect it

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = self.rng.random(x.shape) >= self.p
        self.mask = (keep / (1.0 - self.p)).astype(x.dtype)
        return x * Tensor(self.mask)

    def __repr__(self):
        return f"Dropout(p={self.p})"


# =============================================================================
# 5. Sequential
# =============================================================================
class Sequential(Module):
    """Runs layers one after another: Sequential(a, b, c)(x) == c(b(a(x)))."""

    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __getitem__(self, i):
        return self.layers[i]

    def __len__(self):
        return len(self.layers)

    def __repr__(self):
        inner = "\n".join(f"  ({i}): {layer!r}" for i, layer in enumerate(self.layers))
        return f"Sequential(\n{inner}\n)"


# =============================================================================
# 6. Losses
# =============================================================================
class CrossEntropyLoss(Module):
    """Multi-class loss: logits (N, C) and integer targets (N,)."""

    def forward(self, logits, targets):
        return cross_entropy(logits, targets)


class BCEWithLogitsLoss(Module):
    """Binary loss: raw scores and 0/1 targets of the same shape."""

    def forward(self, logits, targets):
        return binary_cross_entropy_with_logits(logits, targets)
