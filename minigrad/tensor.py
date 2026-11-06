from contextlib import contextmanager

import numpy as np


# =============================================================================
# Section 1 - Core
# =============================================================================

_grad_enabled = True


@contextmanager
def no_grad():
    """Inside `with no_grad():` no graph is recorded (used for evaluation)."""
    global _grad_enabled
    previous = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = previous


def _unbroadcast(grad, shape):
    """Sum `grad` back down to `shape`, undoing NumPy broadcasting.

    If x of shape (5,) was broadcast to (N, 5) in the forward pass, each of its
    entries was used N times, so its gradient is the sum over those N rows.
    """
    # 1) Extra leading axes were added: sum them away.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # 2) Axes of size 1 that were stretched: sum them but keep the axis.
    for i, size in enumerate(shape):
        if size == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    # Makes `numpy_array + tensor` call Tensor.__radd__ instead of NumPy
    # trying (and failing) to treat the Tensor as an array element.
    __array_ufunc__ = None

    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        data = np.asarray(data)
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float32)
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        self._prev = _children          # the tensors this one was computed from
        self._backward = lambda: None   # sends self.grad to the children
        self._op = _op                  # name of the operation, for debugging

    # ---- small conveniences -------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self):
        return self.data.size

    def numpy(self):
        return self.data

    def item(self):
        return self.data.item()

    def detach(self):
        """Same data, cut off from the graph."""
        return Tensor(self.data)

    def zero_grad(self):
        self.grad = None

    def __repr__(self):
        grad_info = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data!r}{grad_info})"

    # ---- graph-building helpers ---------------------------------------------
    def _wrap(self, other):
        """Turn a number or array into a Tensor with the same dtype as self."""
        if isinstance(other, Tensor):
            return other
        return Tensor(np.asarray(other, dtype=self.data.dtype))

    @staticmethod
    def _make(data, parents, op):
        """Create the output tensor of an operation.

        It only joins the graph if grad mode is on and some parent needs a grad.
        """
        needs_grad = _grad_enabled and any(p.requires_grad for p in parents)
        if needs_grad:
            return Tensor(data, requires_grad=True, _children=parents, _op=op)
        return Tensor(data)

    def _accum(self, g):
        """Add g to self.grad. A tensor used twice receives two contributions."""
        if self.grad is None:
            self.grad = np.array(g, dtype=self.data.dtype, copy=True)
        else:
            self.grad = self.grad + g

    # ---- reverse-mode differentiation ---------------------------------------
    def backward(self, grad=None):
        """Compute d(self)/d(t) for every tensor t in the graph behind self."""
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be given for non-scalar outputs")
            grad = np.ones_like(self.data)

        # Topological order: every tensor appears after the tensors it was made from.
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        self._accum(np.asarray(grad, dtype=self.data.dtype))
        # Walk from the output back to the inputs.
        for v in reversed(topo):
            v._backward()

    # =========================================================================
    # Section 2 — Arithmetic
    # =========================================================================
    def __add__(self, other):
        other = self._wrap(other)
        out = Tensor._make(self.data + other.data, (self, other), "add")

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1
            if self.requires_grad:
                self._accum(_unbroadcast(out.grad, self.shape))
            if other.requires_grad:
                other._accum(_unbroadcast(out.grad, other.shape))

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._wrap(other)
        out = Tensor._make(self.data * other.data, (self, other), "mul")

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a
            if self.requires_grad:
                self._accum(_unbroadcast(out.grad * other.data, self.shape))
            if other.requires_grad:
                other._accum(_unbroadcast(out.grad * self.data, other.shape))

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = self._wrap(other)
        out = Tensor._make(self.data / other.data, (self, other), "div")

        def _backward():
            # d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
            if self.requires_grad:
                self._accum(_unbroadcast(out.grad / other.data, self.shape))
            if other.requires_grad:
                g = -out.grad * self.data / other.data**2
                other._accum(_unbroadcast(g, other.shape))

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-self._wrap(other))

    # Reflected versions: called for `2 + t`, `2 * t`, `2 - t`, `2 / t`.
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return self._wrap(other) - self

    def __rtruediv__(self, other):
        return self._wrap(other) / self

    # =========================================================================
    # Section 3 — Shape operations
    # =========================================================================
    def __matmul__(self, other):
        other = self._wrap(other)
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("matmul supports 2-D tensors only")
        out = Tensor._make(self.data @ other.data, (self, other), "matmul")

        def _backward():
            # C = A @ B  ->  dA = dC @ B^T,  dB = A^T @ dC
            if self.requires_grad:
                self._accum(out.grad @ other.data.T)
            if other.requires_grad:
                other._accum(self.data.T @ out.grad)

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor._make(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            # Every input element contributed with weight 1, so each one gets
            # the gradient of the sum it went into: copy grad back to the input shape.
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._accum(np.broadcast_to(g, self.shape))

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            count = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            count = int(np.prod([self.shape[a] for a in axes]))
        return self.sum(axis=axis, keepdims=keepdims) / count

    def reshape(self, *shape):
        out = Tensor._make(self.data.reshape(*shape), (self,), "reshape")

        def _backward():
            self._accum(out.grad.reshape(self.shape))

        out._backward = _backward
        return out

    def transpose(self, *axes):
        axes = axes if axes else tuple(reversed(range(self.ndim)))
        out = Tensor._make(self.data.transpose(axes), (self,), "transpose")

        def _backward():
            # Undo the permutation with its inverse.
            self._accum(out.grad.transpose(np.argsort(axes)))

        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

    def __getitem__(self, index):
        out = Tensor._make(self.data[index], (self,), "getitem")

        def _backward():
            # Picked elements get the gradient; the rest get 0. np.add.at handles
            # an element picked more than once (its gradients are added).
            g = np.zeros_like(self.data)
            np.add.at(g, index, out.grad)
            self._accum(g)

        out._backward = _backward
        return out

    # =========================================================================
    # Section 4 — Element-wise functions
    # =========================================================================
    def exp(self):
        out = Tensor._make(np.exp(self.data), (self,), "exp")

        def _backward():
            self._accum(out.grad * out.data)  # (e^x)' = e^x

        out._backward = _backward
        return out

    def log(self):
        out = Tensor._make(np.log(self.data), (self,), "log")

        def _backward():
            self._accum(out.grad / self.data)  # (ln x)' = 1/x

        out._backward = _backward
        return out

    def __pow__(self, n):
        if not isinstance(n, (int, float)):
            raise TypeError("only a constant (int/float) exponent is supported")
        out = Tensor._make(self.data**n, (self,), f"pow{n}")

        def _backward():
            self._accum(out.grad * n * self.data ** (n - 1))  # (x^n)' = n x^(n-1)

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor._make(np.maximum(self.data, 0), (self,), "relu")

        def _backward():
            self._accum(out.grad * (self.data > 0))  # slope 1 where x>0, else 0

        out._backward = _backward
        return out

    def sigmoid(self):
        # Stable form of 1/(1+e^-x): never computes e^(large number).
        s = np.exp(-np.logaddexp(0, -self.data)).astype(self.data.dtype)
        out = Tensor._make(s, (self,), "sigmoid")

        def _backward():
            self._accum(out.grad * s * (1 - s))  # s' = s(1-s)

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor._make(t, (self,), "tanh")

        def _backward():
            self._accum(out.grad * (1 - t**2))  # tanh' = 1 - tanh^2

        out._backward = _backward
        return out

    # =========================================================================
    # Section 5 — Softmax family
    # =========================================================================
    def softmax(self, axis=-1):
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(shifted)
        s = e / e.sum(axis=axis, keepdims=True)
        out = Tensor._make(s, (self,), "softmax")

        def _backward():
            # Jacobian-vector product: ds = s * (g - sum(g*s))
            g = out.grad
            self._accum(s * (g - (g * s).sum(axis=axis, keepdims=True)))

        out._backward = _backward
        return out

    def log_softmax(self, axis=-1):
        # log softmax(x) = x - logsumexp(x); subtracting the max avoids overflow.
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        logsumexp = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
        y = shifted - logsumexp
        out = Tensor._make(y, (self,), "log_softmax")

        def _backward():
            # dy_i/dx_j = [i==j] - softmax_j  ->  dx = g - softmax * sum(g)
            g = out.grad
            softmax = np.exp(y)
            self._accum(g - softmax * g.sum(axis=axis, keepdims=True))

        out._backward = _backward
        return out


def cross_entropy(logits, targets):
    """Mean cross-entropy loss, softmax and log fused into one operation.

    logits:  Tensor of shape (N, C), raw class scores
    targets: integer array of shape (N,), the correct class of each row
    """
    targets = np.asarray(targets, dtype=np.int64)
    n = logits.shape[0]
    rows = np.arange(n)

    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    loss = -log_probs[rows, targets].mean()

    out = Tensor._make(np.asarray(loss, dtype=logits.dtype), (logits,), "cross_entropy")

    def _backward():
        # The famous result: d(loss)/d(logits) = (softmax - one_hot) / N
        grad = np.exp(log_probs)
        grad[rows, targets] -= 1
        logits._accum(out.grad * grad / n)

    out._backward = _backward
    return out


def cross_entropy_composed(logits, targets):
    """The same loss built from basic ops; used to check the fused version."""
    targets = np.asarray(targets, dtype=np.int64)
    log_probs = logits.log_softmax(axis=1)
    picked = log_probs[np.arange(logits.shape[0]), targets]
    return -picked.mean()

def binary_cross_entropy_with_logits(logits, targets):
    """Mean binary cross-entropy, sigmoid and log fused into one operation.

    logits:  Tensor of any shape, raw scores (before sigmoid)
    targets: array of the same shape with values 0 or 1
    """
    targets = np.asarray(targets, dtype=logits.dtype).reshape(logits.shape)
    z = logits.data
    # loss = -[y log s(z) + (1-y) log(1 - s(z))]  simplifies to  log(1 + e^z) - y z.
    # np.logaddexp(0, z) computes log(1 + e^z) without overflow.
    loss = (np.logaddexp(0, z) - targets * z).mean()
    out = Tensor._make(np.asarray(loss, dtype=logits.dtype), (logits,), "bce_with_logits")

    def _backward():
        # Same pattern as softmax cross-entropy: (sigmoid(z) - y) / N
        s = np.exp(-np.logaddexp(0, -z))
        logits._accum(out.grad * (s - targets) / z.size)

    out._backward = _backward
    return out