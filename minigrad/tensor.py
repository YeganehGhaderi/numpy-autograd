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