"""minigrad: a tensor-level reverse-mode autograd engine in pure NumPy."""

from .tensor import Tensor, cross_entropy, cross_entropy_composed, no_grad

__version__ = "0.1.0"

__all__ = ["Tensor", "cross_entropy", "cross_entropy_composed", "no_grad"]
