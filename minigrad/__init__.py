"""minigrad: a tensor-level reverse-mode autograd engine in pure NumPy."""

from . import nn, optim
from .tensor import (
    Tensor,
    binary_cross_entropy_with_logits,
    cross_entropy,
    cross_entropy_composed,
    no_grad,
)

__version__ = "0.1.0"

__all__ = [
    "Tensor",
    "binary_cross_entropy_with_logits",
    "cross_entropy",
    "cross_entropy_composed",
    "nn",
    "no_grad",
    "optim",
]
