"""Optimizers: they read each parameter's .grad and update its .data.

The update rules follow PyTorch's conventions exactly, so the results can be
compared with torch.optim step by step.
"""

import numpy as np


class Optimizer:
    def __init__(self, params, lr):
        self.params = list(params)
        if not self.params:
            raise ValueError("optimizer got an empty parameter list")
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent, optionally with momentum.

    momentum = 0:   p <- p - lr * g
    momentum = mu:  v <- mu * v + g
                    p <- p - lr * v       (PyTorch's convention)
    """

    def __init__(self, params, lr=0.01, momentum=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, v in zip(self.params, self.velocity):
            if p.grad is None:
                continue
            g = p.grad
            if self.momentum:
                v *= self.momentum  # in place, so self.velocity keeps the new value
                v += g
                g = v
            p.data -= self.lr * g


class Adam(Optimizer):
    """Adam: momentum on the gradient (m) plus a per-parameter step size
    from the running average of squared gradients (v).

    m <- b1 * m + (1 - b1) * g
    v <- b2 * v + (1 - b2) * g^2
    m_hat = m / (1 - b1^t),  v_hat = v / (1 - b2^t)     (bias correction)
    p <- p - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        correction1 = 1 - b1**self.t
        correction2 = 1 - b2**self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            m_hat = m / correction1
            v_hat = v / correction2
            p.data -= (self.lr * m_hat / (np.sqrt(v_hat) + self.eps)).astype(p.dtype)
