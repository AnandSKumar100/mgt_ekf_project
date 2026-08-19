"""
Jacobians of f(x, u) and h(x) via central-difference numerical
differentiation.

Two implementations of each are provided:
  * `_loop` variants use a standard sequential Python for-loop over
    state dimensions.
  * `_vec` variants build a (6, 6) diagonal perturbation matrix and
    evaluate all perturbations in a single batched call to f()/h(),
    using NumPy broadcasting instead of Python loops.

Both variants must agree to within numerical tolerance (see
tests/test_ekf.py).
"""

import numpy as np

from .state_space import f, h

N_STATES = 6


def jacobian_F_loop(x, u, eps=1e-6):
    """df/dx via sequential central differences. Returns (6, 6)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    f0 = f(x, u)
    m = f0.shape[-1]
    F = np.zeros((m, n))
    for j in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += eps
        x_minus[j] -= eps
        F[:, j] = (f(x_plus, u) - f(x_minus, u)) / (2.0 * eps)
    return F


def jacobian_H_loop(x, eps=1e-6):
    """dh/dx via sequential central differences. Returns (16, 6)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    h0 = h(x)
    m = h0.shape[-1]
    H = np.zeros((m, n))
    for j in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += eps
        x_minus[j] -= eps
        H[:, j] = (h(x_plus) - h(x_minus)) / (2.0 * eps)
    return H


def jacobian_F_vec(x, u, eps=1e-6):
    """df/dx via vectorized central differences (no Python loop). Returns (6, 6)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    pert = eps * np.eye(n)
    x_plus = x[None, :] + pert     # (n, n)
    x_minus = x[None, :] - pert    # (n, n)

    f_plus = f(x_plus, u)   # (n, m)
    f_minus = f(x_minus, u)  # (n, m)

    F = (f_plus - f_minus).T / (2.0 * eps)  # (m, n)
    return F


def jacobian_H_vec(x, eps=1e-6):
    """dh/dx via vectorized central differences (no Python loop). Returns (16, 6)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    pert = eps * np.eye(n)
    x_plus = x[None, :] + pert     # (n, n)
    x_minus = x[None, :] - pert    # (n, n)

    h_plus = h(x_plus)    # (n, m)
    h_minus = h(x_minus)  # (n, m)

    H = (h_plus - h_minus).T / (2.0 * eps)  # (m, n)
    return H
