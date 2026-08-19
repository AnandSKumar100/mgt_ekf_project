"""
Array shape assertions and physical boundary sanity checks for the
Project Arka EKF.
"""

import numpy as np

STATE_NAMES = ["N", "Tt4", "P3", "P4", "Pfuel", "P5"]


def assert_shape(arr, expected_last_dim, name="array"):
    """Assert that arr's last dimension equals expected_last_dim."""
    arr = np.asarray(arr)
    if arr.shape[-1] != expected_last_dim:
        raise ValueError(
            f"{name} last dimension must be {expected_last_dim}, "
            f"got shape {arr.shape}"
        )
    return arr


def assert_state_shape(x):
    """Assert x has last dimension 6 (state vector)."""
    return assert_shape(x, 6, "state vector x")


def assert_measurement_shape(z):
    """Assert z has last dimension 16 (measurement vector)."""
    return assert_shape(z, 16, "measurement vector z")


def assert_square(mat, n, name="matrix"):
    """Assert mat is an (n, n) matrix."""
    mat = np.asarray(mat)
    if mat.shape != (n, n):
        raise ValueError(f"{name} must have shape ({n}, {n}), got {mat.shape}")
    return mat


def check_state_bounds(x, bounds):
    """Return the list of state names whose values in x fall outside the
    physical bounds given in `bounds` (a dict of name -> (lo, hi)).
    """
    x = np.asarray(x, dtype=float)
    violations = []
    for i, name in enumerate(STATE_NAMES):
        if name not in bounds:
            continue
        lo, hi = bounds[name]
        val = x[..., i]
        if np.any(val < lo) or np.any(val > hi):
            violations.append(name)
    return violations


def is_symmetric(mat, atol=1e-8):
    """Return True if mat is symmetric within atol."""
    mat = np.asarray(mat)
    return np.allclose(mat, mat.T, atol=atol)


def is_positive_definite(mat):
    """Return True if mat is (numerically) positive definite via Cholesky."""
    mat = np.asarray(mat)
    try:
        np.linalg.cholesky(mat)
        return True
    except np.linalg.LinAlgError:
        return False
