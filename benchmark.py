"""
Comparative benchmark: sequential-loop Jacobians vs. vectorized broadcast
Jacobians, measured over 1,000 predict-update EKF cycles.
"""

import time

import numpy as np

from core_ekf import params as P
from core_ekf.state_space import f, h
from core_ekf.jacobians import (
    jacobian_F_loop,
    jacobian_H_loop,
    jacobian_F_vec,
    jacobian_H_vec,
)

N_CYCLES = 1000
N_WARMUP = 20
U0 = np.array([P.Tamb, P.Pamb])


def run_cycle(F_fn, H_fn):
    """One predict-update EKF cycle using the given Jacobian functions."""
    x = P.x0.copy()
    Pcov = P.P0.copy()
    Q = P.Q
    R = P.R
    I6 = np.eye(6)
    dt = P.dt

    # --- predict ---
    F = F_fn(x, U0)
    x = x + f(x, U0) * dt
    Pcov = F @ Pcov @ F.T + Q

    # --- update ---
    H = H_fn(x)
    z = h(x)
    y = z - h(x)
    S = H @ Pcov @ H.T + R
    K = np.linalg.solve(S, H @ Pcov).T
    x = x + K @ y
    Pcov = (I6 - K @ H) @ Pcov
    Pcov = 0.5 * (Pcov + Pcov.T)

    return x, Pcov


def benchmark(F_fn, H_fn, label):
    for _ in range(N_WARMUP):
        run_cycle(F_fn, H_fn)

    start = time.perf_counter()
    for _ in range(N_CYCLES):
        run_cycle(F_fn, H_fn)
    elapsed = time.perf_counter() - start

    mean_us = (elapsed / N_CYCLES) * 1e6
    print(f"{label:28s} total={elapsed * 1000:8.2f} ms   mean/cycle={mean_us:8.2f} us")
    return mean_us


if __name__ == "__main__":
    print(f"Benchmarking {N_CYCLES} predict-update EKF cycles\n")

    loop_us = benchmark(jacobian_F_loop, jacobian_H_loop, "Sequential-loop Jacobians")
    vec_us = benchmark(jacobian_F_vec, jacobian_H_vec, "Vectorized Jacobians")

    speedup = loop_us / vec_us
    print(f"\nSpeedup (loop / vectorized): {speedup:.2f}x")
