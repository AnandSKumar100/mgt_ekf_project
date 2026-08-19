"""
Continuous-time dynamics f(x, u) and measurement model h(x) for the
Project Arka micro gas turbine EKF.

Both f and h accept either 1D state arrays of shape (6,) or 2D batched
state arrays of shape (N, 6), using ellipsis indexing (x[..., i]) so the
same code path handles both cases via NumPy broadcasting.

All terms that feed a sqrt() are floored at 1e-6 with np.maximum to avoid
NaNs during finite-difference Jacobian perturbation.
"""

import numpy as np

from . import params as P

_FLOOR = 1e-6


def f(x, u):
    """Continuous-time state derivative x_dot = f(x, u).

    x : array_like, shape (6,) or (N, 6)
    u : array_like, shape (2,) or (N, 2) -- [Tamb, Pamb]
    """
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)

    N = x[..., 0]
    Tt4 = x[..., 1]
    P3 = x[..., 2]
    P4 = x[..., 3]
    Pfuel = x[..., 4]
    P5 = x[..., 5]

    Tamb = u[..., 0]
    Pamb = u[..., 1]

    mu = (P.gamma - 1.0) / P.gamma          # compressor-side exponent
    nu = (P.gamma_g - 1.0) / P.gamma_g      # turbine-side exponent

    N_safe = np.maximum(N, _FLOOR)
    P4_safe = np.maximum(P4, _FLOOR)
    Tt4_safe = np.maximum(Tt4, _FLOOR)

    # --- x1: rotor speed ---------------------------------------------------
    turb_arg = np.maximum(P4 * Tt4 / N_safe, _FLOOR)
    pr_turb = np.maximum(P5 / P4_safe, _FLOOR) ** nu
    N_dot = (1.0 / P.Jeff) * (
        P.Cturb * np.sqrt(turb_arg) * (1.0 - pr_turb)
        - (P.kc1 * N ** 2 + P.kc2 * P3)
    )

    # --- x2: turbine inlet temperature --------------------------------------
    T3 = Tamb * (np.maximum(P3 / Pamb, _FLOOR) ** mu)
    Wf = P.Kf * np.sqrt(np.maximum(Pfuel - P3, _FLOOR))
    Wair = np.maximum(P.Ka * N_safe, _FLOOR)
    Tideal = T3 + (P.eta_b * P.LHV * Wf) / (P.Cp * Wair)
    Tt4_dot = (1.0 / P.tau_comb) * (Tideal - Tt4)

    # --- x3, x4: plenum pressures (ICV method) ------------------------------
    liner_flow = np.sqrt(np.maximum(P3 - P4, _FLOOR))
    P3_dot = (P.R_gas * T3 / P.V3) * (P.c1 * N - P.c2 * P3) - P.Kliner * liner_flow
    P4_dot = (P.R_gas * Tt4 / P.V4) * (
        P.Kliner * liner_flow + Wf - (P.ANGV * P.Kgas * P4) / np.sqrt(Tt4_safe)
    )

    # --- x5: fuel line pressure ----------------------------------------------
    Pfuel_dot = (P.beta_fuel / P.Vline) * (
        P.Kpump * N - P.Kinj * np.sqrt(np.maximum(Pfuel - P3, _FLOOR))
    )

    # --- x6: turbine exit pressure (first-order lag) --------------------------
    P5_dot = (1.0 / P.tau5) * (P.Cexp * P4 - P5)

    return np.stack([N_dot, Tt4_dot, P3_dot, P4_dot, Pfuel_dot, P5_dot], axis=-1)


def h(x):
    """Measurement model z = h(x).

    x : array_like, shape (6,) or (N, 6)
    returns array of shape (16,) or (N, 16)
    """
    x = np.asarray(x, dtype=float)

    N = x[..., 0]
    Tt4 = x[..., 1]
    P3 = x[..., 2]
    P4 = x[..., 3]
    Pfuel = x[..., 4]
    P5 = x[..., 5]

    mu = (P.gamma - 1.0) / P.gamma

    P4_safe = np.maximum(P4, _FLOOR)
    pr = np.maximum(P5 / P4_safe, _FLOOR) ** mu
    thermo = Tt4 * (1.0 - P.eta_t * (1.0 - pr))  # z1..z9

    Pfeed = Pfuel                                          # z10
    P3_m = P3                                              # z11
    P4_m = P4                                              # z12
    Wf_m = P.rho_f * P.Kinj * np.sqrt(np.maximum(Pfuel - P3, _FLOOR))  # z13
    N_m = N                                                 # z14
    thrust = P.Kfan * N ** 2 + P.Kcore * (P5 - P.Pamb)       # z15
    P5_m = P5                                               # z16

    thermo9 = np.stack([thermo] * 9, axis=-1)
    rest = np.stack([Pfeed, P3_m, P4_m, Wf_m, N_m, thrust, P5_m], axis=-1)

    return np.concatenate([thermo9, rest], axis=-1)
