"""
Extended Kalman Filter for the Project Arka micro gas turbine digital twin.
"""

import numpy as np

from . import params as P
from .state_space import f, h
from .jacobians import jacobian_F_vec, jacobian_H_vec


class EKF:
    """Extended Kalman Filter over the 6-state / 16-measurement gas
    turbine model defined in core_ekf.state_space.
    """

    def __init__(self, x0=None, P0=None, Q=None, R=None):
        self.x = np.array(P.x0 if x0 is None else x0, dtype=float).copy()
        self.P = np.array(P.P0 if P0 is None else P0, dtype=float).copy()
        self.Q = np.array(P.Q if Q is None else Q, dtype=float).copy()
        self.R = np.array(P.R if R is None else R, dtype=float).copy()
        self.n = self.x.shape[0]

    def predict(self, u, dt=None):
        """Propagate state and covariance forward by dt using Euler
        integration and the vectorized F Jacobian.
        """
        dt = P.dt if dt is None else dt
        u = np.asarray(u, dtype=float)

        F = jacobian_F_vec(self.x, u)
        self.x = self.x + f(self.x, u) * dt
        self.P = F @ self.P @ F.T + self.Q

        return self.x, self.P

    def update(self, z):
        """Assimilate a measurement vector z (shape (16,))."""
        z = np.asarray(z, dtype=float)

        H = jacobian_H_vec(self.x)
        y = z - h(self.x)
        S = H @ self.P @ H.T + self.R

        # Kalman gain via linear solve rather than explicit inversion.
        K = np.linalg.solve(S, H @ self.P).T

        self.x = self.x + K @ y

        I = np.eye(self.n)
        self.P = (I - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)

        return self.x, self.P
