# Micro Gas Turbine Extended Kalman Filter (EKF)

A high-performance, real-time Extended Kalman Filter module designed for state estimation and sensor fusion in micro gas turbine (MGT) systems. The model uses 6 primary continuous-time physical states and 16 measurement outputs, optimized for latency via 2D NumPy array broadcasting.

---

## Key Features

- **Vectorized Numerical Differentiation:** Replaces scalar Python loops with batched matrix evaluations for central-difference Jacobians ($F$ and $H$).
- **Numerically Stable Formulation:** Solves for Kalman gain via linear system decomposition (`np.linalg.solve`) instead of explicit matrix inversion.
- **Physical Boundary Enforcement:** Bounded arguments on square-root and division terms to eliminate divergence during finite-difference perturbations.
- **Comprehensive Unit Testing:** Full coverage verifying state shapes, covariance positive-definiteness, Jacobian numerical equivalence, and multi-cycle Euler integration stability.

---

## State & Measurement Specifications

### State Vector ($x \in \mathbb{R}^6$)
- $N$: Rotor speed $[\text{RPM}]$
- $T_{t4}$: Turbine inlet temperature $[\text{K}]$
- $P_3$: Compressor discharge pressure $[\text{Pa}]$
- $P_4$: Combustor pressure $[\text{Pa}]$
- $P_{\text{fuel}}$: Fuel line pressure $[\text{Pa}]$
- $P_5$: Turbine exit total pressure $[\text{Pa}]$

### Inputs ($u \in \mathbb{R}^2$)
- $T_{\text{amb}}$: Ambient temperature $[\text{K}]$
- $P_{\text{amb}}$: Ambient pressure $[\text{Pa}]$

---

## Benchmark Results

Evaluated over 1,000 continuous predict-update cycles on standard x86-64 hardware:

| Implementation | Total Runtime (1,000 cycles) | Mean Latency / Cycle | Speedup Factor |
| :--- | :--- | :--- | :--- |
| **Sequential Loops** | ~526 ms | 526 $\mu\text{s}$ | 1.00x |
| **Vectorized Broadcast** | ~169 ms | 169 $\mu\text{s}$ | **3.11x** |

---

## Installation & Execution

### 1. Requirements
- Python 3.9+
- NumPy
- Pytest

```powershell
pip install numpy pytest