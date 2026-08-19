"""
Physical constants, thermodynamic parameters, noise covariances, and nominal
baseline state for the Project Arka micro gas turbine EKF.

State vector x = [N, Tt4, P3, P4, Pfuel, P5]
  N      - rotor speed                  [RPM]
  Tt4    - turbine inlet temperature    [K]
  P3     - compressor discharge press.  [Pa]
  P4     - combustor pressure           [Pa]
  Pfuel  - fuel line pressure           [Pa]
  P5     - turbine exit pressure        [Pa]

Input vector u = [Tamb, Pamb]
  Tamb - ambient temperature [K]
  Pamb - ambient pressure    [Pa]

A handful of flow/load coefficients (Cturb, kc1, kc2, Kf, ANGV, c1, c2,
Kpump, Cexp) are solved below from the steady-state (x_dot = 0) balance of
each state equation at the chosen nominal operating point x0, rather than
picked independently. Because several of these equations amplify small
coefficient errors by large factors (e.g. R_gas*T3/V3 ~ 1e8), picking them
independently left the nominal point far from equilibrium and made Euler
propagation diverge over tens of cycles; solving them jointly keeps x0 a
genuine (numerically exact) fixed point of f(x0, u0).
"""

import numpy as np

# --------------------------------------------------------------------------
# Integration
# --------------------------------------------------------------------------
dt = 0.01  # [s] Euler integration step

# --------------------------------------------------------------------------
# Ambient / reference conditions
# --------------------------------------------------------------------------
Tamb = 288.15    # [K]   ISA sea-level ambient temperature
Pamb = 101325.0  # [Pa]  ISA sea-level ambient pressure

# --------------------------------------------------------------------------
# Gas properties
# --------------------------------------------------------------------------
gamma = 1.4      # [-] specific heat ratio, air (compressor side)
gamma_g = 1.33   # [-] specific heat ratio, combustion gas (turbine side)
R_gas = 287.0    # [J/(kg*K)] specific gas constant, air/combustion gas
Cp = 1005.0      # [J/(kg*K)] specific heat at constant pressure, air

mu = (gamma - 1.0) / gamma        # compressor-side pressure exponent
nu = (gamma_g - 1.0) / gamma_g    # turbine-side pressure exponent

# --------------------------------------------------------------------------
# Nominal operating point (also used below to derive equilibrium coefficients)
# --------------------------------------------------------------------------
_N0 = 95000.0     # [RPM]
_Tt4_0 = 950.0    # [K]
_P3_0 = 350000.0  # [Pa]
_P4_0 = 340000.0  # [Pa]
_Pfuel_0 = 500000.0  # [Pa]
_P5_0 = 115000.0  # [Pa]

_T3_0 = Tamb * (_P3_0 / Pamb) ** mu
_pr_turb_0 = (_P5_0 / _P4_0) ** nu
_turb_arg_0 = _P4_0 * _Tt4_0 / _N0

# --------------------------------------------------------------------------
# Rotor dynamics
# --------------------------------------------------------------------------
Jeff = 8.0e-4   # [kg*m^2] effective polar inertia of rotor + compressor + turbine
Cturb = 1.0     # [-] turbine power/flow extraction coefficient

# kc1, kc2 solved from rotor power balance at x0:
#   Cturb*sqrt(turb_arg_0)*(1-pr_turb_0) = kc1*N0^2 + kc2*P3_0
# with a 75/25 split between aerodynamic drag (N^2) and compressor load (P3).
_rotor_bracket_0 = Cturb * np.sqrt(_turb_arg_0) * (1.0 - _pr_turb_0)
_KC1_FRAC = 0.75
kc1 = (_rotor_bracket_0 * _KC1_FRAC) / _N0 ** 2          # [-] N^2 drag coefficient
kc2 = (_rotor_bracket_0 * (1.0 - _KC1_FRAC)) / _P3_0     # [-] P3 load coefficient

# --------------------------------------------------------------------------
# Combustion
# --------------------------------------------------------------------------
tau_comb = 0.25   # [s] combustor thermal time constant
eta_b = 0.98      # [-] combustion efficiency
LHV = 43.0e6      # [J/kg] fuel lower heating value (Jet-A)

Ka = 0.15 / _N0   # [kg/s per RPM] air mass flow coeff., Wair = Ka*N (~0.15 kg/s at N0)
_Wair_0 = Ka * _N0

# Kf solved from the combustor energy balance at x0: T3 + eta_b*LHV*Wf/(Cp*Wair) = Tt4
_Wf_0 = (_Tt4_0 - _T3_0) * Cp * _Wair_0 / (eta_b * LHV)
Kf = _Wf_0 / np.sqrt(_Pfuel_0 - _P3_0)  # [kg/s per sqrt(Pa)]

# --------------------------------------------------------------------------
# Compressor discharge / combustor plena (ICV method)
# --------------------------------------------------------------------------
V3 = 8.0e-4       # [m^3] compressor discharge plenum volume
V4 = 1.5e-3       # [m^3] combustor volume
Kliner = 2.5e-6   # [kg/s per sqrt(Pa)] combustor liner flow coefficient
Kgas = 0.5        # [-] turbine nozzle gas flow coefficient

_liner_flow_0 = Kliner * np.sqrt(_P3_0 - _P4_0)

# ANGV solved from the combustor mass balance at x0:
#   Kliner*sqrt(P3-P4) + Wf = ANGV*Kgas*P4/sqrt(Tt4)
ANGV = (_liner_flow_0 + _Wf_0) * np.sqrt(_Tt4_0) / (Kgas * _P4_0)  # [m^2]

c1 = 0.15 / _N0   # [-] compressor inflow coefficient, c1*N ~ air flow into V3 plenum
_c1N_0 = c1 * _N0

# c2 solved from the V3 mass balance at x0: (R_gas*T3/V3)*(c1*N - c2*P3) = Kliner*sqrt(P3-P4)
_c2_bracket_target_0 = _liner_flow_0 / (R_gas * _T3_0 / V3)
c2 = (_c1N_0 - _c2_bracket_target_0) / _P3_0  # [-]

# --------------------------------------------------------------------------
# Fuel system
# --------------------------------------------------------------------------
Vline = 2.0e-4    # [m^3] fuel line volume
beta_fuel = 1.2e9 # [Pa] fuel bulk modulus
Kinj = 3.5e-7     # [kg/s per sqrt(Pa)] injector flow coefficient
rho_f = 780.0     # [kg/m^3] fuel density (Jet-A)

# Kpump solved from the fuel line balance at x0: Kpump*N = Kinj*sqrt(Pfuel-P3)
Kpump = (Kinj * np.sqrt(_Pfuel_0 - _P3_0)) / _N0  # [kg/s per RPM]

# --------------------------------------------------------------------------
# Turbine exit / exhaust
# --------------------------------------------------------------------------
tau5 = 0.05           # [s] turbine exit pressure first-order lag time constant
Cexp = _P5_0 / _P4_0  # [-] turbine expansion pressure ratio coefficient (exact at x0)
eta_t = 0.85          # [-] turbine efficiency (EGT measurement model)

# --------------------------------------------------------------------------
# Thrust / load
# --------------------------------------------------------------------------
Kfan = 2.0e-8    # [N per RPM^2] thrust coefficient, rotor speed term
Kcore = 8.0e-4   # [N per Pa] thrust coefficient, core exhaust pressure term

# --------------------------------------------------------------------------
# Nominal baseline state and initial covariance
# --------------------------------------------------------------------------
x0 = np.array([_N0, _Tt4_0, _P3_0, _P4_0, _Pfuel_0, _P5_0])

P0 = np.diag([
    500.0 ** 2,   # N
    10.0 ** 2,    # Tt4
    2000.0 ** 2,  # P3
    2000.0 ** 2,  # P4
    3000.0 ** 2,  # Pfuel
    1000.0 ** 2,  # P5
])

# --------------------------------------------------------------------------
# Process noise covariance Q (6x6, diagonal)
# --------------------------------------------------------------------------
q_N = 100.0
q_T = 1.0
q_P3 = 100.0
q_P4 = 100.0
q_Pfuel = 200.0
q_P5 = 50.0

Q = np.diag([q_N, q_T, q_P3, q_P4, q_Pfuel, q_P5])

# --------------------------------------------------------------------------
# Measurement noise covariance R (16x16, diagonal)
# --------------------------------------------------------------------------
r_thermo = 4.0     # [K^2] per thermocouple, x9
r_Pfeed = 2500.0   # [Pa^2]
r_P3 = 2500.0      # [Pa^2]
r_P4 = 2500.0      # [Pa^2]
r_Wf = 1.0e-6      # [(kg/s)^2]
r_N = 25.0         # [RPM^2]
r_thrust = 1.0     # [N^2]
r_P5 = 900.0       # [Pa^2]

R = np.diag(
    [r_thermo] * 9 + [r_Pfeed, r_P3, r_P4, r_Wf, r_N, r_thrust, r_P5]
)

# --------------------------------------------------------------------------
# Physical state bounds (for sanity checking / validation)
# --------------------------------------------------------------------------
STATE_BOUNDS = {
    "N": (0.0, 150000.0),
    "Tt4": (250.0, 1500.0),
    "P3": (50000.0, 700000.0),
    "P4": (50000.0, 700000.0),
    "Pfuel": (50000.0, 1.0e6),
    "P5": (50000.0, 400000.0),
}
