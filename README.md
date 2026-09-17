# Project Arka — Extended Kalman Filter

A directly importable, single-file implementation of the five-state continuous–discrete EKF in **EKF_2.pdf, September 2026**. The same file runs recorded sensor CSVs from the command line. There is no synthetic-data generator, demonstration engine, or dependency on the GPA repository.

**Engine calibration is required.** The supplied document defines equations but does not supply compressor/turbine maps, engine geometry/constants, gas-property data, calibrated noise covariances, or recordings. This implementation requires those inputs explicitly. It is executable software, not a calibrated or experimentally validated model of the physical engine.

## Files

- `arka_ekf.py`: complete physics model, EKF, live/batch APIs and CSV runner.
- `requirements.txt`: versions tested together.
- `engine_config.schema.json`: configuration structure for your calibrated engine data.
- `BENCHMARKS.md`: measured performance, optimization decisions and verification results.
- `.gitignore`: excludes local recordings, calibration and compilation caches.

The previous repository contents are retained in Git history, not in the current working tree.

## Install and run

Tested on 64-bit CPython **3.12.14**, Linux x86-64. Use Python 3.12 and a virtual environment:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python arka_ekf.py --help
python arka_ekf.py --config engine_config.json --input measurements.csv --output estimates.csv
```

Supply your own `engine_config.json` and `measurements.csv` as described below. The CLI limits BLAS to one thread while filtering because these matrices are small. Numba compiles native kernels on first use and caches them beside the module. Use a writable checkout or set `NUMBA_CACHE_DIR` to a writable directory. The initial compilation can take tens of seconds; it is reported separately and is not a per-sample cost.

The CLI prints measured load time, warm-up time, filter time, mean microseconds per sample, throughput, and the number of physically rejected corrections. Filter timing includes array conversion, validation and output-array allocation but excludes compilation, CSV reading/writing, and configuration loading.

## State, inputs and outputs

State order is fixed:

| Index | State | Unit |
|---|---|---|
| 0 | `N_rpm` | rpm |
| 1 | `mdot_air` | kg/s |
| 2 | `Pt3_pa` | Pa, absolute total pressure |
| 3 | `Tt4_k` | K, total temperature |
| 4 | `Pt5_pa` | Pa, absolute total pressure |

The canonical output and measurement order is:

```text
[N_rpm, mdot_air, Pt3_pa, Tt3_k, Tt4_k, Pt5_pa, Tt5_k, thrust_n]
```

Input order is `[mdot_fuel, Ta_k, Pa_pa, M0]`, in kg/s, K, Pa and dimensionless Mach. **Only `M0=0` is supported**: the PDF supplies stationary inlet equations (6–7). Nonzero Mach raises a clear error instead of silently using an incorrect inlet model.

Health order is `[eta_c, flow_c, pi_c, eta_t, flow_t, eta_b, area_n, pi_in]`. Each entry is a positive multiplier, with nominal value 1. Health factors are external parameters, not estimated states. Efficiencies and inlet recovery after scaling must remain physical.

## Input CSV

Required columns:

```text
timestamp_s,mdot_fuel,Ta_k,Pa_pa
```

Optional measurement columns use the eight exact output names above. `M0` is optional and defaults to 0. A complete header is:

```csv
timestamp_s,mdot_fuel,Ta_k,Pa_pa,M0,N_rpm,mdot_air,Pt3_pa,Tt3_k,Tt4_k,Pt5_pa,Tt5_k,thrust_n
```

Timestamps must be finite and strictly increasing; unequal intervals are supported. Values are SI with speed in rpm. Missing measurement columns, empty cells, NaN and infinity are treated as unavailable measurements. Nonpositive speed, flow, pressure and temperature observations are omitted; thrust may be negative because of sensor noise. Required input values cannot be missing or invalid.

Optional health columns are `theta_eta_c`, `theta_flow_c`, `theta_pi_c`, `theta_eta_t`, `theta_flow_t`, `theta_eta_b`, `theta_area_n`, `theta_pi_in`. Missing or empty health values default to 1.

**Timing contract:** row 0 corrects the initial prior at its timestamp without prediction. From row k−1 to row k, the model holds the **previous row's inputs and health** constant, then corrects using row k measurements and row k inputs/health. There is no interpolation of fuel flow. Split intervals at known input changes.

Output CSV columns contain the timestamp, eight physical outputs, five posterior state standard deviations, NIS, used/rejected sensor masks, and `constraint_rejected`. Mask bit i corresponds to output index i; for example, mask 5 represents indices 0 and 2. `rejected_mask` marks statistical or physical-correction rejection; unavailable/nonpositive measurements simply have no used bit. NIS is the score of the **last tested subset**, not necessarily the original full sensor set; it is NaN when no valid measurements were tested. Full covariance matrices are available through the Python API.

## Engine configuration

`engine_config.schema.json` documents the JSON structure. The Python constructor also checks dimensions, positive values, map domains and covariance validity. Schema validation is optional and requires no additional package at runtime. Covariance/map-size checks go beyond what the schema alone enforces.

Top-level keys are `schema_version` (1), `thrust_convention`, `parameters`, `maps`, `gas_properties`, and `filter`. Use actual calibrated numerical arrays, not the symbolic names below.

### Parameters

Every entry is required under `parameters`:

| Key | Meaning | Unit |
|---|---|---|
| `J` | effective spool inertia | kg m² |
| `V3`, `V5` | upstream/downstream effective gas volumes | m³ |
| `tau_m`, `tau_b` | airflow/combustor response times | s |
| `mdot_ref` | airflow dynamic scaling quantity | kg/s |
| `Tref`, `Pref` | component-map reference conditions | K, Pa |
| `pi_in` | nominal inlet total-pressure recovery | dimensionless, (0,1] |
| `delta_b` | combustor fractional pressure loss | [0,1) |
| `eta_b`, `eta_m` | nominal combustion/mechanical efficiency | (0,1] |
| `Wacc` | accessory/parasitic load | W, ≥0 |
| `LHV` | fuel lower heating value | J/kg |
| `An` | nominal convergent-nozzle exit area | m² |
| `Cd` | nozzle discharge coefficient | (0,1] |
| `Ra`, `Rg` | air/combustion-gas constants | J/(kg K) |
| `epsilon_pi` | Eq. 35 denominator safeguard | dimensionless, >0 |

### Component maps

Rectangular, bilinearly interpolated maps; axes must be positive, finite and strictly increasing with at least two knots. Tables are `[speed_index][second_axis_index]`. Extrapolation and clipping are prohibited.

`maps.compressor` contains:

- `speed`: corrected rpm, `N / sqrt(Ta/Tref)`.
- `flow`: corrected kg/s, `mdot_air * sqrt(Ta/Tref) / (Pt2/Pref)` at nominal health.
- `pressure_ratio`: total-pressure-ratio table, shape `(len(speed), len(flow))`, values >1.
- `efficiency`: same shape, values in (0,1].

`maps.turbine` contains:

- `speed`: corrected rpm, `N / sqrt(Tt4/Tref)`.
- `expansion_ratio`: `Pt4/Pt5`, values >1.
- `flow`: corrected turbine kg/s table, shape `(len(speed), len(expansion_ratio))`.
- `efficiency`: same shape, values in (0,1].

Map reference conditions must match `Tref`/`Pref`. Convert a nonrectangular source map to a calibrated rectangular operating domain before use; never include surge/stall regions merely to fill a rectangle. Finite differences and all RK4 stages must remain inside the domain, so initialize and operate with margin from map edges. Piecewise bilinear maps are continuous but not differentiable at cell boundaries; small Jacobian changes there are expected.

### Gas properties and explicit closure assumptions

`gas_properties.air` and `gas_properties.gas` each contain matching `temperature` and `cp` arrays. Temperatures are increasing K knots; cp is J/(kg K) and must exceed the corresponding gas constant. Two equal cp entries over a calibrated temperature range represent a constant-property assumption.

The PDF does not specify its property functions. This implementation evaluates air cp at `Ta` and gas cp at `Tt4`, linearly interpolates the supplied tables, and computes `gamma = cp/(cp-R)`. Those local values are held within one gas-path evaluation, including the nozzle. It implements the PDF's algebraic cp·T balances, not an enthalpy-integral combustor model. The combustor target is therefore evaluated using cp at the current `Tt4`, not solved implicitly at the target temperature. Supply/tune properties consistently and review this approximation against your operating range.

Other explicit choices where the document is incomplete:

- **Compressor flow health:** Eq. 5 names `theta_Wc` but later equations omit its placement. The compressor-map flow coordinate is divided by `theta_flow_c`, implementing flow-capacity scaling. Eq. 12 pressure-ratio scaling is unchanged. Nominal health exactly recovers Eqs. 8–13.
- **Nozzle:** a convergent isentropic nozzle with `Cd` multiplying mass flow. The unchoked exit pressure equals ambient. Choked exit pressure/temperature use the critical ratio. No reverse flow is modelled; require `Pt5 > Pa`.
- **Thrust:** set `thrust_convention` to `pressure` for Eq. 53 (`mdot_n*Ve + (Pe-Pa)*An` at M0=0), or `momentum` for Eq. 52. Match the downstream GPA convention explicitly. This is not automatically inferred from GPA source code.

### Filter configuration

Required under `filter`:

| Key | Shape | Meaning |
|---|---|---|
| `initial_state` | 5 | prior in physical state units |
| `initial_covariance` | 5×5 | symmetric positive-semidefinite P0, physical units² |
| `state_scale` | 5 | positive characteristic physical magnitudes for conditioning |
| `Qc` | 5×5 | continuous process-noise spectral density, physical units²/s |
| `R` | 8×8 | positive-definite sensor covariance in output order, physical units² |
| `max_step_s` | scalar | largest RK4 substep; choose using transient bandwidth/stability |

The PDF uses diagonal Qc; diagonal and full symmetric Qc are both accepted. R may include sensor correlations. Configure unused sensor entries too, with a valid positive variance. The active covariance submatrix is selected automatically. No matrix inverse is explicitly formed.

Optional settings:

| Key | Default | Meaning |
|---|---|---|
| `max_substeps` | 10000 | limit per timestamp interval |
| `fd_relative` | 1e-6 | scale-aware central-difference relative perturbation |
| `fd_absolute` | `state_scale * 1e-6` | five physical-unit perturbation floors |
| `sigma_gate` | 5.0 | individual normalized-innovation threshold |
| `nis_probability` | 0.999 | chi-squared gate probability |

Noise settings and integration steps are engine-specific; benchmark settings are not recommended calibration values. Internally the filter transforms state and covariance by `state_scale`, and scales innovations by sensor standard deviations. Returned states/covariances retain physical units.

## Python integration

```python
from arka_ekf import ProjectArkaEKF, OUTPUT_NAMES

engine = ProjectArkaEKF.from_json("engine_config.json")
# Supply actual first inputs, optionally health multipliers:
engine.warmup(first_inputs)  # compile before acquisition; does not alter the prior

# Inside your acquisition loop:
result = engine.step(timestamp_s, inputs, measurements, health=health)
estimated_outputs = dict(zip(OUTPUT_NAMES, result.y_hat_ekf))
state = result.state
P = result.covariance
```

`measurements` may be an eight-element array (NaN for missing channels), or a dictionary with any subset of exact output names. `inputs` is length 4 and `health` length 8; omit health to use nominal multipliers. The caller supplies acquisition and timestamp handling; the module has no hardware-specific DAQ dependency. Do not call `step` concurrently on the same instance.

For a recording:

```python
# times: (n,), inputs: (n,4), measurements: (n,8), health: optional (n,8)
result = engine.run(times, inputs, measurements, health)
# result.state: (n,5); y_hat_ekf: (n,8); covariance: (n,5,5)
```

`run` always starts from the configured prior and does not alter live `step` state. All arrays and full output histories are stored in memory. For very long acquisitions, use `step` and write results incrementally. Repeated `run` calls on separate chunks **restart** the filter.

Initialize direct states from a first valid measurement using `engine.initialize(measurements, inputs, health, infer_airflow=True)`, or CLI `--initialize-from-first`. Missing states retain their configured initial values; missing airflow can be solved from compressor-map compatibility if exactly one root exists. Ambiguous/unreachable map inversions raise instead of choosing a branch silently. P0 remains explicitly configured; if the same first measurement is also used for correction, choose P0 with that initialization dependence in mind.

For independent recordings:

```python
# Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 before starting Python,
# or use an application-owned threadpoolctl context.
engine.warmup(first_inputs)
results = engine.run_many(
    [(times_a, inputs_a, measurements_a), (times_b, inputs_b, measurements_b)],
    workers=2,
)
```

Each recording starts from the same prior/calibration. Use separate instances for different engines or initial conditions. Output order matches stream order. Worker count is explicit, defaults to 1, and should be benchmarked on the target machine. There is no threading inside an individual EKF time sequence.

Local sensor observability is available with `engine.observability(state, inputs, sensors, dt, health=None)`. It returns the rank and singular values of `[H; H Phi; ...; H Phi^4]`, using scaled states and noise-whitened measurements. Evaluate it at multiple operating points; rank 5 at one point is not a global observability guarantee.

## Numerical algorithm and failure behaviour

1. Evaluate Eqs. 6–41 with calibrated maps and health factors.
2. Propagate each timestamp interval with bounded RK4 substeps (Eqs. 62–66).
3. Differentiate RK4 through all four stages by the chain rule; each continuous process Jacobian uses scale-aware central differences (Eqs. 58–59). This implements the discrete derivative required by Eq. 67. Reuse the first-stage Jacobian for process-noise discretization.
4. Evaluate Van Loan's 10×10 block exponential for Qd (Eq. 70), using locally frozen Fc at the beginning of each substep. Accumulate covariance substep by substep. Use the RK4 derivative for Phi, not `I + Fc*dt`.
5. Form measurements/Jacobians, use exact identity rows for directly measured states, and retain only valid channels.
6. Remove individual extreme innovations. If the remaining joint NIS fails, remove the largest marginal normalized residual and retry until accepted or empty. This is a documented heuristic for locating outliers, especially with correlated sensors; chi-squared assumptions apply only approximately after selection.
7. Solve using Cholesky factors, apply the Joseph covariance update, and symmetrize (Eqs. 76–79).
8. Reject a correction that exits physical/map/property bounds, retaining the prediction and reporting `constraint_rejected`. No silent state clipping occurs.

The Van Loan exponential uses float64 degree-13 Padé scaling/squaring and is independently compared with SciPy's matrix exponential and integrated covariance ODEs in validation. No `fastmath`, reduced precision, stale Jacobian caching, or diagonal-only covariance approximation is used.

If prediction or linearization leaves a calibrated domain, the code raises. A live `step` failure leaves its previously committed state, covariance, time and held inputs unchanged. Reduce the step, check input units/calibration and inspect repeated violations. The batch API raises without returning a partial recording. The CLI writes estimates only after a complete successful run.

The fixed-step RK4 method does not adaptively estimate error. `max_step_s` must resolve the engine's fastest dynamics. Tiny floating-point timestamp roundoff within 1e-12 of an integer substep count is tolerated to avoid accidental extra substeps. Large gaps beyond `max_substeps` fail explicitly.

This is the document's reduced-order single-spool operating model. It does not simulate startup at zero rpm, reverse nozzle flow, surge/stall, actuator dynamics, distributed combustor chemistry, leakage networks, or bearing/vibration dynamics. Synthetic correctness and timing checks do not establish real-engine accuracy or hard real-time deadlines.

## Performance and verification

See [BENCHMARKS.md](BENCHMARKS.md) for measured times and the numerical checks. Benchmark generators and artificial calibration are deliberately absent from this runtime repository. Re-measure on your own hardware with your actual maps, timestamps and sensor set; the CLI reports timing on every real recording.

Primary implementation references: the supplied **EKF_2.pdf**, Eqs. 1–98; [SciPy matrix exponential documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html) for the independent numerical reference; [Numba compilation, GIL release and caching documentation](https://numba.readthedocs.io/en/stable/user/jit.html) for native execution behaviour.
