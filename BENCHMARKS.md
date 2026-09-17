# Benchmark and validation results

Measured during the implementation session on 17 September 2026. These are actual measurements on synthetic verification data, not estimated timings or physical-engine validation. No generated data or synthetic calibration is included in the production code.

## Environment

- AMD EPYC 9V74 host, x86-64 Linux/KVM; 9 logical CPUs visible, cgroup CPU quota equivalent to 8 cores.
- CPython 3.12.14; NumPy 2.3.5; SciPy 1.17.0; Numba 0.67.0; llvmlite 0.49.0; threadpoolctl 3.6.0.
- NumPy/SciPy OpenBLAS 0.3.30, one BLAS thread unless explicitly stated; `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`.
- Shared virtualized runtime, no real-time scheduling or CPU affinity; wall-clock measurements can include scheduling delays.

## Workload and method

The timing workload uses all eight sensors, float64, diagonal Qc/R, a 9×11 compressor map and an 8×10 turbine map. The initial synthetic operating point is 100,000 rpm, 0.2 kg/s air, Pt3=300 kPa, Tt4=1100 K and Pt5=140 kPa, with Ta=300 K and Pa=100 kPa. Synthetic parameters are constructed to balance shaft power and both volume flows. This point is not a calibration recommendation.

The 1,001-point verification recording spans one second at 1 ms spacing. Fuel is perturbed sinusoidally by ±1.5% with a 0.25 s period; sensor noise uses seed 20260917. Sensor standard deviations are [80 rpm, 0.002 kg/s, 250 Pa, 1.5 K, 2 K, 250 Pa, 2 K, 0.6 N]. Truth is integrated independently with SciPy DOP853, rtol=atol=1e-10, holding inputs over each timestamp interval. This tests numerical implementation and estimation on an internally consistent model, not model mismatch or experimental accuracy.

Timing uses the first 100/1,000 points. The 10,000-point stress workload repeats the first 1,000 input/measurement rows at monotonically increasing timestamps. It is a throughput workload, not a ten-second independent accuracy experiment. The final maximum substep is 1 ms. Each run starts from the same prior. Data generation, compilation, and CSV I/O are outside the timed region; input copies, validation, filtering and full output allocation are inside it.

Seven timed repetitions follow warm-up and an untimed run; tables show medians. Parallel and BLAS comparisons use five repetitions. No other benchmark or compilation job ran concurrently with these timed experiments.

## Optimization progression

| Implementation | 100 samples | 1,000 samples |
|---|---:|---:|
| Initial Python/NumPy, JIT disabled | 526.329 ms | 5853.112 ms |
| Initial implementation, compiled | 8.811 ms | 101.872 ms |
| Final optimized code, JIT disabled | 223.972 ms | 2387.710 ms |
| Final optimized code, compiled | 5.173 ms | 45.742 ms |

At 1,000 samples, the complete optimization is **128.0×** faster than the initial uncompiled implementation. With the **same final source**, compilation alone is **52.2×** faster. The additional changes after initial compilation improve that workload by **2.23×**.

The initial implementation differentiated the entire RK4 map using repeated finite differences. The final implementation differentiates the four RK4 stages by the chain rule with finite-difference continuous Jacobians, reuses the first-stage Jacobian for Van Loan, shares interpolation weights between each pair of map surfaces, and prevents timestamp roundoff from spuriously doubling the number of RK4 substeps. The total speedup includes all these changes; it is not a compiler-only claim.

Initial Python profiling placed about 89% of cumulative time inside gas-path evaluations (including map lookups), which motivated these changes. The profile itself was not used for the reported wall-clock timings.

## Final batch throughput

| Samples | Median total | Range across seven runs | Mean per sample from median | Throughput |
|---:|---:|---:|---:|---:|
| 100 | 5.173 ms | 4.302–7.340 ms | 51.73 µs | 19,329 samples/s |
| 1,000 | 45.742 ms | 42.855–54.921 ms | 45.74 µs | 21,862 samples/s |
| 10,000 | 497.475 ms | 473.495–539.470 ms | 49.75 µs | 20,101 samples/s |

Raw timed repetitions (seconds):

| Samples | Runs 1–7 |
|---:|---|
| 100 | 0.004673263, 0.004301654, 0.006005067, 0.005795047, 0.005173440, 0.004649728, 0.007340437 |
| 1,000 | 0.054921134, 0.045742045, 0.042855157, 0.045551053, 0.045968500, 0.044429610, 0.045859589 |
| 10,000 | 0.506312544, 0.539469844, 0.498526021, 0.490012987, 0.473494755, 0.495973361, 0.497475337 |

## Live `step()` latency

Measured over the remaining 1,000 calls after the initial timestamp, with warmed kernels. Includes the Python API, input validation and result construction.

| Percentile | Latency |
|---|---:|
| 50 | 83.63 µs |
| 95 | 168.78 µs |
| 99 | 2219.30 µs |
| 100 | 2306.02 µs |

**Do not interpret average throughput as a guaranteed deadline.** Scheduling delays caused the upper-tail latency to exceed 1 ms. Hardware acquisition, application callbacks and communication are not included. For hard real-time deployment, measure worst-case timing on the target platform and use an appropriate execution environment.

## Parallelization decision

Successive timesteps depend on the previous posterior and remain serial. Parallelizing a single small covariance solve adds coordination overhead without removing that dependency. Increasing BLAS threads did not improve this workload materially:

| BLAS threads | 1,000-sample median |
|---:|---:|
| 1 | 45.385 ms |
| 2 | 47.053 ms |
| 4 | 45.806 ms |

Independent engines or recordings can run concurrently because native kernels release the GIL. Four independent 10,000-sample recordings, identical calibration, one BLAS thread per worker:

| Workers | Total wall time for all 40,000 samples | Speedup vs 1 worker |
|---:|---:|---:|
| 1 | 1.9535 s | 1.00× |
| 2 | 1.1091 s | 1.76× |
| 4 | 0.6496 s | 3.01× |

Consequently the single-stream path stays serial. The production `run_many(..., workers=N)` API provides optional concurrency across recordings; four workers gave the best result among 1, 2 and 4 on this host. The production wrapper was separately checked for output-order preservation and serial/parallel equality. The parallel timing used a pre-created ThreadPoolExecutor around four `run()` calls, excluding executor creation; `run_many()` also creates/destroys the pool once per invocation.

This is the best measured configuration among these experiments, not proof of an absolute performance maximum on every platform or workload.

## Startup and CLI

Cached warm-up in the benchmark process: **0.293 s**, excluding Python import/configuration. A separate CLI smoke test processed 1,001 samples in **43.311 ms**, with **0.397 s** cached warm-up and **9.978 ms** CSV loading. CSV output matched the Python API.
Fresh-cache warm-up: **35.608 s**, excluding imports/configuration. Call `warmup()` before acquisition; compilation must not occur on a deadline-sensitive path.

## Correctness checks completed

1. Steady-state shaft-power, combustor-energy and both mass-flow balances; canonical output ordering.
2. 160 small matrix exponentials compared against `scipy.linalg.expm`, including scaling/squaring cases. Maximum relative Frobenius error: **2.436e-12**.
3. 30 random Van Loan process covariances compared against an independently integrated continuous Lyapunov ODE, plus zero-time and PSD checks.
4. RK4 convergence against DOP853; reducing step size demonstrated fourth-order convergence.
5. Complete discrete-map directional derivative and 20 operating-point comparisons of the optimized stage-chain derivative with direct RK4 finite differences.
6. Choked/unchoked nozzle continuity and pressure-thrust difference against the explicit pressure-area term.
7. Invalid input, covariance and map-domain failures; no extrapolation.
8. Missing sensors and gross-outlier removal; correlated-R update compared with an independent linear-Gaussian solution.
9. Unique map-based airflow initialization and rank-five local observability with direct-state sensors.
10. 1,001-sample transient with perturbed prior, 60 consecutive missing-observation rows and a 100 kPa pressure outlier: covariance symmetry/PSD and live/batch agreement.
11. Physically invalid posterior correction rejected while retaining prediction.
12. Failed prediction/timestamp gap leaves live state, covariance, timestamp and held inputs unchanged.
13. Four-stream production parallel API returns ordered outputs identical to serial execution.
14. End-to-end configuration + CSV CLI run matches the Python API.

State RMSE for the synthetic transient/dropout/outlier validation (not experimental accuracy):

| State | RMSE |
|---|---:|
| N (rpm) | 9.1736049 |
| Airflow (kg/s) | 0.00010015082 |
| Pt3 (Pa) | 31.565371 |
| Tt4 (K) | 0.13809997 |
| Pt5 (Pa) | 15.397911 |

The final JIT and JIT-disabled final-source runs had **0.0e+00** maximum scaled state difference and **5.790e-22** maximum scaled covariance difference. Relative to the initial Python version, the final version's maximum scaled state difference was **8.027e-11**, and the maximum scaled covariance difference was **1.667e-14**. Used/rejected sensor masks agreed.

Scaling here divides each state error by [100000, 0.2, 300000, 1000, 140000], and each covariance entry by the corresponding pair of scales. These comparisons support numerical equivalence for the tested workload; they are not a guarantee across all possible engine calibrations.

## Re-measuring with your real data

Use the README command on the actual engine configuration and recording. Run once to populate the compilation cache, then repeat the same command at least seven times. Compare the reported `filter_s`, keeping dataset, parameters, thread settings and machine fixed. For a same-source uncompiled reference, set `NUMBA_DISABLE_JIT=1` before launching Python (Linux/macOS example):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_DISABLE_JIT=1 python arka_ekf.py --config engine_config.json --input measurements.csv --output estimates.csv
```

Record startup separately. Benchmark the live `step()` path separately if it will be used in acquisition. For throughput, use `run`; use `run_many` only for independent streams. Synthetic generators were used only for the implementation-session checks and were intentionally excluded from this repository.

## Source identity

`arka_ekf.py` SHA-256: `ac283e6bf71772ce924aacf09e515ff438204309381304d8e342196f4e8fc2d2`.
