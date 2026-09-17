"""Project Arka five-state EKF, implementing EKF_2.pdf (September 2026).

Production module and CSV runner. No generated maps, synthetic data or demo mode.
All engine calibration comes from an explicit JSON configuration. See README.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np
from numba import njit
from scipy.optimize import brentq
from scipy.stats import chi2
from threadpoolctl import threadpool_limits

STATE_NAMES = ("N_rpm", "mdot_air", "Pt3_pa", "Tt4_k", "Pt5_pa")
OUTPUT_NAMES = ("N_rpm", "mdot_air", "Pt3_pa", "Tt3_k", "Tt4_k", "Pt5_pa", "Tt5_k", "thrust_n")
INPUT_NAMES = ("mdot_fuel", "Ta_k", "Pa_pa", "M0")
HEALTH_NAMES = ("eta_c", "flow_c", "pi_c", "eta_t", "flow_t", "eta_b", "area_n", "pi_in")
PARAMETER_NAMES = ("J", "V3", "V5", "tau_m", "tau_b", "mdot_ref", "Tref", "Pref", "pi_in", "delta_b", "eta_b", "eta_m", "Wacc", "LHV", "An", "Cd", "Ra", "Rg", "epsilon_pi")
DIRECT = ((0, 0), (1, 1), (2, 2), (4, 3), (5, 4))


@njit(cache=True, nogil=True)
def _interp1(axis, values, point):
    if point < axis[0] or point > axis[-1]:
        raise ValueError("Temperature outside calibrated gas-property domain")
    i = min(max(np.searchsorted(axis, point) - 1, 0), len(axis) - 2)
    w = (point - axis[i]) / (axis[i + 1] - axis[i])
    return values[i] * (1.0 - w) + values[i + 1] * w


@njit(cache=True, nogil=True)
def _interp2(a, b, table, x, y):
    if x < a[0] or x > a[-1] or y < b[0] or y > b[-1]:
        raise ValueError("Operating point outside calibrated component-map domain")
    i = min(max(np.searchsorted(a, x) - 1, 0), len(a) - 2)
    j = min(max(np.searchsorted(b, y) - 1, 0), len(b) - 2)
    s, t = (x - a[i]) / (a[i + 1] - a[i]), (y - b[j]) / (b[j + 1] - b[j])
    return ((1-s) * ((1-t)*table[i,j] + t*table[i,j+1])
            + s * ((1-t)*table[i+1,j] + t*table[i+1,j+1]))


@njit(cache=True, nogil=True)
def _interp2_pair(a, b, first, second, x, y):
    # Share interval lookup and interpolation weights across both map surfaces.
    if x < a[0] or x > a[-1] or y < b[0] or y > b[-1]:
        raise ValueError("Operating point outside calibrated component-map domain")
    i = min(max(np.searchsorted(a, x) - 1, 0), len(a) - 2)
    j = min(max(np.searchsorted(b, y) - 1, 0), len(b) - 2)
    s, t = (x-a[i])/(a[i+1]-a[i]), (y-b[j])/(b[j+1]-b[j])
    w00,w01,w10,w11=(1-s)*(1-t),(1-s)*t,s*(1-t),s*t
    v1=w00*first[i,j]+w01*first[i,j+1]+w10*first[i+1,j]+w11*first[i+1,j+1]
    v2=w00*second[i,j]+w01*second[i,j+1]+w10*second[i+1,j]+w11*second[i+1,j+1]
    return v1,v2


@njit(cache=True, nogil=True)
def _physics(x, u, h, p, maps, gas, pressure_thrust):
    """Equations 6--53, returning f and the canonical eight physical outputs."""
    N, ma, P3, T4, P5 = x
    fuel, Ta, Pa, mach = u
    if not np.all(np.isfinite(x)) or N <= 0 or ma <= 0 or T4 <= 0:
        raise ValueError("Nonphysical state: finite positive speed, airflow and Tt4 required")
    # Stationary inlet equations (6--7) are the scope of this implementation.
    if mach != 0.0:
        raise ValueError("Only stationary operation M0=0 is specified by this inlet model")
    P2 = Pa * p[8] * h[7]
    P4 = P3 * (1.0 - p[9])
    if P3 <= P2 or P4 <= P5 or P5 <= Pa:
        raise ValueError("Pressure constraint violated: Pt3>Pt2, Pt4>Pt5>Pa required")
    cN, cW, cPR, cEta, tN, tER, tW, tEta = maps
    aT, aCp, gT, gCp = gas
    cpa = _interp1(aT, aCp, Ta)
    cpg = _interp1(gT, gCp, T4)
    ga, gg = cpa / (cpa - p[16]), cpg / (cpg - p[17])
    tr = math.sqrt(Ta / p[6])
    nc = N / tr
    # PDF names theta_Wc but omits its application: flow-axis scaling assumption.
    wc = ma * tr / (P2 / p[7]) / h[1]
    pr0, ec0 = _interp2_pair(cN,cW,cPR,cEta,nc,wc)
    pr_map = 1 + (pr0 - 1) * h[2]
    ec = ec0 * h[0]
    er = P4 / P5
    nt = N / math.sqrt(T4 / p[6])
    wt0, et0 = _interp2_pair(tN,tER,tW,tEta,nt,er)
    wt_corr = wt0 * h[4]
    et = et0 * h[3]
    eb = p[10] * h[5]
    if ec <= 0 or ec > 1 or et <= 0 or et > 1 or eb <= 0 or eb > 1 or pr_map <= 1:
        raise ValueError("Invalid health-scaled component efficiency or pressure ratio")
    T3 = Ta * (1 + ((P3/P2)**((ga-1)/ga) - 1) / ec)
    mt = wt_corr * (P4 / p[7]) / math.sqrt(T4 / p[6])
    T5 = T4 * (1 - et * (1 - er**(-(gg-1)/gg)))
    Wc, Wt = ma*cpa*(T3-Ta), mt*cpg*(T4-T5)
    target = (ma*cpa*T3 + fuel*eb*p[13]) / ((ma+fuel)*cpg)
    area = p[14] * h[6]
    critical = (2 / (gg+1))**(gg/(gg-1))
    ratio = Pa/P5
    if ratio <= critical:
        Pe = P5 * critical
        Te = T5 * 2 / (gg+1)
        Ve = math.sqrt(gg*p[17]*Te)
        mn = p[15]*area*P5/math.sqrt(T5)*math.sqrt(gg/p[17])*(2/(gg+1))**((gg+1)/(2*(gg-1)))
    else:
        Pe = Pa
        Te = T5 * ratio**((gg-1)/gg)
        Ve = math.sqrt(2*cpg*(T5-Te))
        mn = p[15]*area*P5/math.sqrt(T5)*math.sqrt(2*gg/(p[17]*(gg-1))*(ratio**(2/gg)-ratio**((gg+1)/gg)))
    thrust = mn*Ve
    if pressure_thrust:
        thrust += (Pe-Pa)*area
    f = np.empty(5)
    f[0] = 900/(math.pi**2) * (p[11]*Wt-Wc-p[12])/(p[0]*N)
    f[1] = p[5]/p[3] * (pr_map-P3/P2)/max(pr_map,p[18])
    f[2] = p[17]*T3/p[1]*(ma+fuel-mt)
    f[3] = (target-T4)/p[4]
    f[4] = p[17]*T5/p[2]*(mt-mn)
    y = np.array([N, ma, P3, T3, T4, P5, T5, thrust])
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(y)):
        raise ValueError("Nonfinite gas-path result")
    return f, y


@njit(cache=True, nogil=True)
def _f(s, u, h, p, maps, gas, pressure_thrust, scale):
    return _physics(s*scale,u,h,p,maps,gas,pressure_thrust)[0]/scale


@njit(cache=True, nogil=True)
def _rk4(s, dt, u, h, p, maps, gas, pt, scale):
    k1 = _f(s,u,h,p,maps,gas,pt,scale)
    k2 = _f(s+dt/2*k1,u,h,p,maps,gas,pt,scale)
    k3 = _f(s+dt/2*k2,u,h,p,maps,gas,pt,scale)
    k4 = _f(s+dt*k3,u,h,p,maps,gas,pt,scale)
    return s + dt/6*(k1+2*k2+2*k3+k4)


@njit(cache=True, nogil=True)
def _jac(s, u, h, p, maps, gas, pt, scale, eps, dt, kind):
    # kind 0: continuous process; 1: entire RK4 map; 2: measurement.
    n = 8 if kind == 2 else 5
    out = np.empty((n,5))
    for j in range(5):
        delta = max(eps[0]*abs(s[j]),eps[j+1])
        a, b = s.copy(), s.copy()
        a[j] += delta
        b[j] -= delta
        if kind == 0:
            fa = _f(a,u,h,p,maps,gas,pt,scale)
            fb = _f(b,u,h,p,maps,gas,pt,scale)
        elif kind == 1:
            fa = _rk4(a,dt,u,h,p,maps,gas,pt,scale)
            fb = _rk4(b,dt,u,h,p,maps,gas,pt,scale)
        else:
            fa = _physics(a*scale,u,h,p,maps,gas,pt)[1]
            fb = _physics(b*scale,u,h,p,maps,gas,pt)[1]
        out[:,j] = (fa-fb)/(2*delta)
    if kind == 2:
        for row, col in ((0,0),(1,1),(2,2),(4,3),(5,4)):
            out[row,:] = 0
            out[row,col] = scale[col]
    return out


@njit(cache=True, nogil=True)
def _expm(a):
    """Float64 degree-13 Pade scaling/squaring for the small Van Loan block."""
    norm = np.max(np.sum(np.abs(a),axis=0))
    if not np.isfinite(norm):
        raise ValueError("Nonfinite Van Loan matrix")
    squarings = max(0,int(math.ceil(math.log2(norm/5.371920351148152)))) if norm > 0 else 0
    if squarings > 50:
        raise ValueError("Excessive Van Loan norm; check units and integration step")
    a = a / 2.0**squarings
    eye = np.eye(a.shape[0])
    a2 = a@a
    a4 = a2@a2
    a6 = a4@a2
    b = (64764752532480000.,32382376266240000.,7771770303897600.,1187353796428800.,129060195264000.,10559470521600.,670442572800.,33522128640.,1323241920.,40840800.,960960.,16380.,182.,1.)
    v = a6@(b[12]*a6+b[10]*a4+b[8]*a2)+b[6]*a6+b[4]*a4+b[2]*a2+b[0]*eye
    u = a@(a6@(b[13]*a6+b[11]*a4+b[9]*a2)+b[7]*a6+b[5]*a4+b[3]*a2+b[1]*eye)
    out = np.linalg.solve(v-u,v+u)
    for _ in range(squarings):
        out = out@out
    return out


@njit(cache=True, nogil=True)
def _van_loan(fc, qc, dt):
    a = np.zeros((10,10))
    a[:5,:5] = fc*dt
    a[:5,5:] = qc*dt
    a[5:,5:] = -fc.T*dt
    e = _expm(a)
    phi = e[:5,:5].copy()
    qd = e[:5,5:].copy()@phi.T
    return phi, (qd+qd.T)*0.5


@njit(cache=True, nogil=True)
def _rk4_transition(s, dt, u, h, p, maps, gas, pt, scale, eps):
    """Differentiate the complete RK4 map by its stage chain rule (Eq. 67).

    Continuous Jacobians use Eq. 58; the first is reused for Van Loan.
    This saves ten process evaluations per substep versus separate RK4-map FD.
    """
    eye=np.eye(5)
    k1=_f(s,u,h,p,maps,gas,pt,scale)
    a1=_jac(s,u,h,p,maps,gas,pt,scale,eps,dt,0)
    s2=s+dt/2*k1
    k2=_f(s2,u,h,p,maps,gas,pt,scale)
    a2=_jac(s2,u,h,p,maps,gas,pt,scale,eps,dt,0)
    b2=a2@(eye+dt/2*a1)
    s3=s+dt/2*k2
    k3=_f(s3,u,h,p,maps,gas,pt,scale)
    a3=_jac(s3,u,h,p,maps,gas,pt,scale,eps,dt,0)
    b3=a3@(eye+dt/2*b2)
    s4=s+dt*k3
    k4=_f(s4,u,h,p,maps,gas,pt,scale)
    a4=_jac(s4,u,h,p,maps,gas,pt,scale,eps,dt,0)
    b4=a4@(eye+dt*b3)
    return s+dt/6*(k1+2*k2+2*k3+k4),eye+dt/6*(a1+2*b2+2*b3+b4),a1


@njit(cache=True, nogil=True)
def _predict(s, cov, dt, u, h, p, maps, gas, pt, scale, eps, qc, max_step, max_substeps):
    count = max(1,int(math.ceil(dt/max_step - 1e-12))) if dt > 0 else 0
    if count > max_substeps:
        raise ValueError("Timestamp gap exceeds max_substeps; process intermediate inputs")
    if count == 0:
        return s.copy(), cov.copy()
    d = dt/count
    for _ in range(count):
        s, phi, fc = _rk4_transition(s,d,u,h,p,maps,gas,pt,scale,eps)
        _, qd = _van_loan(fc,qc,d)
        cov = phi@cov@phi.T + qd
        cov = (cov+cov.T)*0.5
        _physics(s*scale,u,h,p,maps,gas,pt)
    return s, cov


@njit(cache=True, nogil=True)
def _update(s, cov, z, u, h, p, maps, gas, pt, scale, eps, r, gates, sigma_gate):
    y = _physics(s*scale,u,h,p,maps,gas,pt)[1]
    used = np.isfinite(z)
    for j in range(7):
        if z[j] <= 0:
            used[j] = False
    rejected = np.zeros(8, dtype=np.bool_)
    if not np.any(used):
        return s, cov, y, used, rejected, np.nan, False
    Hfull = _jac(s,u,h,p,maps,gas,pt,scale,eps,0.,2)
    nis = np.nan
    while np.any(used):
        idx = np.where(used)[0]
        H = Hfull[idx,:].copy()
        R = r[idx,:][:,idx].copy()
        innovation = z[idx]-y[idx]
        # Whiten by sensor standard deviations, avoiding mixed-unit conditioning.
        std = np.sqrt(np.diag(R))
        H = H/std.reshape((-1,1))
        innovation = innovation/std
        R = R/std.reshape((-1,1))/std.reshape((1,-1))
        B = cov@H.T
        S = H@B + R
        S = (S+S.T)*0.5
        L = np.linalg.cholesky(S)
        whitened = np.linalg.solve(L,innovation)
        nis = whitened@whitened
        standardized = np.abs(innovation)/np.sqrt(np.diag(S))
        bad = standardized > sigma_gate
        if np.any(bad):
            for j in range(len(idx)):
                if bad[j]:
                    used[idx[j]], rejected[idx[j]] = False, True
            continue
        if nis > gates[len(idx)]:
            worst = idx[np.argmax(standardized)]
            used[worst], rejected[worst] = False, True
            continue
        gain = np.linalg.solve(L.T, np.linalg.solve(L,B.T)).T
        candidate = s+gain@innovation
        # Fail closed on an unphysical correction; retain valid prediction.
        try:
            candidate_y = _physics(candidate*scale,u,h,p,maps,gas,pt)[1]
        except Exception:
            rejected |= used
            used[:] = False
            return s,cov,y,used,rejected,nis,True
        a = np.eye(5)-gain@H
        posterior = a@cov@a.T+gain@R@gain.T
        return candidate,(posterior+posterior.T)*0.5,candidate_y,used,rejected,nis,False
    return s,cov,y,used,rejected,nis,False


@njit(cache=True, nogil=True)
def _series(s, cov, times, inputs, health, measurements, p, maps, gas, pt, scale, eps, qc, r, gates, sigma_gate, max_step, max_substeps):
    n = len(times)
    states, outputs, covariances = np.empty((n,5)), np.empty((n,8)), np.empty((n,5,5))
    masks, rejected, nis, constrained = np.empty((n,8),np.bool_), np.empty((n,8),np.bool_), np.empty(n), np.empty(n,np.bool_)
    for k in range(n):
        if k > 0:
            s,cov = _predict(s,cov,times[k]-times[k-1],inputs[k-1],health[k-1],p,maps,gas,pt,scale,eps,qc,max_step,max_substeps)
        s,cov,y,mask,rej,score,con = _update(s,cov,measurements[k],inputs[k],health[k],p,maps,gas,pt,scale,eps,r,gates,sigma_gate)
        if not np.all(np.isfinite(cov)):
            raise ValueError("Nonfinite state covariance")
        states[k],outputs[k],covariances[k] = s*scale,y,cov*scale.reshape((5,1))*scale.reshape((1,5))
        masks[k],rejected[k],nis[k],constrained[k] = mask,rej,score,con
    return states,outputs,covariances,masks,rejected,nis,constrained


def _array(value, shape, name):
    a = np.array(value,dtype=np.float64,order="C",copy=True)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return a


def _covariance(value, n, name, definite):
    a = _array(value,(n,n),name)
    if not np.allclose(a,a.T,rtol=1e-12,atol=1e-14):
        raise ValueError(f"{name} must be symmetric")
    diag = np.diag(a)
    if np.any(diag < 0) or (definite and np.any(diag <= 0)):
        raise ValueError(f"{name} has invalid diagonal")
    scale = np.sqrt(np.where(diag>0,diag,1.0))
    normalized = a/scale[:,None]/scale[None,:]
    vals = np.linalg.eigvalsh(normalized)
    if np.min(vals) < -1e-12 or (definite and np.min(vals) <= 0):
        raise ValueError(f"{name} must be positive {'definite' if definite else 'semidefinite'}")
    return (a+a.T)*0.5


def _axis(value,name):
    a = np.array(value,dtype=np.float64,order="C",copy=True)
    if a.ndim != 1 or len(a)<2 or not np.all(np.isfinite(a)) or np.any(np.diff(a)<=0):
        raise ValueError(f"{name} must contain at least two strictly increasing finite knots")
    return a


@dataclass(frozen=True)
class Result:
    """Arrays use SI units except N in rpm. nis is the last evaluated subset NIS."""
    state: np.ndarray
    y_hat_ekf: np.ndarray
    covariance: np.ndarray
    used: np.ndarray
    rejected: np.ndarray
    nis: np.ndarray | float
    constraint_rejected: np.ndarray | bool


class ProjectArkaEKF:
    """Configured, float64 EKF. Use one instance per live engine stream.

    run() starts a fresh sequence from the configured initial prior. step()
    maintains a live posterior and is transactional on numerical/model errors.
    No process-wide thread settings are changed when importing this module.
    """

    def __init__(self, config: Mapping):
        if config.get("schema_version") != 1:
            raise ValueError("schema_version must be 1")
        par = config["parameters"]
        self.p = _array([par[k] for k in PARAMETER_NAMES],(19,),"parameters")
        for i in (0,1,2,3,4,5,6,7,8,10,11,13,14,15,16,17,18):
            if self.p[i] <= 0:
                raise ValueError(f"{PARAMETER_NAMES[i]} must be positive")
        if not 0 <= self.p[9] < 1 or self.p[12] < 0 or any(self.p[i]>1 for i in (8,10,11,15)):
            raise ValueError("Require delta_b in [0,1), Wacc>=0 and efficiencies/recovery/Cd<=1")
        m = config["maps"]
        c,t = m["compressor"],m["turbine"]
        cn,cw,tn,te = _axis(c["speed"],"compressor speed"),_axis(c["flow"],"compressor flow"),_axis(t["speed"],"turbine speed"),_axis(t["expansion_ratio"],"turbine expansion ratio")
        cpr,ce = _array(c["pressure_ratio"],(len(cn),len(cw)),"compressor pressure_ratio"),_array(c["efficiency"],(len(cn),len(cw)),"compressor efficiency")
        tw,et = _array(t["flow"],(len(tn),len(te)),"turbine flow"),_array(t["efficiency"],(len(tn),len(te)),"turbine efficiency")
        if np.any(cpr<=1) or np.any(tw<=0) or np.any(ce<=0) or np.any(ce>1) or np.any(et<=0) or np.any(et>1):
            raise ValueError("Invalid component-map pressure ratios, flows or efficiencies")
        if min(cn[0],cw[0],tn[0])<=0 or te[0]<=1:
            raise ValueError("Map speed and flow axes must be positive and expansion ratios >1")
        self.maps = (cn,cw,cpr,ce,tn,te,tw,et)
        g = config["gas_properties"]
        at,gt = _axis(g["air"]["temperature"],"air temperature"),_axis(g["gas"]["temperature"],"gas temperature")
        ac,gc = _array(g["air"]["cp"],at.shape,"air cp"),_array(g["gas"]["cp"],gt.shape,"gas cp")
        if at[0]<=0 or gt[0]<=0 or np.any(ac<=self.p[16]) or np.any(gc<=self.p[17]):
            raise ValueError("Require T>0 and cp>R in gas-property tables")
        self.gas = (at,ac,gt,gc)
        filt = config["filter"]
        self.scale = _array(filt["state_scale"],(5,),"state_scale")
        if np.any(self.scale<=0):
            raise ValueError("state_scale must be positive")
        self.x0 = _array(filt["initial_state"],(5,),"initial_state")
        self.P0 = _covariance(filt["initial_covariance"],5,"initial_covariance",False)
        self.qc = _covariance(filt["Qc"],5,"Qc",False)/self.scale[:,None]/self.scale[None,:]
        self.r = _covariance(filt["R"],8,"R",True)
        absolute = _array(filt.get("fd_absolute",self.scale*1e-6),(5,),"fd_absolute")
        relative = float(filt.get("fd_relative",1e-6))
        if np.any(absolute<=0) or not np.isfinite(relative) or relative<=0:
            raise ValueError("Finite-difference perturbations must be positive")
        self.eps = np.r_[relative,absolute/self.scale]
        self.max_step = float(filt["max_step_s"])
        raw_limit = filt.get("max_substeps",10000)
        self.max_substeps = int(raw_limit)
        self.sigma_gate = float(filt.get("sigma_gate",5.0))
        probability = float(filt.get("nis_probability",0.999))
        if not np.isfinite(self.max_step) or self.max_step<=0 or self.max_substeps<1 or self.max_substeps!=raw_limit:
            raise ValueError("Invalid integration step or substep limit")
        if not np.isfinite(self.sigma_gate) or self.sigma_gate<=0 or not 0<probability<1:
            raise ValueError("Invalid measurement gate")
        self.gates = np.r_[0.,chi2.ppf(probability,np.arange(1,9))]
        convention = config["thrust_convention"]
        if convention not in ("momentum","pressure"):
            raise ValueError("thrust_convention must be momentum or pressure")
        self.pt = convention == "pressure"
        self._s,self._cov = self.x0/self.scale,self.P0/self.scale[:,None]/self.scale[None,:]
        self._time,self._u,self._health = None,None,None

    @classmethod
    def from_json(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def _args(self):
        return (self.p,self.maps,self.gas,self.pt,self.scale,self.eps,self.qc,self.r,self.gates,self.sigma_gate,self.max_step,self.max_substeps)

    @staticmethod
    def _validate_inputs(inputs,health):
        if not np.all(np.isfinite(inputs)) or np.any(inputs[...,0]<0) or np.any(inputs[...,1:3]<=0) or np.any(inputs[...,3]!=0):
            raise ValueError("Inputs require fuel>=0, Ta>0, Pa>0, M0=0, all finite")
        if not np.all(np.isfinite(health)) or np.any(health<=0):
            raise ValueError("Health factors must be finite and positive")

    def evaluate(self,state,inputs,health=None):
        """Return continuous derivative f and physical outputs y at a valid point."""
        x,u,h = _array(state,(5,),"state"),_array(inputs,(4,),"inputs"),_array(np.ones(8) if health is None else health,(8,),"health")
        self._validate_inputs(u,h)
        if self.p[8]*h[7]>1:
            raise ValueError("Stationary inlet pressure recovery must be <=1")
        return _physics(x,u,h,self.p,self.maps,self.gas,self.pt)

    def initialize(self,measurements,inputs,health=None,infer_airflow=False):
        """Reset from first direct measurements; missing states retain configured prior.

        Optional map-compatible airflow requires exactly one bracketed root over
        calibrated compressor flow knots. P0 remains the configured covariance.
        This initializes a prior; step/run may subsequently correct it at t0.
        """
        z = self._measurement(measurements)
        x = self.x0.copy()
        for row,col in DIRECT:
            if np.isfinite(z[row]) and z[row]>0:
                x[col] = z[row]
        u = _array(inputs,(4,),"inputs")
        h = _array(np.ones(8) if health is None else health,(8,),"health")
        self._validate_inputs(u,h)
        if infer_airflow and not (np.isfinite(z[1]) and z[1]>0):
            p2 = u[2]*self.p[8]*h[7]
            tr = math.sqrt(u[1]/self.p[6])
            nc = x[0]/tr
            def residual(w):
                return 1+(_interp2(self.maps[0],self.maps[1],self.maps[2],nc,w)-1)*h[2]-x[2]/p2
            roots=[]
            for a,b in zip(self.maps[1][:-1],self.maps[1][1:]):
                ra,rb = residual(a),residual(b)
                if ra == 0:
                    roots.append(a)
                if ra*rb < 0:
                    roots.append(brentq(residual,a,b))
                if rb == 0:
                    roots.append(b)
            roots=np.unique(roots)
            if len(roots)!=1:
                raise ValueError("Airflow initialization needs one unique map-compatible root; supply mdot_air")
            x[1]=roots[0]*h[1]*(p2/self.p[7])/tr
        self.evaluate(x,u,h)
        self.x0 = x
        self.reset()
        return x.copy()

    def reset(self):
        self._s = self.x0/self.scale
        self._cov = self.P0/self.scale[:,None]/self.scale[None,:]
        self._time,self._u,self._health = None,None,None

    @staticmethod
    def _measurement(measurements):
        if isinstance(measurements,Mapping):
            unknown=set(measurements)-set(OUTPUT_NAMES)
            if unknown:
                raise ValueError(f"Unknown sensor names: {sorted(unknown)}")
            measurements=[measurements.get(k,np.nan) for k in OUTPUT_NAMES]
        z=np.array(measurements,dtype=np.float64,copy=True)
        if z.shape!=(8,):
            raise ValueError("Measurements must have eight canonical entries or be a name/value mapping")
        return z

    def step(self,timestamp,inputs,measurements,health=None):
        """Advance one timestamp. Previous inputs/health are held during prediction.

        First call corrects the configured prior at timestamp (no propagation).
        Invalid measurements are omitted. Domain/numerical failures raise and
        leave the previous posterior, timestamp and held inputs unchanged.
        """
        now=float(timestamp)
        if not np.isfinite(now) or (self._time is not None and now<=self._time):
            raise ValueError("Timestamps must be finite and strictly increasing")
        u=_array(inputs,(4,),"inputs")
        h=_array(np.ones(8) if health is None else health,(8,),"health")
        self._validate_inputs(u,h)
        if self.p[8]*h[7]>1:
            raise ValueError("Stationary inlet pressure recovery must be <=1")
        z=self._measurement(measurements)
        s,cov=self._s.copy(),self._cov.copy()
        if self._time is not None:
            s,cov=_predict(s,cov,now-self._time,self._u,self._health,self.p,self.maps,self.gas,self.pt,self.scale,self.eps,self.qc,self.max_step,self.max_substeps)
        s,cov,y,used,rej,nis,con=_update(s,cov,z,u,h,self.p,self.maps,self.gas,self.pt,self.scale,self.eps,self.r,self.gates,self.sigma_gate)
        if not np.all(np.isfinite(cov)):
            raise ValueError("Nonfinite covariance")
        self._s,self._cov,self._time,self._u,self._health=s,cov,now,u,h
        return Result(s*self.scale,y,cov*self.scale[:,None]*self.scale[None,:],used,rej,float(nis),bool(con))

    def run(self,times,inputs,measurements,health=None):
        """Filter a complete recorded stream from x0/P0 in a compiled serial loop.

        Does not mutate live step() state. Outputs include full covariances.
        Inputs/health are zero-order held from row k-1 to row k.
        """
        t=np.array(times,dtype=np.float64,order="C",copy=True)
        if t.ndim!=1 or len(t)==0 or not np.all(np.isfinite(t)) or np.any(np.diff(t)<=0):
            raise ValueError("times must be nonempty, finite and strictly increasing")
        n=len(t)
        u=_array(inputs,(n,4),"inputs")
        h=_array(np.ones((n,8)) if health is None else health,(n,8),"health")
        z=np.array(measurements,dtype=np.float64,order="C",copy=True)
        if z.shape!=(n,8):
            raise ValueError("measurements must have shape (samples,8), with NaN for missing sensors")
        self._validate_inputs(u,h)
        if np.any(self.p[8]*h[:,7]>1):
            raise ValueError("Stationary inlet pressure recovery must be <=1")
        arrays=_series(self.x0/self.scale,self.P0/self.scale[:,None]/self.scale[None,:],t,u,h,z,*self._args())
        return Result(*arrays)

    def run_many(self, streams, workers=1):
        """Run independent recordings concurrently, preserving input order.

        Each stream is a (times, inputs, measurements[, health]) tuple and starts
        from the same configured prior. No timestep parallelization is performed.
        Warm up first. For different engines/calibrations use separate instances.
        """
        if isinstance(workers,bool) or not isinstance(workers,int) or workers<1:
            raise ValueError("workers must be a positive integer")
        streams=list(streams)
        if workers==1 or len(streams)<2:
            return [self.run(*stream) for stream in streams]
        with ThreadPoolExecutor(max_workers=min(workers,len(streams))) as executor:
            return list(executor.map(lambda stream:self.run(*stream),streams))

    def warmup(self,inputs,health=None):
        """Compile/load kernels before acquisition, using the configured real prior.

        Validates one short prediction. Does not alter the filter or synthesize data.
        """
        u=_array(inputs,(4,),"inputs")
        h=_array(np.ones(8) if health is None else health,(8,),"health")
        _,y=self.evaluate(self.x0,u,h)
        _predict(self.x0/self.scale,self.P0/self.scale[:,None]/self.scale[None,:],min(self.max_step,1e-6),u,h,self.p,self.maps,self.gas,self.pt,self.scale,self.eps,self.qc,self.max_step,self.max_substeps)
        _update(self.x0/self.scale,self.P0/self.scale[:,None]/self.scale[None,:],y,u,h,self.p,self.maps,self.gas,self.pt,self.scale,self.eps,self.r,self.gates,self.sigma_gate)
        self.run([0.,min(self.max_step,1e-6)],np.tile(u,(2,1)),np.full((2,8),np.nan),np.tile(h,(2,1)))

    def observability(self,state,inputs,sensors,dt,health=None):
        """Frozen, local rank/singular values with state and sensor-noise scaling.

        This is a local diagnostic, not proof of nonlinear global observability.
        """
        x=_array(state,(5,),"state")
        u=_array(inputs,(4,),"inputs")
        h=_array(np.ones(8) if health is None else health,(8,),"health")
        self.evaluate(x,u,h)
        if not np.isfinite(dt) or not 0<dt<=self.max_step:
            raise ValueError("observability dt must be in (0,max_step_s]")
        names=list(sensors)
        if not names or len(set(names))!=len(names) or any(k not in OUTPUT_NAMES for k in names):
            raise ValueError("Supply a nonempty set of valid sensor names")
        idx=np.array([OUTPUT_NAMES.index(k) for k in names])
        s=x/self.scale
        phi=_jac(s,u,h,self.p,self.maps,self.gas,self.pt,self.scale,self.eps,dt,1)
        H=_jac(s,u,h,self.p,self.maps,self.gas,self.pt,self.scale,self.eps,dt,2)[idx]
        H=np.linalg.solve(np.linalg.cholesky(self.r[np.ix_(idx,idx)]),H)
        blocks=[]
        for _ in range(5):
            blocks.append(H)
            H=H@phi
        O=np.vstack(blocks)
        values=np.linalg.svd(O,compute_uv=False)
        return {"rank":int(np.linalg.matrix_rank(O)),"singular_values":values,"state_dimension":5}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True,type=Path,help="Calibrated engine/filter JSON")
    parser.add_argument("--input",required=True,type=Path,help="Timestamped real sensor CSV")
    parser.add_argument("--output",required=True,type=Path,help="Output estimates/uncertainties CSV")
    parser.add_argument("--initialize-from-first",action="store_true",help="Use valid first-row direct sensors and infer missing airflow from map")
    args=parser.parse_args()
    try:
        if args.output.resolve() in (args.input.resolve(),args.config.resolve()):
            raise ValueError("Output must differ from input and configuration")
        engine=ProjectArkaEKF.from_json(args.config)
        start=perf_counter()
        with args.input.open(newline="",encoding="utf-8-sig") as stream:
            reader=csv.DictReader(stream)
            required=("timestamp_s",)+INPUT_NAMES[:3]
            if not reader.fieldnames or not set(required)<=set(reader.fieldnames):
                raise ValueError(f"CSV requires {required}; M0 defaults to zero")
            times,inputs,measurements,health=[],[],[],[]
            for line,row in enumerate(reader,2):
                try:
                    times.append(float(row["timestamp_s"]))
                    inputs.append([float(row[k]) for k in INPUT_NAMES[:3]]+[float(row.get("M0") or 0)])
                    measurements.append([float(row[k]) if row.get(k) else np.nan for k in OUTPUT_NAMES])
                    health.append([float(row["theta_"+k]) if row.get("theta_"+k) else 1.0 for k in HEALTH_NAMES])
                except (TypeError,ValueError) as error:
                    raise ValueError(f"CSV line {line}: {error}") from error
        if not times:
            raise ValueError("Input CSV contains no samples")
        if args.initialize_from_first:
            engine.initialize(measurements[0],inputs[0],health[0],infer_airflow=True)
        load_s=perf_counter()-start
        start=perf_counter()
        with threadpool_limits(limits=1,user_api="blas"):
            engine.warmup(inputs[0],health[0])
        warmup_s=perf_counter()-start
        start=perf_counter()
        with threadpool_limits(limits=1,user_api="blas"):
            result=engine.run(times,inputs,measurements,health)
        filter_s=perf_counter()-start
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("w",newline="",encoding="utf-8") as stream:
            writer=csv.writer(stream)
            writer.writerow(["timestamp_s",*OUTPUT_NAMES,*["std_"+k for k in STATE_NAMES],"nis","used_mask","rejected_mask","constraint_rejected"])
            for k,t in enumerate(times):
                std=np.sqrt(np.maximum(np.diag(result.covariance[k]),0))
                used=sum(1<<i for i in range(8) if result.used[k,i])
                rejected=sum(1<<i for i in range(8) if result.rejected[k,i])
                writer.writerow([t,*result.y_hat_ekf[k],*std,result.nis[k],used,rejected,int(result.constraint_rejected[k])])
        print(json.dumps({"samples":len(times),"csv_load_s":load_s,"warmup_s":warmup_s,"filter_s":filter_s,"mean_us_per_sample":filter_s/len(times)*1e6,"samples_per_second":len(times)/filter_s,"constraint_rejections":int(np.sum(result.constraint_rejected)),"output":str(args.output)},indent=2))
    except (ValueError,KeyError,TypeError,OSError,np.linalg.LinAlgError) as error:
        parser.exit(2,f"EKF error: {error}\n")


if __name__=="__main__":
    main()
