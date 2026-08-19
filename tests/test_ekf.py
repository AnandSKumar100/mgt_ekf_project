import numpy as np
import pytest

from core_ekf import params as P
from core_ekf.state_space import f, h
from core_ekf.jacobians import (
    jacobian_F_loop,
    jacobian_H_loop,
    jacobian_F_vec,
    jacobian_H_vec,
)
from core_ekf.ekf import EKF
from utils.validation import (
    assert_state_shape,
    assert_measurement_shape,
    is_symmetric,
    is_positive_definite,
    check_state_bounds,
)

X0 = P.x0
U0 = np.array([P.Tamb, P.Pamb])


# --------------------------------------------------------------------------
# f(x, u) / h(x) shapes -- 1D and 2D batched
# --------------------------------------------------------------------------

def test_f_shape_1d():
    out = f(X0, U0)
    assert out.shape == (6,)
    assert np.all(np.isfinite(out))


def test_f_shape_2d_batched():
    X = np.tile(X0, (5, 1))
    U = np.tile(U0, (5, 1))
    out = f(X, U)
    assert out.shape == (5, 6)
    assert np.all(np.isfinite(out))


def test_h_shape_1d():
    out = h(X0)
    assert out.shape == (16,)
    assert_measurement_shape(out)
    assert np.all(np.isfinite(out))


def test_h_shape_2d_batched():
    X = np.tile(X0, (5, 1))
    out = h(X)
    assert out.shape == (5, 16)
    assert np.all(np.isfinite(out))


def test_state_shape_assertion():
    assert_state_shape(X0)
    with pytest.raises(ValueError):
        assert_state_shape(np.zeros(5))


# --------------------------------------------------------------------------
# Jacobian shapes
# --------------------------------------------------------------------------

def test_jacobian_F_vec_shape():
    Fj = jacobian_F_vec(X0, U0)
    assert Fj.shape == (6, 6)
    assert np.all(np.isfinite(Fj))


def test_jacobian_H_vec_shape():
    Hj = jacobian_H_vec(X0)
    assert Hj.shape == (16, 6)
    assert np.all(np.isfinite(Hj))


def test_jacobian_F_loop_shape():
    Fj = jacobian_F_loop(X0, U0)
    assert Fj.shape == (6, 6)


def test_jacobian_H_loop_shape():
    Hj = jacobian_H_loop(X0)
    assert Hj.shape == (16, 6)


# --------------------------------------------------------------------------
# Loop vs vectorized numerical equivalence
# --------------------------------------------------------------------------

def test_jacobian_F_loop_matches_vec():
    F_loop = jacobian_F_loop(X0, U0)
    F_vec = jacobian_F_vec(X0, U0)
    assert np.allclose(F_loop, F_vec, atol=1e-5)


def test_jacobian_H_loop_matches_vec():
    H_loop = jacobian_H_loop(X0)
    H_vec = jacobian_H_vec(X0)
    assert np.allclose(H_loop, H_vec, atol=1e-5)


# --------------------------------------------------------------------------
# EKF predict / update behaviour
# --------------------------------------------------------------------------

def test_ekf_predict_preserves_covariance_properties():
    ekf = EKF()
    ekf.predict(U0, P.dt)
    assert ekf.P.shape == (6, 6)
    assert is_symmetric(ekf.P)
    assert is_positive_definite(ekf.P)


def test_ekf_update_preserves_covariance_properties():
    ekf = EKF()
    ekf.predict(U0, P.dt)
    z = h(ekf.x)
    ekf.update(z)
    assert ekf.P.shape == (6, 6)
    assert is_symmetric(ekf.P)
    assert is_positive_definite(ekf.P)


def test_ekf_update_reduces_uncertainty():
    ekf = EKF()
    ekf.predict(U0, P.dt)
    trace_before = np.trace(ekf.P)
    z = h(ekf.x)
    ekf.update(z)
    trace_after = np.trace(ekf.P)
    assert trace_after <= trace_before + 1e-8


def test_ekf_multi_cycle_stability():
    ekf = EKF()
    for _ in range(50):
        ekf.predict(U0, P.dt)
        z = h(ekf.x)
        ekf.update(z)

    assert np.all(np.isfinite(ekf.x))
    assert np.all(np.isfinite(ekf.P))
    assert is_symmetric(ekf.P)
    assert is_positive_definite(ekf.P)


def test_state_bounds_nominal_within_range():
    violations = check_state_bounds(P.x0, P.STATE_BOUNDS)
    assert violations == []
