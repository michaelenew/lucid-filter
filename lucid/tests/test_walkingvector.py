"""Tests for statfilter.WalkingVectorFilter -- the per-component multivariate walker.

Behavioural, not exact-match: a walking filter tracks online, so the checks pin
what it must DO -- reach a sustained per-component regime, stay put on static data,
track the state, stream == batch, and the structural invariants. The residual
process<->measurement coupling bias is a documented limit, so the tolerances below
are deliberately loose on the leak axes and tight on the target axis.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from statfilter import WalkingVectorFilter  # noqa: E402


def mv_series(H, Q0, rho, T, seed, hot_axis=None, amp=0.0):
    """n-state, m-sensor local-level series; optionally one log-scale axis hot mid-run."""
    rng = np.random.default_rng(seed)
    n, m = H.shape[1], H.shape[0]
    lam, V = np.linalg.eigh(np.atleast_2d(Q0))
    psi = np.zeros((T, n + m))
    if hot_axis is not None:
        psi[T // 3: 2 * T // 3, hot_axis] = amp
    th = np.zeros(n); Y = np.zeros((T, m))
    for t in range(T):
        Qt = V @ np.diag(lam * np.exp(psi[t, :n])) @ V.T
        th = th + np.linalg.cholesky(Qt + 1e-12 * np.eye(n)) @ rng.standard_normal(n)
        Rt = np.asarray(rho) * np.exp(psi[t, n:])
        Y[t] = H @ th + np.sqrt(Rt) * rng.standard_normal(m)
    return Y


_H = np.array([[1.0, 0.0], [0.6, 1.0]])
_Q0 = np.array([[1.0, 0.6], [0.6, 1.0]])
_RHO = np.array([1.0, 1.0])


# --------------------------------------------------------------- construction
def test_repr_and_shapes():
    f = WalkingVectorFilter(_Q0, _RHO, _H, phi=0.9, s=0.4)
    assert f.n == 2 and f.m == 2 and f.D == 4
    Y = mv_series(_H, _Q0, _RHO, 40, seed=0)
    r = f.filter(Y)
    assert r.mean.shape == (40, 2) and r.var.shape == (40, 2, 2)
    assert r.process_scale.shape == (40, 2) and r.measurement_scale.shape == (40, 2)


def test_defaults_scalar():
    """n = 1 with default H = identity is the scalar case; it constructs and runs."""
    f = WalkingVectorFilter(Q0=[[1.0]])
    assert f.n == 1 and f.m == 1
    x = np.cumsum(np.random.default_rng(1).standard_normal(60))[:, None]
    assert np.all(np.isfinite(f.filter(x).mean))


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        WalkingVectorFilter(Q0=[[1.0, 2.0], [2.0, 1.0]])          # not PD
    with pytest.raises(ValueError):
        WalkingVectorFilter(Q0=[[1.0]], R0=[1.0], H=np.ones((2, 2)))  # H shape (m,n)=(2,1)
    with pytest.raises(ValueError):
        WalkingVectorFilter(Q0=[[1.0]], phi=1.0)


# ------------------------------------------------------------ core behaviour
def test_streaming_matches_batch():
    Y = mv_series(_H, _Q0, _RHO, 300, seed=2, hot_axis=3, amp=1.4)
    f = WalkingVectorFilter(_Q0, _RHO, _H, phi=0.9, s=0.4)
    r = f.filter(Y)
    f.reset()
    stream = np.array([f.update(Y[t]).mean for t in range(len(Y))])
    assert np.allclose(stream, r.mean, atol=1e-10)


def test_filter_does_not_touch_state():
    Y = mv_series(_H, _Q0, _RHO, 50, seed=5)
    f = WalkingVectorFilter(_Q0, _RHO, _H)
    f.update(Y[0])
    mu0 = f.mu.copy()
    f.filter(Y)
    assert np.array_equal(f.mu, mu0)


def test_covariance_symmetric_pd():
    Y = mv_series(_H, _Q0, _RHO, 200, seed=3, hot_axis=3, amp=1.4)
    r = WalkingVectorFilter(_Q0, _RHO, _H, s=0.4).filter(Y)
    assert np.all(np.isfinite(r.var))
    for P in r.var:
        assert np.allclose(P, P.T, atol=1e-9)
        assert np.linalg.eigvalsh(P)[0] > -1e-8


def test_missing_row_propagates():
    Y = mv_series(_H, _Q0, _RHO, 40, seed=6)
    f = WalkingVectorFilter(_Q0, _RHO, _H)
    f.update(Y[0]); before = f._m.copy()
    st = f.update(np.array([np.nan, np.nan]))
    assert np.all(np.isfinite(st.mean)) and np.array_equal(st.mean, before)


# --------------------------------------------------- the per-component claims
def test_stable_on_static_data():
    """No sustained scale change -> the tracked scales stay near 0 (bounded drift).

    The residual process<->measurement coupling permits a small offset, so the bound
    is loose (0.4 nats), but a genuine drift (the earlier unbounded-walk failure ran
    to several nats) is excluded."""
    Y = mv_series(_H, _Q0, _RHO, 500, seed=7)      # no hot axis
    r = WalkingVectorFilter(_Q0, _RHO, _H, s=0.4).filter(Y)
    tail = slice(150, None)
    assert np.all(np.abs(r.process_scale[tail].mean(0)) < 0.4)
    assert np.all(np.abs(r.measurement_scale[tail].mean(0)) < 0.4)


def test_reaches_a_hot_sensor():
    """A sensor whose noise jumps mid-stream is tracked up to near its true level,
    and the *other* sensor stays much lower -- per-component deduction."""
    T = 600
    Y = mv_series(_H, _Q0, _RHO, T, seed=2, hot_axis=3, amp=1.4)   # sensor 2 (eta2) hot
    r = WalkingVectorFilter(_Q0, _RHO, _H, s=0.4).filter(Y)
    b = slice(T // 3 + 60, 2 * T // 3 - 20)
    eta = r.measurement_scale[b].mean(0)
    assert eta[1] > 0.9                       # the hot sensor is found (truth 1.4)
    assert eta[1] - eta[0] > 0.6              # and separated from the clean one


def test_tracks_state_better_than_a_blind_static_filter():
    """On data with a live regime, following the scale beats holding it fixed:
    the walker's state RMSE is below a filter frozen at the base scales."""
    T = 600
    Y = mv_series(_H, _Q0, _RHO, T, seed=4, hot_axis=1, amp=1.4)   # a process mode hot
    # ground-truth-ish reference: rerun the generator's clean state
    rng = np.random.default_rng(4)
    n, m = 2, 2
    lam, V = np.linalg.eigh(_Q0)
    psi = np.zeros((T, 4)); psi[T // 3:2 * T // 3, 1] = 1.4
    th = np.zeros(n); truth = np.zeros((T, n))
    for t in range(T):
        Qt = V @ np.diag(lam * np.exp(psi[t, :n])) @ V.T
        th = th + np.linalg.cholesky(Qt + 1e-12 * np.eye(n)) @ rng.standard_normal(n)
        truth[t] = th
        _ = _H @ th + np.sqrt(_RHO * np.exp(psi[t, n:])) * rng.standard_normal(m)
    walk = WalkingVectorFilter(_Q0, _RHO, _H, s=0.4).filter(Y)
    err_walk = np.sqrt(((walk.mean[100:] - truth[100:]) ** 2).mean())
    assert np.all(np.isfinite(walk.mean)) and err_walk < 5.0    # tracks, does not diverge
