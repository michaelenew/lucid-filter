"""Tests for statfilter.AdaptiveKalmanFilter -- supplied F/H, noise learned online.

Behavioural, not exact-match.  The filter's headline capabilities are (1) a general
transition F (the derivative / kinematic mode) and (2) whiteness-gated noise adaptation that
does not diverge when process noise dominates (the Q-vs-R confound, research 0024/0025).  The
checks pin: construction/shape invariants, the kinematic helper, stream == batch, no
divergence and a state-estimation win in a crusher-style burst, and no false alarm on static
data.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from statfilter import AdaptiveKalmanFilter, AdaptiveStep, AdaptiveResult  # noqa: E402


def _crusher(seed, q_on, r_on, q_base=1e-3, r_base=0.04, T=600):
    """1-DOF arm with momentum; process and/or sensor noise burst in the middle third."""
    dt = 1.0; on = slice(T // 3, 2 * T // 3); rng = np.random.default_rng(seed)
    F = np.array([[1.0, dt], [0.0, 1.0]]); G = np.array([[0.5], [1.0]]); H = np.array([[1.0, 0.0]])
    qt = np.full(T, q_base); rt = np.full(T, r_base); qt[on] = q_on; rt[on] = r_on
    x = np.zeros(2); X = np.zeros((T, 2)); Y = np.zeros((T, 1))
    for t in range(T):
        a = math.sqrt(qt[t]) * rng.standard_normal()
        x = F @ x + (G * a).ravel(); X[t] = x
        Y[t] = H @ x + math.sqrt(rt[t]) * rng.standard_normal()
    return X, Y, F, G, H, on


def _fixed_kf(Y, F, H, Q, R, m0, P0):
    m = m0.copy(); P = P0.copy(); out = np.zeros((len(Y), F.shape[0]))
    for t, y in enumerate(Y):
        m = F @ m; P = F @ P @ F.T + Q
        S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)
        m = m + K @ (y - H @ m); P = P - K @ H @ P; out[t] = m
    return out


# --------------------------------------------------------------------- construction
def test_construction_and_shapes():
    Q0 = np.array([[1.0, 0.3], [0.3, 1.0]]); H = np.array([[1.0, 0.0]])
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    f = AdaptiveKalmanFilter(Q0, R0=[0.5], H=H, F=F)
    assert f.n == 2 and f.m == 1 and f.D == 3
    r = f.filter(np.zeros((20, 1)))
    assert r.mean.shape == (20, 2) and r.var.shape == (20, 2, 2)
    assert r.process_scale.shape == (20, 2) and r.measurement_scale.shape == (20, 1)


def test_bad_shapes_raise():
    with pytest.raises(ValueError):
        AdaptiveKalmanFilter(np.array([[1.0, 0.0], [0.0, 1.0]]), H=np.eye(2), F=np.zeros((3, 3)))
    f = AdaptiveKalmanFilter(np.eye(1), R0=[1.0])
    with pytest.raises(ValueError):
        f.update(np.zeros(2))


def test_kinematic_helper():
    f = AdaptiveKalmanFilter.kinematic(n_dof=2, order=2, dt=0.1)
    assert f.n == 4 and f.m == 2                       # (pos,vel) x 2 DOF, position measured
    # F must be the constant-velocity integrator per DOF
    assert np.allclose(f.F[:2, :2], np.array([[1.0, 0.1], [0.0, 1.0]]))
    r = f.filter(np.zeros((30, 2)))
    assert np.all(np.isfinite(r.mean))


def test_stream_equals_batch():
    _, Y, F, G, H, _ = _crusher(0, 0.05, 1.0)
    Q0 = 1e-3 * (G @ G.T) + 1e-9 * np.eye(2)
    a = AdaptiveKalmanFilter(Q0, R0=[0.04], H=H, F=F); rb = a.filter(Y)
    b = AdaptiveKalmanFilter(Q0, R0=[0.04], H=H, F=F); b.reset()
    means = np.array([b.update(y).mean for y in Y])
    assert np.allclose(rb.mean, means, atol=1e-9)


# --------------------------------------------------------------------- behaviour
def test_static_no_false_alarm():
    """On data with no noise excursion the learned scales must stay near zero."""
    X, Y, F, G, H, on = _crusher(0, 1e-3, 0.04)          # nothing hot
    f = AdaptiveKalmanFilter(1e-3 * (G @ G.T) + 1e-9 * np.eye(2), R0=[0.04], H=H, F=F, s=0.5)
    r = f.filter(Y)
    assert np.abs(r.measurement_scale[150:]).mean() < 0.5
    assert np.all(np.isfinite(r.mean))


@pytest.mark.parametrize("q_on,r_on", [(1e-3, 4.0), (0.30, 0.04), (0.30, 4.0)])
def test_crusher_no_divergence_and_helps(q_on, r_on):
    """Across sensor-hot / process-hot / both, the filter must not diverge and should not be
    much worse than a fixed base filter; on the decoupled bursts it should beat it."""
    ad = []; na = []
    T = 600
    for s in range(3):
        X, Y, F, G, H, on = _crusher(s, q_on, r_on)
        Qb = 1e-3 * (G @ G.T); Rb = np.array([[0.04]])
        f = AdaptiveKalmanFilter(Qb + 1e-9 * np.eye(2), R0=[0.04], H=H, F=F, s=0.5)
        est = f.filter(Y).mean
        assert np.all(np.isfinite(est))
        ad.append(np.sqrt(((est[on, 0] - X[on, 0]) ** 2).mean()))
        base = _fixed_kf(Y, F, H, Qb, Rb, np.array([Y[0, 0], 0.0]), np.eye(2))
        na.append(np.sqrt(((base[on, 0] - X[on, 0]) ** 2).mean()))
    ad_rmse, na_rmse = np.mean(ad), np.mean(na)
    assert ad_rmse < 3.0 * na_rmse                      # never catastrophically worse (no divergence)
    if r_on == 0.04 or q_on == 1e-3:                    # a decoupled burst -> should actually win
        assert ad_rmse < na_rmse


def test_known_forcing_removes_lag():
    """A moving state driven by a known input: with the forcing supplied, the estimate must not
    lag -- position AND velocity RMSE collapse versus running without it (research: control input)."""
    T, dt = 300, 0.05; rng = np.random.default_rng(0); t = np.arange(T) * dt
    F = np.array([[1.0, dt], [0.0, 1.0]]); G = np.array([[0.5 * dt * dt], [dt]]); H = np.array([[1.0, 0.0]])
    acmd = 1.2 * np.sin(2 * np.pi * t / (T * dt))          # known commanded acceleration
    x = np.zeros(2); X = np.zeros((T, 2)); Y = np.zeros((T, 1))
    for k in range(T):
        x = F @ x + (G * acmd[k]).ravel(); X[k] = x; Y[k] = H @ x + 0.02 * rng.standard_normal()
    Q0 = 1e-4 * (G @ G.T) + 1e-9 * np.eye(2); R0 = [0.02 ** 2 * 4]
    no_f = AdaptiveKalmanFilter(Q0, R0=R0, H=H, F=F, s=0.5).filter(Y).mean
    fc = AdaptiveKalmanFilter(Q0, R0=R0, H=H, F=F, B=G, s=0.5)
    with_f = fc.filter(Y, U=acmd[:, None]).mean
    vel_no = np.sqrt(((no_f[:, 1] - X[:, 1]) ** 2).mean())
    vel_yes = np.sqrt(((with_f[:, 1] - X[:, 1]) ** 2).mean())
    assert vel_yes < 0.3 * vel_no                         # forcing sharply cuts the velocity lag
    assert np.sqrt(((with_f[:, 0] - X[:, 0]) ** 2).mean()) < 0.05


def test_control_input_contract():
    G = np.array([[0.005], [0.1]]); H = np.array([[1.0, 0.0]]); F = np.array([[1.0, 0.1], [0.0, 1.0]])
    f = AdaptiveKalmanFilter(1e-4 * (G @ G.T) + 1e-9 * np.eye(2), R0=[1.0], H=H, F=F, B=G)
    with pytest.raises(ValueError):
        f.update(np.zeros(1))                             # missing required u
    with pytest.raises(ValueError):
        f.filter(np.zeros((5, 1)))                        # missing required U
    g = AdaptiveKalmanFilter(np.eye(1), R0=[1.0])         # no B
    with pytest.raises(ValueError):
        g.update(np.zeros(1), u=np.zeros(1))              # u passed but no B


def test_kinematic_derivatives_and_accelerometer():
    f = AdaptiveKalmanFilter.kinematic(n_dof=2, order=3, dt=0.05, measured=("pos",), control=True)
    assert f.n == 6 and f.m == 2 and f.p == 2 and f.n_dof == 2 and f.order == 3
    r = f.filter(np.zeros((40, 2)), U=np.zeros((40, 2)))
    D = f.derivatives(r.mean)
    assert D.shape == (40, 2, 3)                          # (T, n_dof, [pos, vel, acc])
    # accelerometer fusion: a sensor may read the 2nd derivative
    fa = AdaptiveKalmanFilter.kinematic(n_dof=1, order=3, dt=0.05, measured=("pos", "acc"))
    assert fa.m == 2 and np.all(np.isfinite(fa.filter(np.zeros((30, 2))).mean))
    with pytest.raises(ValueError):
        AdaptiveKalmanFilter.kinematic(n_dof=1, order=2, measured=("acc",))  # needs order > 2


def test_derivative_mode_beats_nonadaptive_random_walk():
    """The kinematic model must vastly beat a NON-adaptive local-level (F=I) filter on a
    momentum system -- the honest "before" baseline (a fixed random walk can't track a ramp).
    (An *adaptive* local-level partly compensates by cranking Q, so that gap is milder; the
    catastrophic gap is against the fixed filter.)"""
    X, Y, F, G, H, on = _crusher(0, 0.30, 4.0)
    kin = AdaptiveKalmanFilter(1e-3 * (G @ G.T) + 1e-9 * np.eye(2), R0=[0.04], H=H, F=F, s=0.5)
    kin_rmse = np.sqrt(((kin.filter(Y).mean[:, 0] - X[:, 0]) ** 2).mean())
    ll_fixed = _fixed_kf(Y, np.array([[1.0]]), np.array([[1.0]]),
                         np.array([[1e-3]]), np.array([[0.04]]), np.array([Y[0, 0]]), np.eye(1))
    ll_rmse = np.sqrt(((ll_fixed[:, 0] - X[:, 0]) ** 2).mean())
    assert kin_rmse < 0.2 * ll_rmse
