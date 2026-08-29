"""Tests for the public LucidFilter API.

Behavioural, not exact-match: the filter tracks online, so the checks pin
structural properties (finite output, correct shapes, RMSE better than naive,
scale responds to injected noise).

    python -m pytest lucid/tests/test_lucid.py -q
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lucid import LucidFilter, LucidStep, LucidResult  # noqa: E402


# ------------------------------------------------------------------ fixtures

def rng(seed=0):
    return np.random.default_rng(seed)


def local_level(T=300, q=1.0, s2=1.0, seed=0):
    r = rng(seed)
    theta = np.cumsum(r.standard_normal(T) * math.sqrt(q))
    y = theta + r.standard_normal(T) * math.sqrt(s2)
    return y[:, None], theta


def kinematic(T=300, seed=0):
    """Position + velocity; observed through position only."""
    r = rng(seed)
    pos = np.zeros(T); vel = np.zeros(T)
    for t in range(1, T):
        vel[t] = vel[t-1] + r.standard_normal() * 0.1
        pos[t] = pos[t-1] + vel[t] + r.standard_normal() * 0.5
    y = pos + r.standard_normal(T)
    return y[:, None], pos


# ------------------------------------------------------------------ tests

def test_import_path():
    """LucidFilter is importable directly from lucid."""
    from lucid import LucidFilter, LucidStep, LucidResult
    assert LucidFilter is not None


def test_scalar_shapes():
    Y, _ = local_level()
    f = LucidFilter()
    r = f.filter(Y)
    assert isinstance(r, LucidResult)
    assert r.mean.shape == (300, 1)
    assert r.var.shape == (300, 1, 1)
    assert r.process_scale.shape == (300, 1)
    assert r.measurement_scale.shape == (300, 1)
    assert np.all(np.isfinite(r.mean))
    assert np.all(np.isfinite(r.var))


def test_scalar_better_than_naive():
    Y, theta = local_level(T=500, q=1.0, s2=1.0, seed=42)
    f = LucidFilter()
    r = f.filter(Y)
    rmse_filter = float(np.sqrt(np.mean((r.mean[:, 0] - theta) ** 2)))
    rmse_raw = float(np.sqrt(np.mean((Y[:, 0] - theta) ** 2)))
    assert rmse_filter < rmse_raw, f"filter RMSE {rmse_filter:.3f} >= raw {rmse_raw:.3f}"


def test_kinematic_shapes():
    Y, _ = kinematic()
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    f = LucidFilter(dynamics=F, H=H)
    r = f.filter(Y)
    assert r.mean.shape == (300, 2)
    assert r.var.shape == (300, 2, 2)
    assert r.process_scale.shape == (300, 2)
    assert r.measurement_scale.shape == (300, 1)
    assert np.all(np.isfinite(r.mean))


def test_kinematic_pos_rmse():
    Y, pos = kinematic(T=500, seed=7)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    f = LucidFilter(dynamics=F, H=H)
    r = f.filter(Y)
    rmse = float(np.sqrt(np.mean((r.mean[:, 0] - pos) ** 2)))
    assert rmse < 2.0, f"kinematic pos RMSE too high: {rmse:.3f}"


def test_update_step():
    f = LucidFilter()
    f.reset()
    st = f.update(np.array([1.0]))
    assert isinstance(st, LucidStep)
    assert st.mean.shape == (1,)
    assert np.isfinite(st.loglik)


def test_missing_obs():
    Y, _ = local_level(T=100)
    Y[30:35] = np.nan
    f = LucidFilter()
    r = f.filter(Y)
    assert np.all(np.isfinite(r.mean))
    assert np.all(np.isnan(r.innovation[30:35]))


def test_dynamics_none_is_implemented():
    """The cell is filled: dynamics=None learns F online (see the block at the bottom)."""
    f = LucidFilter(dynamics=None)
    assert f._learn and f._nd == 2          # the nominal hedge plus the departure walker


def test_five_dof_arm_polynomial():
    """A 5-DOF arm (n=10 pos/vel, m=5 pots, D=15) must construct and run.

    Pins the polynomial-cost property: the scale window is the caltrop axial star
    (research 0013), linear in the component count.  The joint tensor grid is
    ``(2K+1)**D`` nodes (~3e10 here) and would hang this test.
    """
    n_dof, dt = 5, 0.1
    F = np.kron(np.eye(n_dof), np.array([[1.0, dt], [0.0, 1.0]]))
    H = np.kron(np.eye(n_dof), np.array([[1.0, 0.0]]))
    f = LucidFilter(dynamics=F, H=H)
    for eng in f._members:
        assert eng._G <= 1 + (eng._nn - 1) * eng.D, (
            f"star has {eng._G} nodes; must be linear in D={eng.D}")
    r = rng(5)
    T = 60
    x = np.zeros(2 * n_dof)
    Y = np.empty((T, n_dof))
    for t in range(T):
        x = F @ x + r.standard_normal(2 * n_dof) * 0.05
        Y[t] = H @ x + r.standard_normal(n_dof) * 0.3
    res = f.filter(Y)
    assert res.mean.shape == (T, 2 * n_dof)
    assert np.all(np.isfinite(res.mean))
    assert np.all(np.isfinite(res.var))


def test_meas_scale_rises_on_burst():
    """Measurement log-scale should be higher during a sensor burst than in calm."""
    r = rng(99)
    T = 400
    theta = np.cumsum(r.standard_normal(T) * 0.3)
    noise = r.standard_normal(T)
    noise[150:180] *= 10.0   # sensor burst
    Y = (theta + noise)[:, None]
    f = LucidFilter()
    res = f.filter(Y)
    calm = float(res.measurement_scale[50:130].mean())
    burst = float(res.measurement_scale[150:180].mean())
    assert burst > calm, f"burst scale {burst:.2f} not above calm {calm:.2f}"


# ------------------------------------------------- the dynamics channel (dynamics=None)

def ar1(T=1200, a=0.6, q=0.09, r=0.25, seed=0, a2=None, switch=None):
    """Scalar AR(1), optionally with a step change in `a` at `switch`."""
    g = rng(seed)
    x = np.zeros(T)
    for t in range(1, T):
        at = a if (a2 is None or t < switch) else a2
        x[t] = at * x[t - 1] + math.sqrt(q) * g.standard_normal()
    y = x + math.sqrt(r) * g.standard_normal(T)
    return y[:, None], x


def _rmse(est, truth, lo=300):
    return float(np.sqrt(np.mean((est[lo:] - truth[lo:]) ** 2)))


def test_fixed_dynamics_reports_no_channel():
    """Supplied, fixed dynamics: the channel is off and the report says so."""
    Y, _ = ar1()
    r = LucidFilter(dynamics=[[0.6]], process=[[0.09]], measurement=[0.25]).filter(Y)
    assert r.dynamics is None and r.control is None and r.fault is None
    st = LucidFilter(dynamics=[[0.6]]).update(Y[0])
    assert st.dynamics is None and st.fault == 0.0


def test_dynamics_none_shapes():
    Y, _ = ar1(T=400)
    f = LucidFilter(dynamics=None, process=[[0.09]], measurement=[0.25])
    r = f.filter(Y)
    assert r.dynamics.shape == (400, 1, 1)
    assert r.fault.shape == (400,)
    assert np.all(np.isfinite(r.dynamics)) and np.all(np.isfinite(r.mean))
    assert np.all((r.fault >= 0.0) & (r.fault <= 1.0))


def test_dynamics_none_beats_the_random_walk_it_starts_from():
    """Told nothing, it must learn its way past its own F = I prior."""
    Y, x = ar1(T=1200, a=0.3, seed=2)
    learned = LucidFilter(dynamics=None, process=[[0.09]], measurement=[0.25]).filter(Y)
    walk = LucidFilter(dynamics=[[1.0]], process=[[0.09]], measurement=[0.25]).filter(Y)
    assert _rmse(learned.mean[:, 0], x) < _rmse(walk.mean[:, 0], x)
    assert learned.fault[-1] > 0.5           # the nominal F = I is decisively beaten
    assert abs(learned.dynamics[-1, 0, 0]) < 0.8      # and F has moved well off 1


def test_dynamics_none_tracks_near_the_oracle():
    """Within a few percent of a Kalman filter told the true dynamics."""
    Y, x = ar1(T=1200, a=0.6, seed=1)
    learned = LucidFilter(dynamics=None, process=[[0.09]], measurement=[0.25]).filter(Y)
    oracle = LucidFilter(dynamics=[[0.6]], process=[[0.09]], measurement=[0.25]).filter(Y)
    assert _rmse(learned.mean[:, 0], x) < 1.15 * _rmse(oracle.mean[:, 0], x)


def test_fault_is_detected_and_not_before_it_happens():
    Y, x = ar1(T=2400, a=0.9, a2=0.3, switch=1200, seed=5)
    f = LucidFilter(dynamics=[[0.9]], process=[[0.09]], measurement=[0.25], faults=1 / 2400)
    r = f.filter(Y)
    assert r.fault[300:1200].max() < 0.5      # no false alarm in the calm stretch
    assert r.fault[-200:].mean() > 0.5        # and the fault is found
    assert r.dynamics[-1, 0, 0] < 0.75        # F has moved toward the new truth


def test_fault_recovery_beats_the_frozen_filter():
    Y, x = ar1(T=2400, a=0.9, a2=0.3, switch=1200, seed=5)
    kw = dict(process=[[0.09]], measurement=[0.25])
    live = LucidFilter(dynamics=[[0.9]], faults=1 / 2400, **kw).filter(Y)
    frozen = LucidFilter(dynamics=[[0.9]], **kw).filter(Y)
    post = slice(1600, 2400)
    assert (_rmse(live.mean[:, 0], x, 1600) < _rmse(frozen.mean[:, 0], x, 1600))


def test_calm_costs_nothing():
    """The hedge: carrying the channel through a run with no fault is ~free."""
    Y, x = ar1(T=1500, a=0.9, seed=7)
    kw = dict(dynamics=[[0.9]], process=[[0.09]], measurement=[0.25])
    live = LucidFilter(faults=1 / 1500, **kw).filter(Y)
    fixed = LucidFilter(**kw).filter(Y)
    assert _rmse(live.mean[:, 0], x) < 1.02 * _rmse(fixed.mean[:, 0], x)
    assert live.fault[300:].max() < 0.9       # no confident false detection


def test_named_anchor_pins_the_new_dynamics():
    """A nameable fault mode is carried as its own filter and recovers F closely."""
    Y, x = ar1(T=2400, a=0.9, a2=0.3, switch=1200, seed=5)
    r = LucidFilter(dynamics=[[0.9]], process=[[0.09]], measurement=[0.25],
                    faults=1 / 2400, anchors=[[[0.3]]]).filter(Y)
    assert abs(r.dynamics[-1, 0, 0] - 0.3) < 0.2


def test_low_rank_departures():
    """Mechanism (b): supplying the departure directions is cheaper and still learns."""
    Y, x = ar1(T=800, a=0.3, seed=2)
    kw = dict(dynamics=None, process=[[0.09]], measurement=[0.25])
    full = LucidFilter(**kw)
    low = LucidFilter(departures=[np.array([[1.0]])], **kw)
    assert low._specs[-1][3].k == 1
    r = low.filter(Y)
    assert np.all(np.isfinite(r.dynamics)) and r.dynamics[-1, 0, 0] < 0.8


def test_control_map_is_learned():
    """B is learned alongside F when a control input is supplied."""
    g = rng(11)
    T = 1200
    u = g.standard_normal((T, 1))
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.5 * x[t - 1] + 1.5 * u[t, 0] + 0.2 * g.standard_normal()
    Y = (x + 0.3 * g.standard_normal(T))[:, None]
    r = LucidFilter(dynamics=None, control=[[1.0]], process=[[0.04]],
                    measurement=[0.09]).filter(Y, u)
    assert r.control.shape == (T, 1, 1)
    assert abs(r.control[-1, 0, 0] - 1.5) < 0.5


def test_departure_variance_is_bounded_never_frozen():
    """The class cap bounds the departure's variance; the gain stays live."""
    Y, _ = ar1(T=400)
    f = LucidFilter(dynamics=None, process=[[0.09]], measurement=[0.25])
    f.filter(Y)
    dep = f._specs[-1][3]
    eng = f._members[(f._nd - 1) * f._nc]
    assert np.all(np.diag(eng._P)[dep.gidx] <= dep.cap[dep.gidx] * (1 + 1e-9))
    assert np.all(np.diag(eng._P)[dep.gidx] > 0.0)      # never frozen at zero
    eng.reprice(dep.gidx)                                # the detection restart
    assert np.allclose(np.diag(eng._P)[dep.gidx], dep.cap[dep.gidx])


def test_vector_dynamics_learned():
    """n = 2 oscillator, told nothing, both components observed."""
    g = rng(4)
    T = 1500
    lam, phi = 0.97, 0.25
    F = lam * np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
    x = np.zeros((T, 2))
    for t in range(1, T):
        x[t] = F @ x[t - 1] + 0.2 * g.standard_normal(2)
    Y = x + 0.3 * g.standard_normal((T, 2))
    learned = LucidFilter(dynamics=None, process=np.eye(2) * 0.04,
                          measurement=[0.09, 0.09]).filter(Y)
    walk = LucidFilter(dynamics=np.eye(2), process=np.eye(2) * 0.04,
                       measurement=[0.09, 0.09]).filter(Y)
    lr = float(np.sqrt(np.mean((learned.mean[300:] - x[300:]) ** 2)))
    wr = float(np.sqrt(np.mean((walk.mean[300:] - x[300:]) ** 2)))
    assert lr < wr
    assert np.linalg.norm(learned.dynamics[-1] - F) < np.linalg.norm(np.eye(2) - F)


def test_faults_hazard_validated():
    with pytest.raises(ValueError):
        LucidFilter(dynamics=[[0.5]], faults=1.5)
    with pytest.raises(ValueError):
        LucidFilter(dynamics=[[0.5]], faults=0.0)
    with pytest.raises(ValueError):
        LucidFilter(dynamics=[[0.5]], faults=1e-4, anchors=[np.eye(2)])


def test_callable_dynamics_relinearises():
    """Real F/B arrive linearised per operating point, so `dynamics` may be a callable."""
    g = rng(9)
    T = 800
    x = np.zeros(T)
    for t in range(1, T):                       # a state-dependent (nonlinear) decay
        a = 0.9 - 0.3 * math.tanh(x[t - 1])
        x[t] = a * x[t - 1] + 0.2 * g.standard_normal()
    Y = (x + 0.3 * g.standard_normal(T))[:, None]

    def linearised(state):
        return np.array([[0.9 - 0.3 * math.tanh(float(state[0]))]])

    kw = dict(process=[[0.04]], measurement=[0.09])
    moving = LucidFilter(dynamics=linearised, n=1, **kw).filter(Y)
    frozen = LucidFilter(dynamics=[[0.9]], **kw).filter(Y)
    assert moving.dynamics.shape == (T, 1, 1)   # a moving model is reported, not None
    assert moving.fault is not None
    assert _rmse(moving.mean[:, 0], x) <= _rmse(frozen.mean[:, 0], x)
    # a CONSTANT callable must reduce to the matrix it returns
    same = LucidFilter(dynamics=lambda s: np.array([[0.9]]), n=1, **kw).filter(Y)
    assert np.allclose(same.mean, frozen.mean, atol=1e-8)


def test_callable_dynamics_with_faults():
    """The departure channel rides on top of a moving linearisation."""
    Y, x = ar1(T=1200, a=0.6, seed=3)
    f = LucidFilter(dynamics=lambda s: np.array([[0.6]]), n=1, faults=1e-3,
                    process=[[0.09]], measurement=[0.25])
    r = f.filter(Y)
    assert np.all(np.isfinite(r.dynamics)) and r.dynamics.shape == (1200, 1, 1)
    assert abs(r.dynamics[-1, 0, 0] - 0.6) < 0.4
