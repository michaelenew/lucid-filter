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


def test_bank_matches_the_looped_members():
    """The stacked executor is the same recursion as ``_WalkEngine.update`` -- pinned.

    `LucidFilter` runs its members stacked (`_EngineBank`, one leading member axis) because the
    split ladder multiplies the member count and per-member numpy dispatch was 99% of the step
    cost.  This pins the stack to the looped reference on every path it has: a fresh start, a
    missing observation, the multi-pair star, the dynamics channel with its fault kernel and
    reprice, and a control input.
    """
    from lucid.statfilter.lucid import _WalkEngine

    class _Looped(_WalkEngine):
        def update(self, y, u=None):
            return _WalkEngine.update(self, y, u=u)

    r = rng(3)
    box = {"phis": (0.70, 0.95), "ss": (0.30, 0.80)}
    Y1 = r.standard_normal((40, 1)); Y1[7] = np.nan
    rigs = [
        (dict(box), Y1),
        (dict(n=2, H=np.eye(2), **box), r.standard_normal((30, 2))),
        (dict(dynamics=[[0.9]], process=[[0.09]], measurement=[0.25], faults=1 / 100, **box),
         np.concatenate([r.standard_normal((20, 1)), 4 + 0.3 * r.standard_normal((20, 1))])),
    ]
    for kw, Y in rigs:
        a = LucidFilter(**kw)
        b = LucidFilter(**kw)
        for f in b._members:
            f.__class__ = _Looped
        b.reset()
        assert all(type(bk).__name__ == "_EngineBank" for bk in a._banks)
        assert all(type(bk).__name__ == "_LoopBank" for bk in b._banks)
        ra, rb = a.filter(Y), b.filter(Y)
        assert np.allclose(ra.mean, rb.mean, atol=1e-9, equal_nan=True)
        assert np.allclose(ra.var, rb.var, atol=1e-9)
        assert np.allclose(ra.process_scale, rb.process_scale, atol=1e-8)
        assert np.allclose(ra.measurement_scale, rb.measurement_scale, atol=1e-8)
        assert abs(ra.loglik - rb.loglik) < 1e-6
        if ra.fault is not None:
            assert np.allclose(ra.fault, rb.fault, atol=1e-9)


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


# ------------------------------------------------- a state-dependent measurement map

def test_callable_H_constant_matches_the_matrix():
    """A callable that returns the same matrix every step IS the matrix -- to machine precision.

    The guard on the whole feature: supplying ``H`` as a callable must not change what the
    filter computes, only where the Jacobian comes from.  Everything structural (the
    activation rule, the confounded pairs, the steady-state scale-Fisher) is answered from the
    characteristic linearisation, so the two constructions have to agree exactly.
    """
    r = rng(5)
    Hm = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    F = np.array([[1.0, 0.1, 0.005], [0.0, 1.0, 0.1], [0.0, 0.0, 1.0]])
    Y = r.standard_normal((60, 2))
    box = {"phis": (0.70, 0.95), "ss": (0.30, 0.80)}
    kw = dict(dynamics=F, process=np.eye(3) * 0.01, measurement=[0.2, 0.05], **box)
    a = LucidFilter(H=Hm, **kw).filter(Y)
    b = LucidFilter(H=lambda x: Hm, **kw).filter(Y)
    assert np.allclose(a.mean, b.mean, atol=1e-12)
    assert np.allclose(a.var, b.var, atol=1e-12)
    assert np.allclose(a.process_scale, b.process_scale, atol=1e-12)
    assert np.allclose(a.measurement_scale, b.measurement_scale, atol=1e-12)
    assert abs(a.loglik - b.loglik) < 1e-9


def test_callable_H_tracks_a_rotating_sensor():
    """A sensor whose axis rotates with the state is tracked; a frozen H is not.

    The physical case this exists for: an inertial sensor on a moving linkage reads the chain
    below it through axes that rotate, so its ``H`` is a function of the state.  Here one
    sensor reads ``cos(a) x0 + sin(a) x1`` with the mixing angle swept over the run.  A filter
    told the live Jacobian must beat one frozen at the characteristic (a = 0) map.
    """
    r = rng(11)
    T = 400
    ang = np.linspace(0.0, 1.2, T)
    F = np.array([[1.0, 0.05], [0.0, 1.0]])
    x = np.zeros(2)
    X = np.zeros((T, 2)); Y = np.zeros((T, 2))
    for k in range(T):
        x = F @ x + np.array([0.0, 0.02]) * r.standard_normal()
        X[k] = x
        Y[k] = [x[0] + 0.05 * r.standard_normal(),
                math.cos(ang[k]) * x[0] + math.sin(ang[k]) * x[1] + 0.05 * r.standard_normal()]
    step = {"k": 0}

    def Hof(_x):
        a = ang[min(step["k"], T - 1)]
        return np.array([[1.0, 0.0], [math.cos(a), math.sin(a)]])

    kw = dict(dynamics=F, process=np.diag([1e-9, 4e-4]), measurement=[0.0025, 0.0025],
              phis=(0.85,), ss=(0.40,))
    live = LucidFilter(H=Hof, **kw)
    live.reset()
    est = np.empty((T, 2))
    for k in range(T):
        step["k"] = k
        est[k] = live.update(Y[k]).mean
    frozen = LucidFilter(H=Hof(None), **kw).filter(Y).mean
    err = lambda e: float(np.sqrt(np.mean((e[100:] - X[100:]) ** 2)))   # noqa: E731
    assert err(est) < 0.5 * err(frozen), (err(est), err(frozen))


def test_callable_H_nonlinear_pair_and_stacked_bank():
    """The ``(H, y_pred)`` return, and the stacked bank pinned to the looped members under it.

    ``h(x) = H(x) x`` is not general enough for a real inertial sensor -- a rotating-axis term
    is quadratic in the rates -- so the callable may return the predicted measurement too.
    The stacked executor evaluates a Jacobian per member, at that member's own mean, and must
    still be the same recursion as the loop.
    """
    from lucid.statfilter.lucid import _WalkEngine

    class _Looped(_WalkEngine):
        def update(self, y, u=None):
            return _WalkEngine.update(self, y, u=u)

    r = rng(7)
    F = np.array([[1.0, 0.1], [0.0, 0.98]])

    def Hof(x):
        # a sensor whose axis tilts with the state, plus a small bounded term that is NOT
        # H(x) x -- the shape a rotating-axis (rate-squared) contribution actually has
        a = 0.4 * math.tanh(float(x[1]))
        ca, sa = math.cos(a), math.sin(a)
        Hj = np.array([[1.0, 0.0], [sa, ca]])
        yp = np.array([float(x[0]), sa * float(x[0]) + ca * float(x[1])
                       + 0.05 * math.sin(float(x[0]))])
        return Hj, yp

    Y = np.concatenate([r.standard_normal((25, 2)), 3 + r.standard_normal((25, 2))])
    Y[9] = np.nan
    kw = dict(dynamics=F, H=Hof, process=np.eye(2) * 0.05, measurement=[0.3, 0.3],
              faults=1 / 100, phis=(0.70, 0.95), ss=(0.30, 0.80))
    a, b = LucidFilter(**kw), LucidFilter(**kw)
    for f in b._members:
        f.__class__ = _Looped
    b.reset()
    assert any(type(bk).__name__ == "_EngineBank" for bk in a._banks)
    assert all(type(bk).__name__ == "_LoopBank" for bk in b._banks)
    ra, rb = a.filter(Y), b.filter(Y)
    assert np.all(np.isfinite(ra.mean))
    assert np.allclose(ra.mean, rb.mean, atol=1e-9)
    assert np.allclose(ra.var, rb.var, atol=1e-9)
    assert np.allclose(ra.process_scale, rb.process_scale, atol=1e-8)
    assert np.allclose(ra.measurement_scale, rb.measurement_scale, atol=1e-8)
    assert np.allclose(ra.fault, rb.fault, atol=1e-9)


def test_callable_H_offset_is_not_misread_at_init():
    """h with a constant offset (an accelerometer's gravity term) must not poison the start.

    The cold-start mean is a least squares against the measurement map; with an ``(H, y_pred)``
    hook the map has a value at zero state, and solving ``H x = y`` without subtracting it
    would book gravity as enormous state.  The init linearises h at the origin instead.
    """
    Hm = np.array([[1.0, 0.0], [0.5, 2.0]])
    c = np.array([0.0, 7.0])
    f = LucidFilter(dynamics=np.eye(2), H=lambda x: (Hm, Hm @ x + c),
                    process=np.eye(2) * 1e-4, measurement=[1e-4, 1e-4],
                    phis=(0.85,), ss=(0.40,))
    f.reset()
    st = f.update(Hm @ np.array([1.0, -0.5]) + c)      # the exact reading at x = (1, -0.5)
    assert np.allclose(st.mean, [1.0, -0.5], atol=0.05), st.mean


# ----------------------------------------------------- the offset channel (first moment)
def test_offsets_off_is_bit_identical():
    """The channel is off unless asked for, and then it is not there at all.

    Every noise channel in this filter is second-moment; the offset channel is the first, and
    it rides on the collapsed output rather than inside the recursion precisely so that a
    caller who does not ask for it pays nothing.  Pinned bit-for-bit, not to a tolerance.
    """
    rng = np.random.default_rng(4)
    Y = np.cumsum(rng.normal(0, 0.2, 200))[:, None] + rng.normal(0, 1, (200, 1))
    a = LucidFilter().filter(Y)
    b = LucidFilter(offsets=False).filter(Y)
    assert np.array_equal(a.mean, b.mean)
    assert np.array_equal(a.var, b.var)
    assert a.loglik == b.loglik
    assert a.offset is None


def test_offset_basis_is_the_identifiable_quotient():
    """Activation is structural: the gauge directions are excluded, by construction.

    A sensor bias is gauge on ``H ker(F - I)`` -- a state offset the dynamics hold still reads
    identically -- so a scalar level read by one sensor can carry a DRIFT and cannot carry a
    sensor bias.  Checked against the likelihood in research/bias-channels 0002; the sensor
    entry is not carried at all, for the reason `_mean_basis` records.
    """
    from lucid.statfilter.lucid import _mean_basis

    B1 = _mean_basis(np.eye(1), np.ones((1, 1)))
    assert B1.shape == (2, 1)
    assert np.allclose(np.abs(B1[:, 0]), [1.0, 0.0], atol=1e-9)      # the drift, not the bias

    # the full quotient still knows the sensor entry: one level, two sensors -> the RELATIVE
    # bias is identifiable and the common mode is gauge
    B2 = _mean_basis(np.eye(1), np.ones((2, 1)), process_only=False)
    assert B2.shape == (3, 2)
    assert abs(np.array([0.0, 1.0, 1.0]) @ B2 @ B2.T @ np.array([0.0, 1.0, 1.0])) < 1e-9


def test_offset_channel_finds_a_drift_and_does_not_invent_one():
    """It recovers a climbing bias, and reports zero when there is none.

    The premium for carrying the channel on driftless data is what makes it safe to switch on;
    research/bias-channels 0005 measures it at 0.8% of RMSE, against 49-84% of the distance to
    an oracle told the drift when there is one.
    """
    Q, R, N, T0, rate = 0.02, 1.0, 700, 300, 0.3
    rng = np.random.default_rng(11)
    theta = np.cumsum(rng.normal(0, np.sqrt(Q), N) + rate * (np.arange(N) >= T0))
    Y = (theta + rng.normal(0, np.sqrt(R), N))[:, None]

    on = LucidFilter(offsets=True).filter(Y)
    off = LucidFilter().filter(Y)
    assert abs(on.offset[-1, 0] - rate) < 0.1 * rate                 # the rate, to 10%
    lo = T0 + 100
    e_on = np.sqrt(np.mean((on.mean[lo:, 0] - theta[lo:]) ** 2))
    e_off = np.sqrt(np.mean((off.mean[lo:, 0] - theta[lo:]) ** 2))
    assert e_on < 0.8 * e_off

    flat = np.cumsum(rng.normal(0, np.sqrt(Q), N))
    Yf = (flat + rng.normal(0, np.sqrt(R), N))[:, None]
    quiet = LucidFilter(offsets=True).filter(Yf)
    assert abs(quiet.offset[-1, 0]) < 0.05                           # no hallucinated drift


def test_offset_channel_does_not_disturb_a_biased_sensor_rig():
    """A rig the channel is NOT for must be left alone.

    A miscalibrated sensor is a first-moment fault the channel deliberately does not carry
    (`_mean_basis`), so the guard is that switching the channel on does not degrade the rig
    where it has nothing to offer -- the failure mode measured in research/bias-channels 0006
    when the sensor entry was carried.
    """
    Q, R, N, T0, bias = 0.02, 1.0, 700, 300, 2.0
    rng = np.random.default_rng(7)
    theta = np.cumsum(rng.normal(0, np.sqrt(Q), N))
    Y = np.stack([theta + rng.normal(0, np.sqrt(R), N) for _ in range(3)], axis=1)
    Y[T0:, 2] += bias
    H, R0 = np.ones((3, 1)), np.ones(3)

    off = LucidFilter(H=H, measurement=R0).filter(Y)
    on = LucidFilter(H=H, measurement=R0, offsets=True).filter(Y)
    e_off = np.sqrt(np.mean((off.mean[400:, 0] - theta[400:]) ** 2))
    e_on = np.sqrt(np.mean((on.mean[400:, 0] - theta[400:]) ** 2))
    assert e_on < 1.05 * e_off                                       # no regression
    assert abs(on.offset[-1, 0]) < 0.02                              # and no spurious drift


def test_offset_channel_is_inert_on_a_stable_spectrum():
    """Where a drift cannot be told from a sensor bias, the channel declines to act.

    On a stable `F` a process mean drives the state to the constant ``(I - F)^-1 d``, whose
    reading IS a sensor bias -- the two fit identically and imply different states, so a channel
    carrying only one of them would silently pick it.  `_mean_basis` quotients the process entry
    by the sensor entry for exactly this reason, and on a purely stable spectrum nothing
    survives.  Measured before the quotient existed (research/bias-channels 0007): a real sensor
    bias read as a drift cost 0.786 -> 0.960 RMSE and calibration 2.07 -> 5.25.
    """
    F = np.array([[0.8]])
    rng = np.random.default_rng(5)
    Y = np.empty((300, 1))
    x = 0.0
    for t in range(300):
        x = 0.8 * x + rng.normal(0, 0.2)
        Y[t, 0] = x + rng.normal(0, 1.0)

    f = LucidFilter(dynamics=F, offsets=True)
    assert f._mean is None                                       # k = 0: nothing identifiable
    on, off = f.filter(Y), LucidFilter(dynamics=F).filter(Y)
    assert np.array_equal(on.mean, off.mean)                     # and therefore bit-identical
    assert on.offset is None

    # a unit root beside the stable mode restores it, and only along the unit root
    from lucid.statfilter.lucid import _mean_basis
    B = _mean_basis(np.diag([1.0, 0.7]), np.array([[1.0, 1.0]]))
    assert B.shape[1] == 1
    assert abs(B[1, 0]) < 1e-8                                   # nothing on the stable mode


def test_offsets_do_not_provoke_the_dynamics_channel():
    """The two channels are confounded, so the offset is not fed back when both are on.

    A constant added to the prediction and a departure in `F` explain the same feature, so under
    feedback a departure walker adapts `F` to cancel the injected offset and its adaptation
    registers as a fault -- which the bank's long weight memory then keeps.  Measured over eight
    driftless seeds (research/bias-channels 0008): one locked `fault` at 1.000 under feedback,
    none under the feed-forward the filter now selects structurally.
    """
    Q, R, N = 0.02, 1.0, 500
    rng = np.random.default_rng(12)                              # the seed that locked
    theta = np.cumsum(rng.normal(0, np.sqrt(Q), N))
    Y = (theta + rng.normal(0, np.sqrt(R), N))[:, None]

    plain = LucidFilter(faults=1e-4).filter(Y)
    both = LucidFilter(faults=1e-4, offsets=True).filter(Y)
    assert LucidFilter(faults=1e-4, offsets=True)._mean.feedback is False
    assert both.fault[-1] < 0.5                                  # no locked false detection
    assert float(np.mean(both.fault)) < float(np.mean(plain.fault)) + 0.05
