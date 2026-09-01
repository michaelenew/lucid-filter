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
    assert f._learn and f._ndbase == 2      # per rung: the nominal hedge plus its walker
    assert len(f._specs) == 1 + len(f.hazards)   # the nominal filter is shared across rungs


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
        def update(self, y, u=None, a=1.0):
            return _WalkEngine.update(self, y, u=u, a=a)

    r = rng(3)
    box = {"phis": (0.70, 0.95), "ss": (0.30, 0.80)}
    Y1 = r.standard_normal((40, 1)); Y1[7] = np.nan
    # partly-observed rows, and rows with some sensors absent AND a whole row absent
    Y2 = r.standard_normal((30, 2)); Y2[::3, 0] = np.nan; Y2[11] = np.nan
    rigs = [
        (dict(box), Y1, None),
        (dict(n=2, H=np.eye(2), **box), r.standard_normal((30, 2)), None),
        (dict(dynamics=[[0.9]], process=[[0.09]], measurement=[0.25], faults=1 / 100, **box),
         np.concatenate([r.standard_normal((20, 1)), 4 + 0.3 * r.standard_normal((20, 1))]),
         None),
        # the paths this workstream added: partial rows, and non-nominal gaps
        (dict(n=2, H=np.eye(2), **box), Y2, None),
        (dict(dynamics=[[1.0, 0.1], [0.0, 1.0]], H=np.eye(2), timestep=0.1, **box),
         r.standard_normal((30, 2)), np.cumsum(np.abs(r.normal(0.1, 0.05, 30)) + 1e-3)),
    ]
    for kw, Y, tt in rigs:
        a = LucidFilter(**kw)
        b = LucidFilter(**kw)
        for f in b._members:
            f.__class__ = _Looped
        b.reset()
        assert all(type(bk).__name__ == "_EngineBank" for bk in a._banks)
        assert all(type(bk).__name__ == "_LoopBank" for bk in b._banks)
        ra, rb = a.filter(Y, t=tt), b.filter(Y, t=tt)
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
    T = 30
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
    T = 120
    Y, _ = ar1(T=T)
    f = LucidFilter(dynamics=None, process=[[0.09]], measurement=[0.25],
                    phis=(0.85,), ss=(0.40,))
    r = f.filter(Y)
    assert r.dynamics.shape == (T, 1, 1)
    assert r.fault.shape == (T,)
    assert np.all(np.isfinite(r.dynamics)) and np.all(np.isfinite(r.mean))
    assert np.all((r.fault >= 0.0) & (r.fault <= 1.0))


def test_dynamics_none_beats_the_random_walk_it_starts_from():
    """Told nothing, it must learn its way past its own F = I prior."""
    Y, x = ar1(T=600, a=0.3, seed=2)
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


# The fault rig: 0.9 for the first half, 0.3 for the second, at the hazard the length
# implies.  It is one experiment and three claims are read off it, so the series and the
# run every claim shares are built once per module rather than once per claim.
_SWITCH = 600
_FAULT_T = 2 * _SWITCH


@pytest.fixture(scope="module")
def fault_rig():
    return ar1(T=_FAULT_T, a=0.9, a2=0.3, switch=_SWITCH, seed=5)


@pytest.fixture(scope="module")
def fault_run(fault_rig):
    Y, _ = fault_rig
    return LucidFilter(dynamics=[[0.9]], process=[[0.09]], measurement=[0.25],
                       faults=1 / _FAULT_T).filter(Y)


def test_fault_is_detected_and_not_before_it_happens(fault_run):
    r = fault_run
    assert r.fault[150:_SWITCH].max() < 0.5   # no false alarm in the calm stretch
    assert r.fault[-200:].mean() > 0.5        # and the fault is found
    assert r.dynamics[-1, 0, 0] < 0.75        # F has moved toward the new truth


def test_fault_recovery_beats_the_frozen_filter(fault_rig, fault_run):
    Y, x = fault_rig
    frozen = LucidFilter(dynamics=[[0.9]], process=[[0.09]],
                         measurement=[0.25]).filter(Y)
    lo = _SWITCH + _SWITCH // 3               # past the detection transient
    assert (_rmse(fault_run.mean[:, 0], x, lo) < _rmse(frozen.mean[:, 0], x, lo))


def test_calm_costs_nothing():
    """The hedge: carrying the channel through a run with no fault is ~free."""
    T = 600
    Y, x = ar1(T=T, a=0.9, seed=7)
    kw = dict(dynamics=[[0.9]], process=[[0.09]], measurement=[0.25])
    live = LucidFilter(faults=1 / T, **kw).filter(Y)
    fixed = LucidFilter(**kw).filter(Y)
    assert _rmse(live.mean[:, 0], x, 150) < 1.02 * _rmse(fixed.mean[:, 0], x, 150)
    assert live.fault[150:].max() < 0.9       # no confident false detection


def test_named_anchor_pins_the_new_dynamics(fault_rig):
    """A nameable fault mode is carried as its own filter and recovers F closely."""
    Y, x = fault_rig
    r = LucidFilter(dynamics=[[0.9]], process=[[0.09]], measurement=[0.25],
                    faults=1 / _FAULT_T, anchors=[[[0.3]]]).filter(Y)
    assert abs(r.dynamics[-1, 0, 0] - 0.3) < 0.2


def test_low_rank_departures():
    """Mechanism (b): supplying the departure directions is cheaper and still learns."""
    Y, x = ar1(T=400, a=0.3, seed=2)
    kw = dict(dynamics=None, process=[[0.09]], measurement=[0.25])
    low = LucidFilter(departures=[np.array([[1.0]])], **kw)
    assert low._specs[-1][3].k == 1
    r = low.filter(Y)
    assert np.all(np.isfinite(r.dynamics)) and r.dynamics[-1, 0, 0] < 0.8


def test_control_map_is_learned():
    """B is learned alongside F when a control input is supplied."""
    g = rng(11)
    T = 600
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
    Y, _ = ar1(T=120)
    f = LucidFilter(dynamics=None, process=[[0.09]], measurement=[0.25],
                    phis=(0.85,), ss=(0.40,))
    f.filter(Y)
    dep = f._specs[-1][3]                    # the bottom-rung walker: the last spec
    eng = f._members[(len(f._specs) - 1) * f._nc]
    assert np.all(np.diag(eng._P)[dep.gidx] <= dep.cap[dep.gidx] * (1 + 1e-9))
    assert np.all(np.diag(eng._P)[dep.gidx] > 0.0)      # never frozen at zero
    eng.reprice(dep.gidx)                                # the detection restart
    assert np.allclose(np.diag(eng._P)[dep.gidx], dep.cap[dep.gidx])


def test_vector_dynamics_learned():
    """n = 2 oscillator, told nothing, both components observed.

    n = 2 is where the departure channel is at its most expensive -- four directions, so
    a six-dimensional augmented state per member -- and both claims are made with more
    margin at 600 steps than they were at 1500, so this is the length it runs at.  The
    (phi, s) box is narrowed for the same reason it is in the equivalence tests below:
    the claim is about the DYNAMICS channel and the noise box is the nuisance it
    marginalises, measurably so -- both margins are the same to two figures across the
    default box, this one, and a single cell.
    """
    g = rng(4)
    T = 600
    lam, phi = 0.97, 0.25
    F = lam * np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
    x = np.zeros((T, 2))
    for t in range(1, T):
        x[t] = F @ x[t - 1] + 0.2 * g.standard_normal(2)
    Y = x + 0.3 * g.standard_normal((T, 2))
    kw = dict(process=np.eye(2) * 0.04, measurement=[0.09, 0.09],
              phis=(0.70, 0.95), ss=(0.30, 0.80))
    learned = LucidFilter(dynamics=None, **kw).filter(Y)
    walk = LucidFilter(dynamics=np.eye(2), **kw).filter(Y)
    lr = float(np.sqrt(np.mean((learned.mean[150:] - x[150:]) ** 2)))
    wr = float(np.sqrt(np.mean((walk.mean[150:] - x[150:]) ** 2)))
    assert lr < wr
    assert np.linalg.norm(learned.dynamics[-1] - F) < np.linalg.norm(np.eye(2) - F)


def test_faults_hazard_validated():
    with pytest.raises(ValueError):
        LucidFilter(dynamics=[[0.5]], faults=1.5)
    with pytest.raises(ValueError):
        LucidFilter(dynamics=[[0.5]], faults=0.0)
    with pytest.raises(ValueError):                 # above the class's persistence boundary
        LucidFilter(dynamics=[[0.5]], faults=0.7)
    with pytest.raises(ValueError):                 # one bad rung poisons a supplied ladder
        LucidFilter(dynamics=[[0.5]], faults=[0.05, 0.7])
    with pytest.raises(ValueError):
        LucidFilter(dynamics=[[0.5]], faults=1e-4, anchors=[np.eye(2)])


def test_hazard_box_is_structural_not_forget_derived():
    """faults=True mixes over the fixed broad box: rungs 1.5 nats apart (the Sparrow rule
    at the axis's one-event blur width) down from the class's own persistence boundary.  Nothing structural reads ``forget`` -- the box is identical at any
    memory, and the construction is valid at the NOMINAL filter, ``forget = 1`` (pure Bayes):
    forget is the engineering escape for the stationarity assumption being violated, and it
    is admissible only because nothing depends on it (adaptive-grid 0029, research 0009)."""
    f = LucidFilter(dynamics=[[0.9]], faults=True)
    assert f.hazards[0] == 0.5                      # the class's persistence boundary
    assert np.allclose(np.log(f.hazards[:-1] / f.hazards[1:]), 1.5)   # 1.5-nat rungs
    for fg in (0.99, 1.0):                          # forget never reaches the box
        g = LucidFilter(dynamics=[[0.9]], faults=True, forget=fg)
        assert np.array_equal(g.hazards, f.hazards)
    pure = LucidFilter(dynamics=[[0.9]], faults=True, forget=1.0)
    st = pure.update([0.3])
    assert np.isfinite(st.loglik) and 0.0 <= st.fault <= 1.0 and st.hazard > 0.0


def test_hazard_is_read_not_told():
    """The regime readout: calm drives the posterior-mean hazard toward the ladder's floor,
    and a pinned hazard just reports itself.  Two sensors of one state keep the split ladder
    (an orthogonal mechanism) out of the bank so the test is cheap."""
    r = rng(11)
    T = 900
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.9 * x[t - 1] + 0.3 * r.standard_normal()
    Y = x[:, None] + 0.7 * r.standard_normal((T, 2))
    kw = dict(dynamics=[[0.9]], H=[[1.0], [1.0]], process=[[0.09]],
              measurement=[0.49, 0.49])
    ladder = LucidFilter(faults=True, **kw).filter(Y)
    assert ladder.hazard.shape == (T,)
    assert ladder.hazard[0] > 0.05                  # the log-uniform prior's mean, pre-data
    assert ladder.hazard[-1] < 0.01                 # a quiet run reads as a quiet regime
    pinned = LucidFilter(faults=1e-3, **kw).filter(Y)
    assert np.allclose(pinned.hazard, 1e-3)         # give-what-you-know reports what you gave
    off = LucidFilter(**kw).filter(Y)
    assert off.hazard is None                       # no fault class, no regime to report


def test_pinned_hazard_equals_length_one_ladder():
    Y, _ = ar1(T=400, a=0.9, seed=3)
    kw = dict(dynamics=[[0.9]], process=[[0.09]], measurement=[0.25])
    a = LucidFilter(faults=2e-3, **kw).filter(Y)
    b = LucidFilter(faults=[2e-3], **kw).filter(Y)
    assert np.allclose(a.mean, b.mean) and np.allclose(a.fault, b.fault)


def test_callable_dynamics_relinearises():
    """Real F/B arrive linearised per operating point, so `dynamics` may be a callable."""
    g = rng(9)
    T = 400
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
    assert _rmse(moving.mean[:, 0], x, 150) <= _rmse(frozen.mean[:, 0], x, 150)
    # a CONSTANT callable must reduce to the matrix it returns
    same = LucidFilter(dynamics=lambda s: np.array([[0.9]]), n=1, **kw).filter(Y)
    assert np.allclose(same.mean, frozen.mean, atol=1e-8)


def test_callable_dynamics_with_faults():
    """The departure channel rides on top of a moving linearisation."""
    T = 400
    Y, x = ar1(T=T, a=0.6, seed=3)
    f = LucidFilter(dynamics=lambda s: np.array([[0.6]]), n=1, faults=1e-3,
                    process=[[0.09]], measurement=[0.25])
    r = f.filter(Y)
    assert np.all(np.isfinite(r.dynamics)) and r.dynamics.shape == (T, 1, 1)
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
        def update(self, y, u=None, a=1.0):
            return _WalkEngine.update(self, y, u=u, a=a)

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


def test_sensor_read_out_names_the_biased_sensor():
    """The signed per-sensor offset -- the one thing a second-moment channel cannot report.

    A scale sees only ``e**2``, so a biased sensor and its innocent neighbour move ``eta`` the
    same way (research/bias-channels 0001 measured +0.71 / +0.74 at m = 2).  The read-out is
    relative to the consensus, because the common mode of the biases is gauge on a random walk.
    """
    Q, R, N, T0, bias = 0.02, 1.0, 700, 300, 2.0
    rng = np.random.default_rng(7)
    theta = np.cumsum(rng.normal(0, np.sqrt(Q), N))
    Y = np.stack([theta + rng.normal(0, np.sqrt(R), N) for _ in range(3)], axis=1)
    Y[T0:, 2] += bias

    r = LucidFilter(H=np.ones((3, 1)), measurement=np.ones(3), offsets=True).filter(Y)
    c = r.sensor_offset[-1]
    assert c[2] - 0.5 * (c[0] + c[1]) > 0.7 * bias               # sensor 3 stands out
    assert abs(c[0] - c[1]) < 0.3 * bias                         # the other two agree
    assert abs(float(np.mean(c))) < 0.05 * bias                  # and the gauge is not claimed


def test_sensor_read_out_cannot_change_the_filter():
    """The observer is reported and never acted on, so it must not move a single number.

    Acting on it fails both ways (research/bias-channels 0004, 0006): applied to the state it
    adopts the gauge convention and loses to doing nothing, and left in the innovation it
    corrupts the process entry.  It is therefore built to be inert, and that is pinned here
    bit-for-bit rather than argued.
    """
    Q, R, N, T0 = 0.02, 1.0, 500, 250
    rng = np.random.default_rng(7)
    theta = np.cumsum(rng.normal(0, np.sqrt(Q), N) + 0.1 * (np.arange(N) >= T0))
    Y = np.stack([theta + rng.normal(0, np.sqrt(R), N) for _ in range(3)], axis=1)
    Y[T0:, 2] += 2.0
    H, R0 = np.ones((3, 1)), np.ones(3)

    a = LucidFilter(H=H, measurement=R0, offsets=True)
    b = LucidFilter(H=H, measurement=R0, offsets=True)
    b._sensor = None                                             # the only difference
    ra, rb = a.filter(Y), b.filter(Y)
    assert np.array_equal(ra.mean, rb.mean)
    assert np.array_equal(ra.var, rb.var)
    assert ra.loglik == rb.loglik
    assert np.array_equal(ra.offset, rb.offset)


def test_read_out_and_drift_are_complementary():
    """Whichever of the pair is identifiable is the one that is carried.

    On a stable ``F`` the drift is confounded with a sensor bias and the drift channel is inert
    -- and it is exactly there that ``H ker(F - I)`` is empty, so no bias is gauge and the
    read-out becomes ABSOLUTE rather than relative to the consensus.
    """
    F, H = np.array([[0.8]]), np.ones((2, 1))
    rng = np.random.default_rng(3)
    x, X = 0.0, []
    for _ in range(500):
        x = 0.8 * x + rng.normal(0, 0.3)
        X.append(x)
    X = np.array(X)
    Y = np.stack([X + rng.normal(0, 1.0, 500) for _ in range(2)], axis=1)
    Y[250:, 1] += 1.5

    f = LucidFilter(dynamics=F, H=H, measurement=np.ones(2), offsets=True)
    assert f._mean is None                                       # the drift is not identifiable
    assert f._sensor is not None and f._sensor.k == 2            # both biases are
    r = f.filter(Y)
    assert r.offset is None
    assert abs(r.sensor_offset[-1, 0]) < 0.4                     # absolute, not relative
    assert r.sensor_offset[-1, 1] > 0.7 * 1.5


def test_offset_feedback_is_never_partial_on_a_tower():
    """A Jordan tower's offset basis is kept WHOLE, and feeding it does not corrode the state.

    The defect this guards (research/bias-channels 0015): the earlier sensor-column quotient
    truncated a tower's offset component-wise, and feeding HALF an offset left a permanent
    innovation tension while the channel's own success calmed the scale walk covering it --
    4.1x the settled state error at 10x overconfidence, reproduced with the estimate replaced
    by the exact truth.  The basis rule is now the z = 1 generalized eigenspace, which keeps
    the tower whole; this pins both the basis and the settled-window behaviour.
    """
    import math

    dt, order = 0.01, 3
    F = np.eye(order)
    for i in range(order):
        for j in range(i + 1, order):
            F[i, j] = dt ** (j - i) / math.factorial(j - i)
    G = np.array([dt ** (order - i) / math.factorial(order - i) for i in range(order)])
    Q0 = 0.6 ** 2 * np.outer(G, G) + 1e-12 * np.eye(order)
    H = np.array([[1.0, 0, 0], [0, 0, 1.0]])
    R0 = np.array([0.06 ** 2, 0.02 ** 2])

    from lucid.statfilter.lucid import _mean_basis
    B = _mean_basis(F, H)
    assert B.shape[1] == 2                                       # accel AND velocity means
    # the tower is all one z = 1 block: nothing of it may be truncated away
    span = B[:order] @ B[:order].T
    for v in (np.array([0.0, 1, 0]), np.array([0.0, 0, 1])):
        assert np.linalg.norm(span @ v - v) < 1e-6

    T, T0 = 1500, 300
    rng = np.random.default_rng(0)
    t = np.arange(T) * dt
    U = (2.0 * np.sin(2 * np.pi * 0.35 * t))[:, None]
    s = np.zeros(order)
    S, Y = np.zeros((T, order)), np.zeros((T, 2))
    for k in range(T):
        d = 1.2 if k >= T0 else 0.0
        s = F @ s + G * U[k, 0] + G * (0.6 * rng.standard_normal() + d)
        S[k] = s
        Y[k] = H @ s + np.sqrt(R0) * rng.standard_normal(2)

    kw = dict(dynamics=F, control=G[:, None], H=H, process=Q0, measurement=R0)
    on = LucidFilter(offsets=True, **kw).filter(Y, U=U)
    off = LucidFilter(**kw).filter(Y, U=U)
    sl = slice(900, T)
    e_on = np.sqrt(np.mean((on.mean[sl, 0] - S[sl, 0]) ** 2))
    e_off = np.sqrt(np.mean((off.mean[sl, 0] - S[sl, 0]) ** 2))
    cal = float(np.mean((on.mean[sl, 0] - S[sl, 0]) ** 2 / on.var[sl, 0, 0]))
    assert e_on < 2.2 * e_off                                    # was 4.1x under the defect
    assert cal < 4.0                                             # was 10-18x overconfident
    # (the short window still carries some of the convergence transient; the settled
    #  two-seed measurement in 0015 is 1.5x and 1.16)


# ------------------------------------- partial observation, the clock, and streaming
# The filter's native input is an EVENT -- the sensors that read at one instant, and
# when.  A synchronous, fully-observed row at the nominal step is the special case, and
# these first pin that it stayed EXACTLY the special case it was.

def two_sensor(T=300, seed=0, dt=0.1):
    """Position + velocity, read by a coarse absolute sensor and a precise rate one."""
    r = rng(seed)
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.eye(2)
    x = np.zeros(2); X = np.empty((T, 2)); Y = np.empty((T, 2))
    for t in range(T):
        x = F @ x + r.standard_normal(2) * np.array([0.01, 0.05])
        X[t] = x
        Y[t] = x + r.standard_normal(2) * np.array([0.30, 0.02])
    return F, H, X, Y


def test_uniform_full_rows_are_untouched():
    """Supplying no clock must be the filter that existed before there was one.

    A narrow ``(phi, s)`` box, here and in the two clock tests below: what is compared is
    one construction against itself, so the size of the bank is not part of the claim and
    every cell of it would only say the same thing again.
    """
    F, H, X, Y = two_sensor(T=100)
    kw0 = dict(dynamics=F, H=H, phis=(0.85,), ss=(0.40,))
    base = LucidFilter(**kw0).filter(Y)
    for kw in (dict(dt=1.0), dict(t=np.arange(len(Y), dtype=float))):
        got = LucidFilter(**kw0).filter(Y, **kw)
        assert np.array_equal(base.mean, got.mean)          # bit-for-bit, not "close"
        assert np.array_equal(base.var, got.var)
        assert np.array_equal(base.measurement_scale, got.measurement_scale)
        assert base.loglik == got.loglik


def test_partial_row_uses_the_sensors_that_reported():
    """A NaN entry is one sensor that did not report -- not a lost row."""
    F, H, X, Y = two_sensor()
    keep = (np.arange(len(Y)) % 5) == 0
    part = Y.copy(); part[~keep, 0] = np.nan          # the slow sensor
    drop = Y.copy(); drop[~keep, :] = np.nan          # what the old filter had to do
    p = LucidFilter(dynamics=F, H=H).filter(part)
    d = LucidFilter(dynamics=F, H=H).filter(drop)
    assert np.all(np.isfinite(p.mean))
    # the present sensor is corrected on, the absent one reports no innovation
    assert np.all(np.isnan(p.innovation[~keep, 0]))
    assert np.all(np.isfinite(p.innovation[~keep, 1]))
    def rms(a, b): return float(np.sqrt(np.mean((a[40:] - b[40:]) ** 2)))
    assert rms(p.mean[:, 1], X[:, 1]) < rms(d.mean[:, 1], X[:, 1])
    assert p.loglik > d.loglik                        # more data, more density explained


def test_all_missing_row_still_propagates():
    Y, _ = local_level(T=100)
    Y[30:35] = np.nan
    r = LucidFilter().filter(Y)
    assert np.all(np.isfinite(r.mean))
    assert np.all(np.isnan(r.innovation[30:35]))


def test_observe_is_one_point():
    """(sensor, timestamp, value), one sensor at a time."""
    F, H, X, Y = two_sensor(T=60)
    f = LucidFilter(dynamics=F, H=H)
    st = f.observe(1, Y[0, 1], t=0.0)
    assert isinstance(st, LucidStep) and st.time == 0.0
    assert np.isnan(st.innovation[0]) and np.isfinite(st.innovation[1])
    assert f.time == 0.0
    st = f.observe(0, Y[1, 0], t=1.0)
    assert st.time == 1.0 and np.isfinite(st.innovation[0])
    with pytest.raises(ValueError):
        f.observe(0, 1.0, t=0.5)                      # time may not run backwards
    with pytest.raises(ValueError):
        f.observe(7, 1.0)                             # no such sensor


def test_pointwise_decomposition_tracks_the_joint_row():
    """m points sharing a timestamp track what the row tracks -- but they do not learn
    what it learns, and the gap is the point.

    A joint row splits process from sensor noise because a process mode reaches several
    entries of ``S`` while a sensor reaches one.  Delivered as m points, each event's ``S``
    is a SCALAR and the two are exactly proportional in it, so the split is invisible at
    every such step and the filter declines to move it (see `_subset_groups`).  The state
    therefore stays close -- the state only ever needed the total -- while the attribution
    does not converge to the row's.  Pinning the state ratio and NOT pinning the scales is
    the honest statement of that.
    """
    F, H, X, Y = two_sensor(T=200)
    joint = LucidFilter(dynamics=F, H=H).filter(Y)
    pts = [(i, float(t), Y[t, i]) for t in range(len(Y)) for i in (0, 1)]
    s = LucidFilter(dynamics=F, H=H).stream(pts)
    assert s.mean.shape == (2 * len(Y), 2)
    assert np.array_equal(s.sensor[:4], np.array([0, 1, 0, 1]))
    at_instant = s.mean[1::2]
    def rms(a, b): return float(np.sqrt(np.mean((a[40:] - b[40:]) ** 2)))
    ratio = rms(at_instant, X) / rms(joint.mean, X)
    assert 0.9 < ratio < 1.25                         # tracks it; is not identical to it
    # and it is genuinely the same machinery, not a second filter that happens to be close
    assert np.all(np.isfinite(s.var)) and np.all(np.diagonal(s.var, axis1=1, axis2=2) > 0)


def test_variable_step_beats_assuming_uniformity():
    """Irregular arrivals, with the timestamps and without them."""
    r = rng(4)
    T, nominal = 400, 0.1
    gaps = np.maximum(r.gamma(4.0, nominal / 4.0, T), 1e-3)
    t = np.cumsum(gaps)
    F = np.array([[1.0, nominal], [0.0, 1.0]])
    H = np.eye(2)
    x = np.zeros(2); X = np.empty((T, 2)); Y = np.empty((T, 2))
    for i, a in enumerate(gaps):
        x = np.array([[1.0, a], [0.0, 1.0]]) @ x + r.standard_normal(2) * np.array(
            [0.01, 0.05]) * math.sqrt(a / nominal)
        X[i] = x
        Y[i] = x + r.standard_normal(2) * np.array([0.30, 0.02])
    kw = dict(dynamics=F, H=H)
    told = LucidFilter(timestep=nominal, **kw).filter(Y, t=t)
    guessed = LucidFilter(**kw).filter(Y)
    def rms(a, b): return float(np.sqrt(np.mean((a[40:] - b[40:]) ** 2)))
    assert rms(told.mean[:, 0], X[:, 0]) < rms(guessed.mean[:, 0], X[:, 0])
    assert np.allclose(told.time, t)


def test_zero_gap_moves_no_state():
    """Two readings at one instant: the second must not re-propagate the first."""
    F, H, X, Y = two_sensor(T=10)
    kw = dict(dynamics=F, H=H, phis=(0.85,), ss=(0.40,))
    f = LucidFilter(**kw)
    f.observe(0, Y[0, 0], t=0.0)
    a = f.observe(1, Y[0, 1], t=0.0)
    g = LucidFilter(**kw)
    g.observe(0, Y[0, 0], t=0.0)
    b = g.observe(1, Y[0, 1], dt=0.0)
    assert np.allclose(a.mean, b.mean)
    assert a.time == b.time == 0.0


def test_timestep_sets_the_unit():
    """A model clocked in seconds is the same filter as one clocked in step counts.

    Exactly so when the timestep is binary-exact, and to numerical noise otherwise: a
    decimal rate like 0.01 s does not divide its own differences to exactly 1.0, so the
    gap arrives as 1 + O(eps) and takes the general propagator rather than the identity
    short-circuit.  That is a representation fact, not a modelling one, and it must not
    be papered over with a snap-to-nominal tolerance -- there are no thresholds here.
    """
    F, H, X, Y = two_sensor(T=60)
    kw = dict(dynamics=F, H=H, phis=(0.85,), ss=(0.40,))
    steps = LucidFilter(**kw).filter(Y)
    exact = LucidFilter(timestep=0.25, **kw).filter(
        Y, t=0.25 * np.arange(len(Y)))
    assert np.array_equal(steps.mean, exact.mean)
    decimal = LucidFilter(timestep=0.01, **kw).filter(
        Y, t=0.01 * np.arange(len(Y)))
    assert np.abs(decimal.mean - steps.mean).max() < 1e-9


def test_matrix_power_handles_the_defective_transition():
    """A constant-velocity block has one eigenvector for two states; F**a must still be
    the exact [[1, a dt], [0, 1]] an eigendecomposition would get wrong."""
    from lucid.statfilter.lucid import _Propagator
    dt = 0.1
    p = _Propagator(np.array([[1.0, dt], [0.0, 1.0]]))
    for a in (0.0, 0.25, 1.0, 2.5):
        Fa, _ = p.at(a)
        assert np.allclose(Fa, [[1.0, a * dt], [0.0, 1.0]], atol=1e-12)
    half, _ = p.at(0.5)
    assert np.allclose(half @ half, [[1.0, dt], [0.0, 1.0]], atol=1e-12)
    with pytest.raises(ValueError):                   # no real generator -> say so
        _Propagator(np.array([[-1.0, 0.0], [0.0, -2.0]])).at(0.5)


def test_control_forcing_is_continuous_through_the_nominal_step():
    """B is the ONE-STEP forcing map, so the elapsed map must equal it at a = 1 and
    vanish at a = 0 -- continuous through the step, not merely equal at it."""
    from lucid.statfilter.lucid import _Propagator
    p = _Propagator(np.array([[0.6, 0.2], [0.0, 0.9]]))
    assert np.allclose(p.at(1.0)[1], np.eye(2))
    assert np.allclose(p.at(0.0)[1], np.zeros((2, 2)))
    near = p.at(1.0 - 1e-6)[1]
    assert np.abs(near - np.eye(2)).max() < 1e-5


def test_a_partial_event_moves_the_total_and_holds_the_split():
    """A partial event may move a direction it can see, and may not move one it cannot.

    With one sensor reporting, ``S`` is a scalar: a process mode that sensor sees and the
    sensor's own noise enter it additively, so their scale scores are exactly proportional
    and their SPLIT is invisible.  The event moves the pair's total, which it does see, and
    holds the split at whatever identifiable evidence already made it.  A full row is
    unaffected -- it can see the split, so it moves it.
    """
    F, H, X, Y = two_sensor(T=60)
    kw = dict(dynamics=F, H=H)                     # H = I pairs every mode with a sensor
    f = LucidFilter(**kw)
    eng = f._members[0]
    obs1 = np.array([0])
    assert eng.event_groups(obs1, 1)[0], "a single-sensor event must confound a pair"
    assert eng.event_groups(np.array([0, 1]), 2)[0] is eng._groups, "a full row is the model's"

    # one sensor at a time: each pair's log-odds must not move, though its total may
    f.reset()
    for t in range(12):
        f.update(np.array([Y[t, 0], np.nan]))
        f.update(np.array([np.nan, Y[t, 1]]))
    e = f._members[0]
    los = [lo for _, lo in e._group_read(e.mu)]
    tots = [tt for tt, _ in e._group_read(e.mu)]
    f.update(np.array([Y[12, 0], np.nan]))
    los2 = [lo for _, lo in e._group_read(e.mu)]
    tots2 = [tt for tt, _ in e._group_read(e.mu)]
    assert np.allclose(los, los2, atol=1e-9), "a single-sensor event moved a split it cannot see"
    assert not np.allclose(tots, tots2), "it must still move the total it can see"

    # AND on a rig whose MODEL has no confounded pairs at all -- two sensors reading the
    # same state, so no process mode is read by exactly one of them.  A single-sensor event
    # still confounds, because its S is a scalar; gating the hold on the model's pairs
    # instead of the event's silently drops it here, and full rows stay bit-identical while
    # it does (that is how it got in: research/pointwise-streaming/0005 caught it, the
    # bit-identity audit could not).
    Hb = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])   # every state read by two sensors
    kb = dict(dynamics=F, H=Hb, measurement=[0.3, 0.02, 0.4])
    fb = LucidFilter(**kb)
    eb = fb._members[0]
    assert not eb._groups, "this rig's MODEL must have no pairs, or it tests nothing"
    assert eb.event_groups(np.array([0]), 1)[0], "one sensor must still confound a pair"
    for t in range(10):
        fb.update(np.array([Y[t, 0], np.nan, np.nan]))
    lo_b = [lo for _, lo in eb._group_read(eb.mu, eb.event_groups(np.array([0]), 1)[0])]
    fb.update(np.array([Y[10, 0], np.nan, np.nan]))
    lo_b2 = [lo for _, lo in eb._group_read(eb.mu, eb.event_groups(np.array([0]), 1)[0])]
    assert np.allclose(lo_b, lo_b2, atol=1e-9), "the split moved on an event that cannot see it"

    # the full row is untouched by any of this: it moves the split
    g = LucidFilter(**kw)
    for t in range(12):
        g.update(Y[t])
    eg = g._members[0]
    before = [lo for _, lo in eg._group_read(eg.mu)]
    g.update(Y[12])
    after = [lo for _, lo in eg._group_read(eg.mu)]
    assert not np.allclose(before, after)


def test_stream_with_a_dynamics_channel():
    """Learned dynamics survive being fed one sensor at a time."""
    Y, x = ar1(T=400, a=0.6, seed=2)
    pts = [(0, float(t), Y[t, 0]) for t in range(len(Y))]
    r = LucidFilter(dynamics=None).stream(pts)
    assert r.dynamics.shape == (len(Y), 1, 1)
    assert np.all(np.isfinite(r.mean)) and r.fault is not None
    assert abs(r.dynamics[-1, 0, 0] - 0.6) < 0.35

