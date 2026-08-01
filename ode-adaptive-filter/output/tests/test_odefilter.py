"""Tests for odefilter.

The load-bearing one is `test_reduces_to_parent`: with p = 1 and alpha = 1 this
filter's model IS the parent's local-level model, so the two must agree to
numerical precision on the same data.  If that ever fails, the claim that this
is a strict extension is false.

Run the fast subset with:  pytest -m "not slow"
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from odefilter import OdeFilter, Params, difference_matrix          # noqa: E402
from odefilter.core import _iv_alpha, _moment_noises                # noqa: E402

PARENT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "..", "adaptive-random-walk-filter", "output")

ALPHA3 = (2.785218519281637, -2.6855430450862655, 0.9003245225862656)


def ar(n, alpha, Q, S2, rng):
    """AR(p) plus white measurement noise, with no retroactive state edits."""
    p = len(alpha)
    z = np.zeros(p)
    x = np.zeros(n)
    for t in range(n):
        xn = float(np.dot(alpha, z)) + math.sqrt(Q) * rng.standard_normal()
        z = np.concatenate([[xn], z[:-1]])
        x[t] = xn
    return x, x + math.sqrt(S2) * rng.standard_normal(n)


# ------------------------------------------------------------------ structure
def test_difference_matrix_is_an_involution():
    for p in (1, 2, 3, 5):
        D = difference_matrix(p)
        assert np.allclose(D @ D, np.eye(p))
        assert abs(abs(np.linalg.det(D)) - 1.0) < 1e-12


def test_roots_and_memory():
    pr = Params(alpha=ALPHA3, Q=1.0, s2=1.0)
    r = np.sort_complex(pr.roots)
    assert np.isclose(np.max(np.abs(r)), 1.0, atol=1e-6)   # the offset root
    # a numerically-exact unit root is unattainable from data, so this is very
    # large rather than infinite; only a root at or outside the circle gives inf
    assert pr.memory() > 1e6
    assert math.isinf(Params(alpha=(1.0,), Q=1.0, s2=1.0).memory())
    pr2 = Params(alpha=(0.5,), Q=1.0, s2=1.0)
    assert np.isclose(pr2.memory(), 2.0)


def test_params_validation():
    with pytest.raises(ValueError):
        Params(alpha=(1.0,), Q=-1.0, s2=1.0)
    with pytest.raises(ValueError):
        Params(alpha=(1.0,), Q=1.0, s2=1.0, phi_P=1.0)
    with pytest.raises(ValueError):
        Params(alpha=(1.0,), Q=1.0, s2=1.0, s_M=-0.1)


def test_roundtrip():
    f = OdeFilter(Params(alpha=ALPHA3, Q=1.0, s2=2.0, s_M=0.4, phi_M=0.7), order=5)
    g = OdeFilter.from_dict(f.to_dict())
    assert g.params == f.params and g.order == f.order


# ------------------------------------------------------------------- the core
def test_reduces_to_parent():
    """p = 1, alpha = 1 is exactly the parent's local-level model."""
    sys.path.insert(0, os.path.abspath(PARENT))
    statfilter = pytest.importorskip("statfilter")

    rng = np.random.default_rng(11)
    theta = np.cumsum(rng.standard_normal(500))
    y = theta + 2.0 * rng.standard_normal(500)

    for (Q, s2, sM, phiM) in [(1.0, 4.0, 0.0, 0.0), (0.5, 3.0, 0.6, 0.8)]:
        mine = OdeFilter(Params(alpha=(1.0,), Q=Q, s2=s2, s_M=sM, phi_M=phiM),
                         order=5)
        theirs = statfilter.AdaptiveFilter(
            statfilter.Params(Q=Q, s2=s2, s_M=sM, phi_M=phiM), order=5)
        a, b = mine.filter(y), theirs.filter(y)
        # the two implementations differ only in how they carry the state, so
        # the agreement should be far tighter than any modelling difference
        assert abs(a.loglik - b.loglik) < 1e-6 * abs(b.loglik) + 1e-6
        assert np.allclose(a.mean, b.mean, rtol=1e-8, atol=1e-8)
        assert np.allclose(a.var, b.var, rtol=1e-6, atol=1e-8)


def test_shares_sum_to_one():
    rng = np.random.default_rng(3)
    x, y = ar(400, ALPHA3, 1.0, 9.0, rng)
    r = OdeFilter(Params(alpha=ALPHA3, Q=1.0, s2=9.0), order=5).filter(y)
    s = r.share_prior + r.share_process + r.share_measurement
    assert np.allclose(s, 1.0, atol=1e-10)


def test_missing_observations():
    rng = np.random.default_rng(4)
    x, y = ar(300, ALPHA3, 1.0, 9.0, rng)
    y2 = y.copy()
    y2[100:110] = np.nan
    r = OdeFilter(Params(alpha=ALPHA3, Q=1.0, s2=9.0), order=5).filter(y2)
    assert np.all(np.isfinite(r.mean))
    assert np.isnan(r.innovation[105])
    assert r.var[109] > r.var[99]              # uncertainty grows through a gap


def test_explosive_alpha_is_signalled_not_nan():
    rng = np.random.default_rng(5)
    y = rng.standard_normal(200)
    f = OdeFilter(Params(alpha=(3.0, -3.0, 2.0), Q=1.0, s2=1.0), order=5)
    assert f.loglik(y) == -np.inf


def test_streaming_matches_batch():
    rng = np.random.default_rng(6)
    x, y = ar(250, ALPHA3, 1.0, 9.0, rng)
    f = OdeFilter(Params(alpha=ALPHA3, Q=1.0, s2=9.0), order=5)
    r = f.filter(y)
    f.reset()
    means = [f.update(v).mean for v in y]
    assert np.allclose(means, r.mean, rtol=1e-10, atol=1e-10)


def test_filter_does_not_disturb_streaming_state():
    rng = np.random.default_rng(8)
    x, y = ar(150, ALPHA3, 1.0, 9.0, rng)
    f = OdeFilter(Params(alpha=ALPHA3, Q=1.0, s2=9.0), order=5)
    for v in y[:50]:
        f.update(v)
    before = f.predict(3)
    f.filter(y)
    assert f.predict(3) == before


# ------------------------------------------------------- closed-form starters
def test_iv_beats_ols_on_the_oscillator():
    """Regressing on observed lags deletes the complex pair; IV does not."""
    rng = np.random.default_rng(9)
    a = np.array(ALPHA3)
    x, y = ar(4000, a, 1.0, 9.0, rng)
    p = 3
    idx = np.arange(p, y.size)
    W = np.column_stack([y[idx - i] for i in range(1, p + 1)])
    ols = np.linalg.lstsq(W, y[idx], rcond=None)[0]
    iv = _iv_alpha(y, p)

    def pair(al):
        r = np.roots(np.concatenate([[1.0], -np.asarray(al)]))
        c = r[np.abs(r.imag) > 1e-9]
        return float(np.abs(c[0])) if c.size else 0.0

    truth = pair(a)
    assert abs(pair(iv) - truth) < 0.5 * max(abs(pair(ols) - truth), 1e-9)


def test_moment_noises_get_s2_well_and_Q_badly():
    """The conditioning result recorded in `_moment_noises`' docstring.

    S2 is identified by the k >= 1 residual autocovariances and comes out
    clean.  Q is gamma_0 minus S2 |c|^2 and is only ~0.7% of gamma_0 for this
    smooth a process, so it is amplified ~150x and is a scale hint only.
    """
    rng = np.random.default_rng(10)
    x, y = ar(6000, ALPHA3, 1.0, 9.0, rng)
    Q, s2 = _moment_noises(y, np.array(ALPHA3))
    assert 7.5 < s2 < 10.5              # S2 is genuinely estimated
    assert Q > 0                        # Q is positive and usable as a start
    assert Q < 0.5 * (np.var(np.diff(y)))   # ...and not absurd


# ------------------------------------------------------------- the diagnostic
def test_whiteness_flags_wrong_dynamics_and_not_right_ones():
    """The orthogonality result of exploration/0025, as a test."""
    rng = np.random.default_rng(12)
    x, y = ar(3000, ALPHA3, 1.0, 9.0, rng)

    right = OdeFilter(Params(alpha=ALPHA3, Q=1.0, s2=9.0), order=5).filter(y)
    wrong_a = (2.4, -2.1, 0.68)
    wrong = OdeFilter(Params(alpha=wrong_a, Q=1.0, s2=9.0), order=5).filter(y)

    assert abs(right.whiteness[-1]) < 0.05
    assert abs(wrong.whiteness[-1]) > 4 * abs(right.whiteness[-1])


def test_a_pure_event_leaves_no_whiteness_trace():
    """A one-off kick is process noise: it moves the mean, not the whiteness."""
    rng = np.random.default_rng(13)
    x, y = ar(3000, ALPHA3, 1.0, 9.0, rng)
    y2 = y.copy()
    y2[1500] += 30.0                          # a large one-off event
    f = OdeFilter(Params(alpha=ALPHA3, Q=1.0, s2=9.0), order=5)
    base, hit = f.filter(y), f.filter(y2)
    assert abs(hit.innovation[1500]) > 5 * np.std(base.innovation)
    assert abs(hit.whiteness[-1] - base.whiteness[-1]) < 0.02


# -------------------------------------------------------------------- fitting
@pytest.mark.slow
def test_fit_recovers_the_modes():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(21)
    x, y = ar(1200, ALPHA3, 1.0, 9.0, rng)
    f = OdeFilter.fit(y, p=3, order=5)
    r = np.abs(f.params.roots)
    assert abs(np.max(r) - 1.0) < 0.02                  # the offset root
    comp = f.params.roots[np.abs(f.params.roots.imag) > 1e-6]
    assert comp.size == 2                               # the oscillator survives
    assert abs(np.abs(comp[0]) - 0.9489) < 0.08
    assert 0.5 < f.params.Q < 2.0
    assert 6.0 < f.params.s2 < 13.0


@pytest.mark.slow
def test_fit_finds_no_scale_variation_when_there_is_none():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(22)
    x, y = ar(1000, ALPHA3, 1.0, 9.0, rng)
    f = OdeFilter.fit(y, p=3, order=5)
    assert f.params.s_P < 0.35 and f.params.s_M < 0.35
