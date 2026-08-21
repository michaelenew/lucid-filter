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
from odefilter.core import (_iv_alpha, _moment_noises, _pin_maps,   # noqa: E402
                            _companion)

PARENT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "..", "random-walk-filter", "output")

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
    """p = 1, alpha = 1 is the parent's local-level model.

    The contract is in two parts since the GPB1 collapse was removed.  On
    the s = 0 face the mixture is a single node, no collapse of any kind is
    in play, and the two implementations must agree to round-off.  With a
    live scale channel the two share the MODEL but not the inference: the
    parent collapses to one covariance per step (GPB1 by construction),
    this filter carries one per node, so the outputs differ by the collapse
    -- measured at ~6e-3 nats/pt on this data -- and the assertion is only
    that they describe the same model to that order, not each other's bits.
    """
    sys.path.insert(0, os.path.abspath(PARENT))
    statfilter = pytest.importorskip("statfilter")

    rng = np.random.default_rng(11)
    theta = np.cumsum(rng.standard_normal(500))
    y = theta + 2.0 * rng.standard_normal(500)

    # exact on the s = 0 face
    mine = OdeFilter(Params(alpha=(1.0,), Q=1.0, s2=4.0), order=5)
    theirs = statfilter.AdaptiveFilter(statfilter.Params(Q=1.0, s2=4.0),
                                       order=5)
    a, b = mine.filter(y), theirs.filter(y)
    assert abs(a.loglik - b.loglik) < 1e-6 * abs(b.loglik) + 1e-6
    assert np.allclose(a.mean, b.mean, rtol=1e-8, atol=1e-8)
    assert np.allclose(a.var, b.var, rtol=1e-6, atol=1e-8)

    # same model, finer inference, with the channel live
    mine = OdeFilter(Params(alpha=(1.0,), Q=0.5, s2=3.0, s_M=0.6, phi_M=0.8),
                     order=5)
    theirs = statfilter.AdaptiveFilter(
        statfilter.Params(Q=0.5, s2=3.0, s_M=0.6, phi_M=0.8), order=5)
    a, b = mine.filter(y), theirs.filter(y)
    assert abs(a.loglik - b.loglik) / y.size < 0.05
    assert np.corrcoef(a.mean, b.mean)[0, 1] > 0.999


def test_shares_sum_to_one():
    rng = np.random.default_rng(3)
    x, y = ar(400, ALPHA3, 1.0, 9.0, rng)
    r = OdeFilter(Params(alpha=ALPHA3, Q=1.0, s2=9.0), order=5).filter(y)
    s = r.share_prior + r.share_process + r.share_measurement
    assert np.allclose(s, 1.0, atol=1e-10)


def test_result_convenience_views_match_the_parents():
    """FilterResult exposes the same derived views the parent's does."""
    rng = np.random.default_rng(31)
    x, y = ar(200, ALPHA3, 1.0, 9.0, rng)
    r = OdeFilter(Params(alpha=ALPHA3, Q=1.0, s2=9.0, s_M=0.5, phi_M=0.7),
                  order=5).filter(y)
    assert np.allclose(r.process_scale, r.process_anomaly + r.process_regime)
    assert np.allclose(r.measurement_scale,
                       r.measurement_anomaly + r.measurement_regime)
    assert r.modes.shape == (200, 4)
    assert np.allclose(r.modes[:, 2] + r.modes[:, 3], r.measurement_scale)


def test_pred_var_is_the_variance_loglik_scores():
    """With both scale channels off the grid is one node, so the predictive
    law is exactly Gaussian and `loglik` must be its density at `innovation`.
    That identity pins `pred_var` to the right quantity; on a mixture it is
    the mixture's variance, spread included, so calibration stays near 1.
    """
    rng = np.random.default_rng(17)
    x, y = ar(400, ALPHA3, 1.0, 9.0, rng)

    f = OdeFilter(Params(alpha=ALPHA3, Q=1.0, s2=9.0), order=5).reset()
    for v in y:
        st = f.update(float(v))
        exact = -0.5 * (st.innovation ** 2 / st.pred_var
                        + math.log(st.pred_var) + math.log(2.0 * math.pi))
        assert st.loglik == pytest.approx(exact, rel=1e-10, abs=1e-10)

    r = OdeFilter(Params(alpha=ALPHA3, Q=1.0, s2=9.0, s_M=0.5, phi_M=0.7),
                  order=5).filter(y)
    assert r.pred_var.shape == (400,)
    assert np.all(r.pred_var > 0.0)
    calib = float(np.mean(r.innovation[50:] ** 2 / r.pred_var[50:]))
    assert 0.7 < calib < 1.4


def test_missing_observations():
    rng = np.random.default_rng(4)
    x, y = ar(300, ALPHA3, 1.0, 9.0, rng)
    y2 = y.copy()
    y2[100:110] = np.nan
    r = OdeFilter(Params(alpha=ALPHA3, Q=1.0, s2=9.0), order=5).filter(y2)
    assert np.all(np.isfinite(r.mean))
    assert np.isnan(r.innovation[105])
    assert r.var[109] > r.var[99]              # uncertainty grows through a gap


def test_explosive_alpha_never_yields_nan():
    """An unconstrained search reaches explosive alpha.  The per-node
    recursion keeps a DETECTABLE explosive system's posterior bounded (the
    measurement corrects the state every step), so a moderately explosive
    alpha now carries a genuinely finite likelihood -- the old -inf there
    was the shared-covariance recursion overflowing, not a property of the
    model.  What is guaranteed: never NaN (a NaN sends the optimiser
    chasing it) -- and _loglik_batch's dead-row semantics turn any residual
    overflow into -inf rather than poisoning the batch.  A consequence worth
    knowing: the likelihood no longer walls off the unit disc numerically,
    so a fit CAN land marginally outside it (an applied p=4, unit_roots=2 cell
    did); the disc is a modelling commitment, not an emergent property."""
    rng = np.random.default_rng(5)
    y = rng.standard_normal(200)
    for a in [(3.0, -3.0, 2.0), (40.0, -40.0, 39.0)]:
        ll = OdeFilter(Params(alpha=a, Q=1.0, s2=1.0), order=5).loglik(y)
        assert not math.isnan(ll) and ll < 0.0


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


# ------------------------------------------------------------- pinned roots
def test_pin_maps_are_exact():
    """alpha = base + beta @ M must carry (z-1)^d exactly, coefficient-wise.

    Checked by synthetic division, not by np.roots: root-finding splits a
    multiple root by ~eps^(1/d), which is the ill-conditioning of the QUESTION,
    not of the construction -- the coefficients are exact integers times beta.
    """
    rng = np.random.default_rng(0)
    for p, d in [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (4, 2), (3, 3)]:
        base, M = _pin_maps(p, d)
        beta = rng.normal(size=p - d)
        alpha = base + beta @ M
        c = np.concatenate([[1.0], -alpha])
        for _ in range(d):                       # divide out (z - 1), d times
            c = np.cumsum(c)
            assert abs(c[-1]) < 1e-12            # remainder = poly at z = 1
            c = c[:-1]
        assert np.allclose(c, np.concatenate([[1.0], -beta]))


def test_pinned_parent_face():
    """p = 1, unit_roots = 1 leaves nothing free: alpha IS the parent's (1,)."""
    base, M = _pin_maps(1, 1)
    assert np.allclose(base, [1.0]) and M.shape == (0, 1)
    pr = Params(alpha=(1.0,), Q=1.0, s2=4.0, unit_roots=1)
    assert pr.alpha == (1.0,)


def test_unit_roots_validation_and_roundtrips():
    with pytest.raises(ValueError):              # alpha without the claimed root
        Params(alpha=(0.5, 0.1, 0.0), Q=1.0, s2=1.0, unit_roots=1)
    with pytest.raises(ValueError):
        Params(alpha=(1.0,), Q=1.0, s2=1.0, unit_roots=2)
    v = np.array([0.42, 0.0, 0.1, 0.3, -0.2, -13.8, -13.8, 2.0, -13.8])
    pr = Params._from_vec(v, p=3, unit_roots=2)
    assert pr.unit_roots == 2
    pr2 = Params._from_vec(pr._vec(), 3, 2)      # _vec deconvolves, exactly
    assert np.allclose(pr2.alpha, pr.alpha)
    assert Params.from_dict(pr.to_dict()) == pr


def test_pinned_and_free_agree_where_they_meet():
    """The same alpha scored through the pinned coordinates and the free ones
    must give the same likelihood: the constraint changes the SEARCH SPACE,
    never the model."""
    from odefilter.core import _loglik_batch
    rng = np.random.default_rng(19)
    x, y = ar(200, ALPHA3, 1.0, 9.0, rng)
    pr = Params(alpha=ALPHA3, Q=1.0, s2=9.0, unit_roots=1)
    off = math.log(1e-6)
    noise = [0.0, math.log(9.0), 0.0, 0.0, off, off, 2.0, off]
    v_free = np.concatenate([ALPHA3, noise])
    v_pin = np.concatenate([pr._vec()[:2], noise])
    a = float(_loglik_batch(y, v_free[None, :], 3, 5, with_A=False)[0])
    b = float(_loglik_batch(y, v_pin[None, :], 3, 5, with_A=False,
                            unit_roots=1)[0])
    assert abs(a - b) < 1e-8 * abs(a)


@pytest.mark.slow
def test_fit_pinned_recovers_the_oscillator_behind_a_linear_offset():
    """On the pinned class's own data -- a linear offset whose rate is a state,
    over a damped oscillator -- the constrained fit must recover the quotient
    dynamics cleanly.  This is where the free fit cannot go: its maximum-
    likelihood unit roots land inside the circle (exploration/0040)."""
    pytest.importorskip("scipy")
    beta_true = np.array([1.7852187, -0.9003245])       # ALPHA3's oscillator
    base, M = _pin_maps(4, 2)
    A4 = tuple(base + beta_true @ M)

    rng = np.random.default_rng(30)
    x, y = ar(900, A4, 1.0, 9.0, rng)
    f = OdeFilter.fit(y, p=4, unit_roots=2, dynamics=False, max_iter=200)

    assert f.params.unit_roots == 2
    c = np.concatenate([[1.0], -np.asarray(f.params.alpha)])
    for _ in range(2):                            # (z-1)^2 divides, exactly
        c = np.cumsum(c)
        assert abs(c[-1]) < 1e-9
        c = c[:-1]
    rq = np.roots(c)                              # the quotient's roots
    assert np.abs(rq.imag).max() > 1e-6           # the oscillator survives
    assert abs(np.abs(rq[0]) - 0.9489) < 0.08
    assert 0.5 < f.params.Q < 2.0
    assert 6.0 < f.params.s2 < 13.0


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


# --------------------------------------------------------- the dynamics channel
def test_alpha_at_endpoints_and_clipping():
    """g = 0 must be the parent exactly; g = 1 the fitted alpha exactly."""
    pr = Params(alpha=ALPHA3, Q=1.0, s2=9.0)
    assert np.allclose(pr.alpha_at(0.0), [1.0, 0.0, 0.0])
    assert np.allclose(pr.alpha_at(1.0), ALPHA3)
    for g in (1.2, 1.5, 3.0, -0.5):
        r = np.abs(np.roots(np.concatenate([[1.0], -pr.alpha_at(g)])))
        assert np.max(r) <= 1.0 + 1e-6      # never leaves the unit disc
    # an explosive BASE alpha is left explosive, so fit_'s -inf guard still bites
    bad = Params(alpha=(3.0, -3.0, 2.0), Q=1.0, s2=1.0)
    assert np.allclose(bad.alpha_at(1.0), (3.0, -3.0, 2.0))


def test_dynamics_channel_off_is_inert():
    """s_A = 0 collapses the channel: order_A must then change nothing."""
    rng = np.random.default_rng(41)
    x, y = ar(300, ALPHA3, 1.0, 9.0, rng)
    pr = Params(alpha=ALPHA3, Q=1.0, s2=9.0, s_M=0.4, phi_M=0.8)
    a = OdeFilter(pr, order=5, order_A=3).filter(y)
    b = OdeFilter(pr, order=5, order_A=9).filter(y)
    assert np.allclose(a.mean, b.mean, rtol=1e-12, atol=1e-12)
    assert abs(a.loglik - b.loglik) < 1e-9
    assert np.allclose(a.dynamics, 1.0)


def test_dynamics_reverts_on_affirmative_evidence():
    """A stretch with no ODE governance is a MEMBER of the family, not a gap.

    The point of the channel: when the process goes flat, g must fall, and it
    must do so because the flat member fits better -- not because a statistic
    decayed.  It must also come back.
    """
    rng = np.random.default_rng(7)
    n = 300
    x1, _ = ar(n, ALPHA3, 1.0, 0.0, rng)
    x2 = x1[-1] + np.cumsum(rng.standard_normal(n))          # FLAT: a walk
    x3, _ = ar(n, ALPHA3, 1.0, 0.0, rng)
    x = np.concatenate([x1, x2, x3 - x3[0] + x2[-1]])
    y = x + 3.0 * rng.standard_normal(3 * n)

    f = OdeFilter(Params(alpha=ALPHA3, Q=1.0, s2=9.0, phi_A=0.95, s_A=0.5),
                  order=3, order_A=5)
    d = f.filter(y).dynamics
    ode1, flat, ode2 = d[150:n].mean(), d[n + 150:2 * n].mean(), d[2 * n + 150:].mean()
    assert ode1 > 0.75                       # the ODE is in force
    assert flat < 0.6                        # and demonstrably is not, here
    assert flat < ode1 - 0.3
    assert ode2 > flat + 0.3                 # and it comes back


def test_dynamics_rises_when_alpha_is_too_damped():
    """The other direction: g > 1 when the fitted dynamics decay too fast."""
    rng = np.random.default_rng(8)
    x, y = ar(600, ALPHA3, 1.0, 9.0, rng)
    damped = tuple(Params(alpha=ALPHA3, Q=1.0, s2=9.0).alpha_at(0.85))
    f = OdeFilter(Params(alpha=damped, Q=1.0, s2=9.0, phi_A=0.95, s_A=0.4),
                  order=3, order_A=5)
    assert f.filter(y).dynamics[100:].mean() > 1.05


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
# ------------------------------------------------- the oracle-gap battery
# These four would have detected the holes oracle-gap found: a
# likelihood flat along the ridge Q e^{s_P^2/2} = const, a process-scale
# channel that extracts a fraction of the available evidence, and fits whose
# (Q, s_P) endpoint was chosen by the optimiser's path rather than the data.

def ar_qseq(n, alpha, Qseq, S2, rng):
    """AR(p) with a per-step process variance, plus white measurement noise."""
    p = len(alpha)
    z = np.zeros(p)
    x = np.zeros(n)
    for t in range(n):
        xn = (float(np.dot(alpha, z))
              + math.sqrt(Qseq[t]) * rng.standard_normal())
        z = np.concatenate([[xn], z[:-1]])
        x[t] = xn
    return x, x + math.sqrt(S2) * rng.standard_normal(n)


def oracle_nll(y, alpha, Qseq, S2, burn):
    """Exact Kalman filter told Q_t: the evidence ceiling."""
    a = np.asarray(alpha)
    p = a.size
    F = np.zeros((p, p))
    F[0] = a
    F[1:, :-1] = np.eye(p - 1)
    m = np.zeros(p)
    P = np.eye(p) * (S2 + float(np.max(Qseq))) * p
    e1 = np.zeros(p)
    e1[0] = 1.0
    nll, k = 0.0, 0
    for t, yt in enumerate(y):
        m = F @ m
        A = F @ P @ F.T
        S = A[0, 0] + Qseq[t] + S2
        e = float(yt) - m[0]
        if t >= burn:
            nll += 0.5 * (e * e / S + math.log(S) + math.log(2 * math.pi))
            k += 1
        row = A[:, 0] + Qseq[t] * e1
        m = m + row / S * e
        P = A
        P[0, 0] += Qseq[t]
        P -= np.outer(row / S, row)
    return nll / k


def test_quadrature_resolution_is_inert_at_s_zero():
    """At s = 0 every node is identical, so the grid size must not matter:
    order 3 and order 7 must agree to round-off.  This is the invariance the
    old gpb1-vs-imm equality test pinned, restated without the removed mode:
    a recursion that mishandled per-node state would let the nodes diverge
    and the quadrature resolution would leak into the answer."""
    rng = np.random.default_rng(51)
    x, y = ar(300, ALPHA3, 1.0, 9.0, rng)
    pr = Params(alpha=ALPHA3, Q=1.0, s2=9.0)
    a = OdeFilter(pr, order=3).filter(y)
    b = OdeFilter(pr, order=7).filter(y)
    assert abs(a.loglik - b.loglik) < 1e-9
    assert np.allclose(a.mean, b.mean, rtol=1e-10, atol=1e-10)
    assert np.allclose(a.var, b.var, rtol=1e-10, atol=1e-10)
    assert np.allclose(a.pred_var, b.pred_var, rtol=1e-10, atol=1e-10)


def test_streaming_and_batched_agree():
    from odefilter.core import _loglik_batch
    rng = np.random.default_rng(52)
    x, y = ar(300, ALPHA3, 1.0, 9.0, rng)
    pr = Params(alpha=ALPHA3, Q=1.0, s2=9.0, phi_P=0.9, s_P=0.8,
                phi_M=0.6, s_M=0.4)
    lb = float(_loglik_batch(y, pr._vec()[None, :], 3, 5, with_A=False)[0])
    ls = OdeFilter(pr, order=5).loglik(y)
    assert abs(lb - ls) < 1e-6 * abs(ls)


def test_the_ridge_is_not_flat():
    """THE HOLE (oracle-gap/0004).  Two hypotheses with the same mean
    process variance Q e^{s_P^2/2} -- the truth (a live channel) and its
    homoscedastic ridge-mate -- on data generated with the live channel.
    The removed GPB1 likelihood could not tell them apart (0.0022 nats/pt);
    the per-node one must, and in the right direction.  This is the test
    that would have caught the fitted (Q, s_P) endpoint being the
    optimiser's choice."""
    rng = np.random.default_rng(19)
    n = 900
    lam = np.zeros(n)
    nu = math.sqrt(0.8 ** 2 * (1 - 0.9 ** 2))
    lam[0] = 0.8 * rng.standard_normal()
    for t in range(1, n):
        lam[t] = 0.9 * lam[t - 1] + nu * rng.standard_normal()
    x, y = ar_qseq(n, ALPHA3, np.exp(lam), 9.0, rng)

    qeff = math.exp(0.8 ** 2 / 2.0)
    truth = Params(alpha=ALPHA3, Q=1.0, s2=9.0, phi_P=0.9, s_P=0.8)
    mate = Params(alpha=ALPHA3, Q=qeff, s2=9.0)          # same mean variance
    nll_truth = -OdeFilter(truth, order=5).loglik(y) / n
    nll_mate = -OdeFilter(mate, order=5).loglik(y) / n
    sep = nll_mate - nll_truth               # > 0 means the truth is preferred
    assert sep > 0.004                       # the data can tell, and it sees it


def test_forced_channel_extracts_most_of_the_oracle_gap():
    """THE SYMPTOM (0038/oracle-gap/0002).  A x8 process-noise regime,
    scored against a Kalman filter told Q_t exactly.  The forced channel
    under the removed GPB1 collapse stopped near 80% of the static-to-oracle
    span; the per-node recursion must clear 85%."""
    rng = np.random.default_rng(4)
    n, lo, hi = 900, 400, 560
    Qseq = np.full(n, 1.0)
    Qseq[lo:hi] = 8.0
    x, y = ar_qseq(n, ALPHA3, Qseq, 9.0, rng)
    burn = 60

    nll_o = oracle_nll(y, ALPHA3, Qseq, 9.0, burn)
    nll_s = oracle_nll(y, ALPHA3, np.full(n, 1.0), 9.0, burn)
    span = nll_s - nll_o
    pr = Params(alpha=ALPHA3, Q=1.0, s2=9.0, phi_P=0.9, s_P=0.8)
    f = OdeFilter(pr, order=5).reset()
    ll = [f.update(float(v)).loglik for v in y]
    nll = -float(np.mean(ll[burn:]))
    assert (nll_s - nll) / span > 0.85


@pytest.mark.slow
def test_fit_on_the_imm_likelihood_stays_off_the_boundary():
    """THE ENDPOINT (oracle-gap/0006).  On data generated with a live
    process-scale channel, the fitted s_P must be neither the boundary zero
    (the old self-confirming endpoint, 0029/0039) nor a wild ridge slide,
    and the fitted MEAN process variance must be near the truth."""
    pytest.importorskip("scipy")
    rng = np.random.default_rng(19)
    n = 600
    lam = np.zeros(n)
    nu = math.sqrt(0.8 ** 2 * (1 - 0.9 ** 2))
    lam[0] = 0.8 * rng.standard_normal()
    for t in range(1, n):
        lam[t] = 0.9 * lam[t - 1] + nu * rng.standard_normal()
    x, y = ar_qseq(n, ALPHA3, np.exp(lam), 9.0, rng)

    f = OdeFilter.fit(y, p=3, dynamics=False, max_iter=150)
    pr = f.params
    qeff = pr.Q * math.exp(pr.s_P ** 2 / 2.0)
    assert 0.3 < pr.s_P < 1.6                # off the boundary, off the slide
    assert 0.5 < qeff < 2.0                  # the mean variance is right
    assert pr.phi_P > 0.4                    # and the channel is persistent


@pytest.mark.slow
def test_fit_recovers_the_modes():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(21)
    x, y = ar(1200, ALPHA3, 1.0, 9.0, rng)
    f = OdeFilter.fit(y, p=3, order=5, dynamics=False)
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
    f = OdeFilter.fit(y, p=3, order=5, dynamics=False)
    assert f.params.s_P < 0.35 and f.params.s_M < 0.35


@pytest.mark.slow
def test_fit_finds_the_dynamics_channel_when_the_dynamics_stop():
    """The channel has to be findable by the likelihood, not just usable."""
    pytest.importorskip("scipy")
    rng = np.random.default_rng(23)
    n = 400
    x1, _ = ar(n, ALPHA3, 1.0, 0.0, rng)
    x2 = x1[-1] + np.cumsum(rng.standard_normal(n))       # the dynamics stop
    x = np.concatenate([x1, x2])
    y = x + 3.0 * rng.standard_normal(2 * n)

    f = OdeFilter.fit(y, p=3, order=3, order_A=3, max_iter=120)
    assert f.params.s_A > 0.05                     # it finds a live channel
    d = f.filter(y).dynamics
    assert d[n + 100:].mean() < d[100:n].mean()    # and uses it where it should


# ------------------------------------------- supplied-dynamics (robotics) mode
def test_supplied_constant_F_matches_normal():
    """A supplied filter fed a CONSTANT F = companion(alpha) reproduces the normal
    OdeFilter with that alpha (s_A = 0) to machine precision -- fixed and adaptive
    noise.  This is the supplied-mode analogue of the reduces-to-parent check."""
    rng = np.random.default_rng(0)
    alpha = (1.6, -0.7)
    _, y = ar(300, alpha, 1.0, 1.0, rng)
    Fs = np.broadcast_to(_companion(alpha), (y.size, 2, 2))
    for sP in (0.0, 0.35):
        pr = Params(alpha=alpha, Q=1.0, s2=1.0, phi_P=0.9, phi_M=0.9, s_P=sP, s_M=0.0)
        Ln = OdeFilter(pr).loglik(y)
        Ls = OdeFilter.supplied(2, Q=1.0, s2=1.0, phi_P=0.9, s_P=sP,
                                phi_M=0.9, s_M=0.0).loglik(y, Fs=Fs)
        assert abs(Ln - Ls) < 1e-9


def test_supplied_requires_and_validates_F():
    f = OdeFilter.supplied(2)
    with pytest.raises(ValueError):
        f.update(0.5)                       # supplied mode: F required each step
    with pytest.raises(ValueError):
        f.filter(np.zeros(5))               # batch: Fs required
    with pytest.raises(ValueError):
        f.update(0.5, F=np.eye(3))          # wrong shape


def test_supplied_ltv_runs_and_streams():
    rng = np.random.default_rng(2)
    _, y = ar(200, (1.5, -0.6), 1.0, 1.0, rng)
    Fs = np.array([_companion([1.5 + 0.1 * np.sin(t / 20), -0.6]) for t in range(y.size)])
    r = OdeFilter.supplied(2, s_P=0.3).filter(y, Fs=Fs)
    assert np.all(np.isfinite(r.mean))
    f = OdeFilter.supplied(2, s_P=0.3).reset()
    stream = np.array([f.update(v, Fs[i]).mean for i, v in enumerate(y)])
    assert np.allclose(stream, r.mean, atol=1e-9)


def test_fit_supplied_recovers_noise():
    rng = np.random.default_rng(1)
    T = 1200
    Fs = np.array([_companion([1.5 + 0.15 * np.sin(t / 40), -0.6]) for t in range(T)])
    x = np.zeros(T); st = np.zeros(2)
    for t in range(T):
        st = Fs[t] @ st
        st[0] += rng.standard_normal() * math.sqrt(0.8)
        x[t] = st[0] + rng.standard_normal() * math.sqrt(1.3)
    f = OdeFilter.fit_supplied(x, Fs, p=2, scales=False)
    assert 0.5 < f.params.Q < 1.2 and 1.0 < f.params.s2 < 1.7
    assert f._supplied
    assert f.loglik(x, Fs=Fs) >= OdeFilter.supplied(2, Q=0.8, s2=1.3).loglik(x, Fs=Fs) - 1e-6


def test_supplied_roundtrips_through_dict():
    f = OdeFilter.supplied(2, Q=1.1, s2=0.9, s_P=0.3)
    g = OdeFilter.from_dict(f.to_dict())
    assert g._supplied and g.params.Q == f.params.Q and g.params.s_P == f.params.s_P
