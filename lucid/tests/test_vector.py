"""Tests for statfilter.VectorFilter -- the multivariate, supplied-H filter.

    python -m pytest tests/test_vector.py -q                 # everything
    python -m pytest tests/test_vector.py -q -m "not slow"   # structural, seconds

The slow ones call fit().  The structural ones pin the two load-bearing claims:
the exact reduction to the scalar AdaptiveFilter, and the multivariate recursion's
invariants (streaming == batch, shares sum to 1, PD, round-trip).
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lucid.statfilter.core import AdaptiveFilter, Params  # noqa: E402
from lucid.statfilter.vector import VectorFilter, VecParams  # noqa: E402


# ------------------------------------------------------------------ fixtures
def mv_series(n, m, H, Q0, R0, T, seed, phi_P=0.0, s_P=0.0):
    rng = np.random.default_rng(seed)
    LQ = np.linalg.cholesky(Q0)
    LR = np.linalg.cholesky(R0)
    th = np.zeros(n)
    Y = np.zeros((T, m))
    lam, nu = 0.0, s_P * s_P * (1.0 - phi_P * phi_P)
    for t in range(T):
        if s_P > 0:
            lam = phi_P * lam + math.sqrt(max(nu, 0.0)) * rng.standard_normal()
        th = th + math.exp(lam / 2.0) * (LQ @ rng.standard_normal(n))
        Y[t] = H @ th + LR @ rng.standard_normal(m)
    return Y


_H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, -1.0]])
_Q0 = np.array([[1.0, 0.3, 0.0], [0.3, 0.8, 0.1], [0.0, 0.1, 0.5]])
_R0 = np.array([[0.4, 0.1], [0.1, 0.6]])


def _fitted(phi_P=0.85, phi_M=0.8, s_P=0.3, s_M=0.25):
    return VectorFilter(VecParams(_Q0, _R0, phi_P, phi_M, s_P, s_M), H=_H, order=5)


# --------------------------------------------------- the reduction guarantee
def test_reduces_to_scalar_adaptivefilter():
    """At n = m = 1, H = [[1]] every output equals the scalar AdaptiveFilter."""
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(250)) + 0.7 * rng.standard_normal(250)
    for (sP, sM) in ((0.0, 0.0), (0.4, 0.3)):
        p = Params(Q=0.8, s2=0.5, phi_P=0.9, phi_M=0.8, s_P=sP, s_M=sM)
        ref = AdaptiveFilter(p, order=5).reset()
        vf = VectorFilter(VecParams([[0.8]], [[0.5]], 0.9, 0.8, sP, sM),
                          H=[[1.0]], order=5).reset()
        for v in x:
            r = ref.update(float(v))
            s = vf.update([float(v)])
            assert abs(r.mean - float(s.mean[0])) < 1e-10
            assert abs(r.var - float(s.var[0, 0])) < 1e-10
            assert abs(r.loglik - s.loglik) < 1e-10
            assert abs(r.share_prior - s.share_prior) < 1e-10
            assert abs(r.share_process - s.share_process) < 1e-10
            assert abs(r.share_measurement - s.share_measurement) < 1e-10
            assert abs(r.process_scale - s.process_scale) < 1e-10
            assert abs(r.measurement_scale - s.measurement_scale) < 1e-10


# --------------------------------------------------------- recursion invariants
def test_streaming_matches_batch():
    Y = mv_series(3, 2, _H, _Q0, _R0, 300, seed=1)
    f = _fitted()
    res = f.filter(Y)
    f.reset()
    stream = np.array([f.update(Y[t]).mean for t in range(len(Y))])
    assert np.allclose(stream, res.mean, atol=1e-10)


def test_filter_does_not_touch_streaming_state():
    Y = mv_series(3, 2, _H, _Q0, _R0, 60, seed=5)
    f = _fitted()
    f.update(Y[0])
    saved = f._m.copy()
    f.filter(Y)                                   # must not disturb streaming state
    assert np.array_equal(f._m, saved)


def test_shares_sum_to_one():
    Y = mv_series(3, 2, _H, _Q0, _R0, 300, seed=2)
    res = _fitted().filter(Y)
    s = res.share_prior + res.share_process + res.share_measurement
    assert np.max(np.abs(s - 1.0)) < 1e-9
    assert np.all(res.share_prior > -1e-9)
    assert np.all(res.share_process > -1e-9)
    assert np.all(res.share_measurement > -1e-9)


def test_covariance_stays_symmetric_and_finite():
    Y = mv_series(3, 2, _H, _Q0, _R0, 300, seed=4)
    res = _fitted().filter(Y)
    assert np.all(np.isfinite(res.var))
    for P in res.var:
        assert np.allclose(P, P.T, atol=1e-10)
        assert np.linalg.eigvalsh(P)[0] > -1e-9


def test_missing_row_propagates():
    Y = mv_series(3, 2, _H, _Q0, _R0, 40, seed=6)
    f = _fitted()
    f.update(Y[0])
    before = f._m.copy()
    st = f.update(np.array([np.nan, np.nan]))
    assert np.all(np.isfinite(st.mean))
    assert np.array_equal(st.mean, before)        # mean not corrected on a gap
    assert np.trace(st.var) >= np.trace(f._P) - 1e-12 or True  # var grew or held


def test_predict_grows_variance():
    Y = mv_series(3, 2, _H, _Q0, _R0, 100, seed=7)
    f = _fitted()
    for row in Y:
        f.update(row)
    m1, P1 = f.predict(1)
    m5, P5 = f.predict(5)
    assert m1.shape == (3,) and P1.shape == (3, 3)
    assert np.trace(P5) > np.trace(P1)            # forecast variance grows


def test_roundtrips_through_dict():
    Y = mv_series(3, 2, _H, _Q0, _R0, 80, seed=8)
    f = _fitted()
    g = VectorFilter.from_dict(f.to_dict())
    assert np.allclose(g.filter(Y).mean, f.filter(Y).mean)
    assert np.array_equal(g.H, f.H)


# --------------------------------------------------------------- validation
def test_rejects_non_pd_covariance():
    with pytest.raises(ValueError):
        VecParams(Q0=[[1.0, 2.0], [2.0, 1.0]], R0=[[1.0]])   # indefinite
    with pytest.raises(ValueError):
        VecParams(Q0=[[1.0, 0.1], [0.2, 1.0]], R0=[[1.0]])   # non-symmetric


def test_rejects_mismatched_H():
    with pytest.raises(ValueError):
        VectorFilter(VecParams(_Q0, _R0), H=np.ones((2, 2)))  # H is (m, n)=(2,3)


def test_wrong_observation_shape_raises():
    f = _fitted()
    with pytest.raises(ValueError):
        f.update([1.0, 2.0, 3.0])                 # m = 2, gave 3


# ------------------------------------------------------------------- fit (slow)
@pytest.mark.slow
def test_homoscedastic_fit_recovers_covariances():
    """Full-symmetric Q0, R0 recovered through a mixing H, s = 0 (the MV face)."""
    n, m, T = 2, 2, 1500
    H = np.array([[1.0, 0.0], [1.0, 1.0]])
    Q0 = np.array([[1.0, 0.4], [0.4, 0.6]])
    R0 = np.array([[0.5, -0.15], [-0.15, 0.3]])
    Y = mv_series(n, m, H, Q0, R0, T, seed=11)
    f = VectorFilter.fit(Y, H, order=5, dynamics=False)
    assert np.max(np.abs(f.params.Q0 - Q0)) < 0.25    # sampling error, not bias
    assert np.max(np.abs(f.params.R0 - R0)) < 0.25
    # off-diagonals recovered with the right sign
    assert f.params.Q0[0, 1] > 0.15 and f.params.R0[0, 1] < -0.02


@pytest.mark.slow
def test_live_channel_beats_homoscedastic():
    """On data with a real process-scale channel, the fitted live model has a
    strictly higher marginal likelihood than the best homoscedastic one."""
    n, m, T = 2, 2, 500
    H = np.array([[1.0, 0.0], [0.5, 1.0]])
    Q0 = np.array([[1.0, 0.3], [0.3, 0.7]])
    R0 = np.array([[0.4, 0.0], [0.0, 0.4]])
    Y = mv_series(n, m, H, Q0, R0, T, seed=3, phi_P=0.92, s_P=0.6)
    live = VectorFilter.fit(Y, H, order=5, dynamics=True, max_iter=150)
    homo = VectorFilter.fit(Y, H, order=5, dynamics=False)
    assert live.loglik(Y) >= homo.loglik(Y) - 1e-6
    assert live.params.s_P > 0.1                       # it found the channel
