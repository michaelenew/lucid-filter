"""Tests for statfilter.WalkingFilter.

    python -m pytest tests/test_walking.py -q

All fast: the walking filter has no fit(), so there are no slow-marked tests here.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from statfilter import WalkingFilter, WalkResult  # noqa: E402


# ------------------------------------------------------------------ fixtures
def scale_series(n, lam_star, seed, Q=1.0, s2=1.0):
    """A local level whose process runs at a constant excess log-scale lam_star."""
    rng = np.random.default_rng(seed)
    step = rng.standard_normal(n) * math.sqrt(Q) * math.exp(0.5 * lam_star)
    theta = np.cumsum(step)
    x = theta + rng.standard_normal(n) * math.sqrt(s2)
    return x, theta


# --------------------------------------------------------------- construction
def test_construction_validates():
    with pytest.raises(ValueError):
        WalkingFilter(Q=-1.0, s2=1.0, phi=0.9, s=0.3)
    with pytest.raises(ValueError):
        WalkingFilter(Q=1.0, s2=1.0, phi=1.0, s=0.3)      # phi must be < 1
    with pytest.raises(ValueError):
        WalkingFilter(Q=1.0, s2=1.0, phi=0.9, s=0.0)      # s must be positive
    with pytest.raises(ValueError):
        WalkingFilter(Q=1.0, s2=1.0, phi=0.9, s=0.3, nodes=6)   # nodes must be odd


def test_derived_geometry():
    f = WalkingFilter(Q=1.0, s2=1.0, phi=0.9, s=0.30, nodes=7)
    assert f.gap == pytest.approx(1.5 * 0.30)             # finding 11: gap = 1.5 s
    assert f.cap == pytest.approx(f.gap)                  # overlap: <= one node/step
    assert f.lam.size == 7
    assert f.lam.max() == pytest.approx(3 * f.gap)        # +-(nodes//2) gaps
    # rows of the transition are proper distributions
    assert np.allclose(f.T.sum(1), 1.0)
    assert np.allclose(f.w0.sum(), 1.0)


# ------------------------------------------------------------------ behaviour
def test_streaming_matches_batch():
    x, _ = scale_series(400, 1.0, seed=1)
    f = WalkingFilter(Q=1.0, s2=1.0, phi=0.9, s=0.30)
    r = f.filter(x)
    f.reset()
    streamed = np.array([f.update(v).process_scale for v in x])
    assert np.allclose(streamed, r.process_scale)


def test_filter_does_not_disturb_stream_state():
    x, _ = scale_series(300, 0.5, seed=2)
    f = WalkingFilter(Q=1.0, s2=1.0, phi=0.9, s=0.30)
    f.update(x[0]); f.update(x[1])
    snap = (f.mu, f._m, f._P)
    f.filter(x)                                           # must not touch state
    assert f.mu == snap[0] and f._m == snap[1] and f._P == snap[2]


def test_missing_data_is_propagated_not_corrected():
    x, _ = scale_series(200, 0.0, seed=3)
    x[100] = np.nan
    f = WalkingFilter(Q=1.0, s2=1.0, phi=0.9, s=0.30)
    r = f.filter(x)
    assert np.isfinite(r.mean[100])                       # level carried through
    assert math.isnan(r.innovation[100])                 # no correction applied
    assert np.all(np.isfinite(r.mean))


def test_recovers_a_static_scale():
    """The tracked log-scale locks onto a constant loud regime."""
    x, _ = scale_series(1500, 3.0, seed=4)
    f = WalkingFilter(Q=1.0, s2=1.0, phi=0.9, s=0.30)
    r = f.filter(x)
    assert abs(np.mean(r.process_scale[-400:]) - 3.0) < 0.6


def test_wrong_Q_is_absorbed_by_the_walk():
    """A wrong base Q shifts the reported scale to compensate; the variance
    estimate Q*exp(scale) is unchanged.  (SUMMARY finding 9.)"""
    x, _ = scale_series(1500, 2.0, seed=5, Q=1.0)
    eff = []
    for Qf in (0.1, 1.0, 10.0):
        r = WalkingFilter(Q=Qf, s2=1.0, phi=0.9, s=0.30).filter(x)
        eff.append(Qf * math.exp(np.mean(r.process_scale[-400:])))
    assert max(eff) / min(eff) < 1.05                    # all agree within 5%


def test_unbounded_reach_captures_a_big_jump():
    """A jump far outside the fixed window is still captured -- the window walks."""
    rng = np.random.default_rng(6)
    NT, JT, D = 1200, 100, 9.0                            # +9 nats: way past +-1.35
    lam = np.zeros(NT); lam[JT:] = D
    step = rng.standard_normal(NT) * np.sqrt(np.exp(lam))
    x = np.cumsum(step) + rng.standard_normal(NT)
    r = WalkingFilter(Q=1.0, s2=1.0, phi=0.9, s=0.30).filter(x)
    assert abs(np.mean(r.process_scale[-200:]) - D) < 1.0


def test_result_shape_and_loglik_finite():
    x, _ = scale_series(250, 0.5, seed=7)
    f = WalkingFilter(Q=1.0, s2=1.0, phi=0.9, s=0.30)
    r = f.filter(x)
    assert isinstance(r, WalkResult) and len(r) == 250
    assert math.isfinite(r.loglik)
    assert f.loglik_of(x) == pytest.approx(r.loglik)
    assert np.all(np.abs(r.scale_step) <= f.cap + 1e-9)  # every step within the cap


def test_reset_chains_and_seeds_scale():
    f = WalkingFilter(Q=1.0, s2=1.0, phi=0.9, s=0.30)
    assert f.reset(scale=2.0) is f
    assert f.mu == 2.0


# --------------------------------------------------------------- WalkingBank
from statfilter import WalkingBank, BankResult  # noqa: E402


def test_bank_construction_validates():
    with pytest.raises(ValueError):
        WalkingBank(Q=1.0, s2=1.0, forget=0.0)
    with pytest.raises(ValueError):
        WalkingBank(Q=1.0, s2=1.0, forget=1.5)
    with pytest.raises(ValueError):
        WalkingBank(Q=1.0, s2=1.0, phis=[], ss=[0.3])


def test_bank_defaults_grid():
    b = WalkingBank(Q=1.0, s2=1.0)
    assert len(b.filters) == 3 * 5           # default 3 phis x 5 ss
    assert b.phi_arr.size == len(b.filters) and b.s_arr.size == len(b.filters)


def test_bank_streaming_matches_batch():
    x, _ = scale_series(400, 1.0, seed=11)
    b = WalkingBank(Q=1.0, s2=1.0)
    r = b.filter(x)
    b.reset()
    streamed = np.array([b.update(v).process_scale for v in x])
    assert np.allclose(streamed, r.process_scale)


def test_bank_filter_does_not_disturb_state():
    x, _ = scale_series(300, 0.5, seed=12)
    b = WalkingBank(Q=1.0, s2=1.0)
    b.update(x[0]); b.update(x[1])
    snap = b._logw.copy()
    b.filter(x)
    assert np.allclose(b._logw, snap)


def test_bank_weights_are_a_distribution():
    x, _ = scale_series(500, 2.0, seed=13)
    b = WalkingBank(Q=1.0, s2=1.0)
    r = b.filter(x)
    assert np.all(r.n_eff >= 1.0 - 1e-9) and np.all(r.n_eff <= len(b.filters) + 1e-9)
    # learned (phi, s) stay inside the grid box
    assert b.s_arr.min() - 1e-9 <= r.s_hat[-1] <= b.s_arr.max() + 1e-9
    assert b.phi_arr.min() - 1e-9 <= r.phi_hat[-1] <= b.phi_arr.max() + 1e-9


def test_bank_concentrates_on_the_ridge():
    """With enough data the bank sheds models -- weight collects on the ridge."""
    x, _ = scale_series(2000, 2.5, seed=14)
    r = WalkingBank(Q=1.0, s2=1.0).filter(x)
    assert r.n_eff[-1] < len(WalkingBank(Q=1.0, s2=1.0).filters)   # not uniform


def test_bank_matches_oracle_without_being_told_phi_s():
    """Level tracking at parity with a single filter given the true (phi, s)."""
    rng = np.random.default_rng(15)
    NT = 2000
    lam = np.cumsum(rng.standard_normal(NT) * 0.05)      # a slowly wandering scale
    theta = np.cumsum(rng.standard_normal(NT) * np.sqrt(np.exp(lam)))
    x = theta + rng.standard_normal(NT)
    r_bank = WalkingBank(Q=1.0, s2=1.0).filter(x)
    r_one = WalkingFilter(Q=1.0, s2=1.0, phi=0.95, s=0.30).filter(x)
    def rmse(m):
        return float(np.sqrt(np.mean((m[100:] - theta[100:]) ** 2)))
    assert rmse(r_bank.mean) < 1.25 * rmse(r_one.mean)   # within 25% of a hand-set filter


def test_bank_handles_missing():
    x, _ = scale_series(200, 0.0, seed=16)
    x[100] = np.nan
    r = WalkingBank(Q=1.0, s2=1.0).filter(x)
    assert isinstance(r, BankResult)
    assert np.isfinite(r.mean[100]) and math.isnan(r.innovation[100])


def test_bank_forget_keeps_models_alive():
    """A forgetting factor < 1 stops the weights collapsing onto one model."""
    x, _ = scale_series(3000, 2.0, seed=17)
    r_pure = WalkingBank(Q=1.0, s2=1.0, forget=1.0).filter(x)
    r_forget = WalkingBank(Q=1.0, s2=1.0, forget=0.98).filter(x)
    assert r_forget.n_eff[-1] > r_pure.n_eff[-1]
