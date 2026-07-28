"""Tests for statfilter.

Run with:  python -m pytest tests -q      (or: python tests/test_statfilter.py)
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from statfilter import AdaptiveFilter, Params  # noqa: E402


# ------------------------------------------------------------------ fixtures
def local_level(n, q, s2, seed, jumps=(), hetero=None, outlier_rate=0.0):
    rng = np.random.default_rng(seed)
    theta = np.cumsum(rng.standard_normal(n) * math.sqrt(q))
    for t in jumps:
        theta[t:] += 6.0
    sd = np.full(n, math.sqrt(s2))
    if hetero is not None:
        sd[n // 2:] = math.sqrt(hetero)
    x = theta + rng.standard_normal(n) * sd
    if outlier_rate:
        idx = rng.choice(n, int(n * outlier_rate), replace=False)
        x[idx] += rng.standard_normal(idx.size) * 8.0
    return x, theta


def best_constant_gain_mse(x, theta):
    """MSE of the constant-gain filter whose gain is chosen with hindsight."""
    best = np.inf
    for K in np.linspace(0.01, 0.99, 99):
        m = x[0]
        err = np.empty(len(x))
        for t in range(len(x)):
            m += K * (x[t] - m)
            err[t] = m - theta[t]
        best = min(best, float(np.mean(err ** 2)))
    return best


# -------------------------------------------------------------------- params
def test_params_validate():
    with pytest.raises(ValueError):
        Params(Q=-1.0, s2=1.0)
    with pytest.raises(ValueError):
        Params(Q=1.0, s2=1.0, phi_P=1.0)
    with pytest.raises(ValueError):
        Params(Q=1.0, s2=1.0, s_M=-0.1)


def test_gain_matches_riccati():
    p = Params(Q=0.05, s2=1.0)
    K = p.gain
    # steady state: P_prior = Q/K and K = P_prior / (P_prior + s2)
    P_prior = p.Q / K
    assert K == pytest.approx(P_prior / (P_prior + p.s2), rel=1e-12)


def test_roundtrip_serialisation():
    f = AdaptiveFilter(Params(Q=0.03, s2=1.2, phi_M=0.8, s_M=0.5), order=7)
    g = AdaptiveFilter.from_dict(f.to_dict())
    assert g.params == f.params and g.order == f.order


# -------------------------------------------------------------- correctness
def test_reduces_to_kalman_when_homoscedastic():
    """With s_P = s_M = 0 the filter must equal a plain Kalman filter."""
    x, _ = local_level(400, 0.05, 1.0, seed=0)
    p = Params(Q=0.05, s2=1.0)
    got = AdaptiveFilter(p).filter(x)

    m, P = x[0], p.s2 + p.Q
    means = np.empty(len(x))
    for t, v in enumerate(x):
        Pp = P + p.Q
        K = Pp / (Pp + p.s2)
        m += K * (v - m)
        P = (1 - K) * Pp
        means[t] = m
    assert np.allclose(got.mean[5:], means[5:], atol=1e-9)


def test_amplitude_conservation():
    """The three innovation shares sum to one at every step."""
    x, _ = local_level(300, 0.05, 1.0, seed=1, outlier_rate=0.02)
    r = AdaptiveFilter(Params(0.05, 1.0, 0.5, 0.2, 0.8, 0.9)).filter(x)
    total = r.share_prior + r.share_process + r.share_measurement
    assert np.allclose(total, 1.0, atol=1e-12)
    for s in (r.share_prior, r.share_process, r.share_measurement):
        assert np.all(s >= -1e-12) and np.all(s <= 1 + 1e-12)


def test_scale_conservation():
    """Anomaly plus regime reconstructs the scale coordinate exactly."""
    x, _ = local_level(200, 0.05, 1.0, seed=2, hetero=9.0)
    r = AdaptiveFilter(Params(0.05, 1.0, 0.3, 0.9, 0.6, 0.7)).filter(x)
    assert np.allclose(r.process_anomaly + r.process_regime, r.process_scale)
    assert np.allclose(r.measurement_anomaly + r.measurement_regime,
                       r.measurement_scale)


def test_streaming_matches_batch():
    x, _ = local_level(250, 0.05, 1.0, seed=3, jumps=(120,))
    f = AdaptiveFilter(Params(0.05, 1.0, 0.4, 0.4, 1.0, 0.5))
    batch = f.filter(x)
    f.reset()
    stream = np.array([f.update(v).mean for v in x])
    assert np.allclose(batch.mean, stream, atol=1e-12)


def test_filter_does_not_disturb_stream_state():
    x, _ = local_level(120, 0.05, 1.0, seed=4)
    f = AdaptiveFilter(Params(0.05, 1.0))
    for v in x[:60]:
        f.update(v)
    before = f.update(x[60]).mean
    f.reset()
    for v in x[:60]:
        f.update(v)
    f.filter(x)                      # must not touch streaming state
    after = f.update(x[60]).mean
    assert before == pytest.approx(after, rel=1e-12)


def test_missing_data():
    x, theta = local_level(300, 0.05, 1.0, seed=5)
    holed = x.copy()
    holed[100:130] = np.nan
    r = AdaptiveFilter(Params(0.05, 1.0)).filter(holed)
    assert np.all(np.isfinite(r.mean))
    assert np.all(np.diff(r.var[100:130]) > 0)          # uncertainty grows in the gap
    assert abs(r.mean[99] - r.mean[129]) < 1e-9         # level held flat across it


def test_predict_grows_with_horizon():
    x, _ = local_level(200, 0.05, 1.0, seed=6)
    f = AdaptiveFilter(Params(0.05, 1.0))
    for v in x:
        f.update(v)
    m1, v1 = f.predict(1)
    m5, v5 = f.predict(5)
    assert m1 == pytest.approx(m5)                      # random walk: flat forecast
    assert v5 > v1


# --------------------------------------------------------------------- fitting
def test_fit_recovers_parameters():
    x, _ = local_level(1500, 0.05, 1.0, seed=7)
    f = AdaptiveFilter.fit(x)
    assert 0.015 < f.params.Q < 0.15                    # true 0.05
    assert 0.8 < f.params.s2 < 1.25                     # true 1.0


def test_fit_finds_no_scale_structure_in_clean_data():
    """A homoscedastic series must not produce a large measurement volatility."""
    x, _ = local_level(1500, 0.05, 1.0, seed=8)
    f = AdaptiveFilter.fit(x)
    assert f.params.s_M < 0.5


def test_fit_detects_measurement_scale_structure():
    """Outliers and a variance regime change must both raise s_M well above zero."""
    xo, _ = local_level(1200, 0.05, 1.0, seed=9, outlier_rate=0.01)
    xh, _ = local_level(1200, 0.05, 1.0, seed=10, hetero=9.0)
    assert AdaptiveFilter.fit(xo).params.s_M > 0.5
    assert AdaptiveFilter.fit(xh).params.s_M > 0.4


def test_persistent_noise_change_reads_as_persistent():
    """A sigma^2 regime change should give a high phi_M, not a spiky one."""
    x, _ = local_level(1200, 0.05, 1.0, seed=11, hetero=9.0)
    p = AdaptiveFilter.fit(x).params
    assert p.s_M > 0.3 and p.phi_M > 0.6


def test_mode_coordinate_locates_a_jump():
    """The process-anomaly coordinate should peak at the step, not elsewhere."""
    n, jump = 900, 450
    x, _ = local_level(n, 1e-9, 1.0, seed=12, jumps=(jump,))
    f = AdaptiveFilter.fit(x)
    r = f.filter(x)
    assert abs(int(np.argmax(r.process_anomaly)) - jump) <= 2


# ------------------------------------------------------------------ behaviour
@pytest.mark.parametrize("seed", [21, 22])
def test_competitive_with_hindsight_tuned_kalman(seed):
    """Never much worse than a constant gain chosen with hindsight, often better."""
    cases = {
        "diffusion": dict(q=0.05, s2=1.0),
        "step": dict(q=1e-9, s2=1.0, jumps=(300, 600)),
        "outliers": dict(q=0.05, s2=1.0, outlier_rate=0.01),
    }
    for name, kw in cases.items():
        x, theta = local_level(900, seed=seed, **kw)
        f = AdaptiveFilter.fit(x)
        mse = float(np.mean((f.filter(x).mean - theta) ** 2))
        ratio = mse / best_constant_gain_mse(x, theta)
        assert ratio < 1.15, f"{name}: ratio {ratio:.3f}"


def test_unfitted_filter_refuses_to_run():
    with pytest.raises(ValueError):
        AdaptiveFilter().filter(np.arange(10.0))


def test_fit_rejects_degenerate_input():
    with pytest.raises(ValueError):
        AdaptiveFilter.fit(np.ones(50))
    with pytest.raises(ValueError):
        AdaptiveFilter.fit(np.arange(4.0))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
