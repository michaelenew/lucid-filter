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


def test_dynamics_none_raises():
    with pytest.raises(NotImplementedError):
        LucidFilter(dynamics=None)


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
