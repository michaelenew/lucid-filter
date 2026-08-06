"""Tests for the offset channel.

Each pins a result from exploration/0042-0057 at test scale: the delay row's
exactness (0043a), the anchor (0049), sign detection under uniform deferral
(0054/0055), trust against a matched null (0046), and the FLAT member's
survival on a static offset (0046).
"""
import math

import numpy as np
import pytest

from odefilter import OdeFilter, Params, OffsetFilter, delay_row, cross_anchor

# a damped oscillator: roots 0.95 exp(+/- 0.4i)
ALPHA = (2 * 0.95 * math.cos(0.4), -(0.95 ** 2))
PAR = Params(alpha=ALPHA, Q=1.0, s2=0.09)


def make_pair(n, tau_int, c=1.0, seed=0, coupled=True, s2_2=0.09):
    """In-class discrete data: y2 reads the same latent at an INTEGER offset
    (exact within the class; fractional generation needs a continuous world,
    which the exploration probes cover)."""
    rng = np.random.default_rng(seed)
    pad = 8
    x = np.zeros(n + 2 * pad)
    for t in range(2, len(x)):
        x[t] = ALPHA[0] * x[t - 1] + ALPHA[1] * x[t - 2] \
            + rng.standard_normal()
    src = x if coupled else _fresh(n + 2 * pad, rng)
    t0 = pad
    y1 = x[t0:t0 + n] + 0.3 * rng.standard_normal(n)
    y2 = c * src[t0 - tau_int:t0 - tau_int + n] \
        + math.sqrt(s2_2) * rng.standard_normal(n)
    return y1, y2


def _fresh(n, rng):
    x = np.zeros(n)
    for t in range(2, n):
        x[t] = ALPHA[0] * x[t - 1] + ALPHA[1] * x[t - 2] \
            + rng.standard_normal()
    return x


# ------------------------------------------------------------------ the row
def test_delay_row_exact_on_solution_space():
    # a noiseless real solution x(t) = Re(A z^t), read at a fractional offset
    z = 0.95 * np.exp(0.4j)
    A = 1.3 - 0.7j
    x = lambda t: (A * z ** t).real
    window = np.array([x(10.0 - k) for k in range(6)])
    for s in (0.0, 1.0, 3.0, 0.4, 1.7, 4.6):
        row = delay_row(ALPHA, s, 6)
        assert abs(row @ window - x(10.0 - s)) < 1e-9


def test_delay_row_integer_is_pickout():
    row = delay_row(ALPHA, 2.0, 5)
    expect = np.zeros(5); expect[2] = 1.0
    assert np.allclose(row, expect, atol=1e-9)


def test_delay_row_guards():
    with pytest.raises(ValueError):
        delay_row((2.0, -1.0), 0.5, 4)          # repeated root at z = 1
    with pytest.raises(ValueError):
        delay_row((-0.5,), 0.5, 3)              # negative real root
    with pytest.raises(ValueError):
        delay_row(ALPHA, 9.0, 4)                # outside the stored window


# --------------------------------------------------------------- the anchor
def test_cross_anchor_recovers_tau():
    y1, y2 = make_pair(3000, tau_int=2, seed=3)
    assert abs(cross_anchor(y1, y2, PAR) - 2.0) < 0.2


def test_cross_anchor_differences_near_unit_roots():
    # a latent with an offset root: the anchor must fall back to increments
    par = Params(alpha=(1.0 + ALPHA[0] - 0.0, ), Q=1.0, s2=0.09)  # placeholder
    rng = np.random.default_rng(5)
    n = 3000
    lvl = np.cumsum(0.4 * rng.standard_normal(n + 8))
    osc = _fresh(n + 8, rng)
    x = lvl + osc
    y1 = x[4:4 + n] + 0.3 * rng.standard_normal(n)
    y2 = x[2:2 + n] + 0.3 * rng.standard_normal(n)      # y2 lags by 2
    par3 = Params(alpha=tuple(-np.poly([1.0, 0.95 * np.exp(0.4j),
                                        0.95 * np.exp(-0.4j)])[1:].real),
                  Q=1.0, s2=0.09)
    assert abs(cross_anchor(y1, y2, par3) - 2.0) < 0.35


# --------------------------------------------------------------- the filter
def test_lag_detected():
    y1, y2 = make_pair(400, tau_int=1, seed=7)
    f = OffsetFilter(PAR, s2_2=0.09, window=(-2.0, 2.0),
                     taus=np.arange(-2.0, 2.01, 0.25),
                     c_grid=np.array([0.7, 1.0, 1.4]))
    step = f.filter(y1, y2)[-1]
    assert step.p_lead < 0.05
    assert abs(step.tau_mean - 1.0) < 0.2
    assert abs(step.c_mean - 1.0) < 0.2


def test_lead_detected():
    y1, y2 = make_pair(400, tau_int=-1, seed=8)
    f = OffsetFilter(PAR, s2_2=0.09, window=(-2.0, 2.0),
                     taus=np.arange(-2.0, 2.01, 0.25),
                     c_grid=np.array([0.7, 1.0, 1.4]))
    step = f.filter(y1, y2)[-1]
    assert step.p_lead > 0.95
    assert abs(step.tau_mean + 1.0) < 0.2


def test_trust_against_matched_null():
    # the matched null: the same latent class, fitted face, on y2 alone
    null = OdeFilter(params=Params(alpha=ALPHA, Q=1.0, s2=0.09))
    y1, y2 = make_pair(400, tau_int=1, seed=9)
    f = OffsetFilter(PAR, s2_2=0.09, window=(-2.0, 2.0),
                     taus=np.arange(-2.0, 2.01, 0.25),
                     c_grid=np.array([0.7, 1.0, 1.4]), null=null)
    step = f.filter(y1, y2)[-1]
    assert step.trust > 0.99

    null.reset()
    y1u, y2u = make_pair(400, tau_int=1, seed=10, coupled=False)
    g = OffsetFilter(PAR, s2_2=0.09, window=(-2.0, 2.0),
                     taus=np.arange(-2.0, 2.01, 0.25),
                     c_grid=np.array([0.7, 1.0, 1.4]), null=null)
    stepu = g.filter(y1u, y2u)[-1]
    assert stepu.lam < 0.0
    assert stepu.trust < 0.01


def test_missing_y2_is_tolerated():
    y1, y2 = make_pair(200, tau_int=1, seed=11)
    y2 = y2.copy(); y2[50:150] = math.nan
    f = OffsetFilter(PAR, s2_2=0.09, window=(0.0, 2.0),
                     taus=np.arange(0.0, 2.01, 0.25),
                     c_grid=np.array([1.0]))
    step = f.filter(y1, y2)[-1]
    assert np.isfinite(step.tau_mean)
    assert abs(step.tau_mean - 1.0) < 0.3


def test_static_member_survives():
    # eps = 0 is an explicit member; on a static offset it must not be
    # drowned by the restart members
    y1, y2 = make_pair(400, tau_int=1, seed=12)
    f = OffsetFilter(PAR, s2_2=0.09, window=(0.0, 2.0),
                     taus=np.arange(0.0, 2.01, 0.25),
                     c_grid=np.array([1.0]))
    f.filter(y1, y2)
    hw = np.exp(f._hyper - f._hyper.max()); hw /= hw.sum()
    assert hw[0] > 0.2                          # the eps = 0 member
