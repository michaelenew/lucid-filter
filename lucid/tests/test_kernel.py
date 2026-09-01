"""Tests for the compiled kernel.

There is one claim to check and it is a strong one: **the kernel returns the
same bits as NumPy**.  Not the same number to a tolerance -- `np.allclose`
would pass on a kernel that had quietly reassociated a sum and would say
nothing useful about whether a fit lands in the same place.  So everything
here compares raw IEEE-754 payloads through `.view(np.uint64)`, which also
makes the comparison strict about the two things `==` is wrong about: it
separates 0.0 from -0.0, and it holds NaN equal to itself.

If the kernel is not built, the whole module skips: nothing here is a claim
about the filters, only about the kernel, and without one there is nothing to
compare.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lucid.odefilter import OdeFilter, Params                            # noqa: E402
from lucid.odefilter import core as ode                                  # noqa: E402
from lucid.statfilter import core as stat                                # noqa: E402

K = ode._kernel
pytestmark = pytest.mark.skipif(
    K is None or not K.available(),
    reason=f"kernel not built ({K.reason() if K else 'not importable'})")


def bits(a):
    """The raw payloads, so the comparison is exact about NaN and -0.0."""
    return np.ascontiguousarray(np.asarray(a, dtype=float)).view(np.uint64)


def same(a, b):
    a, b = bits(a), bits(b)
    return a.shape == b.shape and np.array_equal(a, b)


def walk(n, seed=0, s_P=1.2):
    """A random walk with a stochastic volatility -- what the filter is for."""
    rng = np.random.default_rng(seed)
    lam, x, out = 0.0, 0.0, np.empty(n)
    for t in range(n):
        lam = 0.8 * lam + math.sqrt(1.0 - 0.64) * s_P * rng.standard_normal()
        x += math.sqrt(7e-4 * math.exp(lam)) * rng.standard_normal()
        out[t] = x
    return out


# ------------------------------------------------- the NumPy-owned primitives
# These are the operations the kernel must not do itself, because their last
# bit is a property of the local NumPy build rather than of the arithmetic.

def test_sum_is_numpys_pairwise_sum():
    ext = K.ext()
    rng = np.random.default_rng(0)
    for n in list(range(1, 40)) + [49, 75, 127, 128, 129, 175, 256, 375, 1000]:
        a = rng.standard_normal(n)
        assert same(ext.np_prim("sum", a), a.sum()), n


def test_exp_and_log_are_numpys_own_loops():
    ext = K.ext()
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.uniform(-700, 700, 5000), rng.uniform(-1, 1, 5000)])
    with np.errstate(over="ignore"):
        assert same(ext.np_prim("exp", x), np.exp(x))
    xp = np.abs(x) + 1e-300
    assert same(ext.np_prim("log", xp), np.log(xp))


def test_dot_is_numpys_blas():
    ext = K.ext()
    rng = np.random.default_rng(2)
    for n in (1, 2, 3, 25, 49, 75, 175):
        a, b = rng.standard_normal(n), rng.standard_normal(n)
        assert same(ext.np_prim("dot", a, b), a @ b), n


def test_max_propagates_nan_the_way_numpy_does():
    ext = K.ext()
    a = np.array([1.0, np.nan, 3.0])
    assert math.isnan(ext.np_prim("max", a))
    assert same(ext.np_prim("max", np.array([-3.0, 2.0, 2.0])), np.max([-3.0, 2.0, 2.0]))


# ----------------------------------------------------------- the order probe
def test_probe_reports_an_order_that_is_actually_numpys():
    """Whatever the probe claims about einsum has to be true of this NumPy."""
    rng = np.random.default_rng(5)
    for p in (1, 2, 3, 4, 5):
        ap, mp, _sc = ode._einsum_orders(p)
        assert ap is not None, p
        F = rng.random((3, 9, p, p))
        P0 = rng.random((3, 9, p, p))
        ref = np.einsum("bgxw,bgwv,bgzv->bgxz", F, P0, F)
        got = np.zeros_like(ref)
        if ap == K.AP_FLAT:
            for w in range(p):
                for v in range(p):
                    got += (F[:, :, :, None, w] * P0[:, :, None, None, w, v]
                            * F[:, :, None, :, v])
        else:
            for w in range(p):
                inner = np.zeros_like(ref)
                for v in range(p):
                    inner += (F[:, :, :, None, w] * P0[:, :, None, None, w, v]
                              * F[:, :, None, :, v])
                got += inner
        assert same(ref, got), (p, ap)


def test_a_shape_the_kernel_cannot_match_is_refused_not_guessed():
    """The fallback is the whole safety argument, so make it fire."""
    calls = []

    def wrong(*modes):
        calls.append(modes)
        return np.array([1.0, 2.0])

    K.clear_cache()
    with pytest.warns(RuntimeWarning, match="falling back to NumPy"):
        got = K.verify(("test", "impossible"), wrong,
                       lambda: np.array([3.0, 4.0]))
    assert got is None
    assert calls, "verify must actually try the candidates"
    K.clear_cache()


# ---------------------------------------------------- the batched likelihood
@pytest.mark.parametrize("p", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("order,order_A,with_A", [(5, 3, False), (5, 3, True),
                                                  (4, 3, True), (7, 3, False)])
def test_batch_loglik_is_identical(p, order, order_A, with_A):
    rng = np.random.default_rng(100 * p + order)
    B = 7
    V = np.zeros((B, p + 8))
    V[:, :p] = rng.uniform(-0.9, 0.9, (B, p))
    V[:, p] = rng.uniform(-9.0, -4.0, B)
    V[:, p + 1] = rng.uniform(-14.0, -4.0, B)
    V[:, p + 2] = rng.uniform(-3.0, 3.0, B)
    V[:, p + 3] = rng.uniform(-3.0, 3.0, B)
    V[:, p + 4] = rng.uniform(-4.0, 0.4, B)
    V[:, p + 5] = rng.uniform(-4.0, 0.4, B)
    V[:, p + 6] = rng.uniform(-3.0, 3.0, B)
    V[:, p + 7] = rng.uniform(-4.0, 0.0, B)
    y = walk(120, seed=p * 7 + order)
    y[11] = y[12] = y[90] = np.nan           # the missing-observation branch

    g = ode._grid_batch(V, p, order, order_A, with_A, 0)
    modes = ode._kernel_modes(p, order, order_A, with_A)
    assert modes is not None
    assert same(ode._batch_kernel(y, g, p, modes), ode._batch_numpy(y, g, p))


@pytest.mark.parametrize("unit_roots", [1, 2])
def test_batch_loglik_is_identical_with_pinned_roots(unit_roots):
    p = 4
    rng = np.random.default_rng(7 + unit_roots)
    m = p - unit_roots
    V = np.concatenate([rng.uniform(-0.5, 0.5, (6, m)),
                        np.tile([-7.0, -9.0, 0.5, 0.5, -1.0, -1.0, 2.0, -1.5],
                                (6, 1))], axis=1)
    y = walk(110, seed=3)
    g = ode._grid_batch(V, p, 5, 3, True, unit_roots)
    modes = ode._kernel_modes(p, 5, 3, True)
    assert same(ode._batch_kernel(y, g, p, modes), ode._batch_numpy(y, g, p))


def test_a_row_the_recursion_cannot_represent_still_gets_minus_inf():
    """A dead row is -inf in both paths, and does not poison the live ones."""
    p = 3
    good = np.array([0.5, 0.0, 0.0, -7.0, -9.0, 0.0, 0.0, -1.0, -1.0, 2.0, -1.5])
    bad = good.copy()
    # a process variance the recursion cannot represent: exp(700) scaled by the
    # widest node the log-scale grid reaches overflows, so S is not finite
    bad[p] = 700.0
    bad[p + 4] = math.log(20.0)
    V = np.vstack([good, bad, good])
    y = walk(80, seed=9)
    g = ode._grid_batch(V, p, 5, 3, False, 0)
    modes = ode._kernel_modes(p, 5, 3, False)
    ref = ode._batch_numpy(y, g, p)
    assert same(ode._batch_kernel(y, g, p, modes), ref)
    assert np.isneginf(ref[1]) and np.isfinite(ref[0]) and ref[0] == ref[2]


# ------------------------------------------------------ the streaming filter
@pytest.mark.parametrize("p", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("order,order_A,s_A", [(5, 3, 0.0), (5, 3, 0.15),
                                               (7, 3, 0.0), (4, 3, 0.4)])
def test_filter_is_identical_in_every_field(p, order, order_A, s_A):
    pr = Params(alpha=tuple([0.5] + [0.1] * (p - 1)), Q=7e-4, s2=1e-7,
                phi_P=0.8, s_P=0.9, phi_M=0.4, s_M=0.3, phi_A=0.9, s_A=s_A)
    f = OdeFilter(pr, order=order, order_A=order_A)
    modes = f._stream_modes()
    assert modes is not None
    y = walk(130, seed=p + order)
    y[13] = y[14] = np.nan
    assert same(ode._result_bits(f._run_kernel(y, True, modes)),
                ode._result_bits(f._run_numpy(y, True)))
    assert same(f._run_kernel(y, False, modes), f._run_numpy(y, False))


def test_filter_goes_through_the_kernel_by_default():
    """Not just that the two agree -- that `filter()` is the one that ran."""
    pr = Params(alpha=(0.5, 0.1, 0.1), Q=7e-4, s2=1e-7, phi_P=0.8, s_P=0.9)
    f = OdeFilter(pr, order=5)
    y = walk(80, seed=4)
    r = f.filter(y)
    assert same(ode._result_bits(r), ode._result_bits(f._run_numpy(y, True)))
    assert same(f.loglik(y), f._run_numpy(y, False))


def test_supplied_transitions_stay_on_the_numpy_path():
    """The kernel covers the fitted dynamics; a caller's own F is not its job,
    and must still run rather than silently do nothing."""
    pr = Params(alpha=(0.5, 0.1), Q=7e-4, s2=1e-7, phi_P=0.8, s_P=0.9)
    f = OdeFilter(pr, order=5)
    y = walk(60, seed=5)
    Fs = np.tile(np.array([[0.9, 0.0], [1.0, 0.0]]), (y.size, 1, 1))
    r = f.filter(y, Fs=Fs)
    assert np.all(np.isfinite(r.mean))
    assert not same(r.mean, f.filter(y).mean)


# ------------------------------------------------------------- the parent
@pytest.mark.parametrize("order", [3, 5, 7, 9])
def test_statfilter_batch_loglik_is_identical(order):
    rng = np.random.default_rng(order)
    V = (np.array([-7.0, -9.0, 0.0, 0.0, math.log(0.3), math.log(0.3)])
         + 0.5 * rng.standard_normal((12, 6)))
    x = walk(300, seed=order)
    x[17] = np.nan
    assert stat._batch_verified(order)
    g = stat._loglik_batch(x, V, order)
    # rebuild the grid the way _loglik_batch does, to call the NumPy path
    Q, S2 = np.exp(V[:, 0]), np.exp(V[:, 1])
    phP = 1.0 / (1.0 + np.exp(-V[:, 2]))
    phM = 1.0 / (1.0 + np.exp(-V[:, 3]))
    sP, sM = np.exp(V[:, 4]), np.exp(V[:, 5])
    lamP, wP, TP = stat._chain_batch(phP, sP, order)
    lamM, wM, TM = stat._chain_batch(phM, sM, order)
    LP = np.repeat(lamP, order, axis=1)
    LM = np.tile(lamM, (1, order))
    T = (TP[:, :, None, :, None] * TM[:, None, :, None, :]).reshape(
        V.shape[0], order * order, order * order)
    pi = (wP[:, :, None] * wM[:, None, :]).reshape(V.shape[0], order * order)
    Qg = Q[:, None] * np.exp(np.clip(LP, -60.0, 60.0))
    Rg = S2[:, None] * np.exp(np.clip(LM, -60.0, 60.0))
    assert same(g, stat._batch_numpy(x, T, pi, Qg, Rg))


# --------------------------------------------------------------- end to end
@pytest.mark.slow
def test_a_whole_fit_lands_on_the_same_parameters():
    """The one that matters to a reader of a result: the same fit, twice.

    What this adds to the per-mode comparisons above is the CHAINING: a whole staged
    search is ~140 batched evaluations feeding each other, and one bit of divergence
    anywhere sends the optimiser somewhere else entirely.  Every stage still runs.  It is
    p = 1 because the recurrence order buys nothing here -- the kernel's arithmetic at
    every order is pinned bit-for-bit by `test_batch_loglik_is_identical` and
    `test_filter_is_identical_in_every_field` -- and p = 2 costs five times as much.
    """
    y = walk(220, seed=11)
    with_kernel = OdeFilter.fit(y, p=1, max_iter=60)
    saved = ode._kernel
    try:
        ode._kernel = None
        without = OdeFilter.fit(y, p=1, max_iter=60)
    finally:
        ode._kernel = saved
    assert same(np.array(with_kernel.params.alpha),
                np.array(without.params.alpha))
    for name in ("Q", "s2", "phi_P", "phi_M", "s_P", "s_M", "phi_A", "s_A"):
        assert same(np.array([getattr(with_kernel.params, name)]),
                    np.array([getattr(without.params, name)])), name
