"""Tests for the compiled kernel.

What survives here is the part of the kernel that stands on its own: the
NumPy-owned primitives it borrows rather than implements, and the verify-or-fall-back
gate that is the whole safety argument.

Its three NUMERIC entry points -- ``ode_loglik_batch``, ``ode_filter`` and
``stat_loglik_batch`` -- are the recursions of `OdeFilter` and `AdaptiveFilter`, the
prototypes this package no longer ships, so there is at present nothing in the library
for them to be compared against.  Until the kernel carries a `LucidFilter` step, that
comparison cannot be written; see `lucid/lucid_kernel/README.md`.

If the kernel is not built, the whole module skips: nothing here is a claim about the
filter, only about the kernel, and without one there is nothing to compare.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lucid import lucid_kernel as K                                      # noqa: E402

pytestmark = pytest.mark.skipif(
    K is None or not K.available(),
    reason=f"kernel not built ({K.reason() if K else 'not importable'})")


def bits(a):
    """The raw payloads, so the comparison is exact about NaN and -0.0."""
    return np.ascontiguousarray(np.asarray(a, dtype=float)).view(np.uint64)


def same(a, b):
    a, b = bits(a), bits(b)
    return a.shape == b.shape and np.array_equal(a, b)


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


# --------------------------------------------------------- the fallback gate
def test_a_shape_the_kernel_cannot_match_is_refused_not_guessed():
    """The fallback is the whole safety argument, so make it fire.

    Nothing compiled is used for a problem shape until it has been run against the
    NumPy path and compared bit for bit.  A candidate that disagrees is refused, with
    a warning, and the NumPy path runs instead -- which is what makes switching the
    kernel on not a modelling decision.
    """
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


# ------------------------------------------------------ the LucidFilter step
# The kernel carries a compiled transcription of the stacked bank step.  What is
# checked is not that it is close: it is that every field it produces has the
# same IEEE-754 bits as the NumPy the filter would otherwise have run.

from lucid.filter.lucid import (_BANK_WIDTHS, _bank_correct_np,     # noqa: E402
                                _bank_modes, _bank_predict_np,
                                _einsum_orders, _inv_sym)
from lucid import LucidFilter                                        # noqa: E402


def _bank_rig(n, m, G, nw, r, has_cap, seed=0):
    """A random problem of one bank's shape: the state a step is handed."""
    rng = np.random.default_rng(seed)
    M = 5
    A = rng.standard_normal((M, n, n))
    P = np.einsum("bij,bkj->bik", A, A) + n * np.eye(n)
    B = rng.standard_normal((M, G, n, n))
    pi = rng.uniform(0.1, 1.0, (M, r, nw))
    pi /= pi.sum(2, keepdims=True)
    ax = np.ascontiguousarray(
        np.stack([rng.permutation(G)[:nw] for _ in range(r)]).astype(np.intp))
    return dict(
        F=np.ascontiguousarray(rng.standard_normal((M, n, n))),
        P=np.ascontiguousarray(P),
        Qg=np.ascontiguousarray(np.einsum("bgij,bgkj->bgik", B, B) * 0.3),
        rg=np.ascontiguousarray(rng.uniform(0.2, 2.0, (M, G, m))),
        H=np.ascontiguousarray(rng.standard_normal((m, n))),
        mpred=np.ascontiguousarray(rng.standard_normal((M, n))),
        e=np.ascontiguousarray(rng.standard_normal((M, m))),
        pi=np.ascontiguousarray(pi), ax=ax,
        cap=(rng.uniform(0.2, 1.0, n) if has_cap else None), m=m)


@pytest.mark.parametrize("n", list(range(_BANK_WIDTHS[0][0], _BANK_WIDTHS[0][1] + 1)))
@pytest.mark.parametrize("m", list(range(_BANK_WIDTHS[1][0], _BANK_WIDTHS[1][1] + 1)))
def test_the_bank_step_is_identical_in_every_field(n, m):
    """Every width the transcription claims must reproduce NumPy, field by field."""
    G, nw, r = 4 * n + 5, 5, n + m
    modes = _bank_modes(n, m, G, nw, r, True)
    assert modes is not None, (n, m)            # a claimed width may not decline
    ext = K.ext()
    for seed in (0, 1, 2):
        g = _bank_rig(n, m, G, nw, r, True, seed=seed)
        Pp1, PH1, S1 = _bank_predict_np(g["F"], g["P"], g["Qg"], g["rg"], g["H"])
        Pp2, PH2, S2 = ext.lucid_bank_predict(
            g["F"], g["P"], g["Qg"], g["rg"], g["H"], int(modes[0]), int(modes[2]))
        assert same(Pp1, Pp2) and same(PH1, PH2) and same(S1, S2), ("predict", seed)
        Si = _inv_sym(S1, m)
        _, ld = np.linalg.slogdet(S1)
        ld = np.ascontiguousarray(ld)
        piA, piB = g["pi"].copy(), np.array(g["pi"], order="C")
        o1 = _bank_correct_np(Pp1, PH1, Si, ld, g["e"], g["mpred"], g["H"], piA,
                              g["ax"], g["cap"], m)
        o2 = ext.lucid_bank_correct(Pp1, PH1, Si, ld, g["e"], g["mpred"], g["H"],
                                    piB, g["ax"], g["cap"], m, int(modes[1]),
                                    int(modes[2]), int(modes[3]))
        assert same(piA, piB), ("pi", seed)
        for name, a, b in zip(("w", "ll", "m_new", "P_new", "Kbar"), o1, o2):
            assert same(a, b), (name, seed)


def test_the_filter_goes_through_the_kernel_and_gets_the_same_bits():
    """Not that the two agree in the abstract -- that `filter()` is the one that
    ran the compiled step, and that turning it off changes nothing at all."""
    rng = np.random.default_rng(4)
    F = np.array([[1.0, 0.1], [0.0, 1.0]])
    Y = np.cumsum(rng.standard_normal((80, 2)), axis=0)
    f = LucidFilter(dynamics=F, H=np.eye(2))
    assert any(getattr(bk, "_modes", None) is not None for bk in f._banks), (
        "no bank of this rig reached the compiled step")
    with_k = f.filter(Y)
    g = LucidFilter(dynamics=F, H=np.eye(2))
    for bk in g._banks:                          # the only difference
        bk._modes = None
    without = g.filter(Y)
    for name in ("mean", "var", "process_scale", "measurement_scale"):
        assert same(getattr(with_k, name), getattr(without, name)), name
    assert with_k.loglik == without.loglik


def test_a_width_it_does_not_cover_is_declined_not_guessed():
    """The transcription is of the narrow widths; outside them NumPy reduces in
    its vector registers and the kernel says so by standing down."""
    assert _bank_modes(1, 1, 9, 5, 2, False) is None
    assert _bank_modes(12, 6, 41, 5, 9, False) is None
