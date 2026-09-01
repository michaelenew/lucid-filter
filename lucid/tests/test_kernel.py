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
