"""A compiled kernel for the filter recursions, byte-identical to the NumPy one.

Both filters are dispatch-bound rather than arithmetic-bound: a step is a
handful of einsums over arrays small enough that deciding what to do costs
more than doing it.  On a p = 3, order = 5 fit of 500 points, 99% of the wall
clock is inside `_loglik_batch` and 79% of the whole fit is inside
`c_einsum` alone.  This package is that loop in C.

**It is not a second implementation.**  Every result it returns is bit-for-bit
what the NumPy code returns -- the same IEEE-754 bits, not the same number to
within a tolerance -- so a run with the kernel and a run without it are the
same run, and switching it on is not a modelling decision.  Three things make
that true rather than hopeful:

* the kernel calls **NumPy's own** `exp`, `log`, `dot` and pairwise `sum`
  rather than libm's and its own, because those are the operations whose last
  bit is a property of the local NumPy build (its SIMD dispatch, its BLAS)
  rather than of the arithmetic;
* the contraction orders that `np.einsum` chooses for itself -- and it does
  choose differently for different `p` -- are **probed**, not assumed:
  :func:`orders_for` asks NumPy what it does before the kernel does it;
* and the whole recursion is then **verified end to end**: before the kernel
  is used for a given problem shape it is run against the NumPy path on a
  short random series and the two are compared bit for bit.  A shape that
  fails falls back to NumPy and says so once.  There is no configuration in
  which a mismatched kernel is used.

That last point is what makes this safe across NumPy versions and CPUs: a
future NumPy that reassociates one of these sums does not silently change
anyone's numbers, it turns the kernel off.

Build it with ``python3 -m lucid_kernel.build``; without it, everything still
works and is merely slower.  ``LUCID_KERNEL=0`` in the environment turns it
off.
"""
from __future__ import annotations

import os
import warnings

__all__ = ["available", "enabled", "verify", "reason", "AP_FLAT", "AP_NESTED",
           "MP_LANES", "MP_EINSUM", "SC_NAIVE", "SC_EINSUM", "MODES"]

AP_FLAT, AP_NESTED = 0, 1
MP_LANES, MP_EINSUM = 0, 1

#: every (ap_mode, mp_mode) the C side knows how to do, in the order they are
#: tried.  The first that reproduces NumPy exactly wins; the last is the one
#: that hands both undecidable contractions back to NumPy and so is expected
#: to work whatever NumPy does, at the cost of two calls per step.
MODES = ((AP_FLAT, MP_LANES), (AP_NESTED, MP_LANES),
         (AP_FLAT, MP_EINSUM), (AP_NESTED, MP_EINSUM))

SC_NAIVE, SC_EINSUM = 0, 1

_disabled = os.environ.get("LUCID_KERNEL", "").strip() in ("0", "no", "off")
_reason = "disabled by LUCID_KERNEL" if _disabled else None

try:
    if _disabled:
        raise ImportError("disabled")
    from . import _kernel as _ext
except ImportError as exc:                       # pragma: no cover - build-dep
    _ext = None
    if _reason is None:
        _reason = (f"{exc}; build it with `python3 -m lucid_kernel.build`")

if _ext is not None:
    import math

    _ext.configure(math.log(2.0 * math.pi))


def available() -> bool:
    """Whether the compiled extension is importable at all."""
    return _ext is not None


def reason() -> str | None:
    """Why the kernel is not in use, or None when it is."""
    return _reason


def ext():
    """The extension module, or None."""
    return _ext


# --------------------------------------------------------------- verification
_verified: dict = {}


def verify(key, run_kernel, run_reference, candidates=None,
           warn: bool = True):
    """Find a mode pair whose kernel output is bit-identical to NumPy's.

    ``run_kernel(*modes)`` and ``run_reference()`` must return the
    same-shaped float64 array for the same short problem.  ``candidates`` are
    the mode pairs to try, in order -- normally the one the caller's own probe
    of NumPy proposes; omit it to try every pair the C side knows.  Returns
    the winning ``(ap_mode, mp_mode)``, or None when none of them reproduces
    NumPy, in which case the caller must use its own NumPy path.

    Cached on ``key``, which the caller chooses to cover everything that could
    change the answer (the order of the recurrence, the grid sizes).  The
    comparison is on raw bits: `np.array_equal` would call two NaNs unequal
    and 0.0 and -0.0 equal, and neither is what "identical" means here.
    """
    if _ext is None:
        return None
    if key in _verified:
        return _verified[key]

    import numpy as np

    ref = np.asarray(run_reference(), dtype=float)
    want = ref.view(np.uint64)
    found = None
    for modes in (MODES if candidates is None else candidates):
        got = run_kernel(*modes)
        if got is None:
            continue
        got = np.asarray(got, dtype=float)
        if got.shape == ref.shape and np.array_equal(got.view(np.uint64), want):
            found = tuple(modes)
            break
    if found is None and warn:
        warnings.warn(
            "lucid_kernel: no compiled evaluation order reproduces this "
            f"NumPy build bit for bit for {key}; falling back to NumPy. "
            "The results are unaffected; only the speed is.",
            RuntimeWarning, stacklevel=2)
    _verified[key] = found
    return found


def enabled() -> bool:
    """Whether the kernel has been verified for at least one problem shape."""
    return any(v is not None for v in _verified.values())


def clear_cache() -> None:
    """Forget every verification.  For the tests."""
    _verified.clear()
