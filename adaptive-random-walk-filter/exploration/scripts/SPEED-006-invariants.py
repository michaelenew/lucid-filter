"""SPEED-006: the checks that keep the speed work honest.

Three ways a faster fit could be quietly wrong, each checked directly:

1.  The batched evaluator is a different likelihood from the one the filter
    reports.  It should be the same recursion; at B = 1 it should agree with
    AdaptiveFilter.loglik to the last bit, and its columns must not leak into
    each other (a batch of B copies of one vector must give B identical
    numbers, and interleaving unrelated vectors must not change any of them).

2.  The closed-form face optimum is not the face optimum.  Checked in SPEED-004
    against the grid; rechecked here through the public API and with missing
    data present, since fit() calls it on whatever it is handed.

3.  The search box binds.  The bounds in _bounds() are numerical guards, and a
    guard that binds is a prior.  Every coordinate of every battery fit must sit
    strictly inside its bound.

    python exploration/scripts/SPEED-006-invariants.py
"""
import importlib.util
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "output"))

from statfilter import AdaptiveFilter, Params                       # noqa: E402
from statfilter.core import (_loglik_batch, _face_optimum, _face_profile,  # noqa: E402
                             _bounds, _LOG_S_CAP, _LOG_S_FLOOR, _LOGIT_CAP)

spec = importlib.util.spec_from_file_location(
    "sp003", os.path.join(HERE, "SPEED-003-battery.py"))
sp003 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp003)

rng = np.random.default_rng(11)
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"   [{'ok ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


# --------------------------------------------------------- 1. the evaluator
print("1. the batched evaluator is the shipped recursion")
n = 800
x = np.cumsum(rng.standard_normal(n) * 0.2) + rng.standard_normal(n)
xh = x.copy()
xh[200:230] = np.nan

PARS = (Params(0.05, 1.0, 0.5, 0.5, 0.6, 0.6),
        Params(0.004, 2.3, 0.02, 0.95, 0.03, 1.4),
        Params(1.0, 0.1, 0.9, 0.1, 1e-6, 1e-6))

# On a series with no gaps the two agree bit for bit: every operation is the same
# operation on the same values, only the array shape differs.  On a series WITH a
# gap they can differ by an ULP, and by exactly one identifiable step: the
# missing-observation branch propagates the variance with the dot product
# pi @ Qg in update() and with the row sum (pi * Qg).sum(1) in the batch, and
# those two sum 25 numbers in a different order.  So the tolerance is exact
# equality where nothing is missing, and a few ULP where something is.
for p in PARS:
    ref = AdaptiveFilter(p, order=5).loglik(x)
    got = float(_loglik_batch(x, p._vec()[None, :], 5)[0])
    check("B=1 == loglik(), bit for bit [no gaps]", ref == got,
          f"{ref:.12f} vs {got:.12f}")

for p in PARS:
    ref = AdaptiveFilter(p, order=5).loglik(xh)
    got = float(_loglik_batch(xh, p._vec()[None, :], 5)[0])
    ulp = abs(ref - got) / np.spacing(abs(ref))
    check("B=1 == loglik() to a few ULP [with a gap]", ulp <= 4.0,
          f"{ulp:.1f} ULP  ({ref:.12f} vs {got:.12f})")

V = np.array([Params(0.05, 1.0, 0.5, 0.5, 0.6, 0.6)._vec()] * 4)
out = _loglik_batch(x, V, 5)
check("identical rows give identical values", np.ptp(out) == 0.0)

vs = [Params(0.05, 1.0, 0.5, 0.5, 0.6, 0.6)._vec(),
      Params(0.004, 2.3, 0.02, 0.95, 0.03, 1.4)._vec(),
      Params(1.0, 0.1, 0.9, 0.1, 0.2, 0.2)._vec()]
alone = np.array([float(_loglik_batch(x, v[None, :], 5)[0]) for v in vs])
together = _loglik_batch(x, np.array(vs), 5)
check("no leakage between batch rows", np.array_equal(alone, together),
      f"max |d| = {np.abs(alone - together).max():.3e}")

for order in (3, 7, 9):
    p = Params(0.05, 1.0, 0.5, 0.5, 0.6, 0.6)
    ref = AdaptiveFilter(p, order=order).loglik(x)
    got = float(_loglik_batch(x, p._vec()[None, :], order)[0])
    check(f"B=1 == loglik() at order {order}", ref == got)


# ------------------------------------------------------------ 2. the face
print("\n2. the closed-form face optimum is the face optimum")
for label, series in (("clean", x), ("with a gap", xh)):
    Q, s2, _ = _face_optimum(series)
    here = AdaptiveFilter(Params(Q=Q, s2=s2), order=5).loglik(series)
    _, prof, _ = _face_profile(series, Q / s2)
    check(f"profile == grid on the face [{label}]", abs(here - prof) < 1e-8,
          f"{prof:.10f} vs {here:.10f}")
    worse = max(AdaptiveFilter(Params(Q=Q * a, s2=s2 * b), order=5).loglik(series)
                for a in (0.5, 0.9, 1.1, 2.0) for b in (0.9, 1.0, 1.1))
    check(f"nothing nearby on the face is better [{label}]", worse <= here + 1e-7,
          f"best neighbour {worse - here:+.2e} nats")


# ----------------------------------------------------------- 3. the bounds
print("\n3. no fitted coordinate sits on a bound")
path = os.path.join(ROOT, "exploration", "speedbench", "battery_current.json")
if not os.path.exists(path):
    print("   (skipped: run SPEED-003-battery.py current first)")
else:
    rows = json.load(open(path))["rows"]
    worst, where = 1e9, None
    for r in rows:
        # gamma_0 is not stored, so only the four shape bounds are checkable here;
        # the two scale bounds sit 30 and 5 log units from gamma_0 and cannot bind.
        v = Params.from_dict(r["par"])._vec()
        margins = [(_LOGIT_CAP - abs(v[2])), (_LOGIT_CAP - abs(v[3])),
                   min(v[4] - _LOG_S_FLOOR, _LOG_S_CAP - v[4]),
                   min(v[5] - _LOG_S_FLOOR, _LOG_S_CAP - v[5])]
        if min(margins) < worst:
            worst, where = min(margins), (r["probe"], r["seed"],
                                          int(np.argmin(margins)))
    names = ["logit phi_P", "logit phi_M", "log s_P", "log s_M"]
    check("all shape coordinates strictly interior", worst > 1e-6,
          f"tightest {worst:.3f} nats of slack at {names[where[2]]} "
          f"on {where[0]} seed {where[1]}")
    smalls = [Params.from_dict(r["par"]).s_M for r in rows]
    print(f"   smallest fitted s_M = {min(smalls):.2e} "
          f"(floor {math.exp(_LOG_S_FLOOR):.0e}; anything at the floor means "
          f"'no measurement scale structure', which is the same estimate)")

print(f"\n{'ALL CHECKS PASSED' if ok else 'SOMETHING FAILED'}")
sys.exit(0 if ok else 1)
