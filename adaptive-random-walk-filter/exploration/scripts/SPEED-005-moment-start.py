"""SPEED-005: can one online pass guess the scale channels?

SPEED-004 gives the exact optimum on the s = 0 face for free, and with it the
standardised innovations u_t of the homoscedastic fit.  If the data really do
carry log-scale structure it is still in those residuals, and the stochastic
volatility literature's oldest identity reads it off in closed form:

    log u_t^2 = log z_t^2 + lam_t,      z_t ~ N(0,1), independent of lam
    Var(log z^2) = pi^2/2 = 4.9348      (exact, log chi^2_1)

so with c_k the lag-k autocovariance of log u^2,

    s^2 = c_0 - pi^2/2                  (the log-scale variance)
    phi = c_1 / s^2                     (its persistence)

One pass over the residuals: no recursion over the grid, no optimiser.  It
cannot separate the channels -- process and measurement noise both inflate an
innovation -- so it yields a magnitude and a persistence and leaves the split to
the start screen.

The question is only whether it lands near where fit() ends up, often enough to
be worth three extra rows in a batch that costs nothing.  Baseline fits are read
from the battery run rather than recomputed.

    python exploration/scripts/SPEED-003-battery.py baseline   # first
    python exploration/scripts/SPEED-005-moment-start.py
"""
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "output"))

from statfilter.core import _face_optimum, _moment_scale          # noqa: E402

spec = importlib.util.spec_from_file_location(
    "sp003", os.path.join(HERE, "SPEED-003-battery.py"))
sp003 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp003)


if __name__ == "__main__":
    rows = json.load(open(os.path.join(
        ROOT, "exploration", "speedbench", "battery_baseline.json")))["rows"]
    fits = {(r["probe"], r["seed"]): r["par"] for r in rows}

    print("the moment pass against what fit() converges to\n")
    print(f"{'probe':>18} {'seed':>9} {'s hat':>7} {'phi hat':>8} | "
          f"{'s_P':>6} {'s_M':>6} {'phi_P':>6} {'phi_M':>6}")
    for name in sp003.PROBES:
        for sd in sp003.SEEDS:
            rng = np.random.default_rng([sd, sp003.PROBES.index(name)])
            x, _ = sp003.probe(name, rng)
            _, _, u = _face_optimum(x)
            s, phi = _moment_scale(u)
            p = fits[(name, sd)]
            print(f"{name:>18} {sd:>9} {s:>7.3f} {phi:>8.3f} | "
                  f"{p['s_P']:>6.3f} {p['s_M']:>6.3f} "
                  f"{p['phi_P']:>6.3f} {p['phi_M']:>6.3f}")
        print()
