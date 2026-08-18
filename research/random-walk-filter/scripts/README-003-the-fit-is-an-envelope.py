"""Does the fitted parameter vector matter? Mostly not -- and that is the point.

`fit()` is easy to mistake for the thing that makes this filter work: regress on
a stretch of history, get six numbers, done.  That reading is wrong, and this
probe is the check.

What `fit()` chooses is a CLASS -- how fast each noise scale is allowed to move,
and roughly how big it is.  It does not choose the operating point.  Where the
scales actually ARE at time t is a posterior over the quadrature grid, and that
posterior is recomputed from evidence at every single step: `update()` never
touches `self.params`, only `self._pi`, `self._m`, `self._P`.  Everything after
the fit is online.

If that is true, the filter's behaviour should be nearly flat over a wide
envelope of fitted vectors -- being "kind of close" should be enough.  This
sweeps each fitted coordinate over one to three decades, reruns the hero series
with no refitting, and reports the three headline behaviours:

    steady-state RMSE   (regime A, where the oracle-tuned Kalman is optimal)
    jump rise time      (steps to close 90% of the level jump)
    regime-C calibration E[e^2/S]   (1.0 = the reported uncertainty is honest)

Writes figures/fit-envelope.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(ROOT))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "lucid"))

from statfilter import AdaptiveFilter, Params            # noqa: E402

import importlib.util                                    # noqa: E402
spec = importlib.util.spec_from_file_location(
    "hero", os.path.join(HERE, "README-001-hero-lucid-vs-kalman.py"))
hero = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hero)

FIG = os.path.join(ROOT, "figures")


def score(params, theta, y):
    r = AdaptiveFilter(params).filter(y)
    A = slice(0, hero.JUMP_AT)
    C = slice(hero.NOISE_AT, hero.N)
    return {
        "ss_rmse": float(np.sqrt(np.mean((r.mean[A] - theta[A]) ** 2))),
        "rise": hero.rise_time(r.mean, theta, hero.JUMP_AT, hero.JUMP),
        "calib_C": float(np.mean((r.mean[C] - theta[C]) ** 2 / r.var[C])),
    }


def main():
    theta, y = hero.generate()
    base = AdaptiveFilter.fit(hero.history()).params
    kal_m, _ = hero.kalman(y, hero.Q_TRUE, hero.S2_A)
    A = slice(0, hero.JUMP_AT)
    kal_ss = float(np.sqrt(np.mean((kal_m[A] - theta[A]) ** 2)))

    out = {"fitted": base.to_dict(),
           "kalman_steady_state_rmse": kal_ss,
           "fitted_score": score(base, theta, y),
           "sweeps": {}}

    d = base.to_dict()
    sweeps = {
        "Q":     [d["Q"] * f for f in (0.03, 0.1, 0.3, 1, 3, 10, 30)],
        "s2":    [d["s2"] * f for f in (0.2, 0.5, 1, 2, 5)],
        "s_P":   [0.25, 0.5, 1.0, 2.0, d["s_P"], 6.0],
        "s_M":   [0.25, 0.5, 1.0, d["s_M"], 3.0],
        "phi_P": [0.0, 0.2, 0.5, 0.8, 0.95],
        "phi_M": [0.0, 0.3, 0.6, d["phi_M"], 0.99],
    }
    for name, values in sweeps.items():
        rows = []
        for v in values:
            p = dict(d); p[name] = v
            s = score(Params(**p), theta, y)
            s[name] = v
            rows.append(s)
        out["sweeps"][name] = rows

    # a deliberately careless fit: every coordinate wrong, but inside the envelope
    careless = dict(d)
    careless.update(Q=d["Q"] * 10, s2=d["s2"] * 3, s_P=1.5, s_M=1.0,
                    phi_P=0.5, phi_M=0.5)
    out["careless"] = {"params": careless, **score(Params(**careless), theta, y)}

    with open(os.path.join(FIG, "fit-envelope.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    f = out["fitted_score"]
    print("Kalman (told the truth) steady-state RMSE : %.4f" % kal_ss)
    print("fitted                                    : rmse %.4f  rise %2d  calib %.2f"
          % (f["ss_rmse"], f["rise"], f["calib_C"]))
    print()
    for name, rows in out["sweeps"].items():
        print("-- %s" % name)
        for r in rows:
            print("   %-7s %-9.4g  rmse %.4f (%+5.1f%%)  rise %2d  calib %.2f"
                  % (name, r[name], r["ss_rmse"],
                     100 * (r["ss_rmse"] / f["ss_rmse"] - 1), r["rise"], r["calib_C"]))
    c = out["careless"]
    print("\ncareless (Q x10, s2 x3, all phis/s at round numbers):")
    print("   rmse %.4f (%+.1f%% vs fitted, %+.1f%% vs the Kalman oracle)  rise %d  calib %.2f"
          % (c["ss_rmse"], 100 * (c["ss_rmse"] / f["ss_rmse"] - 1),
             100 * (c["ss_rmse"] / kal_ss - 1), c["rise"], c["calib_C"]))


if __name__ == "__main__":
    main()
