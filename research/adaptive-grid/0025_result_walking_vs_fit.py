"""Headline result: online tracking survives a regime shift a one-time fit can't.

The other filters in the package learn their noise structure ONCE, by maximum
likelihood, and then run with it frozen.  The WalkingFilter learns the changing
part ONLINE -- it walks a fine grid to wherever the process scale is, deriving
position, gain, step size and spacing from the data, given only the class pair
(phi, s).  This is the test that shows why that matters, in the exact way a
deployed filter meets it: you fit on the history you have, then stream forward --
and the world moves into a regime your fit never saw.

The process volatility is non-stationary: quiet, then a +3.5-nat shift to a loud
regime partway through.  Three filters run causally:

  * WALKING (online)      -- WalkingFilter(phi, s), no fit, tracks throughout;
  * FIT-ON-HISTORY        -- AdaptiveFilter.fit() on the quiet first half only,
                             then frozen and run forward (the realistic deploy);
  * ORACLE FIT            -- AdaptiveFilter.fit() on the WHOLE series, an upper
                             bound that needs data from the future, so it is not
                             deployable -- shown only to bound the achievable.

The honest point is not that walking beats the oracle (it does not; the oracle
saw everything).  It is that walking nearly MATCHES the oracle without the future,
while the deployable frozen fit goes stale exactly where the user predicted:
"one-time fits don't work because the regime will shift outside their range."

Measured (errors on the SECOND half -- the new regime)
------------------------------------------------------
(a) the level through the shift, true vs the three trackers;
(b) recovered process log-scale: truth, walking, frozen fit, oracle fit;
(c) level- and scale-RMSE on the new regime for the three.

Run: python 0025_result_walking_vs_fit.py   (two fit() calls, ~40 s)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lucid"))
sys.path.insert(0, os.path.join(HERE, "..", "random-walk-filter", "scripts"))

from statfilter import WalkingFilter, AdaptiveFilter  # noqa: E402
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

PHI, S, Q, S2 = 0.9, 0.30, 1.0, 1.0
NT, SHIFT = 1400, 700


def make(seed):
    """A level with a NON-stationary process scale: quiet regime, then loud."""
    rng = np.random.default_rng(seed)
    # true process log-scale: AR(1) fluctuations + a regime step
    lam = np.zeros(NT)
    z = 0.0
    for t in range(NT):
        z = PHI * z + np.sqrt(S * S * (1 - PHI * PHI)) * rng.standard_normal()
        lam[t] = z
    lam[:SHIFT] += -1.0
    lam[SHIFT:] += 2.5
    step = rng.standard_normal(NT) * np.sqrt(Q * np.exp(lam))
    theta = np.cumsum(step)
    x = theta + rng.standard_normal(NT) * np.sqrt(S2)
    return x, theta, lam


def main():
    x, theta, lam = make(7)

    # WALKING -- online, no fit
    t0 = time.time()
    rw = WalkingFilter(Q=Q, s2=S2, phi=PHI, s=S).filter(x)
    t_walk = time.time() - t0

    # FIT-ON-HISTORY -- fit the quiet first half, freeze, run the whole series
    af_h = AdaptiveFilter.fit(x[:SHIFT])
    rh = af_h.filter(x)

    # ORACLE -- fit the whole series (uses the future; not deployable)
    af_o = AdaptiveFilter.fit(x)
    ro = af_o.filter(x)

    # centre each log-scale on its own quiet baseline (removes the arbitrary log-Q
    # offset) so the SHAPE of the recovery is what's compared
    def centre(v):
        return v - np.median(v[:SHIFT])
    lam_c = centre(lam)
    walk_c, hist_c, orac_c = centre(rw.process_scale), centre(rh.process_scale), centre(ro.process_scale)

    new = slice(SHIFT, NT)                               # score the NEW regime only
    def s_rmse(v):
        return float(np.sqrt(np.mean((v[new] - lam_c[new]) ** 2)))
    def l_rmse(m):
        return float(np.sqrt(np.mean((m[new] - theta[new]) ** 2)))

    rows = [("walking (online, no fit)", walk_c, rw.mean, ts.SERIES[3]),
            ("fit-on-history (frozen)", hist_c, rh.mean, ts.SERIES[1]),
            ("oracle fit (sees future)", orac_c, ro.mean, ts.INK2)]
    print(f"[history fit] {af_h.params}")
    print("[new-regime RMSE]  method                       scale   level")
    for name, sc, mn, _ in rows:
        print(f"                   {name:28s} {s_rmse(sc):.3f}   {l_rmse(mn):.3f}")
    print(f"[walking cost] {t_walk*1000:.0f} ms, no fit")

    tt = np.arange(NT)
    fig, ax = plt.subplots(1, 3, figsize=(16.2, 4.6))

    a = ts.tidy(ax[0])
    a.plot(tt, x, color=ts.GRID, lw=0.5, alpha=0.7)
    a.plot(tt, theta, color=ts.INK, lw=1.7, label="true level")
    a.plot(tt, rw.mean, color=ts.SERIES[3], lw=1.1, label="walking")
    a.plot(tt, rh.mean, color=ts.SERIES[1], lw=1.1, label="fit-on-history")
    a.axvspan(SHIFT, NT, color=ts.SEQ[1], alpha=0.4, lw=0)
    a.set_xlabel("step"); a.set_ylabel("level")
    a.set_title("(a) level tracking; shaded = the new regime")
    a.legend(loc="upper left", fontsize=7.4)

    a = ts.tidy(ax[1])
    a.plot(tt, lam_c, color=ts.INK, lw=1.9, label="true process log-scale")
    for name, sc, _, col in rows:
        a.plot(tt, sc, color=col, lw=1.3, label=name)
    a.axvline(SHIFT, color=ts.SERIES[1], lw=1.0, ls=":")
    a.set_xlabel("step"); a.set_ylabel("process log-scale  (centred, nats)")
    a.set_title("(b) walking follows the shift; the frozen fit can't")
    a.legend(loc="upper left", fontsize=7.0)

    a = ts.tidy(ax[2])
    xw = np.arange(2); w = 0.26
    for k, (name, sc, mn, col) in enumerate(rows):
        a.bar(xw + (k - 1) * w, [s_rmse(sc), l_rmse(mn)], w, color=col, label=name)
    a.set_xticks(xw); a.set_xticklabels(["scale RMSE", "level RMSE"])
    a.set_ylabel("RMSE on the new regime")
    a.set_title("(c) walking ≈ oracle, without the future; frozen fit lags")
    a.legend(loc="upper right", fontsize=7.0)
    ts.save(fig, os.path.join(HERE, "figures", "0024-walking-vs-fit.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
