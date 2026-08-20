"""The one residual free parameter (forget), ushered to the least-impactful channel.

WalkingBank averages a grid of (phi, s) by online Bayesian weighting.  With pure
Bayes (forget = 1) the weights concentrate onto the ridge and then FREEZE: a
large sustained shift in the process (phi, s) still re-selects, but only stickily,
so phi_hat/s_hat lock on a long static run.  A forgetting factor < 1 keeps the
weights alive.  That factor is the ONE free parameter left in the whole design.

The point of this probe is where it lives, not that it exists.  It governs the
adaptation rate of (phi, s) -- and (phi, s) are (a) the slowest-varying quantities
in the model (class properties, not the state) and (b) on the flat identification
ridge (finding 14), so their exact value barely reaches the estimate.  The free
parameter has been pushed to the slowest, least consequential channel there is.

Measured
--------
(a) through a mid-stream s-shift (0.25 -> 0.70): s_hat and n_eff for forget = 1.0
    (concentrate then freeze) vs 0.999 (concentrate, then re-select);
(b) it does not matter: new-regime tracking RMSE for forget in {1.0, 0.999, 0.99},
    a single filter frozen at the STALE s, and one at the correct s -- all equal,
    because the mu-walk tracks regardless and the ridge is flat;
(c) forget sweep: static level-RMSE is flat across [0.95, 1.0] (no cost), while
    the post-shift re-selection grows as forget drops below 1 -- so any value near
    1 is free, and the exact choice never reaches the estimate.

Open (recorded in SUMMARY): forget can be ELIMINATED without violating the
no-zero-parameters proof, because doing so only leans harder on the AR(1) shape
assumption already made -- deriving the (phi, s) drift rate from the class rather
than setting it.  Finding the optimal such procedure is a theory task; practically
forget ~ 0.999 will not differ measurably from that optimum.

Run: python 0029_forget_the_last_knob.py   (~30 s)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lucid"))
sys.path.insert(0, os.path.join(HERE, "..", "random-walk-filter", "scripts"))

from statfilter import WalkingBank, WalkingFilter  # noqa: E402
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

PHI, NT, SH = 0.9, 3000, 1500


def shift_series(seed, s_lo=0.25, s_hi=0.70):
    rng = np.random.default_rng(seed)
    z = 0.0; lam = np.zeros(NT)
    for t in range(NT):
        s_t = s_lo if t < SH else s_hi
        z = PHI * z + np.sqrt(s_t * s_t * (1 - PHI ** 2)) * rng.standard_normal()
        lam[t] = z
    theta = np.cumsum(rng.standard_normal(NT) * np.sqrt(np.exp(lam)))
    return theta + rng.standard_normal(NT), theta, lam


def static_level_rmse(forget, nseed=6, s=0.45):
    out = []
    for sd in range(nseed):
        rng = np.random.default_rng(sd); z = 0.0; lam = np.zeros(2500)
        for t in range(2500):
            z = PHI * z + np.sqrt(s * s * (1 - PHI ** 2)) * rng.standard_normal(); lam[t] = z
        th = np.cumsum(rng.standard_normal(2500) * np.sqrt(np.exp(lam)))
        x = th + rng.standard_normal(2500)
        m = WalkingBank(1.0, 1.0, forget=forget).filter(x).mean
        out.append(np.sqrt(np.mean((m[100:] - th[100:]) ** 2)))
    return float(np.mean(out))


def main():
    x, theta, lam = shift_series(4)
    r1 = WalkingBank(1.0, 1.0, forget=1.0).filter(x)
    r9 = WalkingBank(1.0, 1.0, forget=0.999).filter(x)

    # (b) new-regime tracking, several configs
    new = slice(SH + 100, NT)
    def al(v): return v - np.median(v[SH + 100:SH + 400]) + np.median(lam[SH + 100:SH + 400])
    def sr(v): return float(np.sqrt(np.mean((al(v)[new] - lam[new]) ** 2)))
    def lr(m): return float(np.sqrt(np.mean((m[new] - theta[new]) ** 2)))
    configs = [
        ("forget 1.0", WalkingBank(1.0, 1.0, forget=1.0)),
        ("forget 0.999", WalkingBank(1.0, 1.0, forget=0.999)),
        ("forget 0.99", WalkingBank(1.0, 1.0, forget=0.99)),
        ("stale s=0.25", WalkingFilter(1.0, 1.0, phi=0.9, s=0.25)),
        ("correct s=0.70", WalkingFilter(1.0, 1.0, phi=0.9, s=0.70)),
    ]
    res = [(name, sr(f.filter(x).process_scale), lr(f.filter(x).mean)) for name, f in configs]
    for name, s_r, l_r in res:
        print(f"  {name:16s} scale-RMSE {s_r:.3f}  level-RMSE {l_r:.3f}")

    # (c) forget sweep: static cost + post-shift re-selection
    fgs = [0.95, 0.98, 0.99, 0.995, 0.999, 1.0]
    static = [static_level_rmse(f) for f in fgs]
    readapt = []
    for f in fgs:
        r = WalkingBank(1.0, 1.0, forget=f).filter(x)
        readapt.append(float(np.mean(r.s_hat[2800:3000]) - np.mean(r.s_hat[1300:1500])))
    print(f"[static level-RMSE across forget] {dict(zip(fgs, np.round(static,4)))}")
    print(f"[post-shift Δs_hat across forget]  {dict(zip(fgs, np.round(readapt,3)))}")

    tt = np.arange(NT)
    fig, ax = plt.subplots(1, 3, figsize=(16.2, 4.6))

    a = ts.tidy(ax[0])
    a.plot(tt, r1.s_hat, color=ts.SERIES[1], lw=1.5, label="forget 1.0 (freezes)")
    a.plot(tt, r9.s_hat, color=ts.SERIES[3], lw=1.5, label="forget 0.999 (re-selects)")
    a.axvline(SH, color=ts.INK, lw=1.0, ls=":", label="s: 0.25→0.70")
    a.set_xlabel("step"); a.set_ylabel("s_hat  (learned scale swing)")
    a.set_title("(a) forget=1 concentrates then freezes; <1 stays alive")
    a.legend(loc="upper left", fontsize=7.4)
    a2 = a.twinx()
    a2.plot(tt, r1.n_eff, color=ts.SERIES[1], lw=0.8, alpha=0.4)
    a2.plot(tt, r9.n_eff, color=ts.SERIES[3], lw=0.8, alpha=0.4)
    a2.set_ylabel("n_eff (faint)", fontsize=8, color=ts.INK2)

    a = ts.tidy(ax[1])
    names = [r[0] for r in res]; xs = np.arange(len(names)); w = 0.38
    a.bar(xs - w/2, [r[1] for r in res], w, color=ts.SERIES[3], label="scale RMSE")
    a.bar(xs + w/2, [r[2] for r in res], w, color=ts.SEQ[4], label="level RMSE")
    a.set_xticks(xs); a.set_xticklabels(names, rotation=25, ha="right", fontsize=7)
    a.set_ylabel("RMSE on the new regime")
    a.set_title("(b) it doesn't matter — even a stale/frozen (φ,s) tracks")
    a.legend(loc="upper right", fontsize=7.6)

    a = ts.tidy(ax[2])
    a.plot(fgs, static, color=ts.SERIES[2], lw=1.8, marker="o", ms=5, label="static level-RMSE")
    a.set_xlabel("forget"); a.set_ylabel("static level-RMSE", color=ts.SERIES[2])
    a.set_title("(c) tracking flat over [0.95,1]; re-selection grows below 1")
    a.set_ylim(min(static) - 0.01, max(static) + 0.01)
    a3 = a.twinx()
    a3.plot(fgs, readapt, color=ts.SERIES[5], lw=1.8, marker="s", ms=5)
    a3.set_ylabel("post-shift Δs_hat (re-selection)", color=ts.SERIES[5], fontsize=8)
    a.axvline(0.999, color=ts.INK, lw=1.0, ls=":")
    a.text(0.999, min(static), " default", fontsize=7.4, color=ts.INK, va="bottom", ha="right")
    ts.save(fig, os.path.join(HERE, "figures", "0028-forget-the-last-knob.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
