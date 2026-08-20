"""The stiff, asymmetric well: don't set the gain from the LOCAL curvature.

SUPERSEDED by finding 18 (0031_derived_walk_loop.py).  The DIAGNOSIS here stands
-- the observability I swings across the well and the loop gain must not be built
from the local curvature -- but the EMA fix below (q_mu = r*/Ibar) introduced
three un-derived constants (r*, the seed 0.4, the rate 30) and was replaced by a
first-principles derivation: critical damping pins K* = (1-phi)/4, and q_mu
follows from K* and the derived steady Fisher I_char.  This file is kept as the
arc that led there; its "now shipped" labels are no longer accurate.


The tracker descends the scale-family well D(x)=1/2(e^x-1-x) (finding 11): convex
but wildly asymmetric -- an exponential wall on the loud side, a flat plateau on
the quiet side.  Its curvature, and the filter's per-step Fisher information I
(the observability the gain is built from), therefore swing by ~20x across the
reachable range -- high where the process is loud, near-zero where it is quiet.

Finding 10 set the drift variance to the critical-damping point q_mu = r*/I with I
the INSTANTANEOUS grid Fisher.  That over-reacts to the well's local curvature:
where I is momentarily low (a quiet stretch, the soft plateau) q_mu = r*/I blows
up, the gain over-shoots, and the tracker chatters -- it tracks fluctuations worse
AND fails to settle onto a genuinely quiet regime.  This is the last place the
nonlinearity bit.

The fix (finding 17): set the drift variance from the REGIME's steady
observability, q_mu = r*/Ibar, with Ibar a slow EMA of I (rate ~ (1-phi)/30, seeded
at the characteristic locked value).  The per-step gradient step and the
Cramer-Rao down-weight R = 1/I still use the instantaneous I; only the drift
variance is steadied.  Result: uniform damping across the well, better tracking,
and quiet regimes captured.

Measured (rules compared on the same recursion)
-----------------------------------------------
(a) observability I vs regime d (the ~20x 'stiffness'), and q_mu = r*/I (instant,
    huge swing) vs r*/Ibar (steady);
(b) capture % vs destination: instant fails DOWN to quiet regimes, steady does not;
(c) tracking RMSE vs scale swing, and a loud->quiet shift: steady wins throughout.

Run: python 0030_stiff_wall_gain.py   (~2-3 min)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "random-walk-filter", "scripts"))

from gridlab import simulate  # noqa: E402
from moving_grid import MovingChannel  # noqa: E402
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

RSTAR = 3.5e-4
ICHAR = 0.4
RATE = (1 - 0.9) / 30.0


def _run(x, lam_ref, rule, phi=0.9, s=0.30, base_mu=0.0):
    hw = 1.5 * s * 3
    f = MovingChannel(1.0, 1.0, phi=phi, s=s, step="kalman_auto", q_mu=RSTAR / ICHAR,
                      P0=25.0, uniform=(hw, 1.5 * s), mu_cap=1.5 * s)
    f.reset(mu=base_mu)
    Ibar = ICHAR; out = np.empty(x.size)
    for i, v in enumerate(x):
        o = f.update(v); out[i] = o["logscale"]
        I = o["fisher"]
        Ibar += RATE * (I - Ibar)
        f.q_mu = RSTAR / max(I, 1e-6) if rule == "instant" else RSTAR / Ibar
    return out


def observability(d, nseed=40, nt=700):
    acc = []
    for sd in range(nseed):
        x = simulate(np.random.default_rng(sd), d, 1.0, 1.0, nt)
        f = MovingChannel(1.0, 1.0, phi=0.9, s=0.30, step="kalman_auto", q_mu=1e-6, P0=1.0)
        f.reset(mu=d)
        acc.append(np.mean([f.update(v)["fisher"] for v in x][nt // 4:]))
    return float(np.mean(acc))


def capture(rule, D, nseed=40, NT=1000, JT=100):
    g = 0
    for sd in range(nseed):
        rng = np.random.default_rng(sd); lam = np.zeros(NT); lam[JT:] = D
        st = rng.standard_normal(NT) * np.sqrt(np.exp(lam)); x = np.cumsum(st) + rng.standard_normal(NT)
        ls = _run(x, lam, rule)
        if abs(np.mean(ls[-200:]) - D) < 0.7:
            g += 1
    return 100.0 * g / nseed


def track_rmse(rule, s_true, nseed=50, NT=2500, phi=0.9):
    B = []; V = []
    for sd in range(nseed):
        rng = np.random.default_rng(sd); z = 0.0; lam = np.zeros(NT)
        for t in range(NT):
            z = phi * z + np.sqrt(s_true ** 2 * (1 - phi ** 2)) * rng.standard_normal(); lam[t] = z
        th = np.cumsum(rng.standard_normal(NT) * np.sqrt(np.exp(lam))); x = th + rng.standard_normal(NT)
        e = _run(x, lam, rule, s=s_true)[200:] - lam[200:]
        B.append(e.mean()); V.append(e.std())
    return float(np.sqrt(np.mean(B) ** 2 + np.mean(V) ** 2))


def main():
    dd = np.round(np.arange(-2.0, 5.01, 1.0), 1)
    Iof = np.array([observability(d) for d in dd])
    dcap = np.round(np.arange(-3.0, 6.01, 1.0), 1)
    cap_i = np.array([capture("instant", d) for d in dcap])
    cap_s = np.array([capture("steady", d) for d in dcap])
    ss = np.array([0.20, 0.30, 0.45, 0.60])
    tr_i = np.array([track_rmse("instant", s) for s in ss])
    tr_s = np.array([track_rmse("steady", s) for s in ss])
    print("[observability] I range:", Iof.min(), "->", Iof.max(), f"({Iof.max()/Iof.min():.0f}x)")
    print("[capture] instant:", dict(zip(dcap, cap_i.astype(int))))
    print("[capture] steady :", dict(zip(dcap, cap_s.astype(int))))
    print("[tracking] instant:", np.round(tr_i, 3), " steady:", np.round(tr_s, 3))

    fig, ax = plt.subplots(1, 3, figsize=(16.2, 4.6))

    a = ts.tidy(ax[0])
    a.plot(dd, Iof, color=ts.SERIES[2], lw=1.9, marker="o", ms=4, label="observability I(d)")
    a.set_xlabel("regime loudness d  (nats)"); a.set_ylabel("Fisher info I  (the 'stiffness')")
    a.set_title(f"(a) I swings ~{Iof.max()/Iof.min():.0f}× across the well")
    a2 = a.twinx()
    a2.plot(dd, RSTAR / np.maximum(Iof, 1e-6), color=ts.SERIES[1], lw=1.6, ls="--",
            label="q_mu=r*/I (instant)")
    a2.axhline(RSTAR / ICHAR, color=ts.SERIES[3], lw=1.6, label="q_mu=r*/Ibar (steady)")
    a2.set_ylabel("q_mu", fontsize=8); a2.set_yscale("log")
    ln1, lb1 = a.get_legend_handles_labels(); ln2, lb2 = a2.get_legend_handles_labels()
    a.legend(ln1 + ln2, lb1 + lb2, loc="upper center", fontsize=7.0)

    a = ts.tidy(ax[1])
    a.plot(dcap, cap_i, color=ts.SERIES[1], lw=1.9, marker="o", ms=4, label="instant r*/I (old)")
    a.plot(dcap, cap_s, color=ts.SERIES[3], lw=1.9, marker="s", ms=4, label="steady r*/Ibar (now shipped)")
    a.axvspan(-3, 0, color=ts.SEQ[1], alpha=0.3, lw=0)
    a.text(-2.8, 45, "quiet\ndestinations", fontsize=8, color=ts.INK2, va="center")
    a.set_xlabel("jump destination d  (nats)"); a.set_ylabel("capture %")
    a.set_title("(b) instant gain fails DOWN into quiet regimes")
    a.legend(loc="lower right", fontsize=7.6)

    a = ts.tidy(ax[2])
    xw = np.arange(ss.size); w = 0.38
    a.bar(xw - w/2, tr_i, w, color=ts.SERIES[1], label="instant r*/I")
    a.bar(xw + w/2, tr_s, w, color=ts.SERIES[3], label="steady r*/Ibar")
    a.set_xticks(xw); a.set_xticklabels([f"{s:.2f}" for s in ss])
    a.set_xlabel("scale swing  s"); a.set_ylabel("tracking RMSE (nats)")
    a.set_title("(c) steady gain tracks fluctuations better too")
    a.legend(loc="upper left", fontsize=7.6)
    ts.save(fig, os.path.join(HERE, "figures", "0029-stiff-wall-gain.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
