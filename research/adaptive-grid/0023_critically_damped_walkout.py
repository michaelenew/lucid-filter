"""The dense, overlapping walk-out, tuned to critical damping -- and where tau lands.

The reach fix (0022) is a bounded stride.  Tie that stride to the grid: with
mu_cap = gap the window shifts at most ONE node per step, so consecutive windows
overlap on all but one node -- the walk stays dense and overlapping with the
prior step, by construction.  That pins the slew; the only free knob left is the
loop bandwidth q_mu (equivalently tau, via q_mu = 1/(I*tau^2)).

Set it by damping.  The walk-out is a second-order response: mu slews to the
truth, the inner posterior pi lags, and on arrival the pair can overshoot and
ring.  Critical damping (zeta = 1) is the fastest arrival with NO overshoot --
the boundary between the sluggish (over-damped) and the ringing (under-damped)
walk.  Because the cap slew-limits the approach, convergence rate SATURATES: past
the damping knee, more q_mu buys a higher steady floor and a reversion bias, for
almost no speed.  So the defensible operating point is critical damping itself;
tau is then determined, not free.

For contrast the probe also marks the 45-degree knee of the IDEALISED scalar-
Kalman frontier -- in the two dimensionless [0,1] measures, convergence rate
K (fraction of error closed per step) and steady floor sqrt(K), the tangent
d(floor)/d(rate) = 1/(2 sqrt(K)) = 1 sits at K = 1/4 (r = 0.083, tau ~ 3.5).
That knee lands PAST the capped system's rate rollover -- the reversion-limited
walk never reaches it, so critical damping (the more conservative point) binds.

Measured (walk-out to a loud +3 jump; cap = gap = 0.45; I(d=3) ~ 0.41)
------------------------------------------------------------------------
(a) three walk-outs -- over-damped, critically damped, under-damped -- with the
    overlapping window band drawn on the critical one;
(b) arrival overshoot vs r = q_mu*I: the zeta = 1 crossing (critical damping);
(c) convergence rate vs steady floor: the measured curve saturates and rolls
    over; critical damping sits at its knee, the idealised 45-deg knee is past it.

Run: python 0023_critically_damped_walkout.py   (heavy; ~3-4 min)
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

GAP, HALF = 0.45, 1.35                       # cap = gap: dense, overlapping walk
JT, NT, D, BAND = 80, 400, 3.0, 0.6
QMUS = np.array([3e-4, 5e-4, 8e-4, 1.3e-3, 2e-3, 3e-3, 5e-3, 1e-2, 2e-2,
                 5e-2, 1e-1, 2e-1])


def info(d, nseed=40, nt=700):
    acc = []
    for sd in range(nseed):
        rng = np.random.default_rng(sd)
        x = simulate(rng, d, 1.0, 1.0, nt)
        f = MovingChannel(1.0, 1.0, phi=0.9, s=0.30, order=5,
                          step="kalman_auto", q_mu=1e-6, P0=1.0)
        f.reset(mu=d)
        acc.append(np.mean([f.update(v)["fisher"] for v in x][nt // 4:]))
    return float(np.mean(acc))


def walkout(q_mu, nseed=200):
    """Median walk-out to +3 and its (overshoot, settle, floor)."""
    E = np.zeros((nseed, NT))
    for sd in range(nseed):
        rng = np.random.default_rng(sd)
        lam = np.zeros(NT); lam[JT:] = D
        st = rng.normal(0.0, np.sqrt(np.exp(lam)))
        x = np.cumsum(st) + rng.normal(0.0, 1.0, NT)
        f = MovingChannel(1.0, 1.0, phi=0.9, s=0.30, order=5, step="kalman_auto",
                          q_mu=q_mu, P0=25.0, uniform=(HALF, GAP), mu_cap=GAP)
        f.reset(mu=0.0)
        E[sd] = [f.update(v)["logscale"] for v in x]
    e = np.median(E, 0)
    over = max(0.0, float(e[JT:].max()) - D)
    out = np.where(np.abs(e - D) > BAND)[0]; out = out[out >= JT]
    tset = float(out[-1] + 1 - JT) if out.size else 1.0
    floor = float(np.sqrt(((E[:, -60:] - D) ** 2).mean()))
    return e, over, tset, floor


def cross_zero(x, y):
    """Interpolate where y first rises through 0 (overshoot onset)."""
    for i in range(1, len(y)):
        if y[i - 1] <= 1e-3 < y[i]:
            t = (1e-3 - y[i - 1]) / (y[i] - y[i - 1])
            return x[i - 1] * (x[i] / x[i - 1]) ** t
    return x[int(np.argmax(y > 1e-3))]


def main():
    I3 = info(D)
    rr = QMUS * I3
    curves, over, tset, floor = {}, np.zeros(QMUS.size), np.zeros(QMUS.size), np.zeros(QMUS.size)
    for k, q in enumerate(QMUS):
        e, o, tsv, fl = walkout(q)
        curves[q] = e; over[k] = o; tset[k] = tsv; floor[k] = fl
        print(f"  q={q:.1e} r={rr[k]:.4f} overshoot={o:.3f} settle={tsv:4.0f} "
              f"floor*sqrtI={np.sqrt(I3)*fl:.3f}")

    rate = 1.0 / tset
    dfloor = np.sqrt(I3) * floor
    r_crit = cross_zero(rr, over)                 # zeta = 1
    q_crit = r_crit / I3
    k_peak = int(np.argmin(tset))
    r_knee45 = 0.0833                              # idealised K=1/4 knee
    tau_crit = float(np.interp(r_crit, rr, tset))
    print(f"\n[I(d=3)] {I3:.3f}   R=1/I={1/I3:.2f}")
    print(f"[critical damping] r*={r_crit:.4f}  q_mu*={q_crit:.4f}  (= {r_crit:.3f}/I)  "
          f"settle~{tau_crit:.0f} steps, no overshoot")
    print(f"[rate peak] r={rr[k_peak]:.4f} settle={tset[k_peak]:.0f} (only "
          f"{100*(1-tset[k_peak]/tau_crit):.0f}% faster, with overshoot+floor)")
    print(f"[idealised 45-deg knee] r={r_knee45:.3f} -- past the rollover")

    # representative traces: over / critical / under damped
    q_over = QMUS[np.argmin(np.abs(rr - r_crit / 2.5))]
    q_crit_t = QMUS[np.argmin(np.abs(rr - r_crit))]
    q_under = QMUS[np.argmin(np.abs(rr - 30 * r_crit))]

    fig, ax = plt.subplots(1, 3, figsize=(16.2, 4.6))
    tt = np.arange(NT) - JT

    a = ts.tidy(ax[0])
    a.axhline(D, color=ts.INK, lw=1.0, ls=":", label="truth +3")
    a.axvline(0, color=ts.GRID, lw=1.0)
    ec = curves[q_crit_t]
    a.fill_between(tt, ec - HALF, ec + HALF, color=ts.SEQ[3], alpha=0.25, lw=0,
                   label="window ±1.35 (overlaps step-to-step)")
    for q, c, lb in [(q_over, ts.SERIES[3], "over-damped"),
                     (q_crit_t, ts.SERIES[7], "critically damped"),
                     (q_under, ts.SERIES[1], "under-damped (rings)")]:
        a.plot(tt, curves[q], color=c, lw=1.8, label=lb)
    a.set_xlim(-15, 120); a.set_ylim(-0.6, 3.9)
    a.set_xlabel("steps since jump"); a.set_ylabel("log-scale estimate (nats)")
    a.set_title("(a) dense overlapping walk-out, three dampings")
    a.legend(loc="lower right", fontsize=7.6)

    a = ts.tidy(ax[1])
    a.axhline(0.0, color=ts.INK, lw=0.8)
    a.axvspan(r_crit, rr.max() * 1.1, color=ts.SERIES[1], alpha=0.10, lw=0)
    a.plot(rr, over, color=ts.SERIES[2], lw=1.9, marker="o", ms=4)
    a.axvline(r_crit, color=ts.SERIES[7], lw=1.3, ls="--",
              label=f"critical  r*={r_crit:.1e}")
    a.set_xscale("log")
    a.set_xlabel("tracking index  r = q_mu · I")
    a.set_ylabel("arrival overshoot  (nats past truth)")
    a.set_title("(b) ζ=1: overshoot onset → critical damping")
    a.annotate("under-damped\n(rings)", (rr.max() * 0.5, over.max() * 0.75),
               fontsize=7.6, ha="right", color=ts.SERIES[1])
    a.legend(loc="upper left", fontsize=8)

    a = ts.tidy(ax[2])
    a.plot(dfloor, rate, color=ts.INK2, lw=1.7, marker="o", ms=4, zorder=2)
    kc = int(np.argmin(np.abs(rr - r_crit)))
    a.scatter([dfloor[kc]], [rate[kc]], facecolors="none", edgecolors=ts.SERIES[7],
              s=150, lw=2.2, zorder=5, label=f"critical damping (r={r_crit:.1e})")
    a.scatter([dfloor[k_peak]], [rate[k_peak]], color=ts.SERIES[1], s=45, zorder=5,
              label="rate peak (overshoots)")
    a.annotate("more q_mu → higher floor,\nreversion bias, no speed",
               (dfloor[-1], rate[-1]), fontsize=7.4, ha="right", va="bottom",
               color=ts.INK2, xytext=(0, 6), textcoords="offset points")
    a.set_xlabel("dimensionless steady floor  √I · RMS")
    a.set_ylabel("convergence rate  1/settle  (1/steps)")
    a.set_title("(c) rate saturates; critical damping sits at the knee")
    a.legend(loc="lower right", fontsize=7.8)
    ts.save(fig, os.path.join(HERE, "figures", "0022-critically-damped-walkout.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
