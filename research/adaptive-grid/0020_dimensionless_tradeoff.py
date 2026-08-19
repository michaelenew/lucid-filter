"""The config-invariant tradeoff: settling vs floor collapses onto one curve in r.

The window centre is a scalar Kalman filter tracking the true log-scale, driven
by a step whose per-step Fisher information I is read off the grid; its
measurement noise is R = 1/I and its process (drift) noise is q_mu.  For a
random-walk Kalman filter EVERYTHING depends only on the dimensionless ratio

    r = q_mu * I = q_mu / R      (the tracking index),

through the steady prior ratio rho = P/R solving  rho^2 - r*rho - r = 0, giving

    rho = (r + sqrt(r^2 + 4r)) / 2,   gain  K = rho / (rho + 1).

Two consequences, both functions of r ALONE -- so invariant of the filter
configuration (order, spread, regime loudness) once expressed in r:

    dimensionless steady floor    sqrt(I) * RMS_error  =  sqrt(K),
    settling time constant        tau  =  -1 / ln(1 - K)   [steps].

Eliminating r traces a single tradeoff frontier: you cannot settle fast and sit
quiet at once; q_mu picks the point on the curve.  This probe measures I, the
steady floor, and tau across a spread of configurations (order, spread, regime)
and q_mu values, and shows every point landing on the r-parameterised theory --
the graph the choice of q_mu should be read from, independent of config.

Measured
--------
(a) dimensionless floor sqrt(I)*RMS and settling tau vs r, all configs overlaid
    on the closed-form curves -- the collapse;
(b) the invariant frontier: settling tau vs dimensionless floor, one curve, all
    configs on it -- the achievable set, q_mu sliding along it;
(c) the same points vs RAW q_mu do NOT collapse (configs separate) -- r, not
    q_mu, is the invariant knob.

Run: python 0020_dimensionless_tradeoff.py   (heavy; ~3-4 min)
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

# configs span a range of per-step information I: spread, order, regime loudness
CONFIGS = [
    dict(s=0.20, order=5, d=0.0),
    dict(s=0.30, order=5, d=0.0),
    dict(s=0.30, order=7, d=0.0),
    dict(s=0.30, order=5, d=2.0),
    dict(s=0.40, order=7, d=0.0),
]
QMUS = np.logspace(-4.0, -1.5, 6)


def theory_K(r):
    rho = 0.5 * (r + np.sqrt(r * r + 4.0 * r))
    return rho / (rho + 1.0)


def info(cfg, nseed=40, nt=700):
    """Per-step Fisher info I at a centred static regime -- the observability."""
    acc = []
    for sd in range(nseed):
        rng = np.random.default_rng(sd)
        x = simulate(rng, cfg["d"], 1.0, 1.0, nt)
        f = MovingChannel(1.0, 1.0, phi=0.9, s=cfg["s"], order=cfg["order"],
                          step="kalman_auto", q_mu=1e-6, P0=1.0)
        f.reset(mu=cfg["d"])
        fish = [f.update(v)["fisher"] for v in x]
        acc.append(np.mean(fish[nt // 4:]))
    return float(np.mean(acc))


def floor_rms(cfg, q_mu, nseed=60, nt=900):
    """Steady RMS of the log-scale estimate about a static truth."""
    E = np.zeros((nseed, nt))
    for sd in range(nseed):
        rng = np.random.default_rng(400 + sd)
        x = simulate(rng, cfg["d"], 1.0, 1.0, nt)
        f = MovingChannel(1.0, 1.0, phi=0.9, s=cfg["s"], order=cfg["order"],
                          step="kalman_auto", q_mu=q_mu, P0=25.0)
        f.reset(mu=cfg["d"])
        E[sd] = [f.update(v)["logscale"] for v in x]
    return float(np.sqrt(((E[:, -400:] - cfg["d"]) ** 2).mean()))


def settle_tau(cfg, q_mu, nseed=80, nt=500, offset=1.2):
    """1/e settling constant: fit ln|mean error| decay from a reset offset."""
    E = np.zeros((nseed, nt))
    for sd in range(nseed):
        rng = np.random.default_rng(900 + sd)
        x = simulate(rng, cfg["d"], 1.0, 1.0, nt)
        f = MovingChannel(1.0, 1.0, phi=0.9, s=cfg["s"], order=cfg["order"],
                          step="kalman_auto", q_mu=q_mu, P0=25.0)
        f.reset(mu=cfg["d"] + offset)                    # start displaced
        E[sd] = [f.update(v)["logscale"] for v in x]
    err = np.abs(E.mean(0) - cfg["d"])                   # ensemble-mean |error|
    floor = err[-120:].mean()
    y = err - floor
    good = np.where(y > 0.06)[0]                          # decaying, above floor
    good = good[good < 200]
    if good.size < 5:
        return np.nan
    t = good.astype(float)
    sl = np.polyfit(t, np.log(y[good]), 1)[0]
    return float(-1.0 / sl) if sl < 0 else np.nan


def main():
    rows = []
    Iof = {}
    for ci, cfg in enumerate(CONFIGS):
        I = info(cfg)
        Iof[ci] = I
        for q in QMUS:
            rms = floor_rms(cfg, q)
            tau = settle_tau(cfg, q)
            rows.append(dict(ci=ci, I=I, q=q, r=q * I,
                             dfloor=np.sqrt(I) * rms, tau=tau))
            print(f"  cfg{ci} I={I:.3f} q={q:.1e} r={q*I:.2e} "
                  f"floor*sqrtI={np.sqrt(I)*rms:.3f} tau={tau:.1f}")

    R = np.array([x["r"] for x in rows])
    DF = np.array([x["dfloor"] for x in rows])
    TAU = np.array([x["tau"] for x in rows])
    Q = np.array([x["q"] for x in rows])
    CI = np.array([x["ci"] for x in rows])

    rr = np.logspace(np.log10(R.min() / 2), np.log10(R.max() * 2), 200)
    Kth = theory_K(rr)
    dfloor_th = np.sqrt(Kth)
    tau_th = -1.0 / np.log(1.0 - Kth)

    ccols = [ts.SEQ[i] for i in (2, 3, 4, 5, 6)]
    labels = [f"s={c['s']}, ord {c['order']}, d={c['d']:g} (I={Iof[i]:.2f})"
              for i, c in enumerate(CONFIGS)]

    fig, ax = plt.subplots(1, 3, figsize=(16.2, 4.6))

    # (a) both branches vs r, on theory
    a = ts.tidy(ax[0])
    a.plot(rr, dfloor_th, color=ts.INK2, lw=1.6, label="theory  √K")
    a.plot(rr, tau_th / 40.0, color=ts.SERIES[7], lw=1.6, ls="--",
           label="theory  τ/40")
    for i in range(len(CONFIGS)):
        m = CI == i
        a.scatter(R[m], DF[m], color=ccols[i], s=34, zorder=3)
        ok = m & np.isfinite(TAU)
        a.scatter(R[ok], TAU[ok] / 40.0, color=ccols[i], s=34, marker="^", zorder=3)
    a.set_xscale("log")
    a.set_xlabel("tracking index  r = q_mu · I")
    a.set_ylabel("√I·RMS floor  (○)   and   τ/40 steps  (△)")
    a.set_title("(a) both collapse onto r: floor √K (○), settling τ (△)")
    a.legend(loc="center left", fontsize=7.6)

    # (b) the invariant frontier
    a = ts.tidy(ax[1])
    a.plot(dfloor_th, tau_th, color=ts.INK2, lw=1.8, label="theory frontier", zorder=1)
    for i in range(len(CONFIGS)):
        ok = (CI == i) & np.isfinite(TAU)
        a.scatter(DF[ok], TAU[ok], color=ccols[i], s=42, zorder=3, label=labels[i])
    a.set_xlabel("dimensionless steady floor  √I · RMS")
    a.set_ylabel("settling time constant  τ  (steps)")
    a.set_title("(b) one frontier: fast OR quiet; q_mu slides along it")
    a.legend(loc="upper right", fontsize=7.0)
    a.annotate("small r:\nslow & quiet", (0.30, tau_th[np.argmin(np.abs(dfloor_th-0.30))]),
               fontsize=7.5, ha="center", color=ts.INK2,
               xytext=(0, 16), textcoords="offset points")

    # (c) vs raw q_mu -- configs separate, NOT invariant
    a = ts.tidy(ax[2])
    for i in range(len(CONFIGS)):
        m = CI == i
        o = np.argsort(Q[m])
        a.plot(Q[m][o], DF[m][o], color=ccols[i], lw=1.4, marker="o", ms=4,
               label=labels[i])
    a.set_xscale("log")
    a.set_xlabel("raw  q_mu")
    a.set_ylabel("dimensionless steady floor  √I · RMS")
    a.set_title("(c) vs raw q_mu the configs SEPARATE — r is the invariant knob")
    a.legend(loc="upper left", fontsize=7.0)
    ts.save(fig, os.path.join(HERE, "figures", "0019-dimensionless-tradeoff.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
