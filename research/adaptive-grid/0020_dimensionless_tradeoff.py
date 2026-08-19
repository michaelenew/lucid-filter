"""The config-invariant tradeoff: settling vs floor collapses onto one curve in r.

The window centre is a scalar Kalman filter tracking the true log-scale, driven
by a step whose per-step Fisher information I is read off the grid; its
measurement noise is R = 1/I and its process (drift) noise is q_mu.  For a
random-walk Kalman filter EVERYTHING depends only on the dimensionless ratio

    r = q_mu * I = q_mu / R      (the tracking index),

through the steady prior ratio rho = P/R solving  rho^2 - r*rho - r = 0, giving

    rho = (r + sqrt(r^2 + 4r)) / 2,   gain  K = rho / (rho + 1).

Two consequences, both functions of r ALONE -- so invariant of the filter
configuration and the regime once expressed in r:

    dimensionless steady floor    sqrt(I) * RMS_error  =  sqrt(K),
    settling time constant        tau  =  -1 / ln(1 - K)   [steps].

Eliminating r traces a single tradeoff frontier: you cannot settle fast and sit
quiet at once; q_mu picks the point on the curve.  This probe spans I by the
regime loudness d (observability sets I; the grid spread/order barely move it),
sweeps q_mu, and measures -- with the filter WARMED to steady state, so the
settling constant is the closed-loop pole, not the cold-start transient -- the
steady floor and the relaxation time.  Every point lands on the r-curve.

Measured
--------
(a) dimensionless floor sqrt(I)*RMS vs r, all regimes on the closed form sqrt(K);
(b) settling tau vs r, all regimes on -1/ln(1-K) -- warmed, so it is clean;
(c) the invariant frontier: tau vs dimensionless floor, one curve, every regime
    on it -- the achievable set, q_mu sliding along it.

Run: python 0020_dimensionless_tradeoff.py   (heavy; ~4-5 min)
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

# regime loudness d spans the observability I; the linear tracking law holds in
# a band of q_mu -- too small freezes mu off-centre in a loud regime (floor
# becomes a bias, not sqrt(K)); too large chatters (fast, nonlinear relaxation).
# Stay in the band so the collapse is the law, not its breakdown.
DVALS = [-1.0, 0.0, 1.0, 2.0, 3.0]
QMUS = np.logspace(-2.55, -1.7, 6)        # ~[2.8e-3, 2.0e-2]
WARM, SPAN, OFFSET = 400, 320, 1.5


def theory_K(r):
    rho = 0.5 * (r + np.sqrt(r * r + 4.0 * r))
    return rho / (rho + 1.0)


def info(d, nseed=40, nt=700):
    """Per-step Fisher info I at a centred static regime -- the observability."""
    acc = []
    for sd in range(nseed):
        rng = np.random.default_rng(sd)
        x = simulate(rng, d, 1.0, 1.0, nt)
        f = MovingChannel(1.0, 1.0, phi=0.9, s=0.30, order=5,
                          step="kalman_auto", q_mu=1e-6, P0=1.0)
        f.reset(mu=d)
        fish = [f.update(v)["fisher"] for v in x]
        acc.append(np.mean(fish[nt // 4:]))
    return float(np.mean(acc))


def measure(d, q_mu, nseed=150):
    """Warm to steady state; read the floor and steady gain; perturb; time relax.

    Warming means P has reached its steady value before the step, so the fitted
    relaxation is the closed-loop pole, not the P0 acquisition transient.  The
    steady gain K = P_mu/(P_mu+R) (returned each step) gives the theory pole
    -1/ln(1-K) to compare against.  Returns (floor RMS, steady K, relaxation tau).
    """
    tail = np.zeros((nseed, 200))
    gain = np.zeros((nseed, 200))
    relax = np.zeros((nseed, SPAN))
    for sd in range(nseed):
        rng = np.random.default_rng(900 + sd)
        x = simulate(rng, d, 1.0, 1.0, WARM + SPAN)
        f = MovingChannel(1.0, 1.0, phi=0.9, s=0.30, order=5,
                          step="kalman_auto", q_mu=q_mu, P0=25.0)
        f.reset(mu=d)
        for t in range(WARM - 200):
            f.update(x[t])
        for j in range(200):
            o = f.update(x[WARM - 200 + j])
            tail[sd, j] = o["logscale"] - d
            gain[sd, j] = o["gain"]
        f.mu += OFFSET                                   # perturb at steady state
        relax[sd] = [f.update(x[WARM + j])["logscale"] - d for j in range(SPAN)]

    # typical (locked) floor: median over seeds of per-seed RMS.  A rare
    # loss-of-lock (a kick past the fine grid's reach at mid observability) puts
    # a heavy tail on the pooled RMS; that reliability tail is a separate matter
    # from the locked-tracking precision this invariant describes.
    floor = float(np.median(np.sqrt((tail ** 2).mean(1))))
    K = float(np.median(gain))
    err = np.median(relax, 0)                            # robust per-step centre, ~OFFSET -> floor
    fl = err[-80:].mean()
    y = err - fl
    # fit the clean, monotone-decreasing, above-floor stretch (before overshoot)
    good = np.where((y > 0.12) & (np.arange(SPAN) < SPAN - 80))[0]
    if good.size:
        good = good[: int(np.argmax(np.diff(good, prepend=good[0] - 1) > 1) or good.size)]
    tau = np.nan
    if good.size >= 5:
        sl = np.polyfit(good.astype(float), np.log(y[good]), 1)[0]
        tau = float(-1.0 / sl) if sl < 0 else np.nan
    return floor, K, tau


def main():
    Iof = {d: info(d) for d in DVALS}
    rows = []
    for d in DVALS:
        I = Iof[d]
        for q in QMUS:
            floor, K, tau = measure(d, q)
            rows.append(dict(d=d, I=I, q=q, r=q * I, K=K,
                             dfloor=np.sqrt(I) * floor, tau=tau))
            print(f"  d={d:+.0f} I={I:.3f} q={q:.2e} r={q*I:.2e} "
                  f"floor*sqrtI={np.sqrt(I)*floor:.3f} (√Kth={np.sqrt(theory_K(q*I)):.3f}) "
                  f"tau={tau:6.1f} (pole {-1.0/np.log(1.0-K):5.1f})")

    R = np.array([x["r"] for x in rows])
    KM = np.array([x["K"] for x in rows])
    DF = np.array([x["dfloor"] for x in rows])
    TAU = np.array([x["tau"] for x in rows])
    DD = np.array([x["d"] for x in rows])

    rr = np.logspace(np.log10(R.min() / 2), np.log10(R.max() * 2), 240)
    Kth = theory_K(rr)
    dfloor_th = np.sqrt(Kth)
    tau_th = -1.0 / np.log(1.0 - Kth)

    dcol = {d: ts.SEQ[i] for d, i in zip(DVALS, (2, 3, 4, 5, 6))}
    lab = {d: f"d={d:+.0f}  (I={Iof[d]:.2f})" for d in DVALS}

    fig, ax = plt.subplots(1, 3, figsize=(16.2, 4.6))

    # (a) floor law: sqrt(I)*RMS vs r on sqrt(K) -- the empirical collapse
    a = ts.tidy(ax[0])
    a.plot(rr, dfloor_th, color=ts.INK2, lw=1.8, label="theory  √K(r)")
    for d in DVALS:
        m = DD == d
        a.scatter(R[m], DF[m], color=dcol[d], s=42, zorder=3, label=lab[d])
    a.set_xscale("log")
    a.set_xlabel("tracking index  r = q_mu · I")
    a.set_ylabel("dimensionless steady floor  √I · RMS")
    a.set_title("(a) floor collapses in r, ~0.8·√K (sub-grid term beats the bound)")
    a.legend(loc="upper left", fontsize=7.4)

    # (b) settling law: measured relaxation tau vs r on -1/ln(1-K)
    a = ts.tidy(ax[1])
    a.plot(rr, tau_th, color=ts.INK2, lw=1.8, label="theory  −1/ln(1−K(r))")
    for d in DVALS:
        m = (DD == d) & np.isfinite(TAU)
        a.scatter(R[m], TAU[m], color=dcol[d], s=42, zorder=3, label=lab[d])
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("tracking index  r = q_mu · I")
    a.set_ylabel("measured settling τ  (steps)")
    a.set_title("(b) settling (perturb-and-relax) collapses onto the pole")
    a.legend(loc="upper right", fontsize=7.4)

    # (c) the invariant frontier: settling tau vs steady floor
    a = ts.tidy(ax[2])
    a.plot(dfloor_th, tau_th, color=ts.INK2, lw=1.9, label="theory frontier", zorder=1)
    for d in DVALS:
        m = (DD == d) & np.isfinite(TAU)
        a.scatter(DF[m], TAU[m], color=dcol[d], s=46, zorder=3, label=lab[d])
    a.set_yscale("log")
    a.set_xlabel("dimensionless steady floor  √I · RMS")
    a.set_ylabel("settling time constant  τ  (steps)")
    a.set_title("(c) one frontier: fast OR quiet; q_mu slides along it")
    a.legend(loc="upper right", fontsize=7.4)
    a.annotate("small r:\nslow & quiet", (0.12, tau_th[np.argmin(np.abs(dfloor_th - 0.12))]),
               fontsize=7.6, ha="left", color=ts.INK2,
               xytext=(8, 0), textcoords="offset points")
    a.annotate("large r:\nfast & noisy", (0.42, tau_th[np.argmin(np.abs(dfloor_th - 0.42))]),
               fontsize=7.6, ha="right", color=ts.INK2,
               xytext=(-4, 18), textcoords="offset points")
    ts.save(fig, os.path.join(HERE, "figures", "0019-dimensionless-tradeoff.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
