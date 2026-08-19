"""How dramatic is q_mu?  The one irreducible knob, profiled.

q_mu is the class commitment -- how fast the log-scale may drift (Proposition 1
says a bound on this rate is necessary, not optional).  In the self-calibrating
Kalman tracker (0013) it is the ONLY remaining input.  This profiles what it
buys and costs, on the self-calibrating tracker (kalman_auto), truth = 0.

Three views:

  (a) STATIC precision: converge from a start and hold.  Small q_mu keeps
      averaging (low floor); large q_mu keeps the gain up (noisy, high floor);
      q_mu = 0 lets P collapse and can freeze off-target.
  (b) REACTIVITY: the truth jumps 0 -> +3 mid-run.  Small q_mu is sluggish to
      re-acquire; large q_mu snaps to it.
  (c) The trade: steady floor vs jump-recovery time, parameterised by q_mu -- a
      Pareto curve.  q_mu picks a point on it; that is the whole content of the
      knob.

Run: python 0014_q_mu_sweep.py
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

QMUS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
COLORS = [ts.INK2, ts.SEQ[1], ts.SEQ[3], ts.SEQ[4], ts.SERIES[1]]


def run_batch(lam_t, mu0, q_mu, nseed):
    """(nseed, nt) log-scale estimates for a given truth path and q_mu."""
    nt = lam_t.size
    E = np.zeros((nseed, nt))
    for sd in range(nseed):
        rng = np.random.default_rng(700 + sd)
        if np.ptp(lam_t) == 0:
            x = simulate(rng, float(lam_t[0]), 1.0, 1.0, nt)
        else:
            st = rng.normal(0.0, np.sqrt(np.exp(lam_t)))
            x = np.cumsum(st) + rng.normal(0.0, 1.0, nt)
        f = MovingChannel(1.0, 1.0, phi=0.9, s=0.30, order=5,
                          step="kalman_auto", q_mu=q_mu, P0=25.0)
        f.reset(mu=mu0)
        E[sd] = [f.update(v)["logscale"] for v in x]
    return E


def rms(E, truth):
    return np.sqrt(((E - truth) ** 2).mean(0))


def main(nseed=80):
    fig, ax = plt.subplots(1, 3, figsize=(16.0, 4.4))

    # (a) static precision: converge from +3 and hold
    nt = 900
    flat = np.zeros(nt)
    a = ts.tidy(ax[0])
    floors = {}
    for q, c in zip(QMUS, COLORS):
        r = rms(run_batch(flat, 3.0, q, nseed), 0.0)
        floors[q] = r[-200:].mean()
        a.plot(np.maximum(r, 1e-3), color=c, lw=1.7,
               label=f"q_mu={q:g}" + (" (P collapse)" if q == 0 else ""))
    a.set_yscale("log")
    a.set_xlabel("step"); a.set_ylabel("RMS error  (nats)")
    a.set_title("(a) static: small q_mu keeps averaging; q_mu=0 can freeze")
    a.legend(loc="upper right", fontsize=8)

    # (b) reactivity: truth jumps 0 -> +3 at t=250
    nt = 600
    jump = np.zeros(nt); jump[250:] = 3.0
    a = ts.tidy(ax[1])
    recov = {}
    for q, c in zip(QMUS, COLORS):
        r = rms(run_batch(jump, 0.0, q, nseed), jump)
        after = r[250:]
        hit = np.where(after < 0.5)[0]
        recov[q] = int(hit[0]) if hit.size else nt
        a.plot(np.maximum(r, 1e-3), color=c, lw=1.7, label=f"q_mu={q:g}")
    a.axvline(250, color=ts.INK2, lw=0.9, ls="--")
    a.set_yscale("log")
    a.set_xlabel("step"); a.set_ylabel("RMS error  (nats)")
    a.set_title("(b) reactivity: truth jumps +3 at t=250")
    a.legend(loc="lower right", fontsize=8)

    # (c) the trade: floor vs jump-recovery, parameterised by q_mu
    a = ts.tidy(ax[2])
    fl = np.array([floors[q] for q in QMUS])
    rc = np.array([recov[q] for q in QMUS])
    a.plot(rc, fl, color=ts.INK2, lw=1.2, zorder=1)
    for q, c in zip(QMUS, COLORS):
        a.scatter([recov[q]], [floors[q]], color=c, s=55, zorder=3)
        a.annotate(f"{q:g}", (recov[q], floors[q]), fontsize=8,
                   xytext=(6, 4), textcoords="offset points", color=c)
    a.set_yscale("log")
    a.set_xlabel("jump-recovery time  (steps to <0.5 nat)")
    a.set_ylabel("static floor  (RMS, nats)")
    a.set_title("(c) the trade q_mu buys: precision vs reactivity")

    print("[q_mu sweep]  q_mu :  static floor  |  jump-recovery steps")
    for q in QMUS:
        print(f"           {q:>7g} :   {floors[q]:.3f}       |   {recov[q]}")
    ts.save(fig, os.path.join(HERE, "figures", "0013-q-mu-sweep.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
