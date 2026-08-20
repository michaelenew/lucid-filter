"""Online convergence of the move from a random starting grid.

The grid starts at a random centre mu0 (a random first guess at the log-scale),
fine enough to have no dead zone.  Does the move walk it to the truth quickly,
from anywhere?  Profiling drove the design of the move (see 0008 for the full
arc); this probe reports the settled behaviour.

The move is a servo whose step is the recentring signal `pi @ lam + w * score`
(posterior mean, which drives from below and inside coverage; plus the raw
score, which drives from above where the over-variance shelf is flat and the
posterior mean stalls), integrated with a Robbins-Monro step size
eta_t = max(eta_floor, eta/(1 + t/tau)).  The decay is what lets it converge to
a point -- the in-frame AR(1) reverts to the frame centre, so the estimate is
unbiased only when mu sits on the truth, and a constantly-moving servo would
wander around it.  eta_floor > 0 keeps a residual bandwidth for tracking a drift.

What is measured
----------------
A. Error trajectories from a fan of random mu0, fine grid.
B. Convergence time vs SIGNED initial offset, three configs: servo+fine (fast,
   symmetric), raw-score+fine (slow from below: the Qg/S suppression), and
   servo+coarse (stalls in dead zones -- the fineness constraint is load-bearing).
C. The tracking/precision budget, on a static truth: error vs time for a
   decaying step (eta_floor=0) and two constant floors.  Decay settles lowest
   (stochastic approximation converges to a point; the integrating mu even
   time-averages a static truth); a residual floor plateaus higher -- the
   precision it gives up to keep bandwidth for a drift.

Predictions
-----------
- Servo+fine converges from every start; O(10-30) steps within +-2 nats, a few
  hundred from +-5, symmetric in sign.
- Raw-score+fine is slow / stalls from below (mu0 far under the truth).
- Servo+coarse stalls where a dead zone lies between start and truth.
- Decay settles to the lowest floor; larger eta_floor plateaus higher.

Run: python 0007_online_convergence.py
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


def run_from(lam_t, mu0, Q, s2, phi, s, order, seed, **kw):
    """Run the move from centre mu0 over a (possibly time-varying) truth lam_t."""
    rng = np.random.default_rng(seed)
    nt = lam_t.size
    if np.ptp(lam_t) == 0:
        x = simulate(rng, float(lam_t[0]), Q, s2, nt)
    else:
        steps = rng.normal(0.0, np.sqrt(Q * np.exp(lam_t)))
        x = np.cumsum(steps) + rng.normal(0.0, np.sqrt(s2), size=nt)
    f = MovingChannel(Q, s2, phi=phi, s=s, order=order, **kw)
    f.reset(mu=mu0)
    return np.array([f.update(v)["logscale"] for v in x])


def conv_step(err, eps=0.5, hold=15):
    ok = err < eps
    for t in range(err.size - hold):
        if ok[t:t + hold].all():
            return t
    return err.size


# ------------------------------------------------------- A. trajectories
def trajectories(truth=0.0, Q=1.0, s2=1.0, phi=0.9, s=0.30, order=5,
                 nt=400, nseed=24):
    lam_t = np.full(nt, truth)
    mu0s = np.linspace(-6.0, 6.0, 13)
    curves = np.zeros((mu0s.size, nt))
    for i, mu0 in enumerate(mu0s):
        acc = np.zeros(nt)
        for sd in range(nseed):
            acc += np.abs(run_from(lam_t, truth + mu0, Q, s2, phi, s, order,
                                   11000 + 50 * i + sd, eta_floor=0.0) - truth)
        curves[i] = acc / nseed
    print(f"[A] servo, fine grid s={s} (coverage +-{2.857*s:.2f}), truth={truth}")
    print(f"  worst-start error (mean last 60): {curves[:, -60:].mean(1).max():.3f} nats")
    return mu0s, curves


# ------------------------------------------- B. convergence time vs distance
def time_vs_distance(truth=0.0, Q=1.0, s2=1.0, phi=0.9, order=5,
                     nt=500, nseed=40):
    lam_t = np.full(nt, truth)
    dists = np.linspace(-6.0, 6.0, 25)
    configs = (("servo, fine", dict(step="servo", s=0.30, eta_floor=0.0)),
               ("raw score, fine", dict(step="score", s=0.30, eta_floor=0.0)),
               ("servo, coarse", dict(step="servo", s=1.60, eta_floor=0.0)))
    out = {}
    for name, kw in configs:
        s = kw.pop("s")
        tsteps = np.zeros(dists.size)
        stalls = 0
        for i, d in enumerate(dists):
            acc = []
            for sd in range(nseed):
                est = run_from(lam_t, truth + d, Q, s2, phi, s, order,
                               12000 + 50 * i + sd, **kw)
                c = conv_step(np.abs(est - truth))
                acc.append(c)
                stalls += int(c >= nt)
            tsteps[i] = np.median(acc)
        out[name] = (dists, tsteps, stalls)
        print(f"[B] {name:16s}: max median steps={tsteps.max():.0f}, "
              f"stalls={stalls}/{dists.size*nseed}")
    return out


# ---------------------------------------------------- C. Robbins-Monro floor
def rm_floor(Q=1.0, s2=1.0, phi=0.9, s=0.30, order=5, nt=1500, nseed=40):
    """Static-truth error vs time: decay reaches the oracle floor; a constant
    step plateaus at a wander floor set by its bandwidth (the precision cost of
    keeping enough gain to track a drift)."""
    flat = np.zeros(nt)
    curves = {}
    for ef in (0.0, 0.05, 0.15):
        acc = np.zeros(nt)
        for sd in range(nseed):
            acc += np.abs(run_from(flat, 3.0, Q, s2, phi, s, order,
                                   14000 + sd, eta_floor=ef))   # start 3 nats off
        curves[ef] = acc / nseed
    print("[C] static-truth error over last 300 steps (start 3 nats off):")
    for ef, c in curves.items():
        print(f"    eta_floor={ef:.2f}: {c[-300:].mean():.3f}")
    return curves


def plot_all(traj, tvd, budget):
    mu0s, curves = traj
    fig, ax = plt.subplots(1, 3, figsize=(16.8, 4.2))

    a = ts.tidy(ax[0])
    for i, mu0 in enumerate(mu0s):
        c = ts.SEQ[min(int(abs(mu0) / 6 * (len(ts.SEQ) - 1)), len(ts.SEQ) - 1)]
        a.plot(np.maximum(curves[i], 1e-3), color=c, lw=1.2)
    a.set_yscale("log")
    a.set_xlabel("step"); a.set_ylabel("|log-scale estimate - truth|  (nats)")
    a.set_title("(a) convergence from random mu0 (fine grid)")

    a = ts.tidy(ax[1])
    cols = {"servo, fine": ts.SERIES[5], "raw score, fine": ts.SERIES[0],
            "servo, coarse": ts.SERIES[1]}
    for name, (dists, tsteps, stalls) in tvd.items():
        lab = name + (f" ({stalls} stalls)" if stalls else "")
        a.plot(dists, np.minimum(tsteps, 500), color=cols[name], marker="o",
               ms=3.2, label=lab)
    a.set_xlabel("signed initial offset  mu0 - truth  (nats)")
    a.set_ylabel("median steps to converge (<0.5 nat)")
    a.set_title("(b) symmetric; raw score stalls from below; coarse stalls")
    a.legend(loc="upper center", fontsize=7.8)

    a = ts.tidy(ax[2])
    curves = budget
    cmap = {0.0: ts.SERIES[5], 0.05: ts.SERIES[0], 0.15: ts.SERIES[1]}
    for ef, c in curves.items():
        lab = ("decay (eta_floor=0)" if ef == 0 else f"constant floor {ef}")
        a.plot(np.maximum(c, 1e-3), color=cmap[ef], lw=1.6, label=lab)
    a.set_yscale("log")
    a.set_xlabel("step"); a.set_ylabel("|estimate - truth|  (nats)")
    a.set_title("(c) decay settles lowest; floor = tracking budget")
    a.legend(loc="upper right", fontsize=7.8)
    ts.save(fig, os.path.join(HERE, "figures", "0008-online-convergence.png"))


if __name__ == "__main__":
    t0 = time.time()
    traj = trajectories()
    tvd = time_vs_distance()
    budget = rm_floor()
    plot_all(traj, tvd, budget)
    print(f"\ndone in {time.time() - t0:.1f}s")
