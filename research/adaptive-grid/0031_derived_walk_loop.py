"""The walk loop, closed to zero free parameters -- and where the stiff wall bites.

Finding 17 fixed the stiff-wall over-reaction with a slow EMA of the grid Fisher
`I`, but that carried three un-derived constants (r*, the seed 0.4, the EMA rate
30).  This probe derives the whole walk loop from the class pair `(phi, s)` and the
base scales `(Q, s2)` -- no tuned number survives -- and then measures, honestly,
what the derived loop does and does not fix.

The derivation
--------------
The window centre `mu` integrates the grid-shift score, but the grid state relaxes
only at ~phi per step, so the walk is a SECOND-ORDER loop.  With error e = lam - mu
and the grid's lagged offset y,

    e_t = e_{t-1} - K y_{t-1}              (integrator on the lagged offset)
    y_t = phi y_{t-1} + (1-phi) e_{t-1}    (grid relaxes at phi)

the characteristic equation z^2 - (1+phi) z + phi + K(1-phi) = 0 has a double root
-- CRITICAL DAMPING, fastest response with no overshoot -- exactly at

    K* = (1 - phi) / 4     (a pure function of phi; everything else cancels).

The drift variance that settles the mu-Kalman to that steady gain is
q_mu = K*^2 / (I_char (1-K*)), fixed once at reset from K* and I_char -- the grid's
steady Fisher information, evaluated at the scale-free regime (SNR=1, effective
process variance = s2) so it stays Q-invariant (finding 9).  Cold-start prior is the
AR(1) stationary variance Pmu0 = s^2.

Panels
------
(a) EXACT linear loop: overshoot vs K/K* collapses across phi and its onset sits at
    K/K* = 1 -- K* = (1-phi)/4 verified (the derivation, not the noisy filter);
(b) the 84x stiff wall: grid observability I(d) swings ~0.005 (quiet) -> 0.41 (loud),
    with the derived I_char marking the scale-free reference it is pinned at;
(c) the honest consequence -- a FIXED q_mu is critical only at the reference: a
    mid-stream step OVER-shoots going up into a loud (high-I) regime and is mild going
    down into a quiet one; a CONSTANT gain would be uniform but loses quiet capture;
(d) what the derivation cleanly fixes: cold-start overshoot, Pmu0 = s^2 vs old 25.

Run: python 0031_derived_walk_loop.py   (~3-4 min)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "random-walk-filter", "scripts"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lucid"))

from gridlab import simulate                         # noqa: E402
from moving_grid import MovingChannel                # noqa: E402
from statfilter import WalkingFilter                 # noqa: E402
import theory_style as ts                            # noqa: E402
import matplotlib.pyplot as plt                      # noqa: E402


# --- (a) the derivation, on the EXACT linear loop -------------------------------
def linear_overshoot(K, phi, n=600):
    e, y, mn = 1.0, 0.0, 1.0
    for _ in range(n):
        e, y = e - K * y, phi * y + (1.0 - phi) * e
        mn = min(mn, e)
    return max(0.0, -mn)              # how far the step response dips below 0


# --- (b) the 84x observability swing --------------------------------------------
def emp_steady_info(d, phi=0.9, s=0.30, nseed=40, nt=700):
    acc = []
    for sd in range(nseed):
        x = simulate(np.random.default_rng(sd), d, 1.0, 1.0, nt)
        f = MovingChannel(1.0, 1.0, phi=phi, s=s, step="kalman_auto", q_mu=1e-9, P0=1.0)
        f.reset(mu=d)
        acc.append(np.mean([f.update(v)["fisher"] for v in x][nt // 4:]))
    return float(np.mean(acc))


# --- (c) regime-dependent overshoot of the shipped (fixed-q) filter -------------
def mid_overshoot(dest, D0=0.0, nseed=200, burn=400, post=300, phi=0.9, s=0.30):
    """Mean-trajectory overshoot past `dest` after a mid-stream jump D0 -> dest."""
    traj = np.zeros((nseed, post))
    for sd in range(nseed):
        rng = np.random.default_rng(30000 + sd)
        f = WalkingFilter(1.0, 1.0, phi=phi, s=s); f.reset(scale=0.0)
        for v in simulate(rng, D0, 1.0, 1.0, burn):
            f.update(v)
        traj[sd] = [f.update(v).process_scale for v in simulate(rng, dest, 1.0, 1.0, post)]
    m = traj.mean(0) - dest
    past = max(0.0, m.max()) if dest > D0 else max(0.0, -m.min())
    return 100.0 * past / max(abs(dest - D0), 1e-9)


def capture(D, nseed=40, NT=1000, JT=100, phi=0.9, s=0.30):
    """Capture% of a jump to D on the shipped (fixed-q) filter."""
    g = 0
    for sd in range(nseed):
        rng = np.random.default_rng(sd); lam = np.zeros(NT); lam[JT:] = D
        x = np.cumsum(rng.standard_normal(NT) * np.sqrt(np.exp(lam))) + rng.standard_normal(NT)
        ls = WalkingFilter(1.0, 1.0, phi=phi, s=s).filter(x).process_scale
        if abs(np.mean(ls[-200:]) - D) < 0.7:
            g += 1
    return 100.0 * g / nseed


# --- (d) cold-start prior -------------------------------------------------------
def cold_overshoot(amp, Pmu0=None, phi=0.9, s=0.30, nseed=200, nt=200):
    traj = np.zeros((nseed, nt))
    for sd in range(nseed):
        x = simulate(np.random.default_rng(20000 + sd), -amp, 1.0, 1.0, nt)
        f = WalkingFilter(1.0, 1.0, phi=phi, s=s)
        if Pmu0 is not None:
            f.reset(scale=0.0); f._Pmu = Pmu0
            traj[sd] = [f.update(v).process_scale for v in x]
        else:
            traj[sd] = f.filter(x).process_scale
        traj[sd] += amp
    return 100.0 * max(0.0, -traj.mean(0).min()) / amp


def main():
    # (a)
    phis = [0.80, 0.90, 0.95]
    ratios = np.linspace(0.2, 3.0, 40)
    ova = {}
    for phi in phis:
        Kstar = (1 - phi) / 4
        ova[phi] = np.array([linear_overshoot(r * Kstar, phi) for r in ratios])
        onset = np.interp(1e-6, ova[phi], ratios)
        print(f"[linear loop] phi={phi}: overshoot onset at K/K* = {onset:.3f} (K*=(1-phi)/4)")

    # (b)
    dd = np.round(np.arange(-2.0, 4.01, 0.5), 2)
    Iof = np.array([emp_steady_info(float(d)) for d in dd])
    Ich = WalkingFilter(1.0, 1.0, phi=0.9, s=0.30)._Ichar
    print(f"[stiff wall] I swings {Iof.min():.4f} -> {Iof.max():.4f} "
          f"({Iof.max()/Iof.min():.0f}x); derived I_char = {Ich:.4f}")

    # (c) -- jump from d=0 to dest (exclude the no-jump point)
    dests = np.array([-2.0, -1.0, 1.0, 2.0, 3.0])
    ov_c = np.array([mid_overshoot(float(d)) for d in dests])
    cap_fixed = np.array([capture(float(d)) for d in dests])
    print("[fixed-q overshoot vs dest]", dict(zip(dests, ov_c.astype(int))))
    print("[fixed-q capture vs dest ]", dict(zip(dests, cap_fixed.astype(int))))

    # (d)
    amps = np.array([0.5, 1.0, 2.0, 3.0])
    cold_s2 = np.array([cold_overshoot(a) for a in amps])
    cold_25 = np.array([cold_overshoot(a, Pmu0=25.0) for a in amps])
    print("[cold s^2]", dict(zip(amps, cold_s2.astype(int))),
          "[cold 25]", dict(zip(amps, cold_25.astype(int))))

    fig, ax = plt.subplots(1, 4, figsize=(20.5, 4.6))

    a = ts.tidy(ax[0])
    a.axhline(0, color=ts.INK2, lw=0.8); a.axvline(1.0, color=ts.INK2, lw=1.0, ls="--")
    for j, phi in enumerate(phis):
        a.plot(ratios, 100 * ova[phi], color=ts.SERIES[j + 1], lw=1.8, label=f"phi={phi}")
    a.set_xlabel("gain ratio  K / K*,   K* = (1-phi)/4"); a.set_ylabel("step overshoot (%)")
    a.set_title("(a) critical damping verified: onset at K/K* = 1")
    a.legend(loc="upper left", fontsize=7.6)

    a = ts.tidy(ax[1])
    a.plot(dd, Iof, color=ts.SERIES[2], lw=1.9, marker="o", ms=3, label="grid info I(d)")
    a.axhline(Ich, color=ts.SERIES[3], lw=1.8, ls="--",
              label=f"derived I_char = {Ich:.3f}")
    a.set_yscale("log"); a.axvline(0.0, color=ts.INK2, lw=0.8)
    a.set_xlabel("regime loudness d  (nats)"); a.set_ylabel("Fisher info  (log)")
    a.set_title(f"(b) the stiff wall: I swings ~{Iof.max()/Iof.min():.0f}x")
    a.legend(loc="upper left", fontsize=7.6)

    a = ts.tidy(ax[2])
    a.axhline(0, color=ts.INK2, lw=0.8)
    a.plot(dests, ov_c, color=ts.SERIES[1], lw=1.9, marker="o", ms=5, label="mean overshoot")
    a.plot(dests, cap_fixed, color=ts.SERIES[3], lw=1.5, ls=":", marker="s", ms=4, label="capture %")
    a.set_xlabel("mid-stream jump destination d  (nats)"); a.set_ylabel("percent")
    a.set_title("(c) residual: ~20% overshoot on moderate steps, cap-tamed large")
    a.legend(loc="center right", fontsize=7.6)

    a = ts.tidy(ax[3])
    xw = np.arange(amps.size); w = 0.38
    a.bar(xw - w / 2, cold_25, w, color=ts.SERIES[1], label="old Pmu0 = 25")
    a.bar(xw + w / 2, cold_s2, w, color=ts.SERIES[3], label="derived Pmu0 = s^2")
    a.set_xticks(xw); a.set_xticklabels([f"{v:.1f}" for v in amps])
    a.set_xlabel("cold-start offset (nats)"); a.set_ylabel("overshoot (%)")
    a.set_title("(d) what is cleanly fixed: the cold start")
    a.legend(loc="upper right", fontsize=7.6)

    ts.save(fig, os.path.join(HERE, "figures", "0030-derived-walk-loop.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
