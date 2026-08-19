"""Is the settling a real ODE?  Yes -- a nonlinear likelihood-gradient flow.

The settling in 0009 overshoots, which reads as a damped second-order response.
This probe asks whether that is a genuine dynamical system and, if so, what kind
-- because a clean linear one would open a defensible optimality argument
(steady-state Kalman / alpha-beta tracking).

The phase-space force
---------------------
Move the window with a constant step (no decay, no clamp) and bin the mean step
E[d mu | mu - truth]: the RESTORING FORCE that drives the ranging.  This is the
right object, not the frozen-window signal -- with mu held still the in-frame
AR(1) reverts to the frame centre and the static signal is ~0 (a red herring
that first looked like zero restoring gain).

Two findings, one caveat:

1. The force is a proper restoring force -- anti-symmetric, sign(-offset) -- but
   NONLINEAR: harmonic near the optimum (stiffness = the Fisher information I),
   stiffening into anharmonic walls far out.  So the window sits in a stiffening
   potential well, and the damping ratio is AMPLITUDE-DEPENDENT.  With a constant
   step the soft core is the LEAST damped -- small excursions overshoot most
   (~100%), the stiff walls tame the large ones (~37%).  No single constant step
   is critically damped everywhere, which is exactly why the Robbins-Monro decay
   (0007) helps: shrinking the step as the excursion shrinks compensates the
   softening core.

2. The force is sign-aligned with the exact marginal-likelihood gradient
   d loglik / d mu, and shares its local curvature -- the Fisher information
   I = -d^2 loglik / d mu^2 at the optimum.  So the ranging is (approximately)
   GRADIENT ASCENT on the marginal likelihood in the window position: it tracks
   the log-scale by maximum likelihood.  That is the defensible-optimality hook
   (log-loss is the repo's currency), and near the optimum it linearises to a
   spring of stiffness I -- an alpha-beta / steady-state-Kalman tracker whose
   min-variance gains are set by I and the scale-drift variance.

   Caveat: the servo's step is the cheap composite `pi.lam + w*score`, which is
   sign-aligned with the exact gradient but NOT equal to it, so the current move
   is a surrogate for the efficient one.  Making the step the (natural) gradient
   is the route to a defensibly optimal ranging; reshaping the force to be linear
   (e.g. via the far-field log-score, which recovers distance linearly, 0001) is
   the route to one critical damping at all amplitudes.

Run: python 0010_ranging_is_a_likelihood_gradient_flow.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "random-walk-filter", "scripts"))

from gridlab import simulate, grid, exact_shift_gradient  # noqa: E402
from moving_grid import MovingChannel  # noqa: E402
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------- the phase-space force during tracking
def force_profile(eta=0.4, beta=0.6, phi=0.9, s=0.30, order=5,
                  nseed=60, nt=250, rng0=10000):
    """E[d mu | offset] binned, aggregating many constant-step tracking runs."""
    off, dmu = [], []
    for mu0 in np.linspace(-5.0, 5.0, 21):
        for sd in range(nseed):
            rng = np.random.default_rng(rng0 + int(1000 * abs(mu0)) + sd)
            x = simulate(rng, 0.0, 1.0, 1.0, nt)
            f = MovingChannel(1.0, 1.0, phi=phi, s=s, order=order, step="servo",
                              eta=eta, beta=beta, tau=1e9, eta_floor=0.0, cap=10.0)
            f.reset(mu=mu0)
            prev = f.mu
            for v in x:
                m = f.update(v)["mu"]
                off.append(prev); dmu.append(m - prev); prev = m
    off, dmu = np.array(off), np.array(dmu)
    edges = np.linspace(-5.0, 5.0, 41)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    fp = np.array([dmu[(off >= edges[i]) & (off < edges[i + 1])].mean()
                   if ((off >= edges[i]) & (off < edges[i + 1])).sum() > 30 else np.nan
                   for i in range(40)])
    return ctr, fp / eta                        # /eta -> the driving signal


# ------------------------------------- the exact marginal-likelihood gradient
def gradient_profile(phi=0.9, s=0.30, order=5, nseed=80, nt=300):
    lam, w0, T = grid(phi, s, order)
    offs = np.linspace(-3.0, 3.0, 25)
    g = np.zeros(offs.size)
    for j, d in enumerate(offs):
        X = np.array([simulate(np.random.default_rng(sd), 0.0, 1.0, 1.0, nt)
                      for sd in range(nseed)])
        g[j] = exact_shift_gradient(X, lam, w0, T, 1.0, 1.0, mu=d).mean() / nt
    I = -np.polyfit(offs[np.abs(offs) < 0.8], g[np.abs(offs) < 0.8], 1)[0]
    return offs, g, float(I)


# ----------------------------------------- amplitude-dependent damping (overshoot)
def overshoot_vs_amplitude(eta=0.5, beta=0.6, phi=0.9, s=0.30, order=5,
                           nseed=200, nt=200):
    amps = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
    ov = np.zeros(amps.size)
    for k, a in enumerate(amps):
        X = np.zeros((nseed, nt))
        for sd in range(nseed):
            rng = np.random.default_rng(20000 + 100 * k + sd)
            x = simulate(rng, 0.0, 1.0, 1.0, nt)
            f = MovingChannel(1.0, 1.0, phi=phi, s=s, order=order, step="servo",
                              eta=eta, beta=beta, tau=1e9, eta_floor=0.0, cap=10.0)
            f.reset(mu=a)
            X[sd] = [f.update(v)["mu"] for v in x]
        m = X.mean(0)                            # mean trajectory (deterministic response)
        ov[k] = max(0.0, -m.min()) / a           # fractional overshoot past 0
    return amps, ov


def main():
    ctr, fsig = force_profile()
    offs, g, I = gradient_profile()
    amps, ov = overshoot_vs_amplitude()
    print(f"[force] local Fisher info I = {I:.4f} /step; "
          f"corr(force, exact grad) = "
          f"{np.corrcoef(np.interp(offs, ctr, fsig), g)[0,1]:.3f}")
    print(f"[stiffening] exact |grad| at offset 3 is "
          f"{abs(g[-1]) / (I * 3):.1f}x the linear -I*offset")
    print("[overshoot vs start amplitude] "
          + ", ".join(f"{a:.1f}:{o:.2f}" for a, o in zip(amps, ov))
          + "  (falls with amplitude => soft core least damped)")

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))

    # (a) the force is the likelihood gradient (sign-aligned, stiffening)
    a = ts.tidy(ax[0])
    a.axhline(0, color=ts.INK2, lw=0.7); a.axvline(0, color=ts.INK2, lw=0.7)
    a.plot(offs, g, color=ts.SERIES[1], marker="o", ms=3, label="exact d loglik/d mu")
    a.plot(offs, -I * offs, color=ts.INK2, lw=1.1, ls="--",
           label=f"linear -I*offset  (I={I:.3f})")
    keep = np.abs(ctr) <= 2.5                    # well-sampled region
    a.plot(ctr[keep], fsig[keep], color=ts.SERIES[5], lw=1.8, label="servo force / eta")
    a.set_xlabel("offset  mu - truth  (nats)")
    a.set_ylabel("restoring signal per step")
    a.set_title("(a) the ranging force follows the likelihood gradient")
    a.legend(loc="upper right", fontsize=8)

    # (b) the effective potential -- a stiffening well
    a = ts.tidy(ax[1])
    V = -np.concatenate([[0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(offs))])
    V -= V[np.argmin(np.abs(offs))]
    a.plot(offs, V, color=ts.SERIES[1], lw=2.0, label="potential  -integral of grad")
    a.plot(offs, 0.5 * I * offs ** 2, color=ts.INK2, lw=1.1, ls="--",
           label="harmonic  0.5 I offset^2")
    a.set_xlabel("offset  mu - truth  (nats)")
    a.set_ylabel("effective potential")
    a.set_title("(b) a stiffening well: soft core, stiff walls")
    a.legend(loc="upper center", fontsize=8)

    # (c) amplitude-dependent damping -> no single critical gain
    a = ts.tidy(ax[2])
    a.plot(amps, 100 * ov, color=ts.SERIES[5], marker="o", ms=5)
    a.axhline(0, color=ts.INK2, lw=0.8)
    a.set_xlabel("start amplitude  |mu0 - truth|  (nats)")
    a.set_ylabel("overshoot  (% of start)")
    a.set_title("(c) soft core least damped: overshoot 100%->37%, non-uniform")
    ts.save(fig, os.path.join(HERE, "figures", "0010-likelihood-gradient-flow.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
