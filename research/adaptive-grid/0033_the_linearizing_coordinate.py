"""The linearizing coordinate is derived and exact -- and the finite grid breaks it.

Goal (finding 19): remove the amplitude-dependent damping of the stiffening well
(0010 panel c) with a THEORETICALLY DERIVED force-linearization, not an empirical
fit -- because a fitted high/low profile has no scale-invariant justification and
would forfeit the program's defensibility.

The theory (exact, no free constant)
------------------------------------
For a Gaussian with window variance v = Q e^mu and true variance v* = Q e^lam,
offset e = lam - mu, the expected per-step log-likelihood gives

    score   g(e)  = 1/2 (v*/v - 1) = 1/2 (e^e - 1)
    Fisher  I(e)  = -d^2 L/dmu^2   = 1/2 e^e

so the offset is recovered EXACTLY by either

    e = log( I(e) / I(0) )           (log-Fisher-ratio),  and
    e = -log( 1 - g/I )              (un-saturating the Newton step g/I = 1 - e^{-e}).

Both are identities (verified to machine zero below).  Neither the raw score
(~ e^e - 1, explodes loud / saturates quiet) nor the natural gradient
(g/I = 1 - e^{-e}, saturates loud / explodes quiet) is the offset; only these two
derived transforms are.  Stepping mu by gamma * e makes the force de/dt = -gamma e
LINEAR at every amplitude -> one critical damping K* = (1-phi)/4 everywhere
(finding 18's linear loop, now linear globally).

The obstacle (also theory, not fitting)
---------------------------------------
The identities take the IDEAL (g, I).  A finite grid of half-span H can only
supply them while the offset stays inside the window: once |e| > H every node
reports the wrong variance, the prefactors collapse, and the measured g/I no
longer obeys g/I = 1 - e^{-e} -- it OVERSHOOTS 1 (measured ~65 at e=5, vs the
ideal ceiling of 1).  Feeding that into e = -log(1 - g/I) clips at the pole and
emits enormous steps: applied in the walking filter it blows the loud side up
(mid-stream overshoot 21% -> 130%, tracking 0.48 -> 1.25).  So the linearization
is exact on ideal inputs and UNREALISABLE on corrupted ones.

Consequence: the amplitude-dependent damping beyond the window is not a
force-shape problem a transform can fix -- it is a grid-REACH problem.  Inside the
span the shipped Newton step already approximates the exact coordinate (they agree
to first order at e=0); the theoretically clean cure keeps the offset inside the
span (an expanding / hopping grid, the adaptive-grid thesis) so the derived
transform always sees uncorrupted (g, I).

Run: python 0033_the_linearizing_coordinate.py   (~1 min)
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
import theory_style as ts                            # noqa: E402
import matplotlib.pyplot as plt                      # noqa: E402


def grid_force_map(offs, phi=0.9, s=0.30, nseed=60, nt=700):
    """Measured grid score g(e) and Fisher I(e), window frozen at 0, truth at e."""
    g = np.zeros(offs.size); I = np.zeros(offs.size)
    for j, e in enumerate(offs):
        gg, ii = [], []
        for sd in range(nseed):
            x = simulate(np.random.default_rng(sd), e, 1.0, 1.0, nt)
            f = MovingChannel(1.0, 1.0, phi=phi, s=s, step="kalman_auto", q_mu=0.0, P0=1e-9)
            f.reset(mu=0.0)
            o = [f.update(v) for v in x]
            gg.append(np.mean([r["score"] for r in o][nt // 4:]))
            ii.append(np.mean([r["fisher"] for r in o][nt // 4:]))
        g[j] = np.mean(gg); I[j] = np.mean(ii)
    return g, I


def main():
    # analytic identities
    e = np.linspace(-3, 3, 25)
    g_id = 0.5 * (np.exp(e) - 1.0)
    I_id = 0.5 * np.exp(e)
    err_logfish = float(np.max(np.abs(np.log(I_id / 0.5) - e)))
    err_lognewt = float(np.max(np.abs(-np.log1p(-(g_id / I_id)) - e)))
    print(f"[identity] max|log(I/I0) - e|      = {err_logfish:.2e}")
    print(f"[identity] max|-log(1-g/I) - e|    = {err_lognewt:.2e}")

    # grid corruption
    offs = np.round(np.arange(-2.5, 5.01, 0.5), 2)
    g_gr, I_gr = grid_force_map(offs)
    newton_gr = g_gr / np.maximum(I_gr, 1e-9)
    print(f"[grid] Newton g/I at e=+5 ~ {newton_gr[-1]:.1f} (ideal ceiling 1.0) -> identity broken")

    fig, ax = plt.subplots(1, 3, figsize=(15.8, 4.5))

    a = ts.tidy(ax[0]); a.plot(e, e, color=ts.INK2, lw=1.1, ls=":", label="identity")
    a.plot(e, np.log(I_id / 0.5), color=ts.SERIES[3], lw=2.2, label="log(I/I0)")
    a.plot(e, -np.log1p(-(g_id / I_id)), color=ts.SERIES[5], lw=1.4, ls="--",
           label="-log(1-g/I)")
    a.set_xlabel("true offset e  (nats)"); a.set_ylabel("derived estimate")
    a.set_title("(a) both transforms = e exactly (ideal well)")
    a.legend(loc="upper left", fontsize=8)

    a = ts.tidy(ax[1]); a.axhline(1.0, color=ts.INK2, lw=1.0, ls=":", label="ideal ceiling g/I=1")
    a.plot(offs, np.clip(newton_gr, -2, 70), color=ts.SERIES[1], lw=2.0, marker="o", ms=3,
           label="grid g/I (measured)")
    a.plot(e, 1 - np.exp(-e), color=ts.SERIES[3], lw=1.4, ls="--", label="ideal 1-e^{-e}")
    a.set_yscale("symlog", linthresh=1.0)
    a.set_xlabel("true offset e  (nats)"); a.set_ylabel("Newton step g/I  (symlog)")
    a.set_title("(b) the grid breaks the identity: g/I overshoots 1")
    a.legend(loc="upper left", fontsize=8)

    a = ts.tidy(ax[2])
    safe = np.minimum(newton_gr, 0.999)
    a.plot(offs, offs, color=ts.INK2, lw=1.1, ls=":", label="identity")
    a.plot(offs, -np.log1p(-safe), color=ts.SERIES[5], lw=2.0, marker="o", ms=3,
           label="-log(1 - grid g/I)")
    a.set_xlabel("true offset e  (nats)"); a.set_ylabel("transform on grid inputs")
    a.set_title("(c) exact transform on corrupted inputs diverges")
    a.legend(loc="upper left", fontsize=8)

    ts.save(fig, os.path.join(HERE, "figures", "0032-linearizing-coordinate.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
