"""Design the force-linearizing coordinate that flattens the stiffening well.

The stiffening well (0010 panel c) makes the damping ratio amplitude-dependent:
the Newton step grad/info saturates.  For the exponential wall the Fisher is
I ~= 1/2 e^e (e = offset), so grad/info = 1 - e^{-e} -> 1: a single step can never
move more than ~1 nat, and the effective per-step fractional progress falls with
amplitude.  0001 showed the cure: far above the window log(score) is linear in the
true offset (slope ~1), an UNSATURATED distance signal.

This probe (measurement only, no filter change yet):
  (1) maps grad(e) and info(e) for the shipped grid (window pinned at mu=0, truth
      held at a constant offset e), and
  (2) searches for a transform e_hat(grad, info) that is linear in the true e at
      ALL amplitudes -- the coordinate a constant gain would critically damp
      everywhere.  Candidates: Newton grad/info; log-Fisher log(info/I_char);
      signed log-score asinh(grad / g0).

Run: python 0032_linearize_the_wall.py   (~1-2 min)
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


def force_map(offs, phi=0.9, s=0.30, nseed=60, nt=700):
    """Time-averaged grid-shift score grad(e) and Fisher info(e), window at mu=0,
    truth held at offset e.  Returns grad, info arrays over `offs`."""
    grad = np.zeros(offs.size); info = np.zeros(offs.size)
    for j, e in enumerate(offs):
        gg = []; ii = []
        for sd in range(nseed):
            x = simulate(np.random.default_rng(sd), e, 1.0, 1.0, nt)
            f = MovingChannel(1.0, 1.0, phi=phi, s=s, step="kalman_auto", q_mu=0.0, P0=1e-9)
            f.reset(mu=0.0)                       # window frozen at 0; measure the force at offset e
            g = []; iu = []
            for v in x:
                o = f.update(v); g.append(o["score"]); iu.append(o["fisher"])
            gg.append(np.mean(g[nt // 4:])); ii.append(np.mean(iu[nt // 4:]))
        grad[j] = np.mean(gg); info[j] = np.mean(ii)
    return grad, info


def main():
    offs = np.round(np.arange(-3.0, 5.01, 0.5), 2)
    grad, info = force_map(offs)
    Ich = WalkingFilter(1.0, 1.0, phi=0.9, s=0.30)._Ichar

    newton = grad / np.maximum(info, 1e-9)                 # current step (saturates)
    logfish = np.log(np.maximum(info, 1e-9) / Ich)         # log-Fisher distance
    g0 = float(np.interp(0.0, offs, np.gradient(grad, offs)))  # near-field slope ~ I(0)
    asinhs = np.arcsinh(grad / max(g0 * 0.5, 1e-6))        # signed log-score distance

    # linearity: fit each candidate to the identity over |e|<=3 and report slope/R^2
    def lin(y):
        m = np.abs(offs) <= 3.0
        a = np.polyfit(offs[m], y[m], 1)
        r = np.corrcoef(offs[m], y[m])[0, 1] ** 2
        return a[0], r
    for name, y in [("newton grad/info", newton), ("log-Fisher", logfish),
                    ("asinh score", asinhs)]:
        sl, r2 = lin(y)
        print(f"[{name:16s}] slope={sl:+.3f}  R^2={r2:.4f}  "
              f"range {y.min():+.2f}..{y.max():+.2f}")
    print(f"[near-field slope g0 ~ I(0)] {g0:.4f};  I_char={Ich:.4f}")

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))

    a = ts.tidy(ax[0]); a.axhline(0, color=ts.INK2, lw=0.8); a.axvline(0, color=ts.INK2, lw=0.8)
    a.plot(offs, grad, color=ts.SERIES[1], lw=1.9, marker="o", ms=3, label="grid-shift score grad(e)")
    a.set_xlabel("true offset e = truth - mu  (nats)"); a.set_ylabel("score")
    a.set_yscale("symlog", linthresh=0.05)
    a.set_title("(a) the stiffening force (symlog)"); a.legend(loc="upper left", fontsize=8)

    a = ts.tidy(ax[1])
    a.plot(offs, info, color=ts.SERIES[2], lw=1.9, marker="o", ms=3)
    a.axhline(Ich, color=ts.SERIES[3], lw=1.3, ls="--", label=f"I_char={Ich:.3f}")
    a.set_yscale("log"); a.set_xlabel("true offset e  (nats)"); a.set_ylabel("Fisher info (log)")
    a.set_title("(b) observability I(e)"); a.legend(loc="upper left", fontsize=8)

    a = ts.tidy(ax[2]); a.plot(offs, offs, color=ts.INK2, lw=1.1, ls=":", label="identity (target)")
    a.plot(offs, newton, color=ts.SERIES[1], lw=1.7, marker="o", ms=3, label="Newton grad/info")
    a.plot(offs, logfish, color=ts.SERIES[3], lw=1.7, marker="s", ms=3, label="log-Fisher")
    a.plot(offs, asinhs, color=ts.SERIES[5], lw=1.7, marker="^", ms=3, label="asinh score")
    a.set_xlabel("true offset e  (nats)"); a.set_ylabel("distance estimate")
    a.set_title("(c) which coordinate is linear in e?"); a.legend(loc="upper left", fontsize=8)

    ts.save(fig, os.path.join(HERE, "figures", "0031-linearize-the-wall.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
