"""0031 -- The gut check: the data, and what each filter believes about it.

Numbers settle arguments; pictures catch mistakes.  This puts the parent and
the candidate on the same series and draws what each one thinks, in the spirit
of the parent's own figures.

The layout is chosen so the figure cannot flatter the candidate:

  A  TRACKING.  Truth, observations, and both posterior means with +-2 SD.
     Per `0006` this is where the two should look nearly identical -- tracking
     error is almost blind to the dynamics.  If the candidate looked much
     better here, something would be wrong.
  B  FORECASTING.  From one origin, both filters projected 40 steps with their
     own predictive bands, against what the series actually did.  This is
     where alpha lives: the parent's forecast is flat by construction, the
     candidate's oscillates and decays toward the same level.
  C  The h = 20 forecast error of each, over time.  The running version of B.
  D  The candidate's posterior in (x, x', x'') coordinates -- a fixed
     involutive change of basis, so nothing is created; the parent has no
     analogue because at p = 1 there is nothing to convert.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ode-adaptive-filter", "output"))
sys.path.insert(0, os.path.join(ROOT, "adaptive-random-walk-filter", "output"))
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from odefilter import OdeFilter  # noqa: E402
import statfilter  # noqa: E402

ALPHA3 = np.array([2.785218519281637, -2.6855430450862655, 0.9003245225862656])


def gen(n, Q, S2, rng):
    z = np.zeros(3)
    x = np.zeros(n)
    for t in range(n):
        xn = float(ALPHA3 @ z) + np.sqrt(Q) * rng.standard_normal()
        z = np.concatenate([[xn], z[:-1]])
        x[t] = xn
    return x, x + np.sqrt(S2) * rng.standard_normal(n)


def main():
    n, Q, kappa = 700, 1.0, 0.25
    rng = np.random.default_rng(4242)
    xr, _ = gen(40000, Q, 0.0, np.random.default_rng(3))
    S2 = (kappa * float(np.std(np.diff(xr)))) ** 2
    x, y = gen(n, Q, S2, rng)

    t0 = time.time()
    fo = OdeFilter.fit(y, p=3, order=5, max_iter=250)
    t1 = time.time()
    fp = statfilter.AdaptiveFilter.fit(y, order=5, max_iter=250)
    t2 = time.time()
    print(f"fitted: ode {t1-t0:.0f}s, parent {t2-t1:.0f}s")
    print(f"  roots {np.round(fo.params.roots, 4)}")
    print(f"  memory {fo.params.memory():.1f} steps; Q={fo.params.Q:.3f} "
          f"s2={fo.params.s2:.3f}  (truth Q={Q}, s2={S2:.3f})")

    ro, rp = fo.filter(y), fp.filter(y)

    # ------------------------------------------------------- forecast records
    H, HMAX = 20, 40
    lo = n // 2
    fo.reset(); fp.reset()
    eo, ep = np.full(n, np.nan), np.full(n, np.nan)
    origin = lo + 60
    fan_o = fan_p = None
    for t, v in enumerate(y):
        fo.update(v); fp.update(v)
        if t >= lo:
            if t + H < n:
                eo[t + H] = fo.predict(H)[0] - x[t + H]
                ep[t + H] = fp.predict(1)[0] - x[t + H]
            if t == origin:
                fan_o = np.array([fo.predict(h) for h in range(1, HMAX + 1)])
                fan_p = np.array([fp.predict(h) for h in range(1, HMAX + 1)])

    mo = float(np.nanmean(eo ** 2))
    mp = float(np.nanmean(ep ** 2))
    print(f"  h={H} forecast MSE: ode {mo:.2f}, parent {mp:.2f} "
          f"(ratio {mo/mp:.3f})")

    # ------------------------------------------------------------- the figure
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.2))

    # --- A: tracking
    ax = axes[0, 0]
    w = slice(origin - 90, origin + 50)
    tt = np.arange(n)[w]
    ax.scatter(tt, y[w], s=7, color=ts.INK2, alpha=0.45, label="observed",
               zorder=2, linewidths=0)
    ax.plot(tt, x[w], color=ts.INK, lw=1.6, label="truth", zorder=3)
    for r, c, nm in ((rp, ts.SERIES[1], "parent"), (ro, ts.SERIES[0], "odefilter")):
        sd = np.sqrt(np.maximum(r.var[w], 0.0))
        ax.fill_between(tt, r.mean[w] - 2 * sd, r.mean[w] + 2 * sd,
                        color=c, alpha=0.16, linewidth=0, zorder=1)
        ax.plot(tt, r.mean[w], color=c, lw=1.5, label=nm, zorder=4)
    ax.set_title("A · Tracking: nearly the same picture, as 0006 predicted")
    ax.set_xlabel("t")
    ax.set_ylabel("level")
    ax.legend(fontsize=8, ncol=2)
    ts.tidy(ax)

    # --- B: the forecast fan
    ax = axes[0, 1]
    back = np.arange(origin - 40, origin + 1)
    ax.plot(back, x[back], color=ts.INK, lw=1.6, zorder=3)
    fh = np.arange(origin + 1, origin + HMAX + 1)
    ax.plot(fh, x[fh], color=ts.INK, lw=1.6, label="truth", zorder=3)
    ax.axvline(origin, color=ts.GRID, lw=1.2, zorder=0)
    for fan, c, nm in ((fan_p, ts.SERIES[1], "parent"),
                       (fan_o, ts.SERIES[0], "odefilter")):
        sd = np.sqrt(np.maximum(fan[:, 1], 0.0))
        ax.fill_between(fh, fan[:, 0] - 2 * sd, fan[:, 0] + 2 * sd,
                        color=c, alpha=0.16, linewidth=0, zorder=1)
        ax.plot(fh, fan[:, 0], color=c, lw=1.8, label=nm, zorder=4)
    ax.set_title("B · Forecasting from one origin: where $\\alpha$ lives")
    ax.set_xlabel("t")
    ax.legend(fontsize=8)
    ts.tidy(ax)

    # --- C: running h-step error
    ax = axes[1, 0]
    ax.axhline(0, color=ts.INK, lw=1.0, zorder=0)
    ax.plot(np.arange(n), ep, color=ts.SERIES[1], lw=1.1, alpha=0.85,
            label=f"parent  (MSE {mp:.0f})")
    ax.plot(np.arange(n), eo, color=ts.SERIES[0], lw=1.1, alpha=0.9,
            label=f"odefilter  (MSE {mo:.0f})")
    ax.set_title(f"C · $h={H}$ forecast error, running "
                 f"(ratio {mo/mp:.2f})")
    ax.set_xlabel("t")
    ax.set_ylabel("forecast minus truth")
    ax.legend(fontsize=8)
    ts.tidy(ax)

    # --- D: the derivative posterior
    ax = axes[1, 1]
    fo.reset()
    D = []
    for t, v in enumerate(y):
        fo.update(v)
        m, P = fo.derivatives()
        D.append([m[1], np.sqrt(max(P[1, 1], 0.0)),
                  m[2], np.sqrt(max(P[2, 2], 0.0))])
    D = np.array(D)
    tt = np.arange(n)[w]
    for k, (c, nm) in enumerate(((ts.SERIES[2], "$\\hat x'$"),
                                 (ts.SERIES[4], "$\\hat x''$"))):
        mu, sd = D[w, 2 * k], D[w, 2 * k + 1]
        ax.fill_between(tt, mu - 2 * sd, mu + 2 * sd, color=c, alpha=0.16,
                        linewidth=0)
        ax.plot(tt, mu, color=c, lw=1.5, label=nm)
    dx = np.gradient(x)
    ax.plot(tt, dx[w], color=ts.INK, lw=1.0, ls="--", label="true $x'$")
    ax.axhline(0, color=ts.INK, lw=0.8, zorder=0)
    ax.set_title("D · The candidate's derivative posterior "
                 "(no parent analogue)")
    ax.set_xlabel("t")
    ax.legend(fontsize=8, ncol=3)
    ts.tidy(ax)

    ts.save(fig, os.path.join(HERE, "figures", "fig20-two-beliefs.png"))

    with open(os.path.join(HERE, "figures", "ode031.json"), "w") as f:
        json.dump(dict(n=n, kappa=kappa, Q=Q, S2=S2, H=H, origin=origin,
                       mse_ode=mo, mse_parent=mp,
                       roots=[[float(z.real), float(z.imag)]
                              for z in fo.params.roots],
                       ode=dict(Q=fo.params.Q, s2=fo.params.s2,
                                s_P=fo.params.s_P, s_M=fo.params.s_M),
                       parent=dict(Q=fp.params.Q, s2=fp.params.s2,
                                   s_P=fp.params.s_P, s_M=fp.params.s_M)),
                  f, indent=1)


if __name__ == "__main__":
    main()
