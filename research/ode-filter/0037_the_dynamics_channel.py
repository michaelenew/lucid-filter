"""0037 -- Does the dynamics channel actually make the filter responsive?

`0036` argued that `whiteness` was the wrong instrument: a cumulative residual
statistic can only accumulate, so it cannot say "the evidence now favours a
different member of the family".  The fix was to make alpha a gridded channel
with FLAT as an explicit member, evolved by a learned-persistence kernel.  That
is now in `output/odefilter`, as

    alpha(g) = (1 - g) * (1, 0, 0) + g * alpha,   g ~ AR(1)(phi_A, s_A)

with g = 1 the fitted dynamics, g = 0 the parent's local-level model exactly,
and g > 1 more persistent than fitted.

Two things to measure, one per figure:

  A  REVERSION.  A series that runs ODE, then flat, then ODE again.  Does g
     fall on affirmative evidence and come back?  Does it beat the static
     filter on log-loss while it does?

  B  THE FLAT-FORECAST COMPLAINT.  A fitted alpha that is too damped makes
     forecasts decay faster than the truth does.  Given a deliberately damped
     alpha, does the channel push g above 1 and recover the forecast?

Scored by log predictive density throughout, per `0036` section 2: these are
distributional claims and MSE cannot see them.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "lucid"))
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from odefilter import OdeFilter, Params  # noqa: E402

ALPHA3 = np.array([2.785218519281637, -2.6855430450862655, 0.9003245225862656])
LOG2PI = float(np.log(2.0 * np.pi))
Q0, S20 = 1.0, 9.0


def ode(n, alpha, Q, rng, z=None):
    z = np.zeros(3) if z is None else z.copy()
    x = np.zeros(n)
    for t in range(n):
        z = np.concatenate([[alpha @ z + math.sqrt(Q) * rng.standard_normal()],
                            z[:-1]])
        x[t] = z[0]
    return x, z


def score(m, S, x):
    """Mean log-loss, and calibration E[e^2/S] (1 = honest)."""
    e = m - x
    ok = np.isfinite(e) & np.isfinite(S) & (S > 0)
    e, s = e[ok], S[ok]
    return (float(np.mean(0.5 * (e * e / s + np.log(s)) + 0.5 * LOG2PI)),
            float(np.mean(e * e / s)), float(np.mean(e * e)))


def run(f, y, h):
    f.reset()
    n = len(y)
    m = np.empty(n)
    v = np.empty(n)
    d = np.empty(n)
    fm = np.full(n, np.nan)
    fS = np.full(n, np.nan)
    for t, val in enumerate(y):
        st = f.update(val)
        m[t], v[t], d[t] = st.mean, st.var, st.dynamics
        if t + h < n:
            fm[t + h], fS[t + h] = f.predict(h)
    return m, v, d, fm, fS


# ------------------------------------------------------------------ part A
def part_a(h=10):
    n = 300
    rng = np.random.default_rng(2027)
    x1, z = ode(n, ALPHA3, Q0, rng)
    x2 = x1[-1] + np.cumsum(math.sqrt(Q0) * rng.standard_normal(n))   # FLAT
    z = np.full(3, x2[-1])
    x3, _ = ode(n, ALPHA3, Q0, rng, z=z)
    x = np.concatenate([x1, x2, x3])
    y = x + math.sqrt(S20) * rng.standard_normal(3 * n)

    base = dict(alpha=tuple(ALPHA3), Q=Q0, s2=S20)
    static = OdeFilter(Params(**base), order=3)
    adapt = OdeFilter(Params(**base, phi_A=0.95, s_A=0.5), order=3, order_A=5)
    out = {}
    for nm, f in (("static", static), ("adaptive", adapt)):
        out[nm] = run(f, y, h)

    seg = [("ODE", 60, n), ("FLAT", n + 60, 2 * n), ("ODE again", 2 * n + 60, 3 * n)]
    rows = []
    print("=== A. reversion: ODE -> flat -> ODE ===")
    print(f"  {'segment':>10} | {'g':>6} | {'nll stat':>9} {'nll adapt':>9} "
          f"{'diff':>7} | {'cal stat':>9} {'cal adapt':>9}")
    for nm, a, b in seg:
        sl = slice(a, b)
        st = score(out["static"][3][sl], out["static"][4][sl], x[sl])
        ad = score(out["adaptive"][3][sl], out["adaptive"][4][sl], x[sl])
        gm = float(out["adaptive"][2][sl].mean())
        rows.append(dict(segment=nm, g=gm, nll_static=st[0], nll_adapt=ad[0],
                         cal_static=st[1], cal_adapt=ad[1],
                         mse_static=st[2], mse_adapt=ad[2]))
        print(f"  {nm:>10} | {gm:6.3f} | {st[0]:9.3f} {ad[0]:9.3f} "
              f"{ad[0]-st[0]:+7.3f} | {st[1]:9.2f} {ad[1]:9.2f}")

    # ------------------------------------------------------------ figure
    T = np.arange(3 * n)
    fig, axes = plt.subplots(3, 1, figsize=(11.6, 8.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.4, 1.5, 1.5],
                                              hspace=0.28))
    ax = axes[0]
    ax.scatter(T, y, s=3, color=ts.INK2, alpha=0.28, linewidths=0, zorder=1)
    ax.plot(T, x, color=ts.INK, lw=1.2, zorder=3, label="truth")
    ax.plot(T, out["adaptive"][0], color=ts.SERIES[0], lw=1.1, zorder=4,
            label="adaptive")
    ax.axvspan(n, 2 * n, color=ts.SERIES[3], alpha=0.10, lw=0, zorder=0)
    ax.annotate("no ODE governance:\na plain random walk",
                (1.5 * n, ax.get_ylim()[0]), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=8,
                color=ts.SERIES[3])
    ax.set_ylabel("level")
    ax.set_title("A · The dynamics stop, then resume")
    ax.legend(fontsize=8, ncol=2)
    ts.tidy(ax)

    ax = axes[1]
    ax.plot(T, out["adaptive"][2], color=ts.SERIES[0], lw=1.6)
    ax.axhline(1.0, color=ts.INK, lw=1.0, ls="--", zorder=0)
    ax.axhline(0.0, color=ts.SERIES[7], lw=1.0, ls=":", zorder=0)
    ax.axvspan(n, 2 * n, color=ts.SERIES[3], alpha=0.10, lw=0, zorder=0)
    ax.annotate("g = 1: the fitted ODE", (10, 1.0), xytext=(0, 5),
                textcoords="offset points", fontsize=8, color=ts.INK2)
    ax.annotate("g = 0: the parent's model, exactly", (10, 0.0),
                xytext=(0, -14), textcoords="offset points", fontsize=8,
                color=ts.SERIES[7])
    ax.set_ylabel("$g$  (dynamics\nin force)")
    ax.set_title("It falls on affirmative evidence and comes back — "
                 "no forgetting factor")
    ts.tidy(ax)

    ax = axes[2]
    k = 31
    ker = np.ones(k) / k
    for nm, c in (("static", ts.SERIES[1]), ("adaptive", ts.SERIES[0])):
        fm, fS = out[nm][3], out[nm][4]
        e = fm - x
        nll = 0.5 * (e * e / fS + np.log(fS)) + 0.5 * LOG2PI
        ax.plot(T, np.convolve(np.nan_to_num(nll), ker, mode="same"),
                color=c, lw=1.5, label=nm)
    ax.axvspan(n, 2 * n, color=ts.SERIES[3], alpha=0.10, lw=0, zorder=0)
    ax.set_ylabel(f"$h={h}$ log-loss\n({k}-pt mean)")
    ax.set_xlabel("t")
    ax.set_title("Lower is better")
    ax.legend(fontsize=8, ncol=2)
    ts.tidy(ax)
    ts.save(fig, os.path.join(HERE, "figures", "fig24-dynamics-reversion.png"))
    return rows


# ------------------------------------------------------------------ part B
def part_b(h=25):
    """A deliberately over-damped alpha: does g go above 1 and fix the fan?"""
    n = 900
    rng = np.random.default_rng(99)
    x, _ = ode(n, ALPHA3, Q0, rng)
    y = x + math.sqrt(S20) * rng.standard_normal(n)

    truth = Params(alpha=tuple(ALPHA3), Q=Q0, s2=S20)
    damped = tuple(truth.alpha_at(0.85))          # a too-damped fit
    print("\n=== B. a too-damped alpha ===")
    print(f"  true   oscillator "
          f"{np.abs(truth.roots[np.abs(truth.roots.imag) > 1e-9][0]):.4f}")
    dp = Params(alpha=damped, Q=Q0, s2=S20)
    print(f"  damped oscillator {np.abs(dp.roots[np.abs(dp.roots.imag) > 1e-9][0]):.4f}"
          f"  (memory {dp.memory():.1f} steps)")

    static = OdeFilter(Params(alpha=damped, Q=Q0, s2=S20), order=3)
    adapt = OdeFilter(Params(alpha=damped, Q=Q0, s2=S20, phi_A=0.95, s_A=0.4),
                      order=3, order_A=5)
    oracle = OdeFilter(truth, order=3)
    out = {nm: run(f, y, h) for nm, f in (("static", static),
                                          ("adaptive", adapt),
                                          ("oracle", oracle))}
    sl = slice(100, n)
    rows = []
    print(f"  {'filter':>10} | {'g':>6} | {'nll':>8} {'calib':>7} {'MSE':>9}")
    for nm in ("static", "adaptive", "oracle"):
        s_ = score(out[nm][3][sl], out[nm][4][sl], x[sl])
        gm = float(out[nm][2][sl].mean())
        rows.append(dict(filter=nm, g=gm, nll=s_[0], cal=s_[1], mse=s_[2]))
        print(f"  {nm:>10} | {gm:6.3f} | {s_[0]:8.3f} {s_[1]:7.2f} {s_[2]:9.1f}")
    closed = ((rows[0]["nll"] - rows[1]["nll"])
              / max(rows[0]["nll"] - rows[2]["nll"], 1e-9))
    print(f"  the channel closes {100 * closed:.0f}% of the "
          f"static-to-oracle log-loss gap")

    # forecast fans from one origin
    HMAX = 60
    org = 500
    fans = {}
    for nm, f in (("static", static), ("adaptive", adapt), ("oracle", oracle)):
        f.reset()
        for t in range(org + 1):
            f.update(y[t])
        fans[nm] = np.array([f.predict(k) for k in range(1, HMAX + 1)])

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.2),
                             gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    back = np.arange(org - 60, org + 1)
    fh = np.arange(org + 1, org + HMAX + 1)
    ax.plot(back, x[back], color=ts.INK, lw=1.5, zorder=3)
    ax.plot(fh, x[fh], color=ts.INK, lw=1.5, label="truth", zorder=3)
    ax.axvline(org, color=ts.GRID, lw=1.2, zorder=0)
    for nm, c in (("static", ts.SERIES[1]), ("adaptive", ts.SERIES[0]),
                  ("oracle", ts.SERIES[2])):
        ax.plot(fh, fans[nm][:, 0], color=c, lw=1.8, label=nm, zorder=4)
    ax.set_xlabel("t")
    ax.set_ylabel("level")
    ax.set_title("B · A too-damped fit forecasts too flat")
    ax.legend(fontsize=8, ncol=2)
    ts.tidy(ax)

    ax = axes[1]
    T = np.arange(n)
    ax.plot(T, out["adaptive"][2], color=ts.SERIES[0], lw=1.5)
    ax.axhline(1.0, color=ts.INK, lw=1.0, ls="--", zorder=0)
    ax.axhline(1.0 / 0.85, color=ts.SERIES[2], lw=1.2, ls=":", zorder=0)
    ax.annotate("g needed to undo the damping", (n * 0.05, 1 / 0.85),
                xytext=(0, 5), textcoords="offset points", fontsize=8,
                color=ts.SERIES[2])
    ax.set_xlabel("t")
    ax.set_ylabel("$g$")
    ax.set_title("The channel pushes past 1 to recover it")
    ts.tidy(ax)
    ts.save(fig, os.path.join(HERE, "figures", "fig25-too-damped.png"))
    return rows, float(closed)


def main():
    a = part_a()
    b, closed = part_b()
    with open(os.path.join(HERE, "figures", "ode037.json"), "w") as f:
        json.dump(dict(A=a, B=b, gap_closed=closed), f, indent=1)


if __name__ == "__main__":
    main()
