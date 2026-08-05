"""0034 -- Re-score 0032 as a distribution, not a point.

0032 scored forecasts by MSE.  That was wrong by this workstream's own stated
standard (SUMMARY item 7) and wrong for the reason the parent's `fit_` docstring
already records about its `pem` criterion: squared error depends on the
parameters only through the predicted mean, so it cannot see anything living in
the predictive variance.  A forecast is a distribution; it should be scored as
one.

The log predictive density decomposes exactly into the two things worth
separating:

    -log p = 1/2 ( e^2 / S )  +  1/2 log S  +  1/2 log 2 pi
             \_____________/     \________/
              CALIBRATION         SHARPNESS
              was it wrong        how confident did it claim to be

  E[e^2/S] = 1 for an honestly calibrated forecaster, whatever its accuracy.
  > 1 is overconfidence, < 1 is timidity.

This matters for the 0033 result specifically.  odefilter's h-step forecast MSE
after a jump in alpha was 1.365x the parent's, and the natural reading was "it
got worse".  But a model that is wrong while *saying* it is uncertain is doing
something different from one that is wrong while claiming to be right -- both
practically and as evidence about whether the theory is on track.  MSE cannot
tell those apart.  This can.

Reuses 0032's series and its cached fit, so nothing is refitted.
"""
import json
import os
import sys
from importlib import import_module

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ode-adaptive-filter", "output"))
sys.path.insert(0, os.path.join(ROOT, "adaptive-random-walk-filter", "output"))
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

_m32 = import_module("0032_a_hard_series")

LOG2PI = float(np.log(2.0 * np.pi))


def forecasts(f, y, h, parent):
    """h-step predictive mean AND variance, indexed by the target time."""
    f.reset()
    m = np.full(len(y), np.nan)
    S = np.full(len(y), np.nan)
    for t, v in enumerate(y):
        f.update(v)
        if t + h < len(y):
            mm, ss = f.predict(h)
            m[t + h], S[t + h] = mm, ss
    return m, S


def score(m, S, x, sl):
    """Mean log-loss over a slice, and its two components."""
    e = m[sl] - x[sl]
    s = S[sl]
    ok = np.isfinite(e) & np.isfinite(s) & (s > 0)
    e, s = e[ok], s[ok]
    cal = float(np.mean(e * e / s))            # 1.0 when honest
    sharp = float(np.mean(0.5 * np.log(s)))
    return dict(nll=float(np.mean(0.5 * (e * e / s + np.log(s)) + 0.5 * LOG2PI)),
                calibration=cal, sharpness=sharp,
                mse=float(np.mean(e * e)),
                claimed_sd=float(np.mean(np.sqrt(s))), n=int(e.size))


def main():
    h = _m32.H
    rng = np.random.default_rng(20260801)
    x, y, qmul, smul = _m32.simulate(rng)
    fo, fp = _m32.fit_and_cache(y[:_m32.JUMPS[0]])

    mo, So = forecasts(fo, y, h, parent=False)
    mp, Sp = forecasts(fp, y, h, parent=True)
    ro, rp = fo.filter(y), fp.filter(y)      # filtered posterior, for tracking

    phases = [("baseline", 60, 190), ("kicks", 195, 460),
              ("meas. regime", 470, 600), ("dyn. jump 1", 620, 720),
              ("proc. regime", 720, 850), ("dyn. jump 2", 880, 980),
              ("all", 60, _m32.N)]

    print(f"h = {h}.  log-loss in nats/point, lower is better.  "
          f"calibration E[e^2/S] = 1 is honest.\n")
    hdr = (f"{'phase':>14} | {'nll ode':>8} {'nll par':>8} {'diff':>7} | "
           f"{'cal ode':>8} {'cal par':>8} | {'sd ode':>7} {'sd par':>7} | "
           f"{'MSE ratio':>9}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for nm, a, b in phases:
        sl = slice(a, b)
        so, sp = score(mo, So, x, sl), score(mp, Sp, x, sl)
        rows.append(dict(phase=nm, lo=a, hi=b, ode=so, parent=sp))
        print(f"{nm:>14} | {so['nll']:8.3f} {sp['nll']:8.3f} "
              f"{so['nll']-sp['nll']:+7.3f} | {so['calibration']:8.2f} "
              f"{sp['calibration']:8.2f} | {so['claimed_sd']:7.1f} "
              f"{sp['claimed_sd']:7.1f} | {so['mse']/sp['mse']:9.3f}")

    print("\n  A negative `diff` means odefilter's forecast DISTRIBUTION is "
          "better,\n  whatever the point forecast did.")

    # the same treatment for the FILTERED state, since 0033's headline loss
    # (1.35x worse tracking) is also a point-estimate claim
    print(f"\n  --- the filtered state, scored the same way ---")
    print(hdr)
    print("-" * len(hdr))
    for nm, a, b in phases:
        sl = slice(a, b)
        to, tp = (score(ro.mean, ro.var, x, sl), score(rp.mean, rp.var, x, sl))
        for r in rows:
            if r["phase"] == nm:
                r["track_ode"], r["track_parent"] = to, tp
        print(f"{nm:>14} | {to['nll']:8.3f} {tp['nll']:8.3f} "
              f"{to['nll']-tp['nll']:+7.3f} | {to['calibration']:8.2f} "
              f"{tp['calibration']:8.2f} | {to['claimed_sd']:7.2f} "
              f"{tp['claimed_sd']:7.2f} | {to['mse']/tp['mse']:9.3f}")

    flips = [r for r in rows
             if (r["ode"]["mse"] / r["parent"]["mse"] > 1.0)
             and (r["ode"]["nll"] < r["parent"]["nll"])]
    if flips:
        print("\n  === phases where MSE and log-loss DISAGREE ===")
        for r in flips:
            print(f"  {r['phase']:>14}: MSE ratio "
                  f"{r['ode']['mse']/r['parent']['mse']:.3f} (worse) but "
                  f"log-loss {r['ode']['nll']-r['parent']['nll']:+.3f} nats "
                  f"(better).  odefilter claimed SD "
                  f"{r['ode']['claimed_sd']:.1f} vs the parent's "
                  f"{r['parent']['claimed_sd']:.1f}; calibration "
                  f"{r['ode']['calibration']:.2f} vs "
                  f"{r['parent']['calibration']:.2f}.")
    else:
        print("\n  No phase flips: MSE and log-loss agree everywhere here.")

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0))
    sel = [r for r in rows if r["phase"] != "all"]
    xs = np.arange(len(sel))
    names = [r["phase"].replace(" ", "\n") for r in sel]

    ax = axes[0]
    ax.bar(xs - 0.2, [r["ode"]["nll"] for r in sel], width=0.4,
           color=ts.SERIES[0], label="odefilter")
    ax.bar(xs + 0.2, [r["parent"]["nll"] for r in sel], width=0.4,
           color=ts.SERIES[1], label="parent")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylabel("log-loss (nats/pt)")
    ax.set_title(f"The score that sees the whole forecast ($h={h}$)")
    ax.legend(fontsize=8)
    ts.tidy(ax)

    ax = axes[1]
    ax.axhline(1.0, color=ts.INK, lw=1.2, ls="--", zorder=0)
    ax.bar(xs - 0.2, [r["ode"]["calibration"] for r in sel], width=0.4,
           color=ts.SERIES[0], label="odefilter")
    ax.bar(xs + 0.2, [r["parent"]["calibration"] for r in sel], width=0.4,
           color=ts.SERIES[1], label="parent")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylabel("$E[e^2/S]$   (1 = honest)")
    ax.set_title("Was it wrong, or wrong *and confident*?")
    ax.legend(fontsize=8)
    ts.tidy(ax)

    ax = axes[2]
    r_mse = [r["ode"]["mse"] / r["parent"]["mse"] for r in sel]
    d_nll = [r["ode"]["nll"] - r["parent"]["nll"] for r in sel]
    ax.axhline(0, color=ts.INK, lw=1.0, zorder=0)
    ax.axvline(1, color=ts.INK, lw=1.0, ls="--", zorder=0)
    ax.scatter(r_mse, d_nll, s=70, color=ts.SERIES[0], zorder=3)
    for i, r in enumerate(sel):
        ax.annotate(r["phase"], (r_mse[i], d_nll[i]), xytext=(5, 4),
                    textcoords="offset points", fontsize=7.5, color=ts.INK2)
    ax.set_xlabel("MSE ratio (>1: point forecast worse)")
    ax.set_ylabel("log-loss difference (<0: distribution better)")
    ax.set_title("Lower-right = wrong, but honestly so")
    ts.tidy(ax)
    ts.save(fig, os.path.join(HERE, "figures", "fig22-distributional-score.png"))

    with open(os.path.join(HERE, "figures", "ode034.json"), "w") as f:
        json.dump(dict(h=h, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
