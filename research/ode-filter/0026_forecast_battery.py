"""0026 -- Does modelling the dynamics actually pay?  The candidate against the parent.

Everything in 0021-0025 is about what is distinguishable.  This is the first
measurement of whether any of it buys accuracy, and it is the comparison the
workstream exists to make: `odefilter` (which models the recurrence) against the
parent `statfilter` (which models a random walk) on data that genuinely has
second-order dynamics.

Per 0006 the comparison must be a FORECAST comparison -- tracking error is
nearly blind to alpha -- so everything is scored at h = 1, 5, 20.

Two datasets:

  ODE     unit root plus a damped oscillator: the target class.  The parent's
          model is wrong here, and the question is by how much.
  WALK    a plain random walk with noise: the parent's own model, where
          odefilter's extra two orders are pure overhead.  Adaptivity being
          nearly free when it is not needed is the parent's own standard
          (1.001-1.005 on stationary diffusions) and this workstream should
          meet it.

Baselines: the parent, fitted; and last-value, which is the h-step-optimal
forecast for a random walk and therefore the honest floor.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "lucid"))
sys.path.insert(0, os.path.join(ROOT, "lucid"))
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from odefilter import OdeFilter  # noqa: E402
import statfilter  # noqa: E402

HORIZONS = (1, 5, 20)
ALPHA3 = np.array([2.785218519281637, -2.6855430450862655, 0.9003245225862656])


def gen(kind, n, Q, S2, rng):
    if kind == "WALK":
        x = np.cumsum(np.sqrt(Q) * rng.standard_normal(n))
        return x, x + np.sqrt(S2) * rng.standard_normal(n)
    z = np.zeros(3)
    x = np.zeros(n)
    for t in range(n):
        xn = float(ALPHA3 @ z) + np.sqrt(Q) * rng.standard_normal()
        z = np.concatenate([[xn], z[:-1]])
        x[t] = xn
    return x, x + np.sqrt(S2) * rng.standard_normal(n)


def score(fc, x, lo):
    """Forecast MSE by horizon over t >= lo."""
    out = {}
    for h, d in fc.items():
        ts_ = [t for t in d if t >= lo and t + h < len(x)]
        out[h] = float(np.mean([(d[t] - x[t + h]) ** 2 for t in ts_]))
    return out


def forecasts_ode(f, y, lo):
    f.reset()
    fc = {h: {} for h in HORIZONS}
    for t, v in enumerate(y):
        f.update(v)
        if t >= lo:
            for h in HORIZONS:
                fc[h][t] = f.predict(h)[0]
    return fc


def forecasts_parent(f, y, lo):
    f.reset()
    fc = {h: {} for h in HORIZONS}
    for t, v in enumerate(y):
        f.update(v)
        if t >= lo:
            m = f.predict(1)[0]        # a random walk's forecast is flat in h
            for h in HORIZONS:
                fc[h][t] = m
    return fc


def forecasts_naive(y, lo):
    return {h: {t: y[t] for t in range(lo, len(y))} for h in HORIZONS}


def main():
    n, R, Q = 900, 3, 1.0
    lo = n // 2
    master = np.random.default_rng(24601)
    rows = []

    for kind in ("ODE", "WALK"):
        for kappa in (0.25, 1.0):
            xr, _ = gen(kind, 40000, Q, 0.0, np.random.default_rng(3))
            S2 = (kappa * float(np.std(np.diff(xr)))) ** 2
            acc = {m: {h: [] for h in HORIZONS} for m in ("ode", "parent", "naive")}
            info = []
            for r in range(R):
                rng = np.random.default_rng(master.integers(2 ** 63))
                x, y = gen(kind, n, Q, S2, rng)

                t0 = time.time()
                fo = OdeFilter.fit(y, p=3, order=5, max_iter=250)
                t1 = time.time()
                fp = statfilter.AdaptiveFilter.fit(y, order=5, max_iter=250)
                t2 = time.time()

                for m, fc in (("ode", forecasts_ode(fo, y, lo)),
                              ("parent", forecasts_parent(fp, y, lo)),
                              ("naive", forecasts_naive(y, lo))):
                    s = score(fc, x, lo)
                    for h in HORIZONS:
                        acc[m][h].append(s[h])
                info.append(dict(fit_ode_s=t1 - t0, fit_parent_s=t2 - t1,
                                 roots=[[float(z.real), float(z.imag)]
                                        for z in fo.params.roots],
                                 memory=fo.params.memory(),
                                 Q=fo.params.Q, s2=fo.params.s2,
                                 s_P=fo.params.s_P, s_M=fo.params.s_M))
                print(f"  {kind} k={kappa} seed {r}: fit {t1-t0:.0f}s/"
                      f"{t2-t1:.0f}s  |z|max={max(abs(complex(*z)) for z in info[-1]['roots']):.4f}",
                      flush=True)

            rec = dict(kind=kind, kappa=kappa, info=info)
            for m in acc:
                for h in HORIZONS:
                    v = np.array(acc[m][h])
                    rec[f"{m}_h{h}"] = float(v.mean())
                    rec[f"{m}_h{h}_se"] = float(v.std() / np.sqrt(len(v)))
            rows.append(rec)

    print("\n=== forecast MSE, ratio to the parent (lower is better) ===")
    hdr = (f"{'data':>5} {'kappa':>6} {'method':>7} "
           + " ".join(f"{'h='+str(h):>18}" for h in HORIZONS))
    print(hdr + "\n" + "-" * len(hdr))
    for rec in rows:
        for m in ("parent", "naive", "ode"):
            cells = []
            for h in HORIZONS:
                v, b = rec[f"{m}_h{h}"], rec[f"parent_h{h}"]
                cells.append(f"{v:10.2f}({v/b:5.3f})")
            print(f"{rec['kind']:>5} {rec['kappa']:6.2f} {m:>7} " + " ".join(cells))
        print()

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), sharey=True)
    for ax, kind in zip(axes, ("ODE", "WALK")):
        sel = [r for r in rows if r["kind"] == kind]
        for i, kap in enumerate([r["kappa"] for r in sel]):
            rec = sel[i]
            ys = [rec[f"ode_h{h}"] / rec[f"parent_h{h}"] for h in HORIZONS]
            ax.plot(HORIZONS, ys, marker="o", color=ts.SERIES[i],
                    label=fr"$\kappa$={kap}")
        ax.axhline(1.0, color=ts.INK, lw=1.0, ls="--", zorder=0)
        ax.set_xscale("log")
        ax.set_xticks(list(HORIZONS))
        ax.set_xticklabels([str(h) for h in HORIZONS])
        ax.set_xlabel("forecast horizon h")
        ax.set_title(f"{kind} data")
        ts.tidy(ax)
    axes[0].set_ylabel("odefilter MSE / parent MSE")
    axes[0].legend(fontsize=8)
    ts.save(fig, os.path.join(HERE, "figures", "fig18-forecast-battery.png"))

    with open(os.path.join(HERE, "figures", "ode026.json"), "w") as f:
        json.dump(dict(rows=rows, n=n, R=R, horizons=list(HORIZONS)), f, indent=1)


if __name__ == "__main__":
    main()
