"""0006 -- alpha barely shows up in tracking.  Measure it where it lives.

0005 compared drift coordinates on theta-MSE and found nothing: ratios
1.0000-1.0001 at |t| < 0.7, and allowing drift at all bought at most 1.3%.
That is not evidence that the coordinate does not matter -- it is evidence that
theta-MSE cannot see alpha.

Why it cannot.  The one-step filtering gain solves a Riccati equation that,
for p=1, depends on alpha only through how much prior variance survives one
step.  With Q and S2 fixed, moving a from 0.90 to 0.99 barely moves the
steady-state gain, so the level estimate barely moves either.  What alpha
controls is where the process is going, and that is invisible at lag 0.

    alpha is a FORECASTING parameter, not a filtering parameter.  A loss that
    only scores the current level is nearly blind to it.

This matters beyond the coordinate question: it says the ODE filter cannot be
tuned or validated on tracking error, and it is consistent with the previous
construction's reported behaviour -- decent tracking, "some predictive power out
to a few steps".  Tracking was never the part that depended on getting A right.

So: the same three drift coordinates, the same marginal-likelihood choice of
nu, scored on h-step forecast MSE for h = 1, 5, 20, at two noise levels.

The h-step forecast is taken under the current posterior over a, held fixed
across the horizon: E[x_{t+h}] = sum_a pi_a a^h m_t.  (Propagating the drift
across the horizon as well is defensible and slightly different; it is not what
distinguishes the coordinates, which is pi_a.)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from importlib import import_module  # noqa: E402
_m = import_module("0005_fisher_vs_coefficient_drift")
AGRID, COORD, transition, simulate = _m.AGRID, _m.COORD, _m.transition, _m.simulate

HORIZONS = (1, 5, 20)


def run_fc(y, T, Q, S2):
    """As 0005's run(), plus h-step forecast means."""
    G = len(AGRID)
    n = len(y)
    pi = np.full(G, 1.0 / G)
    m, P = 0.0, 1e6
    ll = 0.0
    fc = {h: np.full(n, np.nan) for h in HORIZONS}
    ah_pow = {h: AGRID ** h for h in HORIZONS}
    for t, yt in enumerate(y):
        pi = pi @ T
        m_a = AGRID * m
        P_a = AGRID ** 2 * P + Q
        S = P_a + S2
        e = yt - m_a
        lg = -0.5 * (np.log(S) + e * e / S)
        mx = lg.max()
        w = pi * np.exp(lg - mx)
        Z = w.sum()
        ll += np.log(Z) + mx - 0.5 * np.log(2 * np.pi)
        pi = w / Z
        K = P_a / S
        mpost = m_a + K * e
        m = float(pi @ mpost)
        P = float(pi @ (P_a * (1.0 - K) + (mpost - m) ** 2))
        for h in HORIZONS:
            fc[h][t] = float(pi @ ah_pow[h]) * m
    return ll, fc


def score(fc, x):
    """Forecast MSE per horizon, over the second half only (post-shift)."""
    n = len(x)
    lo = n // 2
    out = {}
    for h, f in fc.items():
        idx = np.arange(lo, n - h)
        out[h] = float(np.mean((f[idx] - x[idx + h]) ** 2))
    return out


SCEN = {
    "0.50 -> 0.90": (0.50, 0.90),
    "0.90 -> 0.99": (0.90, 0.99),
    "0.99 -> 0.90": (0.99, 0.90),
    "no shift, a=0.90": (0.90, 0.90),
}
NUS = np.concatenate([[0.0], np.logspace(-3.0, -0.5, 6)])


def main():
    n, R, Q = 1500, 16, 1.0
    master = np.random.default_rng(4242)
    rows = []

    for kappa in (0.2, 0.5):
        for sname, (a0, a1) in SCEN.items():
            a_path = np.where(np.arange(n) < n // 2, a0, a1)
            ref, _ = simulate(a_path, Q, 0.0, np.random.default_rng(1))
            S2 = (kappa * np.std(np.diff(ref))) ** 2
            data = [simulate(a_path, Q, S2,
                             np.random.default_rng(master.integers(2**63)))
                    for _ in range(R)]

            # oracle: alpha known exactly after the shift
            orc = transition(None, 0.0)
            j = int(np.argmin(np.abs(AGRID - a1)))
            pin = np.zeros((len(AGRID), len(AGRID)))
            pin[:, j] = 1.0
            oracle_sc = []
            for x, y in data:
                _, fc = run_fc(y, pin, Q, S2)
                oracle_sc.append(score(fc, x))

            for coord in COORD:
                best, best_ll = None, -np.inf
                for nu in ([0.0] if coord == "static" else NUS):
                    T = transition(coord, nu)
                    lls, scs = [], []
                    for x, y in data:
                        ll, fc = run_fc(y, T, Q, S2)
                        lls.append(ll / n)
                        scs.append(score(fc, x))
                    if np.mean(lls) > best_ll:
                        best_ll = float(np.mean(lls))
                        best = dict(nu=float(nu), ll=best_ll,
                                    **{f"h{h}": float(np.mean([s[h] for s in scs]))
                                       for h in HORIZONS},
                                    **{f"h{h}_se": float(np.std([s[h] for s in scs])
                                                         / np.sqrt(R))
                                       for h in HORIZONS})
                rows.append(dict(kappa=kappa, scenario=sname, coord=coord, **best))
            rows.append(dict(kappa=kappa, scenario=sname, coord="oracle",
                             nu=float("nan"), ll=float("nan"),
                             **{f"h{h}": float(np.mean([s[h] for s in oracle_sc]))
                                for h in HORIZONS},
                             **{f"h{h}_se": float(np.std([s[h] for s in oracle_sc])
                                                  / np.sqrt(R)) for h in HORIZONS}))

    hdr = (f"{'k':>4} {'scenario':>18} {'coord':>10} {'nu*':>7} "
           + " ".join(f"{'h='+str(h):>16}" for h in HORIZONS))
    print(hdr + "\n" + "-" * len(hdr))
    for kappa in (0.2, 0.5):
        for sname in SCEN:
            base = [r for r in rows if r["kappa"] == kappa
                    and r["scenario"] == sname and r["coord"] == "static"][0]
            for coord in list(COORD) + ["oracle"]:
                r = [q for q in rows if q["kappa"] == kappa
                     and q["scenario"] == sname and q["coord"] == coord][0]
                cells = " ".join(f"{r[f'h{h}']:9.3f}({r[f'h{h}']/base[f'h{h}']:.3f})"
                                 for h in HORIZONS)
                print(f"{kappa:4.1f} {sname:>18} {coord:>10} {r['nu']:7.4f} {cells}")
            print()

    # paired arcsin vs a, on h=20, where the separation should be largest
    print("paired, arcsin(a) against a, h=20 forecast MSE (each at its own nu*):")
    for kappa in (0.2, 0.5):
        for sname, (a0, a1) in SCEN.items():
            a_path = np.where(np.arange(n) < n // 2, a0, a1)
            ref, _ = simulate(a_path, Q, 0.0, np.random.default_rng(1))
            S2 = (kappa * np.std(np.diff(ref))) ** 2
            nus = {c: [r for r in rows if r["kappa"] == kappa
                       and r["scenario"] == sname and r["coord"] == c][0]["nu"]
                   for c in ("a", "arcsin(a)")}
            Ts = {c: transition(c, nus[c]) for c in nus}
            rng = np.random.default_rng(777)
            d = []
            for _ in range(R):
                x, y = simulate(a_path, Q, S2, rng)
                e = {c: score(run_fc(y, Ts[c], Q, S2)[1], x)[20] for c in nus}
                d.append(e["arcsin(a)"] / e["a"] - 1.0)
            d = np.array(d)
            t = d.mean() / (d.std(ddof=1) / np.sqrt(R))
            print(f"  k={kappa}  {sname:>18}: ratio {1+d.mean():.4f}"
                  f" +-{d.std(ddof=1)/np.sqrt(R):.4f}  t = {t:+.2f}")
            rows.append(dict(kappa=kappa, scenario=sname, coord="paired h20",
                             ratio=float(1 + d.mean()),
                             se=float(d.std(ddof=1) / np.sqrt(R)), t=float(t)))

    # --------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    for ax, kappa in zip(axes, (0.2, 0.5)):
        xs = np.arange(len(SCEN))
        for i, coord in enumerate(["static", "a", "arcsin(a)", "oracle"]):
            ys = []
            for sname in SCEN:
                r = [q for q in rows if q.get("kappa") == kappa
                     and q.get("scenario") == sname and q.get("coord") == coord][0]
                b = [q for q in rows if q.get("kappa") == kappa
                     and q.get("scenario") == sname and q.get("coord") == "static"][0]
                ys.append(r["h20"] / b["h20"])
            ax.plot(xs, ys, marker="o", color=ts.SERIES[i], label=coord)
        ax.set_xticks(xs)
        ax.set_xticklabels(list(SCEN), rotation=20, ha="right")
        ax.axhline(1.0, color=ts.INK, lw=1.0, ls="--", zorder=0)
        ax.set_title(fr"$\kappa$ = {kappa}")
        ts.tidy(ax)
    axes[0].set_ylabel("h=20 forecast MSE, relative to static")
    axes[0].legend()
    ts.save(fig, os.path.join(HERE, "figures", "fig05-forecast-loss.png"))

    with open(os.path.join(HERE, "figures", "ode006.json"), "w") as f:
        json.dump(dict(rows=rows, n=n, R=R, horizons=list(HORIZONS)), f, indent=1)


if __name__ == "__main__":
    main()
