"""0013 -- A measured minimax over the direction the dynamics move in.

0011 section 3 rests the whole case for the invariant drift law on a worst case
taken over THREE hand-chosen scenarios ("damping shift", "frequency shift", "no
shift").  Three points chosen by me is not a worst case; it is an anecdote with
a min() on it.  The proper object sweeps the direction of the shift.

Fix a base alpha_0 and a shift magnitude r, and move

    alpha_1 = alpha_0 + r (cos psi, sin psi)

around the circle.  At each angle, compare the drift laws at their own
marginal-likelihood nu, against a static filter and against an oracle that knows
alpha_1.  Then read off the worst angle for each law.  A law that is better in
the minimax sense is better at ITS worst angle than the other is at ITS worst
angle -- which is the notion filter-optimality-proof adopted, "since the premise
is that no prior over the class is available."  A direction in alpha-space is
precisely such a prior.

Two base points, because the answer turned out to depend on one of them.

  interior   rho = 0.85, theta = 0.50, r = 0.15
  boundary   rho = 0.95, theta = 0.35, r = 0.08  -- 0008's base point

Both are chosen so the whole circle stays admissible; otherwise the infeasible
wedge, not the metric, would decide the worst case.  That is what forces the
smaller radius at the boundary point, where alpha_1 + alpha_2 = 0.882 is already
close to the stationarity limit of 1.  The Fisher metric's anisotropy
|gamma_1|/gamma_0 = alpha_1/(1-alpha_2) is 0.866 at the interior point and 0.938
at the boundary one, so if the invariant law's advantage is a near-boundary
effect this pair should show it.
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
_m = import_module("0008_anisotropy_at_p2")

BASES = {"interior": ((0.85, 0.50), 0.15),
         "boundary": ((0.95, 0.35), 0.08)}
ANGLES = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
NUS = [0.004, 0.010, 0.025, 0.060]
METHODS = ["iso", "fisher-shape"]
H = 5


def stationary(a):
    return abs(a[1]) < 0.999 and a[0] + a[1] < 0.999 and a[1] - a[0] < 0.999


def sweep(al, g, Ts, Tstat, base, radius, n, R, kappa, Q, master):
    a0 = _m.alpha_osc(*base)
    print(f"\nbase alpha = {np.round(a0, 4)}  r = {radius}  "
          f"anisotropy |g1|/g0 = {a0[0] / (1 - a0[1]):.4f}")
    rows = []
    for psi in ANGLES:
        a1 = a0 + radius * np.array([np.cos(psi), np.sin(psi)])
        if not stationary(a1):
            print(f"psi={psi:.3f} infeasible, skipped")
            continue
        ref, _ = _m.simulate(a0, a1, 20000, Q, 0.0, np.random.default_rng(2))
        S2 = (kappa * np.std(np.diff(ref))) ** 2
        data = [_m.simulate(a0, a1, n, Q, S2,
                            np.random.default_rng(master.integers(2 ** 63)))
                for _ in range(R)]

        stat = float(np.mean([_m.score(_m.run(y, g, Tstat, Q, S2)[1], x)[H]
                              for x, y in data]))
        j = int(np.argmin(((al - a1) ** 2).sum(1)))
        pin = __import__("scipy.sparse", fromlist=["x"]).csr_matrix(
            (np.ones(len(al)), (np.arange(len(al)), np.full(len(al), j))),
            shape=(len(al),) * 2)
        orac = float(np.mean([_m.score(_m.run(y, g, pin, Q, S2)[1], x)[H]
                              for x, y in data]))

        rec = dict(psi=float(psi), a1=a1.tolist(), static=stat, oracle=orac)
        for mth in METHODS:
            best, bl = None, -np.inf
            for nu in NUS:
                lls, sc = [], []
                for x, y in data:
                    ll, fc, _ = _m.run(y, g, Ts[(mth, nu)], Q, S2)
                    lls.append(ll / n)
                    sc.append(_m.score(fc, x)[H])
                if np.mean(lls) > bl:
                    bl = float(np.mean(lls))
                    best = (float(nu), float(np.mean(sc)),
                            float(np.std(sc) / np.sqrt(R)))
            rec[mth] = dict(nu=best[0], mse=best[1], se=best[2], ll=bl,
                            closed=(stat - best[1]) / max(stat - orac, 1e-12))
        rows.append(rec)
        print(f"psi={psi:5.3f}  static {stat:7.3f}  oracle {orac:7.3f}  " +
              "  ".join(f"{m} {rec[m]['mse']:7.3f} ({rec[m]['closed']:+.2f}, "
                        f"nu={rec[m]['nu']:.3f})" for m in METHODS))

    summary = {}
    for mth in METHODS:
        c = np.array([r[mth]["closed"] for r in rows])
        summary[mth] = dict(best=float(c.max()), median=float(np.median(c)),
                            worst=float(c.min()),
                            worst_psi=float(rows[int(np.argmin(c))]["psi"]))
    return rows, summary


def main():
    n, R, kappa, Q = 1000, 6, 0.35, 1.0
    al = _m.build_grid()
    g = _m.Grid2(al)
    print(f"grid {len(al)} nodes")
    Ts = {(mth, nu): _m.build_T(al, mth, nu) for mth in METHODS for nu in NUS}
    Tstat = _m.build_T(al, "static", 0.0)
    master = np.random.default_rng(112358)

    allrows, allsum = {}, {}
    for name, (base, radius) in BASES.items():
        allrows[name], allsum[name] = sweep(al, g, Ts, Tstat, base, radius,
                                            n, R, kappa, Q, master)

    print("\n=== minimax over direction: fraction of static-to-oracle gap "
          "closed ===")
    print(f"{'base':>10} {'method':>14} {'best':>8} {'median':>9} {'WORST':>9}")
    print("-" * 54)
    for name in BASES:
        for mth in METHODS:
            s_ = allsum[name][mth]
            print(f"{name:>10} {mth:>14} {s_['best']:8.3f} "
                  f"{s_['median']:9.3f} {s_['worst']:9.3f}")

    # ------------------------------------------------------------- figure
    fig = plt.figure(figsize=(11.5, 4.0))
    for k, name in enumerate(BASES):
        rows = allrows[name]
        ax = fig.add_subplot(1, 3, k + 1, projection="polar")
        for i, mth in enumerate(METHODS):
            p = [r["psi"] for r in rows] + [rows[0]["psi"] + 2 * np.pi]
            c = [max(r[mth]["closed"], 0.0) for r in rows]
            ax.plot(p, c + [c[0]], marker="o", color=ts.SERIES[i + 1], label=mth)
        ax.set_title(f"{name} base", pad=18)
        ax.set_rlim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.18))

    ax = fig.add_subplot(1, 3, 3)
    for k, name in enumerate(BASES):
        for i, mth in enumerate(METHODS):
            c = np.array([r[mth]["closed"] for r in allrows[name]])
            ax.plot(np.sort(c), np.linspace(0, 1, len(c)),
                    marker="o" if k == 0 else "s",
                    ls="-" if k == 0 else "--",
                    color=ts.SERIES[i + 1], label=f"{name[:4]} {mth}")
    ax.set_xlabel("fraction of static-to-oracle gap closed")
    ax.set_ylabel("fraction of directions at or below")
    ax.set_title("Minimax is the left-hand edge")
    ax.legend(fontsize=7)
    ts.tidy(ax)
    ts.save(fig, os.path.join(HERE, "figures", "fig10-minimax-directions.png"))

    with open(os.path.join(HERE, "figures", "ode013.json"), "w") as f:
        json.dump(dict(rows=allrows, summary=allsum,
                       bases={k: [list(v[0]), v[1]] for k, v in BASES.items()},
                       n=n, R=R, kappa=kappa, horizon=H), f, indent=1)


if __name__ == "__main__":
    main()
