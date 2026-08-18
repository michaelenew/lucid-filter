"""0019 -- Does drift matter in proportion to its alignment with the metric?

0018's five conditions fall into a pattern that predicts all of them.  At base B
the information metric Gt = [[g0,g1],[g1,g0]] has principal axes (1,1)/sqrt2 at
psi = pi/4 = 0.785 (eigenvalue 7.04) and (1,-1)/sqrt2 at psi = 3pi/4 = 2.356
(eigenvalue 0.816) -- an 8.6:1 ratio.  Sorting the measurements by the
generating drift's angle to the HIGH-information axis:

    psi_true   angle to 0.785   headroom (mnats/pt)   shape evidence
    0.9        0.115            (not measured)        6.69   (0015)
    1.5        0.715            +3.58                 2.78   (0018)
    2.5        1.715 -> 0.144 from the LOW axis  -1.37   0.60   (0018)

and at base A, where the metric is isotropic AND smaller (g0 = 1.46 against
3.93), the headroom is negative for every psi_true tested.

The law this suggests:

    **Drift is worth modelling, and is measurable, in proportion to its
    alignment with the information metric's principal axis.**

That resurrects the Fisher metric in a different role from the one 0014
refuted.  It is not a law for how alpha MOVES -- that was refuted.  It is the
law for which movements can be SEEN and which are worth tracking.

Three points is a pattern, not a law.  This sweeps psi_true around the circle at
base B, including both metric axes exactly, and reports headroom, shape
evidence and psi-hat at each.  Predictions, falsifiable:

  1. headroom and evidence peak at psi_true = 0.785 and trough at 2.356;
  2. psi-hat lands on the grid node nearest psi_true throughout -- which also
     supplies the cell missing from 0018, psi_true = 0.9 profiled at tau = 4,
     the one 0015 got apparently wrong at tau = 8.
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
_p = import_module("0015_is_the_drift_shape_estimable")
_o = import_module("0017_orientation_at_an_isotropic_metric")
_h = import_module("0018_headroom_check")

BASE = (0.75, 0.60)
NUS = [0.010, 0.025, 0.050]
PSIS = np.linspace(0.0, np.pi, 6, endpoint=False)        # kernel grid
PSI_TRUE = [0.262, 0.785, 0.900, 1.309, 1.833, 2.356, 2.880]
TAU_TRUE = 4.0
H = 5


def main():
    n, R, kappa, Q = 1000, 5, 0.35, 1.0
    al = _m.build_grid()
    g = _m.Grid2(al)
    a0 = _m.alpha_osc(*BASE)
    g0, g1, r1, cond = _o.metric(a0)
    G = np.array([[g0, g1], [g1, g0]])
    w, V = np.linalg.eigh(G)
    hi = V[:, int(np.argmax(w))]
    psi_hi = float(np.arctan2(hi[1], hi[0]) % np.pi)
    print(f"base alpha = {np.round(a0,4)}   r1 = {r1:.4f}   metric cond "
          f"= {cond:.2f}")
    print(f"metric axes: high-info psi = {psi_hi:.3f} (eig {w.max():.3f}), "
          f"low-info psi = {(psi_hi + np.pi/2) % np.pi:.3f} "
          f"(eig {w.min():.3f})\n")

    Tstat = _m.build_T(al, "static", 0.0)
    Tiso = [_p.build_T_const(al, nu, 1.0, 0.0) for nu in NUS]
    Tsh = {(nu, float(p)): _p.build_T_const(al, nu, TAU_TRUE, float(p))
           for nu in NUS for p in PSIS}
    print(f"grid {len(al)} nodes; {1 + len(Tiso) + len(Tsh)} kernels\n",
          flush=True)

    master = np.random.default_rng(161803)
    rows = []
    for pt in PSI_TRUE:
        rngref = np.random.default_rng(11)
        ap = _p.gen_path(a0, TAU_TRUE, pt, 20000, rngref)
        xr, _ = _p.simulate_path(ap, Q, 0.0, rngref)
        S2 = (kappa * np.std(np.diff(xr))) ** 2
        data = []
        for _ in range(R):
            rng = np.random.default_rng(master.integers(2 ** 63))
            ap = _p.gen_path(a0, TAU_TRUE, pt, n, rng)
            data.append(_p.simulate_path(ap, Q, S2, rng))

        st = _h.best_over(data, g, [Tstat], Q, S2)
        iso = _h.best_over(data, g, Tiso, Q, S2)
        prow = [(float(p), _h.best_over(data, g, [Tsh[(nu, float(p))]
                                                  for nu in NUS], Q, S2))
                for p in PSIS]
        psi_hat, shb = max(prow, key=lambda z: z[1][0])
        nearest = float(PSIS[int(np.argmin(np.abs(
            (PSIS - pt + np.pi / 2) % np.pi - np.pi / 2)))])
        # angle to the high-information axis, folded to [0, pi/2]
        dhi = abs((pt - psi_hi + np.pi / 2) % np.pi - np.pi / 2)
        rec = dict(psi_true=pt, angle_to_hi=float(dhi),
                   headroom=iso[0] - st[0], shape_evidence=shb[0] - iso[0],
                   psi_hat=psi_hat, psi_nearest=nearest,
                   hit=bool(abs(psi_hat - nearest) < 1e-9),
                   iso_mse_ratio=iso[1] / st[1], shape_mse_ratio=shb[1] / st[1],
                   psi_row=[[p, v[0]] for p, v in prow])
        rows.append(rec)
        pk = max(v[0] for _, v in prow)
        print(f"psi_true={pt:.3f} (angle to hi-info axis {dhi:.3f}): "
              f"headroom {(iso[0]-st[0])*1000:+7.2f}  shape "
              f"{(shb[0]-iso[0])*1000:+6.2f}  psi-hat {psi_hat:.3f} "
              f"(nearest {nearest:.3f}){'  HIT' if rec['hit'] else '  miss'}")
        print("            profile: " + "  ".join(
            f"{p:.2f}:{(v[0]-pk)*1000:+.1f}" for p, v in prow), flush=True)

    print("\n=== predictions ===")
    hits = sum(r["hit"] for r in rows)
    print(f"  1. psi-hat on the nearest node: {hits}/{len(rows)}")
    hr = np.array([r["headroom"] for r in rows])
    dh = np.array([r["angle_to_hi"] for r in rows])
    o = np.argsort(dh)
    print("  2. headroom against angle to the high-information axis "
          "(should decrease):")
    for i in o:
        print(f"       angle {dh[i]:.3f}  ->  headroom "
              f"{hr[i]*1000:+7.2f} mnats/pt   "
              f"(psi_true {rows[i]['psi_true']:.3f})")
    rho = np.corrcoef(dh, hr)[0, 1]
    print(f"     correlation(angle, headroom) = {rho:+.3f}")

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    ax = axes[0]
    ax.plot(dh[o], hr[o] * 1000, marker="o", color=ts.SERIES[0],
            label="isotropic drift over static")
    ax.plot(dh[o], np.array([r["shape_evidence"] for r in rows])[o] * 1000,
            marker="s", color=ts.SERIES[1], label="shape over isotropic")
    ax.axhline(0.0, color=ts.INK, lw=1.0)
    ax.set_xlabel("angle between the drift and the high-information axis")
    ax.set_ylabel("millinats/point")
    ax.set_title("Drift pays in proportion to alignment")
    ax.legend(fontsize=8)
    ts.tidy(ax)

    ax = axes[1]
    ax.plot([0, np.pi], [0, np.pi], color=ts.GRID, lw=1.2, zorder=0)
    ax.scatter([r["psi_true"] for r in rows], [r["psi_hat"] for r in rows],
               color=ts.SERIES[0], s=40, zorder=3)
    for r in rows:
        ax.plot([r["psi_true"]] * 2, [r["psi_true"], r["psi_hat"]],
                color=ts.SERIES[0], lw=1.0, alpha=0.5)
    for p in PSIS:
        ax.axhline(p, color=ts.GRID, lw=0.7, ls=":")
    ax.set_xlabel(r"generating $\psi_{\rm true}$")
    ax.set_ylabel(r"profile argmax $\hat\psi$")
    ax.set_title(r"Orientation recovery (dotted = kernel $\psi$ grid)")
    ts.tidy(ax)
    ts.save(fig, os.path.join(HERE, "figures", "fig14-alignment-law.png"))

    with open(os.path.join(HERE, "figures", "ode019.json"), "w") as f:
        json.dump(dict(rows=rows, base=list(BASE), psi_hi=psi_hi,
                       eig=[float(w.min()), float(w.max())],
                       tau_true=TAU_TRUE, psis=PSIS.tolist(), nus=NUS,
                       n=n, R=R, kappa=kappa, horizon=H), f, indent=1)


if __name__ == "__main__":
    main()
