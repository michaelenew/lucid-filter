"""0018 -- The two controls 0017 was missing.

Control 1 -- headroom.

0017 finds essentially no drift-shape evidence at the isotropic-metric base A
(0.00 and 0.21 millinats/point, against 6.69 at base B in 0015).  That admits
two very different readings:

  1. the metric is what makes shape readable, and with an isotropic metric
     there is nothing to read -- an identifiability statement;
  2. base A has no adaptivity headroom AT ALL, so there is no benefit for the
     shape to improve on -- in which case 0017 measured nothing.

0017 compares shapes against the ISOTROPIC kernel, never against a STATIC one,
so it cannot separate these.  This does.  For each condition: static, the best
isotropic kernel, and the best shaped kernel, on loglik and h-step forecast MSE.

If the isotropic kernel beats static at base A by a normal amount and only the
SHAPE is invisible, reading 1 holds.  If isotropic ties static at base A, then
0017's null is uninformative and the orientation question is still open.

Control 2 -- a third orientation at base B.  0015 (psi_true = 0.9) and 0017
(psi_true = 1.5) BOTH returned psi-hat = 1.571 at base B.  Two points suggest
the answer is a fixed property of the base point rather than a reading of the
drift, but 1.5 is itself close to 1.571, so the two points are not independent
evidence.  A third generating orientation far from pi/2 settles it: psi_true =
2.5.  If psi-hat is 1.571 again, the orientation channel is reading the base
point, not the truth.

The psi profile at tau = 4 comes free from the kernels the headroom check
already needs.
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

NUS = [0.010, 0.025, 0.050]
PSIS = np.linspace(0.0, np.pi, 6, endpoint=False)
SHAPES = [(4.0, float(p)) for p in PSIS]
EXTRA = [("B (anisotropic metric)", 4.0, 2.5)]
H = 5


def best_over(data, g, kernels, Q, S2):
    out = None
    for T in kernels:
        lls, sc = [], []
        for x, y in data:
            ll, fc, _ = _m.run(y, g, T, Q, S2)
            lls.append(ll / len(y))
            sc.append(_m.score(fc, x)[H])
        v = (float(np.mean(lls)), float(np.mean(sc)),
             float(np.std(sc) / np.sqrt(len(sc))))
        if out is None or v[0] > out[0]:
            out = v
    return out


def main():
    n, R, kappa, Q = 1000, 5, 0.35, 1.0
    al = _m.build_grid()
    g = _m.Grid2(al)
    Tstat = _m.build_T(al, "static", 0.0)
    Tiso = [_p.build_T_const(al, nu, 1.0, 0.0) for nu in NUS]
    Tshape = {(nu, t, p): _p.build_T_const(al, nu, t, p)
              for nu in NUS for t, p in SHAPES}
    print(f"grid {len(al)} nodes; {1 + len(Tiso) + len(Tshape)} kernels\n")

    master = np.random.default_rng(3141592)      # same stream as 0017
    rows = []
    for bname, tau_t, psi_t in list(_o.COND) + EXTRA:
        a0 = _m.alpha_osc(*_o.BASES[bname])
        rngref = np.random.default_rng(11)
        ap = _p.gen_path(a0, tau_t, psi_t, 20000, rngref)
        xr, _ = _p.simulate_path(ap, Q, 0.0, rngref)
        S2 = (kappa * np.std(np.diff(xr))) ** 2
        data = []
        for _ in range(R):
            rng = np.random.default_rng(master.integers(2 ** 63))
            ap = _p.gen_path(a0, tau_t, psi_t, n, rng)
            data.append(_p.simulate_path(ap, Q, S2, rng))

        st = best_over(data, g, [Tstat], Q, S2)
        iso = best_over(data, g, Tiso, Q, S2)
        sh = best_over(data, g, list(Tshape.values()), Q, S2)
        # psi profile at tau = 4, nu maximised out
        prow = []
        for psi in PSIS:
            v = best_over(data, g, [Tshape[(nu, 4.0, float(psi))]
                                    for nu in NUS], Q, S2)
            prow.append((float(psi), v[0]))
        psi_hat = max(prow, key=lambda z: z[1])[0]
        rec = dict(base=bname, tau_true=tau_t, psi_true=psi_t,
                   static_ll=st[0], iso_ll=iso[0], shape_ll=sh[0],
                   static_mse=st[1], iso_mse=iso[1], shape_mse=sh[1],
                   iso_mse_se=iso[2],
                   iso_over_static_nats=iso[0] - st[0],
                   shape_over_iso_nats=sh[0] - iso[0],
                   psi_hat=psi_hat,
                   psi_row=[[a, b] for a, b in prow])
        rows.append(rec)
        print(f"--- {bname} | tau={tau_t:.0f} psi={psi_t} ---")
        print(f"  loglik/pt   static {st[0]:.5f}   iso {iso[0]:.5f}   "
              f"shape {sh[0]:.5f}")
        print(f"  iso over static: {(iso[0]-st[0])*1000:8.2f} millinats/pt"
              f"   shape over iso: {(sh[0]-iso[0])*1000:6.2f}")
        print(f"  h={H} MSE   static {st[1]:.3f}   iso {iso[1]:.3f} "
              f"({iso[1]/st[1]:.4f})   shape {sh[1]:.3f} ({sh[1]/st[1]:.4f})")
        pk = max(v for _, v in prow)
        print(f"  psi profile at tau=4 (truth {psi_t}), argmax {psi_hat:.3f}: "
              + "  ".join(f"{a:.2f}:{(b-pk)*1000:+.1f}" for a, b in prow))
        print(flush=True)

    print("=== verdict ===")
    for r in rows:
        tag = "A" if r["base"].startswith("A") else "B"
        print(f"  {tag} psi_true={r['psi_true']}: headroom "
              f"{r['iso_over_static_nats']*1000:7.2f} mnats/pt "
              f"(MSE {r['iso_mse']/r['static_mse']:.4f});  shape adds "
              f"{r['shape_over_iso_nats']*1000:6.2f};  psi-hat "
              f"{r['psi_hat']:.3f}")
    bs = [r for r in rows if r["base"].startswith("B")]
    print("\n  base B psi-hat against psi_true: "
          + ", ".join(f"{r['psi_true']} -> {r['psi_hat']:.3f}" for r in bs)
          + "   (0015 also had 0.9 -> 1.571)")

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    lab = [("A" if r["base"].startswith("A") else "B")
           + f"  $\\psi$={r['psi_true']}" for r in rows]
    xs = np.arange(len(rows))
    ax.bar(xs - 0.19, [r["iso_over_static_nats"] * 1000 for r in rows],
           width=0.36, color=ts.SERIES[0], label="isotropic drift over static")
    ax.bar(xs + 0.19, [r["shape_over_iso_nats"] * 1000 for r in rows],
           width=0.36, color=ts.SERIES[1], label="shape over isotropic")
    ax.axhline(0.0, color=ts.INK, lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(lab, fontsize=8)
    ax.set_ylabel("millinats/point")
    ax.set_title("Headroom, and how much of it the shape can reach")
    ax.legend(fontsize=8)
    ts.tidy(ax)
    ts.save(fig, os.path.join(HERE, "figures", "fig13-headroom.png"))

    with open(os.path.join(HERE, "figures", "ode018.json"), "w") as f:
        json.dump(dict(rows=rows, n=n, R=R, kappa=kappa, horizon=H,
                       nus=NUS), f, indent=1)


if __name__ == "__main__":
    main()
