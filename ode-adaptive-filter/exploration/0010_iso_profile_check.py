"""0010 -- Robustness check on 0008's sharpest number.

0008 reports that on the damping shift the ISOTROPIC drift chose nu* = 0 --
marginal likelihood declined to adapt at all, closing 0% of the static-to-oracle
gap, while the Fisher-shaped drift closed 70%.  That single cell carries the
minimax reading of the whole comparison, and 0008's nu scan had six points.

This profiles both kernels on a fine nu grid over the damping scenario, so the
claim rests on a profile rather than on a scan landing between points.
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


def main():
    n, R, kappa, Q = 1000, 8, 0.35, 1.0
    al = _m.build_grid()
    g = _m.Grid2(al)
    pre, post = _m.SCEN["damping 0.95->0.85"]
    a_pre, a_post = _m.alpha_osc(*pre), _m.alpha_osc(*post)
    ref, _ = _m.simulate(a_pre, a_post, 20000, Q, 0.0, np.random.default_rng(2))
    S2 = (kappa * np.std(np.diff(ref))) ** 2
    master = np.random.default_rng(90210)      # same stream as 0008
    data = [_m.simulate(a_pre, a_post, n, Q, S2,
                        np.random.default_rng(master.integers(2**63)))
            for _ in range(R)]

    nus = np.concatenate([[0.0], np.logspace(np.log10(0.002), np.log10(0.25), 16)])
    prof = {}
    for method in ("iso", "fisher-shape"):
        rows = []
        for nu in nus:
            T = _m.build_T(al, method, nu)
            lls, h5 = [], []
            for x, y in data:
                ll, fc, _ = _m.run(y, g, T, Q, S2)
                lls.append(ll / n)
                h5.append(_m.score(fc, x)[5])
            rows.append(dict(nu=float(nu), ll=float(np.mean(lls)),
                             h5=float(np.mean(h5)),
                             h5_se=float(np.std(h5) / np.sqrt(R))))
            print(f"{method:>13} nu={nu:7.4f} ll={rows[-1]['ll']:.5f} "
                  f"h5={rows[-1]['h5']:.4f}")
        prof[method] = rows

    print()
    for method, rows in prof.items():
        i = int(np.argmax([r["ll"] for r in rows]))
        j = int(np.argmin([r["h5"] for r in rows]))
        print(f"{method:>13}: loglik-optimal nu = {rows[i]['nu']:.4f} "
              f"(h5 {rows[i]['h5']:.3f}); "
              f"h5-optimal nu = {rows[j]['nu']:.4f} (h5 {rows[j]['h5']:.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for i, method in enumerate(prof):
        r = prof[method]
        axes[0].plot([q["nu"] for q in r], [q["ll"] for q in r], marker="o",
                     color=ts.SERIES[i + 1], label=method)
        axes[1].errorbar([q["nu"] for q in r], [q["h5"] for q in r],
                         yerr=[q["h5_se"] for q in r], marker="o",
                         color=ts.SERIES[i + 1], capsize=2, label=method)
    for ax, lab in zip(axes, ["log-likelihood per point", "h=5 forecast MSE"]):
        ax.set_xscale("symlog", linthresh=2e-3)
        ax.set_xlabel(r"drift SD $\nu$")
        ax.set_ylabel(lab)
        ax.legend()
        ts.tidy(ax)
    axes[0].set_title("Damping shift: does isotropic drift ever pay?")
    ts.save(fig, os.path.join(HERE, "figures", "fig08-iso-profile.png"))

    with open(os.path.join(HERE, "figures", "ode010.json"), "w") as f:
        json.dump(prof, f, indent=1)


if __name__ == "__main__":
    main()
