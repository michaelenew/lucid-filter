"""0017 -- Is the drift ORIENTATION readable when the metric cannot distort it?

0015 found the drift anisotropy's MAGNITUDE readable (6.69 millinats/point
against a 0.38 null floor) but its ORIENTATION not: the profile peaked at
psi = pi/2 against a generating 0.9.  Two candidate explanations were left
untested -- that the likelihood sees drift only through the information metric
Gamma, or that a narrow kernel is badly represented by a square grid.

The decisive test is to remove the metric from the picture.  For p = 2,

    Gt = [[g0, g1], [g1, g0]]   ->   eigenvalues g0 +- g1,
                                     eigenvectors (1, +-1)/sqrt(2),
                                     condition number (1 + r1)/(1 - r1)

where r1 = g1/g0 = a1/(1 - a2) is the process's own lag-1 autocorrelation.  So

    **the information metric's anisotropy IS the process's smoothness**,

the same r1 that sets the differencing cost (1 - r1) in 0011 section 1.  An
isotropic metric therefore means r1 = 0, which for a second-order system means
a1 = 0 exactly: poles at +- i rho, four samples per period.  There is no such
thing as an isotropic metric on a smooth process.  That caveat is the point of
this file as much as the measurement is -- a positive result here does NOT
transfer automatically to the smooth regime this workstream targets.

A controlled pair, sharing rho and a2 and differing only in a1:

    A  (rho, theta) = (0.75, pi/2)   alpha = ( 0.000, -0.5625)   r1 = 0.000
    B  (rho, theta) = (0.75, 0.60)   alpha = ( 1.238, -0.5625)   r1 = 0.792

Four conditions.  Conditions 2 and 3 are the paired comparison: same generating
orientation, same damping, different metric.

    1. A, tau = 4, psi_true = 0.5    does psi-hat track psi_true at A ...
    2. A, tau = 4, psi_true = 1.5    ... across two well-separated values?
    3. B, tau = 4, psi_true = 1.5    the metric is the only difference from 2
    4. A, tau = 1                    isotropic control, the null floor

Reading:
  psi-hat tracks truth at A but not at B   -> the metric explanation is right,
                                              orientation is readable after a
                                              known correction, and the
                                              correction is unavailable exactly
                                              where the workstream needs it
  psi-hat is the same for 1 and 2          -> orientation is not readable at
                                              all; the shape is one number
                                              (tau) and psi stays isotropic

Both generating orientations sit near a psi grid node (0.524 and 1.571), so a
correct answer is representable; failing to find a representable truth is
decisive in a way that failing to find an unrepresentable one is not.
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

BASES = {"A (isotropic metric)": (0.75, np.pi / 2),
         "B (anisotropic metric)": (0.75, 0.60)}
COND = [("A (isotropic metric)", 4.0, 0.5),
        ("A (isotropic metric)", 4.0, 1.5),
        ("B (anisotropic metric)", 4.0, 1.5),
        ("A (isotropic metric)", 1.0, 0.0)]

TAUS = [1.0, 2.0, 4.0, 8.0]
PSIS = np.linspace(0.0, np.pi, 6, endpoint=False)
NUS = [0.010, 0.025, 0.050]
H = 5


def metric(a):
    """(g0, g1, r1, condition number) of Gamma-tilde at alpha."""
    a1, a2 = a
    den = (1.0 + a2) * ((1.0 - a2) ** 2 - a1 ** 2)
    g0 = (1.0 - a2) / den
    g1 = a1 * g0 / (1.0 - a2)
    r1 = g1 / g0
    return g0, g1, r1, (1 + abs(r1)) / (1 - abs(r1))


def main():
    n, R, kappa, Q = 1000, 5, 0.35, 1.0
    al = _m.build_grid()
    g = _m.Grid2(al)
    print(f"grid {len(al)} nodes, step {_m.STEP}")
    for name, bt in BASES.items():
        a0 = _m.alpha_osc(*bt)
        g0, g1, r1, cond = metric(a0)
        print(f"  {name:<24} alpha = {np.round(a0, 4)}  "
              f"r1 = {r1:+.4f}  metric cond = {cond:.2f}")
    print(f"\ngenerating: OU phi={_p.PHI_GEN}, s={_p.S_GEN}; "
          f"axis SDs {_p.S_GEN*2:.3f}/{_p.S_GEN/2:.3f} at tau=4\n")

    print("building kernels...", end=" ", flush=True)
    Ts = {}
    for nu in NUS:
        Ts[(nu, 1.0, 0.0)] = _p.build_T_const(al, nu, 1.0, 0.0)
        for tau in TAUS[1:]:
            for psi in PSIS:
                Ts[(nu, tau, float(psi))] = _p.build_T_const(al, nu, tau,
                                                             float(psi))
    print(f"{len(Ts)} kernels", flush=True)

    master = np.random.default_rng(3141592)
    out, summary = {}, []
    for bname, tau_t, psi_t in COND:
        a0 = _m.alpha_osc(*BASES[bname])
        label = f"{bname} | tau={tau_t:.0f} psi={psi_t}"
        rngref = np.random.default_rng(11)
        ap = _p.gen_path(a0, tau_t, psi_t, 20000, rngref)
        xr, _ = _p.simulate_path(ap, Q, 0.0, rngref)
        S2 = (kappa * np.std(np.diff(xr))) ** 2
        data = []
        for _ in range(R):
            rng = np.random.default_rng(master.integers(2 ** 63))
            ap = _p.gen_path(a0, tau_t, psi_t, n, rng)
            data.append(_p.simulate_path(ap, Q, S2, rng))

        prof = {}
        for key, T in Ts.items():
            nu, tau, psi = key
            lls, sc = [], []
            for x, y in data:
                ll, fc, _ = _m.run(y, g, T, Q, S2)
                lls.append(ll / n)
                sc.append(_m.score(fc, x)[H])
            k = (tau, psi if tau > 1.0 else 0.0)
            v = (float(np.mean(lls)), float(np.mean(sc)), nu)
            if k not in prof or v[0] > prof[k][0]:
                prof[k] = v

        best = max(prof, key=lambda k: prof[k][0])
        iso = prof[(1.0, 0.0)]
        # psi profile at the best tau, and the psi that maximises it
        if best[0] > 1.0:
            row = sorted((p, prof[(best[0], p)][0])
                         for p in (k[1] for k in prof if k[0] == best[0]))
            span = max(v for _, v in row) - min(v for _, v in row)
        else:
            row, span = [], 0.0
        rec = dict(label=label, base=bname, tau_true=tau_t, psi_true=psi_t,
                   tau_hat=best[0], psi_hat=best[1], nu_hat=prof[best][2],
                   evidence=prof[best][0] - iso[0], psi_span=span,
                   mse_ratio=prof[best][1] / iso[1],
                   psi_row=[[float(p), float(v)] for p, v in row])
        summary.append(rec)
        out[label] = {f"{k[0]}|{k[1]:.4f}": list(v) for k, v in prof.items()}
        print(f"--- {label} ---")
        print(f"  argmax tau={best[0]:.0f} psi={best[1]:.3f} "
              f"(truth psi={psi_t})   nu*={prof[best][2]:.3f}")
        print(f"  evidence over isotropic: {rec['evidence']*1000:.2f} "
              f"millinats/pt   psi span {span*1000:.2f}   "
              f"h={H} MSE ratio {rec['mse_ratio']:.4f}")
        if row:
            print("  psi profile: " + "  ".join(
                f"{p:.2f}:{(v - max(vv for _, vv in row))*1000:+.1f}"
                for p, v in row) + "  (millinats/pt from the peak)")
        print(flush=True)

    print("=== verdict ===")
    # A psi-hat is only meaningful when the argmax is anisotropic; at tau-hat = 1
    # the member has no orientation and the reported 0.0 is a placeholder, not
    # an estimate.  Reporting it as one would fake a signal.
    def psi_str(r):
        return (f"{r['psi_hat']:.3f}" if r["tau_hat"] > 1.0
                else "n/a (tau-hat = 1, no orientation)")

    for r in summary:
        tag = "A" if r["base"].startswith("A") else "B"
        print(f"  {tag} psi_true={r['psi_true']}: evidence "
              f"{r['evidence']*1000:5.2f} mnats/pt  tau-hat={r['tau_hat']:.0f}"
              f"  psi-hat={psi_str(r)}")
    print("\n  0015's null floor on an isotropic-generated control was 0.38 "
          "mnats/pt;\n  anything at or below that is not a measurement.")

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    ax = axes[0]
    for i, r in enumerate([r for r in summary if r["tau_true"] > 1]):
        if not r["psi_row"]:
            continue
        pk = max(v for _, v in r["psi_row"])
        ax.plot([p for p, _ in r["psi_row"]],
                [(v - pk) * 1000 for _, v in r["psi_row"]],
                marker="o", color=ts.SERIES[i],
                label=f"{r['base'][0]}, "
                      + r"$\psi_{\rm true}$=" + f"{r['psi_true']}")
        ax.axvline(r["psi_true"], color=ts.SERIES[i], ls=":", lw=1.2)
    ax.set_xlabel(r"kernel major axis $\psi$")
    ax.set_ylabel("millinats/pt from the profile peak")
    ax.set_title(r"Does the $\psi$ profile peak at the truth?")
    ax.legend(fontsize=8)
    ts.tidy(ax)

    ax = axes[1]
    lab = [r["label"].split("|")[0].strip()[0] + f" {r['psi_true']}"
           for r in summary]
    ax.bar(range(len(summary)), [r["evidence"] * 1000 for r in summary],
           color=[ts.SERIES[0] if r["tau_true"] > 1 else ts.SERIES[3]
                  for r in summary])
    ax.axhline(0.0, color=ts.INK, lw=1.0)
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(lab, fontsize=8)
    ax.set_ylabel("millinats/pt over isotropic")
    ax.set_title("Shape evidence (orange = isotropic control)")
    ts.tidy(ax)
    ts.save(fig, os.path.join(HERE, "figures", "fig12-orientation-test.png"))

    with open(os.path.join(HERE, "figures", "ode017.json"), "w") as f:
        json.dump(dict(summary=summary, profiles=out,
                       bases={k: list(v) for k, v in BASES.items()},
                       taus=TAUS, psis=PSIS.tolist(), nus=NUS,
                       n=n, R=R, kappa=kappa, horizon=H), f, indent=1)


if __name__ == "__main__":
    main()
