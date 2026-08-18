"""0003 -- Where does the errors-in-variables damage land, and what is the prize?

0002 found something sharper than plain attenuation.  Regressing the observed
series on its own observed lags leaves the UNIT root essentially intact
(0.996-0.999 against a truth of 1) while destroying the oscillator: at
sigma/SD(dx) = 0.25 and above the complex pair collapses to two real roots
outright, so a lightly damped oscillation is read as over-damped relaxation.

Two questions.

A. Is the unit root's immunity superconsistency?  An integrated regressor has
   second moment O(n) while the measurement noise contributes O(1), so the
   attenuation ratio gamma/(gamma + S2) -> 1 for that direction alone.  If that
   is the mechanism, then a purely stationary target should show no such
   protection, and a pure random walk (the parent workstream's own model)
   should be almost unharmed.  Three truths, matched noise:

     offset only     p=1, root at 1          -- the parent filter's model
     oscillator only p=2, stationary pair    -- no integrated direction
     offset + osc.   p=3, both               -- the target class

B. How much is left on the table by IV?  IV throws away lags 1..p.  Exact
   Gaussian ML on the state-space form uses everything.  With (Q, S2) held at
   truth so that only the dynamics are in question, how far apart are they?
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from importlib import import_module  # noqa: E402
_m = import_module("0002_identifying_the_dynamics")
simulate, est_ols, est_iv = _m.simulate, _m.est_ols, _m.est_iv
alpha_from_ode, roots_of = _m.alpha_from_ode, _m.roots_of


# --------------------------------------------------------- exact Gaussian ML
def kalman_loglik(y, a, Q, S2, diffuse=1e8):
    """Log-likelihood of an AR(p)+noise model in companion form.

    z_t = A z_{t-1} + e1 w_t,  y_t = e1' z_t + v_t.  Diffuse start, with the
    first p contributions dropped so the arbitrary prior does not enter.
    """
    p = len(a)
    A = np.zeros((p, p))
    A[0] = a
    if p > 1:
        A[1:, :-1] = np.eye(p - 1)
    Qm = np.zeros((p, p))
    Qm[0, 0] = Q
    m = np.zeros(p)
    P = diffuse * np.eye(p)
    ll = 0.0
    for i, yt in enumerate(y):
        m = A @ m
        P = A @ P @ A.T + Qm
        S = P[0, 0] + S2
        e = yt - m[0]
        if i >= p:
            ll += -0.5 * (np.log(2 * np.pi * S) + e * e / S)
        K = P[:, 0] / S
        m = m + K * e
        P = P - np.outer(K, P[0, :])
    return ll


def est_ml(y, p, Q, S2, start):
    r = minimize(lambda v: -kalman_loglik(y, v, Q, S2) / len(y), start,
                 method="Nelder-Mead",
                 options=dict(maxiter=600, xatol=1e-4, fatol=1e-7))
    return r.x


# ------------------------------------------------------------------ truths
def build_truths():
    zeta, omega = 0.15, 0.35
    rho, theta = np.exp(-zeta * omega), omega * np.sqrt(1 - zeta**2)
    osc = np.array([2 * rho * np.cos(theta), -rho * rho])       # z^2 - a1 z - a2
    return {
        "offset only": dict(a=np.array([1.0]), p=1),
        "oscillator only": dict(a=osc, p=2),
        "offset + osc.": dict(a=alpha_from_ode(rho, theta), p=3),
    }, rho, theta


def inc_sd(a, Q, n=200_000, seed=7):
    x, _ = simulate(a, n, Q, 0.0, np.random.default_rng(seed))
    return float(np.std(np.diff(x)))


# --------------------------------------------------------------- part A
def part_a(truths, kappas, n=4000, R=40, seed=11):
    master = np.random.default_rng(seed)
    rows = []
    for name, spec in truths.items():
        a, p = spec["a"], spec["p"]
        d_sd = inc_sd(a, 1.0)
        for kap in kappas:
            S2 = (kap * d_sd) ** 2
            acc = {"ols": [], "iv": []}
            for _ in range(R):
                rng = np.random.default_rng(master.integers(2**63))
                _, y = simulate(a, n, 1.0, S2, rng)
                acc["ols"].append(est_ols(y, p=p))
                acc["iv"].append(est_iv(y, p=p, m=2 * p))
            for meth, A in acc.items():
                A = np.array(A)
                # relative error in the largest root modulus: the quantity that
                # governs how fast a forecast decays
                rt = np.array([np.abs(roots_of(v)).max() for v in A])
                rows.append(dict(truth=name, kappa=kap, method=meth,
                                 a=A.mean(0).tolist(),
                                 a_se=(A.std(0) / np.sqrt(R)).tolist(),
                                 rmax=float(rt.mean()),
                                 rmax_se=float(rt.std() / np.sqrt(R)),
                                 rmax_true=float(np.abs(roots_of(a)).max())))
    return rows


# --------------------------------------------------------------- part B
def part_b(kappas, n=2000, R=20, seed=23):
    master = np.random.default_rng(seed)
    truths, rho, theta = build_truths()
    a = truths["offset + osc."]["a"]
    p = 3
    d_sd = inc_sd(a, 1.0)
    rows = []
    for kap in kappas:
        S2 = (kap * d_sd) ** 2
        acc = {"iv(6)": [], "ml": []}
        for _ in range(R):
            rng = np.random.default_rng(master.integers(2**63))
            _, y = simulate(a, n, 1.0, S2, rng)
            aiv = est_iv(y, p=p, m=6)
            acc["iv(6)"].append(aiv)
            acc["ml"].append(est_ml(y, p, 1.0, S2, aiv))
        for meth, A in acc.items():
            A = np.array(A)
            err = A - a
            rows.append(dict(kappa=kap, method=meth,
                             a=A.mean(0).tolist(),
                             bias=err.mean(0).tolist(),
                             rmse=float(np.sqrt((err ** 2).sum(1).mean())),
                             rmse_se=float(np.std(np.sqrt((err ** 2).sum(1)))
                                           / np.sqrt(R))))
    return rows, a


def main():
    truths, rho, theta = build_truths()
    kappas = [0.1, 0.25, 0.5, 1.0, 2.0]

    print("=== A. where the damage lands (largest root modulus) ===")
    rows_a = part_a(truths, kappas)
    hdr = f"{'truth':>16} {'kappa':>6} {'method':>5} {'|z|max':>17} {'truth':>8}"
    print(hdr + "\n" + "-" * len(hdr))
    for r in rows_a:
        print(f"{r['truth']:>16} {r['kappa']:6.2f} {r['method']:>5} "
              f"{r['rmax']:10.4f}+-{r['rmax_se']:.4f} {r['rmax_true']:8.4f}")

    print("\n=== B. IV against exact ML, (Q, S2) at truth ===")
    rows_b, a_true = part_b([0.25, 0.5, 1.0])
    print(f"a_true = {np.round(a_true, 4)}")
    hdr = f"{'kappa':>6} {'method':>7} {'||a_hat - a||  rmse':>22} {'bias':>34}"
    print(hdr + "\n" + "-" * len(hdr))
    for r in rows_b:
        print(f"{r['kappa']:6.2f} {r['method']:>7} "
              f"{r['rmse']:12.4f}+-{r['rmse_se']:.4f}   "
              f"{np.round(r['bias'], 4)}")

    # ------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ax = axes[0]
    for i, name in enumerate(truths):
        for meth, ls in (("ols", "-"), ("iv", "--")):
            sel = [r for r in rows_a if r["truth"] == name and r["method"] == meth]
            rel = [r["rmax"] / r["rmax_true"] - 1.0 for r in sel]
            er = [r["rmax_se"] / r["rmax_true"] for r in sel]
            ax.errorbar(kappas, rel, yerr=er, marker="o" if meth == "ols" else "s",
                        ls=ls, color=ts.SERIES[i], capsize=2,
                        label=f"{name} [{meth}]")
    ax.axhline(0.0, color=ts.INK, lw=1.0, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\sigma\,/\,\mathrm{SD}(\Delta x)$")
    ax.set_ylabel(r"relative error in $|z|_{\max}$")
    ax.set_title("The integrated direction is protected; the stationary one is not")
    ax.legend(ncol=1)
    ts.tidy(ax)

    ax = axes[1]
    ks = sorted({r["kappa"] for r in rows_b})
    for i, meth in enumerate(["iv(6)", "ml"]):
        sel = [r for r in rows_b if r["method"] == meth]
        ax.errorbar(ks, [r["rmse"] for r in sel], yerr=[r["rmse_se"] for r in sel],
                    marker="o", color=ts.SERIES[i + 3], capsize=2, label=meth)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\sigma\,/\,\mathrm{SD}(\Delta x)$")
    ax.set_ylabel(r"RMSE of $\hat a$")
    ax.set_title("What IV leaves on the table")
    ax.legend()
    ts.tidy(ax)
    ts.save(fig, os.path.join(HERE, "figures", "fig02-damage-and-prize.png"))

    with open(os.path.join(HERE, "figures", "ode003.json"), "w") as f:
        json.dump(dict(part_a=rows_a, part_b=rows_b, a_true=a_true.tolist()),
                  f, indent=1)


if __name__ == "__main__":
    main()
