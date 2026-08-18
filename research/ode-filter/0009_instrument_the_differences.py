"""0009 -- Instrument the differences, and ask whether the offset root is real.

0003 section 2 left a specific defect.  Lagged LEVELS of an integrated series
are dominated by the common trend, so as instruments they are near-collinear
for exactly the stationary coordinates that need instrumenting: IV recovered a
pure stationary oscillator to 0.9469-0.9488 against 0.9489 at every noise level
tested, but the same oscillator sitting under a unit root degraded to 0.880 at
kappa = 1 and went spuriously explosive at kappa = 2.

The repair is to impose the unit root and instrument the DIFFERENCED series.
Writing A(L) = 1 - a1 L - a2 L^2 - a3 L^3 with A(1) = 0 gives A(L) = B(L)(1-L),
B(L) = 1 - b1 L - b2 L^2, and matching coefficients,

    a1 = 1 + b1,      a2 = b2 - b1,      a3 = -b2

so the oscillator is exactly the AR(2) of the differenced process.

The valid instrument lag is unchanged.  For y = x + v with x AR(p) the residual
touches v over a window of p+1 steps, so lag >= p+1 works.  Differencing once
makes the AR order p-1 and the noise MA(1), and (p-1) + 1 + 1 = p+1 again --
the same lag, but now on a stationary series whose lags are not all the same
trend.

Part B asks the question differencing presupposes: **is the offset root really
at 1?**  Exact Gaussian ML with the root free (3 parameters) against the root
pinned (2 parameters, differenced), compared by likelihood on data generated
with a true unit root and with a near-unit root at 0.98.  Same machinery as the
filter, no new theory -- this is what "is the offset constant or drifting"
looks like when it is made answerable.
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
_m2 = import_module("0002_identifying_the_dynamics")
_m3 = import_module("0003_where_the_eiv_damage_lands")
simulate, est_iv, roots_of = _m2.simulate, _m2.est_iv, _m2.roots_of
alpha_from_ode = _m2.alpha_from_ode
kalman_loglik = _m3.kalman_loglik


def beta_to_alpha(b):
    return np.array([1.0 + b[0], b[1] - b[0], -b[1]])


def est_iv_diff(y, m=6):
    """Impose the unit root: IV on the differenced series, AR order p-1 = 2."""
    return beta_to_alpha(est_iv(np.diff(y), p=2, m=m))


def osc_from_alpha(a):
    """(modulus, angle) of the complex pair, or (geometric mean, 0) if real."""
    r = roots_of(a)
    c = r[r.imag > 1e-9]
    if c.size:
        return float(np.abs(c[0])), float(np.angle(c[0]))
    mods = np.sort(np.abs(r[np.abs(r.imag) < 1e-9]))
    keep = mods[:2] if len(mods) > 2 else mods
    return float(np.sqrt(keep[0] * keep[1])), 0.0


# ------------------------------------------------------------------ part A
def part_a(a_true, rho, theta, kappas, n=4000, R=40, seed=17):
    master = np.random.default_rng(seed)
    d_sd = _m3.inc_sd(a_true, 1.0)
    rows = []
    for kap in kappas:
        S2 = (kap * d_sd) ** 2
        acc = {"iv-levels(6)": [], "iv-diff(6)": []}
        for _ in range(R):
            rng = np.random.default_rng(master.integers(2**63))
            _, y = simulate(a_true, n, 1.0, S2, rng)
            acc["iv-levels(6)"].append(est_iv(y, p=3, m=6))
            acc["iv-diff(6)"].append(est_iv_diff(y, m=6))
        for meth, A in acc.items():
            A = np.array(A)
            oc = np.array([osc_from_alpha(v) for v in A])
            err = A - a_true
            rows.append(dict(
                kappa=kap, method=meth,
                rho=float(oc[:, 0].mean()), rho_se=float(oc[:, 0].std() / np.sqrt(R)),
                theta=float(oc[:, 1].mean()),
                theta_se=float(oc[:, 1].std() / np.sqrt(R)),
                n_real=int((oc[:, 1] == 0.0).sum()),
                rmse=float(np.sqrt((err ** 2).sum(1).mean()))))
    return rows


# ------------------------------------------------------------------ part B
def ml_free(y, start, Q, S2):
    r = minimize(lambda v: -kalman_loglik(y, v, Q, S2) / len(y), start,
                 method="Nelder-Mead",
                 options=dict(maxiter=800, xatol=1e-4, fatol=1e-8))
    return r.x, -r.fun * len(y)


def ml_pinned(y, start_b, Q, S2):
    """Root pinned at 1: search over b (2-D), evaluate the induced a (3-D)."""
    r = minimize(lambda v: -kalman_loglik(y, beta_to_alpha(v), Q, S2) / len(y),
                 start_b, method="Nelder-Mead",
                 options=dict(maxiter=800, xatol=1e-4, fatol=1e-8))
    return beta_to_alpha(r.x), -r.fun * len(y)


def part_b(rho, theta, kappa=0.5, n=2000, R=20, seed=29):
    master = np.random.default_rng(seed)
    out = []
    for z0 in (1.0, 0.98):
        # characteristic polynomial (z - z0)(z^2 - 2 rho cos t z + rho^2)
        c = 2.0 * rho * np.cos(theta)
        a = np.array([z0 + c, -(rho * rho + c * z0), rho * rho * z0])
        d_sd = _m3.inc_sd(a, 1.0)
        S2 = (kappa * d_sd) ** 2
        d = []
        for _ in range(R):
            rng = np.random.default_rng(master.integers(2**63))
            _, y = simulate(a, n, 1.0, S2, rng)
            b0 = est_iv(np.diff(y), p=2, m=6)
            af, llf = ml_free(y, beta_to_alpha(b0), 1.0, S2)
            ap, llp = ml_pinned(y, b0, 1.0, S2)
            d.append(dict(llr=2.0 * (llf - llp),
                          z_free=float(np.real(
                              roots_of(af)[np.argmin(np.abs(
                                  np.imag(roots_of(af))))])),
                          err_free=float(np.linalg.norm(af - a)),
                          err_pin=float(np.linalg.norm(ap - a))))
        out.append(dict(z0=z0, n=n, R=R,
                        llr=float(np.mean([q["llr"] for q in d])),
                        llr_se=float(np.std([q["llr"] for q in d]) / np.sqrt(R)),
                        z_free=float(np.mean([q["z_free"] for q in d])),
                        z_free_se=float(np.std([q["z_free"] for q in d])
                                        / np.sqrt(R)),
                        err_free=float(np.mean([q["err_free"] for q in d])),
                        err_pin=float(np.mean([q["err_pin"] for q in d]))))
    return out


def main():
    zeta, omega = 0.15, 0.35
    rho, theta = np.exp(-zeta * omega), omega * np.sqrt(1 - zeta ** 2)
    a_true = alpha_from_ode(rho, theta)
    kappas = [0.1, 0.25, 0.5, 1.0, 2.0]

    print("=== A. instruments on levels against instruments on differences ===")
    print(f"truth: rho = {rho:.4f}, theta = {theta:.4f}\n")
    rows = part_a(a_true, rho, theta, kappas)
    hdr = (f"{'kappa':>6} {'method':>14} {'rho':>16} {'theta':>16} "
           f"{'all-real':>9} {'||da||':>8}")
    print(hdr + "\n" + "-" * len(hdr))
    for r in rows:
        print(f"{r['kappa']:6.2f} {r['method']:>14} "
              f"{r['rho']:9.4f}+-{r['rho_se']:.4f} "
              f"{r['theta']:9.4f}+-{r['theta_se']:.4f} "
              f"{r['n_real']:6d}/40 {r['rmse']:8.4f}")

    print("\n=== B. is the offset root at 1?  free (3 par) vs pinned (2 par) ===")
    rb = part_b(rho, theta)
    print(f"{'true z0':>8} {'2*LLR (free-pinned)':>22} {'z0_hat (free)':>20} "
          f"{'||da|| free':>12} {'||da|| pinned':>14}")
    print("-" * 80)
    for r in rb:
        print(f"{r['z0']:8.2f} {r['llr']:14.3f}+-{r['llr_se']:.3f} "
              f"{r['z_free']:14.4f}+-{r['z_free_se']:.4f} "
              f"{r['err_free']:12.4f} {r['err_pin']:14.4f}")
    print("\n(one extra parameter: chi2_1 has mean 1.0, 95th percentile 3.84)")

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for j, (key, tv, lab) in enumerate(
            [("rho", rho, r"oscillator modulus $\rho$"),
             ("theta", theta, r"oscillator angle $\theta$")]):
        ax = axes[j]
        for i, meth in enumerate(["iv-levels(6)", "iv-diff(6)"]):
            sel = [r for r in rows if r["method"] == meth]
            ax.errorbar(kappas, [r[key] for r in sel],
                        yerr=[r[key + "_se"] for r in sel], marker="o",
                        color=ts.SERIES[i + 1], capsize=2, label=meth)
        ax.axhline(tv, color=ts.INK, lw=1.0, ls="--", zorder=0)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\sigma\,/\,\mathrm{SD}(\Delta x)$")
        ax.set_title(lab)
        ts.tidy(ax)
    axes[0].legend()
    ts.save(fig, os.path.join(HERE, "figures", "fig07-instrument-differences.png"))

    with open(os.path.join(HERE, "figures", "ode009.json"), "w") as f:
        json.dump(dict(part_a=rows, part_b=rb, rho=rho, theta=theta,
                       a_true=a_true.tolist()), f, indent=1)


if __name__ == "__main__":
    main()
