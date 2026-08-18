"""0002 -- Can the dynamics be identified under measurement noise, and how?

The parent workstream's spine was: increments annihilate the unknown level
exactly, leaving a stationary MA(1) whose two autocovariances give (Q, s2).
This probe looks for the analogue for the dynamics.

Setting.  A second-order linear ODE in one variable with a constant offset,

    xdd + 2 zeta omega xd + omega^2 x = r,

sampled uniformly.  Its solution space is span{1, e^{l1 t}, e^{l2 t}}, so the
sampled sequence is annihilated by (z - 1)(z - z1)(z - z2): an order-3
homogeneous recurrence with one root pinned at 1.  Forced by noise and observed
with noise:

    x_t = a1 x_{t-1} + a2 x_{t-2} + a3 x_{t-3} + w_t     w ~ N(0, Q)
    y_t = x_t + v_t                                       v ~ N(0, S2)

The question: what does the data say about a = (a1, a2, a3)?

Three estimators.

  oracle  OLS of x_t on true lags.  Unachievable; the benchmark.
  ols     OLS of y_t on observed lags.  This is what the previous filter did
          (a pseudo-inverse of a noisy regressor block is OLS with noisy
          regressors), and it is the classic errors-in-variables setup.
  iv(m)   instrumental variables using lags p+1 .. p+m as instruments.

The IV moment condition is exact and needs nothing but the model:

    y_t - sum_i a_i y_{t-i} = w_t + v_t - sum_i a_i v_{t-i}

depends on v only at times t, t-1, ..., t-p.  So ANY observation at lag >= p+1
is uncorrelated with it:

    E[ y_{t-k} ( y_t - sum_i a_i y_{t-i} ) ] = 0     for all k >= p+1.

That is the exact analogue of "increments annihilate the level": lagging by
more than the order annihilates the measurement noise.  It holds for every
(Q, S2) without knowing either, and it does not need stationarity.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

P = 3  # order of the recurrence: 2nd-order ODE + constant offset


# ---------------------------------------------------------------- the truth
def alpha_from_ode(rho: float, theta: float) -> np.ndarray:
    """AR(3) coefficients of the sequence annihilated by (z-1)(z^2-2 rho cos t z + rho^2)."""
    c = 2.0 * rho * np.cos(theta)
    return np.array([1.0 + c, -(rho * rho + c), rho * rho])


def simulate(a, n, Q, S2, rng, burn=500):
    p = len(a)
    x = np.zeros(n + burn)
    sw = np.sqrt(Q)
    lags = np.arange(1, p + 1)
    for t in range(p, n + burn):
        x[t] = a @ x[t - lags] + sw * rng.standard_normal()
    x = x[burn:]
    y = x + np.sqrt(S2) * rng.standard_normal(n)
    return x, y


def increment_sd(a, Q, n=200_000, seed=0):
    """SD of the stationary increment process; the natural scale for S2."""
    x, _ = simulate(a, n, Q, 0.0, np.random.default_rng(seed))
    return float(np.std(np.diff(x)))


# ----------------------------------------------------------- the estimators
def _design(s, p, n_extra_lags):
    """Rows t: target s_t, regressors s_{t-1..t-p}, instruments s_{t-p-1..t-p-m}."""
    m = n_extra_lags
    lo = p + m
    T = len(s)
    idx = np.arange(lo, T)
    yv = s[idx]
    W = np.column_stack([s[idx - i] for i in range(1, p + 1)])
    Z = np.column_stack([s[idx - p - j] for j in range(1, m + 1)])
    return yv, W, Z


def est_ols(s, p=P):
    yv, W, _ = _design(s, p, 1)
    return np.linalg.lstsq(W, yv, rcond=None)[0]


def est_iv(s, p=P, m=P):
    """Two-stage least squares with lags p+1..p+m as instruments."""
    yv, W, Z = _design(s, p, m)
    # project W onto Z, then regress y on the projection
    Wh = Z @ np.linalg.lstsq(Z, W, rcond=None)[0]
    return np.linalg.lstsq(Wh, yv, rcond=None)[0]


def roots_of(a):
    """Roots of z^p - a1 z^{p-1} - ... - ap, sorted by |arg| then -|z|."""
    r = np.roots(np.concatenate([[1.0], -np.asarray(a)]))
    return r[np.lexsort((-np.abs(r), np.abs(np.angle(r))))]


def summarise_roots(a):
    """(modulus of the real root nearest 1, modulus of the complex pair, its angle)."""
    r = roots_of(a)
    real = r[np.abs(r.imag) < 1e-9]
    comp = r[r.imag > 1e-9]
    z_off = float(np.real(real[np.argmin(np.abs(real - 1.0))])) if real.size else np.nan
    if comp.size:
        return z_off, float(np.abs(comp[0])), float(np.angle(comp[0]))
    # over-damped: no complex pair, report the two remaining real roots' geometric mean
    other = np.sort(np.abs(real))[::-1]
    return z_off, float(np.sqrt(other[0] * other[1])) if other.size > 1 else np.nan, 0.0


# ------------------------------------------------------------------ the run
def main():
    rng_master = np.random.default_rng(20260801)

    zeta, omega, dt = 0.15, 0.35, 1.0          # lightly damped, ~18-step period
    rho, theta = np.exp(-zeta * omega * dt), omega * np.sqrt(1 - zeta**2) * dt
    a_true = alpha_from_ode(rho, theta)
    Q = 1.0
    d_sd = increment_sd(a_true, Q)

    print(f"true poles: unit root, rho={rho:.4f} theta={theta:.4f} "
          f"(period {2*np.pi/theta:.1f} steps)")
    print(f"a_true = {a_true}")
    print(f"SD(increment) = {d_sd:.4f}  (with Q=1)\n")

    kappas = [0.1, 0.25, 0.5, 1.0, 2.0]        # S2^0.5 / SD(increment)
    n, R = 4000, 60
    methods = {
        "oracle": lambda s: est_ols(s),
        "ols": lambda s: est_ols(s),
        "iv(3)": lambda s: est_iv(s, m=3),
        "iv(6)": lambda s: est_iv(s, m=6),
        "iv(12)": lambda s: est_iv(s, m=12),
    }

    out = {k: {"a": [], "roots": []} for k in methods}
    rows = []
    for kap in kappas:
        S2 = (kap * d_sd) ** 2
        acc = {k: {"a": [], "roots": []} for k in methods}
        for r in range(R):
            rng = np.random.default_rng(rng_master.integers(2**63))
            x, y = simulate(a_true, n, Q, S2, rng)
            for name, fn in methods.items():
                s = x if name == "oracle" else y
                ah = fn(s)
                acc[name]["a"].append(ah)
                acc[name]["roots"].append(summarise_roots(ah))
        for name in methods:
            A = np.array(acc[name]["a"])
            Rt = np.array(acc[name]["roots"])
            out[name]["a"].append([A.mean(0).tolist(),
                                   (A.std(0) / np.sqrt(R)).tolist()])
            out[name]["roots"].append([Rt.mean(0).tolist(),
                                       (Rt.std(0) / np.sqrt(R)).tolist()])
            rows.append(dict(kappa=kap, method=name,
                             a=A.mean(0).tolist(),
                             a_se=(A.std(0) / np.sqrt(R)).tolist(),
                             z_offset=Rt[:, 0].mean(), z_offset_se=Rt[:, 0].std() / np.sqrt(R),
                             rho=Rt[:, 1].mean(), rho_se=Rt[:, 1].std() / np.sqrt(R),
                             theta=Rt[:, 2].mean(), theta_se=Rt[:, 2].std() / np.sqrt(R)))

    hdr = f"{'kappa':>6} {'method':>8} {'z_offset':>16} {'rho':>16} {'theta':>16}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['kappa']:6.2f} {r['method']:>8} "
              f"{r['z_offset']:9.4f}+-{r['z_offset_se']:.4f} "
              f"{r['rho']:9.4f}+-{r['rho_se']:.4f} "
              f"{r['theta']:9.4f}+-{r['theta_se']:.4f}")
    print(f"\ntruth:   z_offset = 1.0000   rho = {rho:.4f}   theta = {theta:.4f}")

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    order = ["oracle", "ols", "iv(3)", "iv(6)", "iv(12)"]
    truth = [1.0, rho, theta]
    labels = [r"offset root  $|z_0|$", r"oscillator modulus  $\rho$",
              r"oscillator angle  $\theta$"]
    keys = ["z_offset", "rho", "theta"]
    for j, (ax, key, lab, tv) in enumerate(zip(axes, keys, labels, truth)):
        for i, name in enumerate(order):
            ys = [r[key] for r in rows if r["method"] == name]
            es = [r[key + "_se"] for r in rows if r["method"] == name]
            ax.errorbar(kappas, ys, yerr=es, marker="o", color=ts.SERIES[i],
                        label=name, capsize=2)
        ax.axhline(tv, color=ts.INK, lw=1.0, ls="--", zorder=0)
        ax.set_xscale("log")
        ax.set_xlabel(r"measurement noise  $\sigma\,/\,\mathrm{SD}(\Delta x)$")
        ax.set_title(lab)
        ts.tidy(ax)
    axes[0].legend(ncol=2)
    ts.save(fig, os.path.join(HERE, "figures", "fig01-eiv-vs-iv.png"))

    with open(os.path.join(HERE, "figures", "ode002.json"), "w") as f:
        json.dump(dict(a_true=a_true.tolist(), rho=rho, theta=theta,
                       increment_sd=d_sd, n=n, R=R, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
