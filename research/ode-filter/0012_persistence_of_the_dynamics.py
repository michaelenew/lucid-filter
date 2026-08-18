"""0012 -- The trust/belief split for the dynamics channel.

The parent gives each noise channel a persistence phi and reads the two ends as
different events: phi -> 0 is an impulse (an outlier, a jump), phi -> 1 is a
regime (a drift-rate change, a noise-level change).  Every probe so far in this
workstream has used a pure random walk in alpha, i.e. phi_A = 1 with no dial.
This is the hole the previous construction had, and the one the user named
first.

What the two ends MEAN here is worth stating before measuring, because it is
not obvious and it is the point.  Write alpha_t = abar + delta_t.  Then

    x_t = (abar + delta_t)' z_{t-1} + w_t
        = abar' z_{t-1} + (delta_t' z_{t-1} + w_t)

so the deviation enters as a noise term delta_t' z_{t-1} whose magnitude is
proportional to the STATE.  Therefore:

    phi_A -> 0   delta is white, the extra noise is z' Sig z per step: process
                 noise PROPORTIONAL TO SIGNAL POWER.  Multiplicative, not
                 additive.  This is 0004's identity read as a generative model.

    phi_A -> 1   delta persists: the ODE coefficients themselves changed.

**The impulsive end of the dynamics channel is relative noise; the persistent
end is a change in the dynamics.  They are one coordinate.**  That is the
parent's insight one level up, and it is something the parent's own model cannot
express: its log-scale modulates the variance by an exogenous process, never in
proportion to the signal.

The channel therefore carries the same centre / magnitude / persistence triple
as each of the parent's noise channels:  (abar, s_A, phi_A).

Two questions.

A. **Does phi_A identify?**  Generate from the impulsive end, the persistent
   end, and a static alpha; fit (abar, s_A, phi_A) by marginal likelihood in
   each case, and see whether the fitted phi_A separates them.

B. **Does the dial earn its keep?**  Compare the three-parameter fit against
   the pure random walk in alpha (phi_A pinned at 1) that every earlier probe
   used, on h-step forecast MSE.  If the split is real, pinning phi_A = 1
   should cost on impulsive data -- the filter chases a wobble it should be
   reverting from.

p = 1 throughout: alpha is a scalar, so the grid is exact and the question is
about phi, not about shape.
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

AGRID = np.linspace(-0.99, 0.99, 199)
HORIZONS = (1, 5, 20)
ABAR, SA = 0.80, 0.12          # generating centre and stationary SD


# --------------------------------------------------------------- generation
def gen_alpha(kind, n, rng):
    if kind == "static":
        return np.full(n, ABAR)
    if kind == "impulsive":                     # phi = 0, white
        a = ABAR + SA * rng.standard_normal(n)
    else:                                       # "persistent", phi = 0.99
        phi = 0.99
        e = np.sqrt(SA * SA * (1 - phi * phi)) * rng.standard_normal(n)
        d = np.empty(n)
        d[0] = SA * rng.standard_normal()
        for t in range(1, n):
            d[t] = phi * d[t - 1] + e[t]
        a = ABAR + d
    return np.clip(a, -0.99, 0.99)


def simulate(a_path, Q, S2, rng):
    n = len(a_path)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = a_path[t] * x[t - 1] + np.sqrt(Q) * rng.standard_normal()
    return x, x + np.sqrt(S2) * rng.standard_normal(n)


# ------------------------------------------------------------------- kernel
def ou_kernel(abar, s, phi):
    """Stationary OU on AGRID: centre abar, stationary SD s, persistence phi."""
    if s <= 1e-6:
        j = int(np.argmin(np.abs(AGRID - abar)))
        T = np.zeros((len(AGRID),) * 2)
        T[:, j] = 1.0
        pi0 = T[0].copy()
        return pi0, T
    nu = max(s * s * (1.0 - phi * phi), 1e-12)
    mu = abar + phi * (AGRID - abar)
    # subtract the row max before exponentiating.  Without it, a row whose OU
    # mean falls more than a few sqrt(nu) from every node underflows to all
    # zeros and the normalisation produces nan, which then poisons the fit --
    # the same failure mode as the parent's _expit overflow.
    q = -0.5 * (AGRID[None, :] - mu[:, None]) ** 2 / nu
    T = np.exp(q - q.max(1, keepdims=True))
    T /= T.sum(1, keepdims=True)
    z = -0.5 * ((AGRID - abar) / s) ** 2
    pi0 = np.exp(z - z.max())
    return pi0 / pi0.sum(), T


def rw_kernel(nu):
    """Pure random walk in alpha: the kernel every earlier probe used."""
    if nu <= 1e-6:
        return np.full(len(AGRID), 1.0 / len(AGRID)), np.eye(len(AGRID))
    T = np.exp(-0.5 * ((AGRID[None, :] - AGRID[:, None]) / nu) ** 2)
    T /= T.sum(1, keepdims=True)
    return np.full(len(AGRID), 1.0 / len(AGRID)), T


# ------------------------------------------------------------------- filter
def run(y, pi0, T, Q, S2, forecast_every=0):
    n = len(y)
    pi = pi0.copy()
    m, P = 0.0, 1e6
    ll = 0.0
    fc = {h: {} for h in HORIZONS}
    for t in range(n):
        pi = pi @ T
        m_a = AGRID * m
        P_a = AGRID ** 2 * P + Q
        S = P_a + S2
        e = y[t] - m_a
        lg = -0.5 * (np.log(S) + e * e / S)
        mx = lg.max()
        w = pi * np.exp(lg - mx)
        Z = w.sum()
        ll += np.log(Z) + mx - 0.5 * np.log(2 * np.pi)
        pi = w / Z
        K = P_a / S
        q = m_a + K * e
        m = float(pi @ q)
        P = float(pi @ (P_a * (1.0 - K) + (q - m) ** 2))
        if forecast_every and t % forecast_every == 0 and t >= n // 2:
            # mu_j^(k) = E[x_{t+k} 1{a_{t+k}=j}];  exact given the chain and
            # the collapsed level, and it reverts to abar^h when phi < 1
            mu = pi * m
            for k in range(1, max(HORIZONS) + 1):
                mu = AGRID * (mu @ T)
                if k in HORIZONS:
                    fc[k][t] = float(mu.sum())
    return ll, fc


def fc_mse(fc, x):
    out = {}
    for h, d in fc.items():
        idx = [t for t in d if t + h < len(x)]
        out[h] = float(np.mean([(d[t] - x[t + h]) ** 2 for t in idx]))
    return out


# --------------------------------------------------------------------- fits
def fit_ou(y, Q, S2, n_iter=250):
    def neg(v):
        abar = np.tanh(v[0]) * 0.99
        s = np.exp(v[1])
        phi = 1.0 / (1.0 + np.exp(-v[2]))
        if not np.isfinite(s) or s > 2.0:
            return 1e9
        pi0, T = ou_kernel(abar, s, phi)
        return -run(y, pi0, T, Q, S2)[0] / len(y)

    best, bv = None, np.inf
    for st in ([0.9, np.log(0.05), 0.0], [0.9, np.log(0.15), 3.0]):
        r = minimize(neg, np.array(st), method="Nelder-Mead",
                     options=dict(maxiter=n_iter, xatol=2e-3, fatol=1e-6))
        if r.fun < bv:
            best, bv = r.x, r.fun
    return (float(np.tanh(best[0]) * 0.99), float(np.exp(best[1])),
            float(1.0 / (1.0 + np.exp(-best[2])))), -bv * len(y)


def fit_rw(y, Q, S2):
    nus = np.concatenate([[0.0], np.logspace(-3.0, -0.4, 9)])
    best, bl = 0.0, -np.inf
    for nu in nus:
        pi0, T = rw_kernel(nu)
        ll = run(y, pi0, T, Q, S2)[0]
        if ll > bl:
            best, bl = float(nu), ll
    return best, bl


# --------------------------------------------------------------------- main
def main():
    n, R, kappa, Q = 1200, 12, 0.35, 1.0
    kinds = ["static", "impulsive", "persistent"]
    master = np.random.default_rng(60613)
    rows, fits = [], {k: [] for k in kinds}

    for kind in kinds:
        ref_a = gen_alpha(kind, 40000, np.random.default_rng(3))
        refx, _ = simulate(ref_a, Q, 0.0, np.random.default_rng(4))
        S2 = (kappa * np.std(np.diff(refx))) ** 2
        acc = {"ou": [], "rw": []}
        for _ in range(R):
            rng = np.random.default_rng(master.integers(2 ** 63))
            a_path = gen_alpha(kind, n, rng)
            x, y = simulate(a_path, Q, S2, rng)
            (abar, s, phi), llo = fit_ou(y, Q, S2)
            nu, llr = fit_rw(y, Q, S2)
            fits[kind].append((abar, s, phi))
            pi0, T = ou_kernel(abar, s, phi)
            acc["ou"].append((llo, fc_mse(run(y, pi0, T, Q, S2, 5)[1], x)))
            pi0, T = rw_kernel(nu)
            acc["rw"].append((llr, fc_mse(run(y, pi0, T, Q, S2, 5)[1], x)))
        F = np.array(fits[kind])
        print(f"{kind:>11}: abar {F[:,0].mean():.3f}+-{F[:,0].std()/np.sqrt(R):.3f}  "
              f"s_A {F[:,1].mean():.3f}+-{F[:,1].std()/np.sqrt(R):.3f}  "
              f"phi_A {F[:,2].mean():.3f}+-{F[:,2].std()/np.sqrt(R):.3f}  "
              f"[median {np.median(F[:,2]):.3f}]")
        for meth, v in acc.items():
            rows.append(dict(kind=kind, method=meth,
                             ll=float(np.mean([q[0] for q in v]) / n),
                             **{f"h{h}": float(np.mean([q[1][h] for q in v]))
                                for h in HORIZONS},
                             **{f"h{h}_se": float(np.std([q[1][h] for q in v])
                                                  / np.sqrt(R))
                                for h in HORIZONS}))

    print("\n=== B. does the dial earn its keep? (h-step forecast MSE) ===")
    hdr = (f"{'generated'.rjust(11)} {'filter':>6} {'loglik/pt':>10} "
           + " ".join(f"{'h='+str(h):>17}" for h in HORIZONS))
    print(hdr + "\n" + "-" * len(hdr))
    for kind in kinds:
        base = [r for r in rows if r["kind"] == kind and r["method"] == "rw"][0]
        for meth in ("rw", "ou"):
            r = [q for q in rows if q["kind"] == kind and q["method"] == meth][0]
            cells = " ".join(
                f"{r[f'h{h}']:8.3f}+-{r[f'h{h}_se']:.3f}({r[f'h{h}']/base[f'h{h}']:.3f})"
                for h in HORIZONS)
            print(f"{kind:>11} {meth:>6} {r['ll']:10.4f} {cells}")
    print("\n  rw = pure random walk in alpha (phi_A pinned at 1), what every")
    print("  earlier probe used.  ou = the (abar, s_A, phi_A) triple.")

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ax = axes[0]
    for i, kind in enumerate(kinds):
        F = np.array(fits[kind])
        ax.scatter(F[:, 2] + 0.004 * np.random.default_rng(i).standard_normal(len(F)),
                   F[:, 1], color=ts.SERIES[i], s=26, label=kind, alpha=0.85)
    ax.set_xlabel(r"fitted persistence $\hat\varphi_A$")
    ax.set_ylabel(r"fitted magnitude $\hat s_A$")
    ax.set_title("Does the dynamics channel's persistence identify?")
    ax.legend()
    ts.tidy(ax)

    ax = axes[1]
    xs = np.arange(len(kinds))
    for i, h in enumerate(HORIZONS):
        ys = []
        for kind in kinds:
            r = [q for q in rows if q["kind"] == kind and q["method"] == "ou"][0]
            b = [q for q in rows if q["kind"] == kind and q["method"] == "rw"][0]
            ys.append(r[f"h{h}"] / b[f"h{h}"])
        ax.plot(xs, ys, marker="o", color=ts.SERIES[i], label=f"h={h}")
    ax.axhline(1.0, color=ts.INK, lw=1.0, ls="--", zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels(kinds)
    ax.set_ylabel("forecast MSE, triple / random walk")
    ax.set_title("Does the dial earn its keep?")
    ax.legend()
    ts.tidy(ax)
    ts.save(fig, os.path.join(HERE, "figures", "fig09-dynamics-persistence.png"))

    with open(os.path.join(HERE, "figures", "ode012.json"), "w") as f:
        json.dump(dict(rows=rows, fits={k: np.array(v).tolist()
                                        for k, v in fits.items()},
                       n=n, R=R, kappa=kappa, abar=ABAR, s_A=SA), f, indent=1)


if __name__ == "__main__":
    main()
