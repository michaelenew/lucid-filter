"""0005 -- In what coordinate should the dynamics be allowed to drift?

0004 measured that a CONSTANT alpha is an easy problem: eta (dynamics ignorance
in units of process noise) falls to 0.001 by t=1500 and the exact posterior
correlation between alpha and the state stays near 0.01.  So the content of the
ODE filter is not "estimate alpha" -- it is "how fast is alpha allowed to move",
which is exactly where the parent workstream's trust/belief split lives.

The parent forced its answer with scale equivariance: the class cannot know
whether x is in metres or feet, so the constraint on how the noise scales move
must live on the LOG scale, leaving two numbers per channel (magnitude,
persistence).  For alpha there is no scale to be equivariant to.  The proposed
replacement:

    A drift law for a parameter must not depend on how the parameter is
    written down.  The unique reparameterisation-invariant metric on a
    statistical model is the Fisher metric (Cencov), so the only
    coordinate-free statement available is that alpha diffuses ISOTROPICALLY IN
    ITS OWN FISHER METRIC, with one magnitude and one persistence.

Two checks it must pass.

  Consistency  For a Gaussian scale family, I(log sigma^2) = 1/2, a constant.
               Fisher-isotropic diffusion in sigma^2 IS a constant-variance
               random walk in log sigma^2 -- the parent's law exactly.  So the
               principle reproduces the parent where both apply.  (Analytic;
               derivation in 0006 section 2.)

  Prediction   For an AR(1) with observed state, I(a) = 1/(1-a^2), so the
               Fisher arc length is d(arcsin a).  The principle therefore says
               a should random-walk in arcsin(a), NOT in a.  Near |a| = 1 those
               differ a lot, and near |a| = 1 is where this workstream lives.
               This script measures whether it is a real effect.

Design.  p = 1 so that alpha is a scalar and the nuisance grid is exact.
Architecture is the parent's: grid the nuisance, run the conditional Kalman
recursion, collapse the level to one Gaussian per step (GPB1).  Three drift
coordinates -- none, a, arcsin(a) -- each given its own drift magnitude nu,
each nu chosen by marginal likelihood, so nothing is hand-tuned.

Note the architecture reproduces 0004's identity for free: the collapse variance
picks up the spread of the conditional means, which is (a - abar)^2 m^2 -- that
is exactly the zh' Sig zh term.  Nothing has to be coded for it.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

AGRID = np.linspace(-0.995, 0.995, 241)

COORD = {
    "static": None,
    "a": (lambda a: a, lambda a: np.ones_like(a)),
    "arcsin(a)": (lambda a: np.arcsin(a), lambda a: 1.0 / np.sqrt(1.0 - a * a)),
}


def transition(coord, nu):
    """Random-walk kernel on AGRID, Gaussian with SD nu in coordinate `coord`."""
    if coord is None or nu <= 0:
        return np.eye(len(AGRID))
    c, jac = COORD[coord]
    cv = c(AGRID)
    T = np.exp(-0.5 * ((cv[None, :] - cv[:, None]) / nu) ** 2) * jac(AGRID)[None, :]
    return T / T.sum(1, keepdims=True)


def run(y, T, Q, S2):
    """Grid-over-a filter with a single collapsed Gaussian level (GPB1)."""
    G = len(AGRID)
    pi = np.full(G, 1.0 / G)
    m, P = 0.0, 1e6
    ll = 0.0
    mm = np.empty(len(y))
    ab = np.empty(len(y))
    for t, yt in enumerate(y):
        pi = pi @ T
        m_a = AGRID * m                      # per-node prior mean
        P_a = AGRID ** 2 * P + Q             # per-node prior variance
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
        mm[t] = m
        ab[t] = float(pi @ AGRID)
    return ll, mm, ab


def simulate(a_path, Q, S2, rng):
    n = len(a_path)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = a_path[t] * x[t - 1] + np.sqrt(Q) * rng.standard_normal()
    return x, x + np.sqrt(S2) * rng.standard_normal(n)


SCENARIOS = {
    "no shift, a=0.90": (0.90, 0.90),
    "0.50 -> 0.90": (0.50, 0.90),
    "0.90 -> 0.99": (0.90, 0.99),
    "0.99 -> 0.90": (0.99, 0.90),
}
NUS = np.concatenate([[0.0], np.logspace(-3.5, -0.3, 11)])


def main():
    n, R, kappa, Q = 1500, 24, 0.5, 1.0
    master = np.random.default_rng(31337)
    rows = []
    curves = {}

    for sname, (a0, a1) in SCENARIOS.items():
        a_path = np.where(np.arange(n) < n // 2, a0, a1)
        # a scale for the measurement noise that does not depend on the filter
        ref, _ = simulate(a_path, Q, 0.0, np.random.default_rng(1))
        S2 = (kappa * np.std(np.diff(ref))) ** 2

        data = []
        for _ in range(R):
            rng = np.random.default_rng(master.integers(2**63))
            data.append(simulate(a_path, Q, S2, rng))

        for coord in COORD:
            best = None
            prof = []
            for nu in ([0.0] if coord == "static" else NUS):
                T = transition(coord, nu)
                lls, mses = [], []
                for x, y in data:
                    ll, mm, _ = run(y, T, Q, S2)
                    lls.append(ll / n)
                    mses.append(np.mean((mm - x) ** 2))
                rec = dict(nu=float(nu), ll=float(np.mean(lls)),
                           mse=float(np.mean(mses)),
                           mse_se=float(np.std(mses) / np.sqrt(R)))
                prof.append(rec)
                if best is None or rec["ll"] > best["ll"]:
                    best = rec
            curves[(sname, coord)] = prof
            rows.append(dict(scenario=sname, coord=coord, **best))

    # ------------------------------------------------------------- report
    print(f"{'scenario':>18} {'coord':>10} {'nu*':>8} {'loglik/pt':>11} "
          f"{'theta-MSE':>18} {'vs static':>10}")
    print("-" * 82)
    for sname in SCENARIOS:
        base = [r for r in rows if r["scenario"] == sname
                and r["coord"] == "static"][0]["mse"]
        for coord in COORD:
            r = [q for q in rows if q["scenario"] == sname
                 and q["coord"] == coord][0]
            print(f"{sname:>18} {coord:>10} {r['nu']:8.4f} {r['ll']:11.4f} "
                  f"{r['mse']:11.4f}+-{r['mse_se']:.4f} {r['mse'] / base:10.3f}")

    # paired test on the one scenario that separates the coordinates
    print("\npaired comparison, arcsin(a) against a, at each one's own nu*:")
    for sname, (a0, a1) in SCENARIOS.items():
        a_path = np.where(np.arange(n) < n // 2, a0, a1)
        ref, _ = simulate(a_path, Q, 0.0, np.random.default_rng(1))
        S2 = (kappa * np.std(np.diff(ref))) ** 2
        nus = {c: [r for r in rows if r["scenario"] == sname
                   and r["coord"] == c][0]["nu"] for c in ("a", "arcsin(a)")}
        Ts = {c: transition(c, nus[c]) for c in nus}
        rng2 = np.random.default_rng(999)
        d = []
        for _ in range(R):
            x, y = simulate(a_path, Q, S2, rng2)
            e = {c: np.mean((run(y, Ts[c], Q, S2)[1] - x) ** 2) for c in nus}
            d.append(e["arcsin(a)"] / e["a"] - 1.0)
        d = np.array(d)
        t = d.mean() / (d.std(ddof=1) / np.sqrt(R))
        print(f"  {sname:>18}: MSE ratio {1 + d.mean():.4f} "
              f"+-{d.std(ddof=1) / np.sqrt(R):.4f}   t = {t:+.2f}")
        rows.append(dict(scenario=sname, coord="paired arcsin/a",
                         nu=float("nan"), ll=float("nan"),
                         mse=float(1 + d.mean()),
                         mse_se=float(d.std(ddof=1) / np.sqrt(R)), t=float(t)))

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(14, 3.3), sharey=False)
    for ax, sname in zip(axes, SCENARIOS):
        for i, coord in enumerate(["a", "arcsin(a)"]):
            prof = curves[(sname, coord)]
            ax.plot([p["nu"] for p in prof], [p["ll"] for p in prof],
                    marker="o", color=ts.SERIES[i], label=coord)
        st = curves[(sname, "static")][0]["ll"]
        ax.axhline(st, color=ts.INK, ls="--", lw=1.0, label="static")
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_xlabel(r"drift SD $\nu$")
        ax.set_title(sname)
        ts.tidy(ax)
    axes[0].set_ylabel("log-likelihood per point")
    axes[0].legend()
    ts.save(fig, os.path.join(HERE, "figures", "fig04-drift-coordinate.png"))

    with open(os.path.join(HERE, "figures", "ode005.json"), "w") as f:
        json.dump(dict(rows=rows,
                       curves={f"{k[0]}|{k[1]}": v for k, v in curves.items()},
                       n=n, R=R, kappa=kappa), f, indent=1)


if __name__ == "__main__":
    main()
