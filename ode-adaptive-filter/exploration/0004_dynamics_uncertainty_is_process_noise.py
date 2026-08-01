"""0004 -- Uncertainty in the dynamics is exactly a process-noise term.

Claim (exact, second moments, a independent of z).  Let the state be the lag
vector z = (x_t, ..., x_{t-p+1}) with posterior (zh, P), and let the free row of
the companion matrix be a with posterior (ah, Sig).  Write F = companion(ah).
Then the one-step predictive moments of z are exactly

    E[z']    = F zh
    Cov[z']  = F P F' + e1 e1' * ( Q + zh' Sig zh + tr(Sig P) )

The dynamics uncertainty does not spread across the state.  It enters through
e1 e1' -- the SAME channel as the process noise -- so at the level of second
moments it IS process noise, with an effective magnitude

    Q_eff = Q + zh' Sig zh + tr(Sig P).

Two readings of the extra terms:

    zh' Sig zh   "I do not know the dynamics", acting on the state I do know.
                 Proportional to signal POWER, so it is a fixed RELATIVE noise
                 floor, where Q is a fixed ABSOLUTE one.
    tr(Sig P)    "I know neither", the interaction.  Always positive.

That gives the ODE filter a third dimensionless number alongside the parent's
q = Q/S2:

    eta = (zh' Sig zh + tr(Sig P)) / Q      dynamics ignorance, in units of
                                            process noise.

Part A verifies the identity by Monte Carlo.
Part B measures the one assumption it rests on.  In the exact posterior a and z
are NOT independent, and nothing forces them to be.  Done exactly in p=1 (a is
a scalar, so it can be gridded finely and the joint posterior computed with no
approximation at all): how large does Corr(a, z_t | y_{1:t}) actually get?
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def companion(a):
    p = len(a)
    F = np.zeros((p, p))
    F[0] = a
    if p > 1:
        F[1:, :-1] = np.eye(p - 1)
    return F


# ------------------------------------------------------------------- part A
def part_a(seed=3, N=4_000_000):
    """Monte Carlo the predictive covariance against the closed form."""
    rng = np.random.default_rng(seed)
    p = 3
    ah = np.array([2.785, -2.686, 0.900])
    L = np.array([[0.09, 0.0, 0.0], [0.04, 0.06, 0.0], [-0.02, 0.03, 0.05]])
    Sig = L @ L.T
    zh = np.array([3.0, 2.2, 1.1])
    Lp = np.array([[0.5, 0.0, 0.0], [0.3, 0.4, 0.0], [0.1, 0.2, 0.35]])
    P = Lp @ Lp.T
    Q = 0.7

    a = ah + rng.standard_normal((N, p)) @ L.T
    z = zh + rng.standard_normal((N, p)) @ Lp.T
    w = np.sqrt(Q) * rng.standard_normal(N)
    x_new = np.einsum("ij,ij->i", a, z) + w
    Znew = np.column_stack([x_new, z[:, :-1]])

    emp_mean = Znew.mean(0)
    emp_cov = np.cov(Znew.T, bias=False)

    F = companion(ah)
    Qeff = Q + zh @ Sig @ zh + np.trace(Sig @ P)
    pred_mean = F @ zh
    pred_cov = F @ P @ F.T
    pred_cov[0, 0] += Qeff

    se_cov = np.sqrt(2.0 / N) * np.sqrt(np.outer(np.diag(emp_cov), np.diag(emp_cov)))
    return dict(
        Sig=Sig.tolist(), P=P.tolist(), zh=zh.tolist(), ah=ah.tolist(), Q=Q,
        Q_term=Q, quad_term=float(zh @ Sig @ zh), trace_term=float(np.trace(Sig @ P)),
        Qeff=float(Qeff),
        emp_mean=emp_mean.tolist(), pred_mean=pred_mean.tolist(),
        emp_cov=emp_cov.tolist(), pred_cov=pred_cov.tolist(),
        max_abs_dev_in_se=float(np.max(np.abs(emp_cov - pred_cov) / se_cov)),
        N=N)


# ------------------------------------------------------------------- part B
def exact_joint_ar1(y, a_grid, Q, S2, diffuse=1e6):
    """Exact p(a, x_t | y_{1:t}) for x_t = a x_{t-1} + w, y = x + v.

    a is scalar, so a fine grid makes this exact up to quadrature.  One Kalman
    filter per node; the node weights are the exact conditional likelihoods.
    """
    G = len(a_grid)
    m = np.zeros(G)
    P = np.full(G, diffuse)
    logw = np.zeros(G)
    out = {k: [] for k in ("corr", "sd_a", "mean_a", "share_dyn")}
    for t, yt in enumerate(y):
        m = a_grid * m
        P = a_grid ** 2 * P + Q
        S = P + S2
        e = yt - m
        logw = logw - 0.5 * (np.log(S) + e * e / S)
        logw -= logw.max()
        pi = np.exp(logw)
        pi /= pi.sum()
        K = P / S
        m = m + K * e
        P = P * (1.0 - K)

        abar = float(pi @ a_grid)
        va = float(pi @ (a_grid - abar) ** 2)
        mbar = float(pi @ m)
        vz = float(pi @ (P + (m - mbar) ** 2))
        cov = float(pi @ ((a_grid - abar) * (m - mbar)))
        out["corr"].append(cov / np.sqrt(max(va, 1e-300) * max(vz, 1e-300)))
        out["sd_a"].append(np.sqrt(va))
        out["mean_a"].append(abar)
        # eta, in the p=1 case: (zh^2 Sig + Sig P) / Q
        out["share_dyn"].append((mbar * mbar * va + va * vz) / Q)
    return {k: np.array(v) for k, v in out.items()}


def part_b(a_true=0.97, Q=1.0, kappas=(0.25, 1.0), n=1500, seed=5):
    a_grid = np.linspace(0.0, 1.2, 1201)
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = a_true * x[t - 1] + np.sqrt(Q) * rng.standard_normal()
    d_sd = float(np.std(np.diff(x)))
    res = {}
    for kap in kappas:
        S2 = (kap * d_sd) ** 2
        y = x + np.sqrt(S2) * rng.standard_normal(n)
        res[kap] = exact_joint_ar1(y, a_grid, Q, S2)
    return res, a_true


def main():
    print("=== A. the predictive-covariance identity, Monte Carlo ===")
    A = part_a()
    print(f"Q          = {A['Q_term']:.6f}")
    print(f"zh' Sig zh = {A['quad_term']:.6f}")
    print(f"tr(Sig P)  = {A['trace_term']:.6f}")
    print(f"Q_eff      = {A['Qeff']:.6f}   (eta = "
          f"{(A['quad_term'] + A['trace_term']) / A['Q_term']:.3f})")
    print("\npredicted Cov:")
    print(np.round(np.array(A["pred_cov"]), 5))
    print("empirical Cov (N = %.0e):" % A["N"])
    print(np.round(np.array(A["emp_cov"]), 5))
    print(f"\nlargest deviation, in Monte Carlo standard errors: "
          f"{A['max_abs_dev_in_se']:.2f}")

    print("\n=== B. how dependent are a and z in the exact posterior? (p=1) ===")
    B, a_true = part_b()
    for kap, r in B.items():
        c = r["corr"]
        print(f"kappa={kap}:  a_hat -> {r['mean_a'][-1]:.4f} (truth {a_true}), "
              f"SD(a) {r['sd_a'][-1]:.4f}")
        print(f"           |corr(a, x_t)|  median {np.median(np.abs(c[50:])):.3f}  "
              f"p90 {np.quantile(np.abs(c[50:]), 0.9):.3f}  "
              f"max {np.max(np.abs(c[50:])):.3f}")
        print(f"           eta at t=100/500/1500: "
              f"{r['share_dyn'][99]:.3f} / {r['share_dyn'][499]:.3f} / "
              f"{r['share_dyn'][-1]:.3f}")

    # ------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    ks = list(B)
    ax = axes[0]
    for i, kap in enumerate(ks):
        ax.plot(np.abs(B[kap]["corr"]), color=ts.SERIES[i], lw=1.0,
                label=fr"$\kappa$={kap}")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$|\mathrm{corr}(a,\ x_t\ |\ y_{1:t})|$")
    ax.set_title("The independence assumption, measured")
    ax.legend()
    ts.tidy(ax)

    ax = axes[1]
    for i, kap in enumerate(ks):
        ax.plot(B[kap]["sd_a"], color=ts.SERIES[i], label=fr"$\kappa$={kap}")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel(r"posterior SD of $a$")
    ax.set_title("Dynamics uncertainty shrinks")
    ax.legend()
    ts.tidy(ax)

    ax = axes[2]
    for i, kap in enumerate(ks):
        ax.plot(B[kap]["share_dyn"], color=ts.SERIES[i], label=fr"$\kappa$={kap}")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\eta$")
    ax.set_title(r"Dynamics ignorance, in units of process noise")
    ax.legend()
    ts.tidy(ax)
    ts.save(fig, os.path.join(HERE, "figures", "fig03-dynamics-as-process-noise.png"))

    with open(os.path.join(HERE, "figures", "ode004.json"), "w") as f:
        json.dump(dict(part_a=A,
                       part_b={str(k): {kk: vv.tolist()[::10] for kk, vv in v.items()}
                               for k, v in B.items()}), f, indent=1)


if __name__ == "__main__":
    main()
