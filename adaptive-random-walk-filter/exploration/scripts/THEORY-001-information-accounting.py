"""THEORY-001: how many nats does each point carry about (Q, sigma^2)?

Model (unchanged from the whole thread):
    theta_t = theta_{t-1} + w_t,   w ~ N(0, Q)
    x_t     = theta_t + v_t,       v ~ N(0, s2)

Exact finite-n information, no simulation. Take first differences
    d_t = x_t - x_{t-1},  t = 1..n-1
which annihilates the unknown initial level exactly (this is the REML /
marginal likelihood -- the right likelihood when theta_0 is a nuisance).
d is a stationary MA(1) with
    gamma_0 = Q + 2 s2,   gamma_1 = -s2,   gamma_k = 0 for k >= 2

so Sigma_n is tridiagonal Toeplitz and
    dSigma/dQ  = I
    dSigma/ds2 = T   (2 on diagonal, -1 on the off-diagonals)

For a zero-mean Gaussian the Fisher information is
    I_ab = 1/2 tr( S^-1 S_a S^-1 S_b )

Everything below is a consequence of those four lines.
"""
import json
import numpy as np
from theory_style import plt, tidy, save, SERIES, SEQ

OUT = "figures"


# ---------------------------------------------------------------- exact Fisher
def fisher(n, Q, s2):
    """Exact 2x2 Fisher information about (Q, s2) from n observations.

    Sigma = Q*I + s2*T with T = tridiag(-1, 2, -1), so Sigma and both of its
    derivatives are simultaneously diagonalised by the discrete sine basis.
    Eigenvalues of T are tau_j = 2 - 2 cos(j pi/(m+1)), hence lambda_j =
    Q + s2*tau_j and every trace collapses to an O(m) sum:
        I_QQ  = 1/2 sum lam^-2
        I_Qs  = 1/2 sum tau lam^-2
        I_ss  = 1/2 sum tau^2 lam^-2
    """
    m = n - 1
    if m < 2:
        return np.full((2, 2), np.nan)
    j = np.arange(1, m + 1)
    tau = 2.0 - 2.0 * np.cos(j * np.pi / (m + 1))
    lam2 = (Q + s2 * tau) ** -2
    return 0.5 * np.array([[lam2.sum(), (tau * lam2).sum()],
                           [(tau * lam2).sum(), (tau ** 2 * lam2).sum()]])


def _fisher_dense(n, Q, s2):
    """Reference implementation, used once to verify the spectral form."""
    m = n - 1
    S = (np.diag(np.full(m, Q + 2 * s2)) + np.diag(np.full(m - 1, -s2), 1)
         + np.diag(np.full(m - 1, -s2), -1))
    T = (np.diag(np.full(m, 2.0)) + np.diag(np.full(m - 1, -1.0), 1)
         + np.diag(np.full(m - 1, -1.0), -1))
    Si = np.linalg.inv(S)
    A, B = Si, Si @ T
    return 0.5 * np.array([[np.trace(A @ A), np.trace(A @ B)],
                           [np.trace(B @ A), np.trace(B @ B)]])


assert np.allclose(fisher(37, 0.05, 1.0), _fisher_dense(37, 0.05, 1.0)), "spectral form wrong"


# ------------------------------------------------------- steady-state quantities
def gain(q):
    """Steady-state Kalman gain as a function of q = Q/s2.  K^2 s2 + K Q - Q = 0."""
    return (-q + np.sqrt(q * q + 4 * q)) / 2.0


def dgain(q):
    return 0.5 * (-1.0 + (q + 2.0) / np.sqrt(q * q + 4 * q))


def mse(a, Q, s2):
    """Steady-state tracking MSE of the constant-gain filter m_t=(1-a)m_{t-1}+a x_t."""
    return ((1 - a) ** 2 * Q + a ** 2 * s2) / (1 - (1 - a) ** 2)


# ---------------------------------------------------------------- the accounting
QS = [0.005, 0.05, 0.5]          # matches the probe battery (s2 = 1)
S2 = 1.0
NS = np.array([5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 800, 1200, 2000])

rows, curves = [], {}
for q in QS:
    Q = q * S2
    K = gain(q)
    Sopt = mse(K, Q, S2)
    gK = np.array([dgain(q) / S2, -dgain(q) * Q / S2 ** 2])   # dK/d(Q,s2)
    glq = np.array([1.0 / Q, -1.0 / S2])                      # dlog(q)/d(Q,s2)
    # 21-node Gauss-Hermite over log q -- q is positive, so log is its natural
    # scale.  A quadratic expansion in K is useless here: at n=20, sd(K)/K is
    # already O(1) and the expansion reports penalties of several hundred x.
    gh_x, gh_w = np.polynomial.hermite_e.hermegauss(21)
    gh_w = gh_w / gh_w.sum()

    rel_q, rel_s, rel_k, excess, sdlq = [], [], [], [], []
    for n in NS:
        I = fisher(n, Q, S2)
        C = np.linalg.inv(I)
        rel_q.append(np.sqrt(C[0, 0]) / Q)
        rel_s.append(np.sqrt(C[1, 1]) / S2)
        vK = gK @ C @ gK
        rel_k.append(np.sqrt(vK) / K)
        sd_lq = np.sqrt(glq @ C @ glq)
        sdlq.append(sd_lq)
        qhat = q * np.exp(np.clip(gh_x * sd_lq, -40, 40))
        # Floor the gain at 1/n: an estimator that has only seen n points cannot
        # justify a memory longer than n.  Without this floor the penalty is
        # literally infinite for n <= 10 -- the sampling distribution of qhat puts
        # real mass on gains near zero, and a near-zero gain never tracks at all.
        Khat = np.maximum(gain(qhat), 1.0 / n)
        excess.append(float((gh_w * mse(Khat, Q, S2)).sum() / Sopt - 1.0))
        rows.append(dict(q=q, n=int(n), rel_Q=rel_q[-1], rel_s2=rel_s[-1],
                         rel_K=rel_k[-1], excess_mse=excess[-1],
                         corr=C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])))
    # marginal nats per point: d/dn of (1/2) log det I_n  (Laplace posterior volume)
    fine = np.arange(4, 2001)
    J = np.array([0.5 * np.log(np.linalg.det(fisher(n, Q, S2))) for n in fine])
    curves[q] = dict(n=NS, sdlq=np.array(sdlq), rel_q=np.array(rel_q), rel_s=np.array(rel_s),
                     rel_k=np.array(rel_k), excess=np.array(excess),
                     fine=fine[1:], dJ=np.diff(J), K=K,
                     level_nats=0.5 * np.log(1.0 / (1.0 - K)))

# n needed for a 5% MSE penalty
need = {}
for q in QS:
    c = curves[q]
    ok = np.where(c["excess"] < 0.05)[0]
    need[q] = int(NS[ok[0]]) if len(ok) else None

json.dump(dict(rows=rows, n_for_5pct_excess_mse=need,
               K={q: curves[q]["K"] for q in QS},
               level_nats_per_point={q: curves[q]["level_nats"] for q in QS}),
          open("figures/theory001.json", "w"), indent=1)

print(f"{'q':>7} {'n':>5} {'sd(Q)/Q':>9} {'sd(s2)/s2':>10} {'sd(K)/K':>9} "
      f"{'excessMSE':>10} {'corr':>7}")
for r in rows:
    if r["n"] in (5, 10, 20, 50, 200, 2000):
        print(f"{r['q']:>7} {r['n']:>5} {r['rel_Q']:>9.3f} {r['rel_s2']:>10.4f} "
              f"{r['rel_K']:>9.3f} {r['excess_mse']:>10.4f} {r['corr']:>7.3f}")
print("\nn for <5% excess MSE:", need)
print("K:", {q: round(curves[q]['K'], 4) for q in QS})


# ---------------------------------------------------------------------- figures
# Fig 1: marginal nats per point -- static parameters vs the moving level
fig, ax = plt.subplots(figsize=(6.4, 4.0))
tidy(ax)
for i, q in enumerate(QS):
    c = curves[q]
    ax.loglog(c["fine"], c["dJ"], color=SERIES[i], label=f"(Q, $\\sigma^2$),  q={q}")
    ax.axhline(c["level_nats"], color=SERIES[i], lw=1.4, ls=":")
ax.loglog(curves[QS[0]]["fine"], 1.0 / curves[QS[0]]["fine"],
          color="#8a8880", lw=1.2, ls="--", label="$1/n$ reference")
ax.text(300, curves[0.5]["level_nats"] * 1.15, "level channel (dotted): constant forever",
        fontsize=8.5, color="#52514e")
ax.set_xlabel("n  (observations held)")
ax.set_ylabel("marginal nats contributed by point n")
ax.set_title("Per-point information: static noise parameters decay as $1/n$,\n"
             "the level channel never decays")
ax.legend(loc="lower left")
save(fig, f"{OUT}/fig01-nats-per-point.png")

# Fig 2: small multiples -- relative accuracy of each estimand (no dual axis)
fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.5), sharex=True, sharey=True)
for ax, key, name in zip(axes, ["rel_q", "rel_s", "rel_k"],
                         ["$\\hat Q$", "$\\hat\\sigma^2$", "$\\hat K$"]):
    tidy(ax)
    for i, q in enumerate(QS):
        ax.loglog(curves[q]["n"], curves[q][key], color=SERIES[i],
                  marker="o", label=f"q={q}")
    ax.axhline(0.05, color="#d03b3b", lw=1.2, ls="--")
    ax.text(6, 0.056, "5% error", fontsize=8, color="#d03b3b")
    ax.set_title(f"relative SD of {name}")
    ax.set_xlabel("n")
axes[0].set_ylabel("Cramer-Rao relative SD")
axes[0].legend(loc="lower left")
fig.suptitle("How well can the noises be known from n points?  (exact CRLB, no estimator can beat this)",
             fontsize=11, color="#0b0b0b", y=1.04)
save(fig, f"{OUT}/fig02-crlb-vs-n.png")

# Fig 3: decision-relevant information -- excess MSE from parameter uncertainty
fig, ax = plt.subplots(figsize=(6.4, 4.0))
tidy(ax)
for i, q in enumerate(QS):
    c = curves[q]
    good = c["sdlq"] <= 1.5      # beyond this the normal approx to log q is fiction
    ax.loglog(c["n"][good], c["excess"][good], color=SERIES[i], marker="o",
              label=f"q={q}   (n*={need[q]})")
    ax.loglog(c["n"][~good], np.clip(c["excess"][~good], None, 30), color=SERIES[i],
              marker="o", mfc="none", ls=":", lw=1.2)
ax.axhline(0.05, color="#d03b3b", lw=1.3, ls="--")
ax.text(6, 0.056, "5% excess MSE", fontsize=8.5, color="#d03b3b")
ax.set_xlabel("n  (observations held)")
ax.set_ylabel("$E[S(\\hat K)]/S(K^*) - 1$")
ax.set_title("Decision-relevant information: cost of not knowing the noises\n"
             "is far cheaper than knowing them")
ax.legend()
save(fig, f"{OUT}/fig03-excess-mse.png")
