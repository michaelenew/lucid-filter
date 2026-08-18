"""THEORY-002: how fast does a past measurement stop mattering, and why?

Three separate decay laws, all exact:

A. LEVEL CHANNEL.  Relevance of x_{t-k} for theta_t.
   - marginal (that point alone):   Fisher = 1/(sigma^2 + kQ)   -- hyperbolic
   - incremental (that point given every newer point): geometric
   - optimal influence weight:       a_k = K(1-K)^k              -- geometric
   The gap between marginal and incremental *is* the redundancy the user asked
   to track: "correlation to more recent points".

B. PARAMETER CHANNEL.  Relevance of the n-th point for (Q, sigma^2): 1/n.
   Never geometric.  This is why a level tracker forgets and a noise tracker
   does not.

C. WHAT MAKES A FINITE TAIL OPTIMAL.  Under a stationary (Q, sigma^2) the
   answer is: nothing.  L* = infinity.  A finite tail is optimal only if the
   parameters themselves drift, and then L* has a closed form in the drift rate.
"""
import json
import numpy as np
from theory_style import plt, tidy, save, SERIES

OUT = "figures"
S2 = 1.0
QS = [0.005, 0.05, 0.5]


def gain(q):
    return (-q + np.sqrt(q * q + 4 * q)) / 2.0


def fisher(n, Q, s2):
    m = n - 1
    j = np.arange(1, m + 1)
    tau = 2.0 - 2.0 * np.cos(j * np.pi / (m + 1))
    lam2 = (Q + s2 * tau) ** -2
    return 0.5 * np.array([[lam2.sum(), (tau * lam2).sum()],
                           [(tau * lam2).sum(), (tau ** 2 * lam2).sum()]])


# ------------------------------------------------------------------ A. level
def level_posterior_var(kmax, Q, s2):
    """P_k = Var(theta_t | x_{t-k} ... x_t), exact.

    Estimating theta_t from the k+1 most recent points is the mirror image of
    filtering forward from a diffuse prior for k+1 steps -- the random walk is
    time-reversible, so this is just the ordinary Riccati recursion.
    """
    P = [s2]                                   # after the oldest point alone
    for _ in range(kmax):
        pm = P[-1] + Q
        P.append(pm * s2 / (pm + s2))
    return np.array(P)                          # P[k] uses k+1 points


KMAX = 120
levelA = {}
for q in QS:
    Q, K = q * S2, gain(q)
    P = level_posterior_var(KMAX, Q, S2)
    k = np.arange(KMAX + 1)
    incr = np.concatenate([[np.nan], 0.5 * np.log(P[:-1] / P[1:])])   # nats added by the k-th oldest
    marg = 1.0 / (S2 + k * Q)                                          # that point alone
    weight = K * (1 - K) ** k                                          # BLUE influence
    levelA[q] = dict(k=k, P=P, incr=incr, marg=marg / marg[0],
                     weight=weight / weight[0], K=K,
                     P_inf=(1 - K) * Q / K if K > 0 else np.inf)

# Is the incremental decay rate really (1-K)^2 per step, i.e. weight^2?
for q in QS:
    d = levelA[q]
    j = int(np.max(np.where(np.nan_to_num(d["incr"]) > 1e-12)[0]))   # avoid underflow
    r = d["incr"][j] / d["incr"][j - 1]
    print(f"q={q:<6} K={d['K']:.4f}   incremental-nats ratio {r:.5f} "
          f"vs (1-K)^2={(1-d['K'])**2:.5f}   weight ratio {(1-d['K']):.5f}   "
          f"sqrt(nats) ratio {np.sqrt(r):.5f}")

# ------------------------------------------------------------- B. parameters
NF = np.arange(4, 1201)
paramB = {}
for q in QS:
    J = np.array([0.5 * np.log(np.linalg.det(fisher(n, q * S2, S2))) for n in NF])
    paramB[q] = dict(n=NF[1:], dJ=np.diff(J))

# ------------------------------------------------- C. optimal tail from drift
# per-point Fisher in (log Q, log s2) -- the scale-free coordinates
I1log = {}
for q in QS:
    Q = q * S2
    I1 = (fisher(4001, Q, S2) - fisher(2001, Q, S2)) / 2000.0
    D = np.diag([Q, S2])
    I1log[q] = D @ I1 @ D

D_PARAM = 2
OMEGAS = np.logspace(-4, -1, 40)      # per-step SD of the drift in log Q / log s2


def tail_loss(L, Ilog, omega):
    """Expected excess nats from using a rectangular window of the last L points.

    estimation term : (1/2) tr(Ilog (L Ilog)^-1) = d / (2L)
    staleness term  : (1/2) tr(Ilog S(L)),  S(L) = omega^2 (L-1)(2L-1)/(6L) I
    """
    stale = omega ** 2 * (L - 1) * (2 * L - 1) / (6.0 * L)
    return D_PARAM / (2.0 * L) + 0.5 * np.trace(Ilog) * stale


Lgrid = np.unique(np.round(np.logspace(0.5, 5, 400)).astype(int))
tailC = {}
for q in QS:
    Ilog = I1log[q]
    Lstar_num, Lstar_cf = [], []
    for w in OMEGAS:
        loss = np.array([tail_loss(L, Ilog, w) for L in Lgrid])
        Lstar_num.append(Lgrid[np.argmin(loss)])
        Lstar_cf.append(np.sqrt(3.0 * D_PARAM / (np.trace(Ilog) * w ** 2)))
    tailC[q] = dict(omega=OMEGAS, num=np.array(Lstar_num), cf=np.array(Lstar_cf),
                    trI=float(np.trace(Ilog)))
    print(f"q={q:<6} tr(I1_log)={np.trace(Ilog):.4f}  "
          f"L*(omega=1e-3)={np.sqrt(6.0/(np.trace(Ilog)*1e-6)):.0f}  "
          f"L*(omega=1e-2)={np.sqrt(6.0/(np.trace(Ilog)*1e-4)):.0f}")

json.dump({str(q): dict(K=levelA[q]["K"], trI1log=tailC[q]["trI"],
                        Lstar_1em3=float(np.sqrt(6.0 / (tailC[q]["trI"] * 1e-6))),
                        Lstar_1em2=float(np.sqrt(6.0 / (tailC[q]["trI"] * 1e-4))))
           for q in QS}, open("figures/theory002.json", "w"), indent=1)


# --------------------------------------------------------------------- figures
# Fig 4: the three level-channel decay laws
fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.5), sharex=True)
for ax, q in zip(axes, QS):
    tidy(ax)
    d = levelA[q]
    ax.semilogy(d["k"], d["marg"], color=SERIES[0],
                label="marginal relevance  $1/(\\sigma^2+kQ)$")
    ax.semilogy(d["k"], d["weight"], color=SERIES[1],
                label="optimal influence  $(1-K)^k$")
    ax.semilogy(d["k"][1:], d["incr"][1:] / d["incr"][1], color=SERIES[2],
                label="incremental nats (given newer)")
    ax.set_ylim(1e-6, 2)
    ax.set_title(f"q = {q}   (K = {d['K']:.3f})")
    ax.set_xlabel("k  (lag into the past)")
axes[0].set_ylabel("relevance, normalised to k=0")
axes[0].legend(loc="lower left")
fig.suptitle("Level channel: a point's raw relevance decays hyperbolically, but once you\n"
             "condition on newer points it decays geometrically -- the gap is redundancy",
             fontsize=11, color="#0b0b0b", y=1.07)
save(fig, f"{OUT}/fig04-level-relevance-decay.png")

# Fig 5: level vs parameter memory, same axes
fig, ax = plt.subplots(figsize=(6.6, 4.2))
tidy(ax)
for i, q in enumerate(QS):
    d = levelA[q]
    ax.loglog(d["k"][1:], d["incr"][1:], color=SERIES[i], ls="-",
              label=f"level channel, q={q}")
    b = paramB[q]
    ax.loglog(b["n"], b["dJ"], color=SERIES[i], ls="--")
ax.text(200, 0.004, "dashed: $(Q,\\sigma^2)$ channel, $\\propto 1/n$", fontsize=9, color="#52514e")
ax.text(9, 2e-9, "solid: level channel,\ngeometric cutoff at $k\\sim1/K$", fontsize=9, color="#52514e")
ax.set_ylim(1e-10, 1)
ax.set_xlabel("k or n  (points into the past)")
ax.set_ylabel("marginal nats from that point")
ax.set_title("The two channels have incompatible memory laws\n"
             "-- one truncation length cannot serve both")
ax.legend(loc="lower left")
save(fig, f"{OUT}/fig05-two-memory-laws.png")

# Fig 6: optimal tail length vs parameter drift rate
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
ax = tidy(axes[0])
for i, q in enumerate(QS):
    t = tailC[q]
    ax.loglog(t["omega"], t["num"], color=SERIES[i], marker="o", ms=3, ls="none",
              label=f"q={q}, numeric")
    ax.loglog(t["omega"], t["cf"], color=SERIES[i], lw=1.4)
ax.set_xlabel("$\\omega$  (per-step SD of drift in $\\log Q,\\ \\log\\sigma^2$)")
ax.set_ylabel("$L^*$  (optimal tail length)")
ax.set_title("Optimal tail length is set by the hyper-drift,\nnot by the estimation problem")
ax.legend(loc="upper right")

ax = tidy(axes[1])
q = 0.05
for i, w in enumerate([1e-3, 3e-3, 1e-2]):
    loss = np.array([tail_loss(L, I1log[q], w) for L in Lgrid])
    ax.loglog(Lgrid, loss, color=SERIES[i], label=f"$\\omega$={w:g}")
    Ls = Lgrid[np.argmin(loss)]
    ax.plot([Ls], [loss.min()], marker="o", ms=7, color=SERIES[i],
            mec="#fcfcfb", mew=1.5)
ax.loglog(Lgrid, D_PARAM / (2.0 * Lgrid), color="#8a8880", ls="--", lw=1.2,
          label="estimation term alone ($\\omega$=0)")
ax.set_xlabel("L  (tail length)")
ax.set_ylabel("expected excess nats")
ax.set_title(f"Loss vs tail length, q={q}\n(with $\\omega$=0 the minimum is at $L=\\infty$)")
ax.legend(loc="lower left")
save(fig, f"{OUT}/fig06-optimal-tail.png")
