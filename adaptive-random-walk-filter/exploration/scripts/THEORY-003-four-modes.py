"""THEORY-003: the four deviation modes as an exact geometry.

An event occurs between x_{-1} and x_0.  Observe the m increments
d_0 .. d_{m-1},  d_t = x_t - x_{t-1} = w_t + v_t - v_{t-1}.
Under H0 these are stationary MA(1): Sigma0 = tridiag(-s2, Q+2s2, -s2).

The four modes, each a one-parameter family through H0:

  PA  process anomaly      theta_0 += delta        -> mean  += delta*(e0)
  MA  measurement anomaly  x_0     += delta        -> mean  += delta*(e0 - e1)
  PR  process regime       Q -> rho Q  for t>=0    -> Sigma += (rho-1)Q * I
  MR  measurement regime   s2 -> rho s2 for t>=0   -> Sigma += (rho-1)s2 * Ttilde

where Ttilde = tridiag(-1, 2, -1) with the (0,0) entry equal to 1, because d_0
straddles the event (v_{-1} is old noise, v_0 is new).

Two facts fall straight out of Gaussian score algebra and drive everything:

  1. Mean-parameters and covariance-parameters have ORTHOGONAL scores.  So the
     4x4 Fisher matrix is block diagonal: {PA,MA} never mixes with {PR,MR}.
     Location events and scale events are never confusable with each other.
  2. At m=1 the two modes within each block are COLLINEAR -- correlation
     exactly 1, Fisher block singular.  One post-event point carries zero
     discriminating information.  The whole discrimination is manufactured by
     the second point.
"""
import json
import numpy as np
from theory_style import plt, tidy, save, SERIES, SEQ

OUT = "figures"
S2 = 1.0
MODES = ["PA", "MA", "PR", "MR"]
LABEL = {"PA": "process anomaly", "MA": "measurement anomaly",
         "PR": "process regime", "MR": "measurement regime"}


def sigma0(m, Q, s2):
    return (np.diag(np.full(m, Q + 2 * s2)) + np.diag(np.full(m - 1, -s2), 1)
            + np.diag(np.full(m - 1, -s2), -1))


def dmean(mode, m):
    v = np.zeros(m)
    if mode == "PA":
        v[0] = 1.0
    elif mode == "MA":
        v[0] = 1.0
        if m > 1:
            v[1] = -1.0
    return v


def dcov(mode, m, Q, s2):
    if mode == "PR":
        return Q * np.eye(m)                      # d/dlog(rho) at rho=1
    if mode == "MR":
        T = (np.diag(np.full(m, 2.0)) + np.diag(np.full(m - 1, -1.0), 1)
             + np.diag(np.full(m - 1, -1.0), -1))
        T[0, 0] = 1.0                             # d_0 straddles the event
        return s2 * T
    return np.zeros((m, m))


def fisher4(m, Q, s2):
    """4x4 Fisher information from m post-event increments, at H0."""
    S = sigma0(m, Q, s2)
    Si = np.linalg.inv(S)
    F = np.zeros((4, 4))
    for i, a in enumerate(MODES):
        for j, b in enumerate(MODES):
            if a in ("PA", "MA") and b in ("PA", "MA"):
                F[i, j] = dmean(a, m) @ Si @ dmean(b, m)
            elif a in ("PR", "MR") and b in ("PR", "MR"):
                F[i, j] = 0.5 * np.trace(Si @ dcov(a, m, Q, s2) @ Si @ dcov(b, m, Q, s2))
            # cross terms are exactly zero for a Gaussian
    return F


def kl(m1, S1, m2, S2_):
    """KL( N(m1,S1) || N(m2,S2) ) in nats."""
    n = len(m1)
    S2i = np.linalg.inv(S2_)
    dm = m2 - m1
    _, ld1 = np.linalg.slogdet(S1)
    _, ld2 = np.linalg.slogdet(S2_)
    return 0.5 * (np.trace(S2i @ S1) - n + dm @ S2i @ dm + ld2 - ld1)


def hypothesis(mode, m, Q, s2, delta, rho):
    mu = np.zeros(m)
    S = sigma0(m, Q, s2)
    if mode == "PA":
        mu = delta * dmean("PA", m)
    elif mode == "MA":
        mu = delta * dmean("MA", m)
    elif mode == "PR":
        S = S + (rho - 1.0) * Q * np.eye(m)
    elif mode == "MR":
        T = (np.diag(np.full(m, 2.0)) + np.diag(np.full(m - 1, -1.0), 1)
             + np.diag(np.full(m - 1, -1.0), -1))
        T[0, 0] = 1.0
        S = S + (rho - 1.0) * s2 * T
    return mu, S


# ------------------------------------------------------------------- geometry
Q, MS = 0.05, np.arange(1, 41)
res = {"rho_loc": [], "rho_scale": [], "F": {k: [] for k in MODES}}
for m in MS:
    F = fisher4(m, Q, S2)
    rl = F[0, 1] / np.sqrt(F[0, 0] * F[1, 1])
    rs = F[2, 3] / np.sqrt(F[2, 2] * F[3, 3])
    res["rho_loc"].append(rl)
    res["rho_scale"].append(rs)
    for i, k in enumerate(MODES):
        res["F"][k].append(F[i, i])
for k in res:
    if k != "F":
        res[k] = np.array(res[k])
for k in MODES:
    res["F"][k] = np.array(res["F"][k])

# verify the block structure and the m=1 degeneracy
F1, F5 = fisher4(1, Q, S2), fisher4(5, Q, S2)
assert np.allclose(F5[:2, 2:], 0.0), "location/scale blocks are not orthogonal"
print("block-diagonal (location x scale cross terms):", np.abs(F5[:2, 2:]).max())
print(f"m=1  rho_loc={res['rho_loc'][0]:+.6f}   rho_scale={res['rho_scale'][0]:+.6f}")
print(f"m=2  rho_loc={res['rho_loc'][1]:+.6f}   rho_scale={res['rho_scale'][1]:+.6f}")
print(f"m=5  rho_loc={res['rho_loc'][4]:+.6f}   rho_scale={res['rho_scale'][4]:+.6f}")
print(f"m=40 rho_loc={res['rho_loc'][-1]:+.6f}  rho_scale={res['rho_scale'][-1]:+.6f}")

# saturating vs linear: total information available to each mode
print("\nmode information vs m (own-Fisher, arbitrary common units)")
print(f"{'m':>4} " + " ".join(f"{k:>10}" for k in MODES))
for idx in [0, 1, 2, 4, 9, 19, 39]:
    print(f"{MS[idx]:>4} " + " ".join(f"{res['F'][k][idx]:>10.4f}" for k in MODES))

# fraction of the infinite series' information that lives in the first m points
Fbig = fisher4(4000, Q, S2)
frac = {k: res["F"][k] / Fbig[i, i] for i, k in enumerate(MODES)}
print("\nfraction of total (m=4000) information contained in the first m points")
print(f"{'m':>4} " + " ".join(f"{k:>10}" for k in MODES))
for idx in [0, 1, 2, 4, 9, 19]:
    print(f"{MS[idx]:>4} " + " ".join(f"{frac[k][idx]:>10.4f}" for k in MODES))

# ------------------------------------------------- exact pairwise LLR matrices
DELTA = 4.0 * np.sqrt(Q + 2 * S2)     # a 4-sigma location event
RHO = 3.0                             # a 3x scale change
llr = {}
for m in (2, 5, 20):
    M = np.zeros((4, 4))
    for i, a in enumerate(MODES):
        ma, Sa = hypothesis(a, m, Q, S2, DELTA, RHO)
        for j, b in enumerate(MODES):
            mb, Sb = hypothesis(b, m, Q, S2, DELTA, RHO)
            M[i, j] = kl(ma, Sa, mb, Sb)
    llr[m] = M
print(f"\nexpected LLR (nats) separating true row-mode from alt col-mode, "
      f"delta={DELTA:.2f}, rho={RHO}")
for m in (2, 5, 20):
    print(f"  m={m}: " + " | ".join(
        f"{MODES[i]}->{MODES[j]}:{llr[m][i, j]:6.2f}"
        for i in range(4) for j in range(4) if i != j))

json.dump(dict(rho_loc=res["rho_loc"].tolist(), rho_scale=res["rho_scale"].tolist(),
               frac={k: frac[k].tolist() for k in MODES},
               llr={str(m): llr[m].tolist() for m in llr},
               delta=DELTA, rho=RHO, Q=Q),
          open("figures/theory003.json", "w"), indent=1)


# --------------------------------------------------------------------- figures
# Fig 7: within-block confusion and its cost
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
ax = tidy(axes[0])
ax.plot(MS, res["rho_loc"], color=SERIES[0], marker="o", ms=4,
        label="location plane:  PA vs MA")
ax.plot(MS, res["rho_scale"], color=SERIES[1], marker="o", ms=4,
        label="scale plane:  PR vs MR")
ax.axhline(1.0, color="#d03b3b", lw=1.2, ls="--")
ax.text(12, 1.02, "$\\rho=1$: not identifiable", fontsize=8.5, color="#d03b3b")
ax.set_xlabel("m  (post-event increments observed)")
ax.set_ylabel("Fisher correlation $\\rho$ between the two modes")
ax.set_title("One point after an event discriminates nothing.\nThe second point creates the whole distinction.")
ax.legend(loc="center right")

ax = tidy(axes[1])
ax.semilogy(MS[1:], 1.0 / (1.0 - res["rho_loc"][1:] ** 2), color=SERIES[0], marker="o", ms=4,
            label="location plane")
ax.semilogy(MS[1:], 1.0 / (1.0 - res["rho_scale"][1:] ** 2), color=SERIES[1], marker="o", ms=4,
            label="scale plane")
ax.set_xlabel("m  (post-event increments observed)")
ax.set_ylabel("variance inflation  $1/(1-\\rho^2)$")
ax.set_title("Cost of the confusion, in effective sample size\n(infinite at m=1)")
ax.legend()
save(fig, f"{OUT}/fig07-mode-confusion.png")

# Fig 8: the information budget of each mode -- bounded vs unbounded
fig, ax = plt.subplots(figsize=(6.6, 4.2))
tidy(ax)
for i, k in enumerate(MODES):
    ax.plot(MS, res["F"][k] / res["F"][k][1], color=SERIES[i],
            marker="o", ms=3.5, label=f"{k} -- {LABEL[k]}")
ax.set_xlabel("m  (post-event increments observed)")
ax.set_ylabel("information available, relative to m=2")
ax.set_title("Anomaly evidence is a fixed budget, delivered at once.\n"
             "Regime evidence accrues at a constant rate, forever.")
ax.legend(loc="upper left")
save(fig, f"{OUT}/fig08-information-budget.png")

# Fig 9: the "first m points vs the infinite series" question, answered
fig, ax = plt.subplots(figsize=(6.6, 4.2))
tidy(ax)
for i, k in enumerate(MODES):
    ax.semilogy(MS, frac[k], color=SERIES[i], marker="o", ms=3.5,
                label=f"{k} -- {LABEL[k]}")
ax.axhline(1.0, color="#8a8880", lw=1.0, ls="--")
for i, k in enumerate(MODES):
    ax.annotate(f"{frac[k][1]:.3f}", (2, frac[k][1]), textcoords="offset points",
                xytext=(6, -2), fontsize=8, color=SERIES[i])
ax.set_xlabel("m  (post-event increments observed)")
ax.set_ylabel("share of the whole future's information")
ax.set_title("How much do the first m points carry\nversus the rest of the infinite series?")
ax.legend(loc="lower right")
save(fig, f"{OUT}/fig09-share-of-future.png")

# Fig 10: exact pairwise LLR, small multiples over m
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
vmax = max(llr[m].max() for m in llr)
cmap = matplotlib_cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
for ax, m in zip(axes, (2, 5, 20)):
    M = llr[m]
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(range(4), MODES)
    ax.set_yticks(range(4), MODES)
    ax.grid(False)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8.5,
                    color="#0b0b0b" if M[i, j] < 0.55 * vmax else "#fcfcfb")
    ax.set_title(f"m = {m}")
    ax.set_xlabel("alternative")
axes[0].set_ylabel("true mode")
fig.suptitle(f"Expected evidence in nats separating the true mode (row) from the alternative (column)\n"
             f"$\\delta$ = 4 increment-SD, $\\rho$ = 3x,  Q={Q}, $\\sigma^2$={S2}",
             fontsize=10.5, color="#0b0b0b", y=1.10)
fig.colorbar(im, ax=axes, shrink=0.85, label="nats")
fig.savefig(f"{OUT}/fig10-llr-matrix.png", bbox_inches="tight")
plt.close(fig)
print("wrote figures/fig10-llr-matrix.png")
