"""THEORY-005: the four modes as corners of one smooth 2-parameter family.

The previous turn parameterised the two anomalies as MEAN shifts (an oracle who
knows delta) and the two regime changes as COVARIANCE changes.  That split is
what made the location and scale blocks exactly orthogonal -- it was an artefact
of the parameterisation, not a fact about the process.

Marginalise the unknown event size, which is what a non-oracle must do, and an
anomaly becomes a covariance bump too.  Then ALL FOUR modes live in one family:

    Sigma(a, phi, A) = Sigma_0 + A * M(a, phi)
    M(a, phi) = sum_t phi^t [ a G^P_t + (1-a) G^M_t ]

    G^P_t = e_t e_t^T                        process-noise excess at time t
    G^M_t = (e_t - e_{t+1})(e_t - e_{t+1})^T  measurement-noise excess at time t
                                              (truncated to e_t e_t^T at the edge)

Two continuous coordinates, and the four named modes are the CORNERS:

              phi = 0 (impulse)      phi = 1 (step)
    a = 1     PA  process anomaly    PR  process regime
    a = 0     MA  measurement anom.  MR  measurement regime

  * a  in [0,1] is the CHANNEL coordinate  -- which noise source
  * phi in [0,1] is the PERSISTENCE coordinate -- anomaly at 0, regime at 1,
    a genuine continuum in between (an exponential of time constant 1/|log phi|)

All four corners are exactly representable, verified below.  There is no event
detection anywhere in this file: the output is a posterior density over the unit
square, and the allocation is its marginals.
"""
import json
import numpy as np
from theory_style import plt, tidy, save, SERIES, SEQ

OUT = "figures"
S2, Q = 1.0, 0.05
VINC = Q + 2 * S2
CORNERS = {"PA": (1.0, 0.0), "MA": (0.0, 0.0), "PR": (1.0, 1.0), "MR": (0.0, 1.0)}
LAB = {"PA": "process anomaly", "MA": "measurement anomaly",
       "PR": "process regime", "MR": "measurement regime"}


def sigma0(m):
    return (np.diag(np.full(m, VINC)) + np.diag(np.full(m - 1, -S2), 1)
            + np.diag(np.full(m - 1, -S2), -1))


def GP(t, m):
    G = np.zeros((m, m)); G[t, t] = 1.0
    return G


def GM(t, m):
    u = np.zeros(m); u[t] = 1.0
    if t + 1 < m:
        u[t + 1] = -1.0
    return np.outer(u, u)


def M(a, phi, m):
    out = np.zeros((m, m))
    for t in range(m):
        w = phi ** t if (phi > 0 or t == 0) else 0.0
        if w == 0.0:
            continue
        out += w * (a * GP(t, m) + (1 - a) * GM(t, m))
    return out


def kl(S1, S2_):
    """KL( N(0,S1) || N(0,S2) ) -- S1 is the truth."""
    m = S1.shape[0]
    _, l1 = np.linalg.slogdet(S1)
    _, l2 = np.linalg.slogdet(S2_)
    return 0.5 * (np.trace(np.linalg.solve(S2_, S1)) - m + l2 - l1)


# ------------------------------------------------- corners are exact, verify it
m0 = 6
Tt = (np.diag(np.full(m0, 2.0)) + np.diag(np.full(m0 - 1, -1.0), 1)
      + np.diag(np.full(m0 - 1, -1.0), -1))
Tt[0, 0] = 1.0
e0 = np.zeros(m0); e0[0] = 1.0
u01 = np.zeros(m0); u01[0] = 1.0; u01[1] = -1.0
assert np.allclose(M(1, 0, m0), np.outer(e0, e0)), "PA corner wrong"
assert np.allclose(M(0, 0, m0), np.outer(u01, u01)), "MA corner wrong"
assert np.allclose(M(1, 1, m0), np.eye(m0)), "PR corner wrong"
assert np.allclose(M(0, 1, m0), Tt), "MR corner wrong"
print("all four corners exactly representable in M(a, phi)")


# ------------------- the Gram matrix that replaces last turn's orthogonality
def gram(m, ref=None):
    """Fisher correlations among the four corner directions, evaluated at H0."""
    S = sigma0(m)
    Si = np.linalg.inv(S)
    D = {k: M(*v, m) for k, v in CORNERS.items()}
    ks = list(CORNERS)
    F = np.array([[0.5 * np.trace(Si @ D[i] @ Si @ D[j]) for j in ks] for i in ks])
    d = np.sqrt(np.diag(F))
    return F / np.outer(d, d), ks


for m in (2, 5, 20):
    C, ks = gram(m)
    print(f"\nm={m}  Fisher correlation between corner directions")
    print("      " + " ".join(f"{k:>7}" for k in ks))
    for i, k in enumerate(ks):
        print(f"  {k:>3} " + " ".join(f"{C[i, j]:>7.3f}" for j in range(4)))

# ------------------------------------------ the identifiability landscape
# Fisher metric on (a, phi) at a reference deviation, over the unit square
def metric_logdet(a, phi, A, m):
    S = sigma0(m) + A * M(a, phi, m)
    Si = np.linalg.inv(S)
    h = 1e-4
    da = (M(min(a + h, 1.0), phi, m) - M(max(a - h, 0.0), phi, m)) / (min(a + h, 1) - max(a - h, 0))
    dp = (M(a, min(phi + h, 1.0), m) - M(a, max(phi - h, 0.0), m)) / (min(phi + h, 1) - max(phi - h, 0))
    J = np.zeros((2, 2))
    for i, X in enumerate((A * da, A * dp)):
        for j, Y in enumerate((A * da, A * dp)):
            J[i, j] = 0.5 * np.trace(Si @ X @ Si @ Y)
    ev = np.linalg.eigvalsh(J)
    return np.log10(max(ev[0], 1e-300)), np.log10(max(ev[1], 1e-300))


AG = np.linspace(0.0, 1.0, 41)
PG = np.linspace(0.0, 0.985, 41)
land = {}
for m in (5, 20):
    Z = np.zeros((len(PG), len(AG)))
    for i, p in enumerate(PG):
        for j, a in enumerate(AG):
            lo, _ = metric_logdet(a, p, 1.0, m)
            Z[i, j] = lo
    land[m] = Z
print(f"\nweakest-direction Fisher eigenvalue over the square (log10), m=20: "
      f"min {land[20].min():.2f} at a={AG[land[20].min(0).argmin()]:.2f}, "
      f"max {land[20].max():.2f}")


# ---------------------------- expected posterior over the square, per corner
def truth_sigma(corner, m):
    """Sigma under a corner event of a realistic size, built from first principles."""
    S = sigma0(m)
    if corner == "PA":                       # level jump, sd = 4 increment-SD
        return S + 16 * VINC * M(1, 0, m)
    if corner == "MA":                       # outlier, sd = 4 increment-SD
        return S + 16 * VINC * M(0, 0, m)
    if corner == "PR":                       # Q tripled
        return S + 2 * Q * M(1, 1, m)
    if corner == "MR":                       # sigma^2 tripled
        return S + 2 * S2 * M(0, 1, m)


AMPS = np.logspace(-3, 2.2, 60)


def profile_surface(corner, m):
    """Expected log-posterior over (a, phi): -min_A KL(truth || model), exact."""
    St = truth_sigma(corner, m)
    Z = np.zeros((len(PG), len(AG)))
    for i, p in enumerate(PG):
        for j, a in enumerate(AG):
            Mm = M(a, p, m)
            Z[i, j] = -min(kl(St, sigma0(m) + A * Mm) for A in AMPS)
    return Z


surf = {}
for corner in CORNERS:
    for m in (2, 5, 20):
        surf[(corner, m)] = profile_surface(corner, m)


def allocation(Z):
    """Posterior marginals over the channel and persistence coordinates."""
    W = np.exp(Z - Z.max())
    W /= W.sum()
    return (W.sum(0) @ AG), (W.sum(1) @ PG), W


print("\nposterior-mean allocation on the unit square (no thresholds anywhere)")
print(f"{'truth':>6} {'m':>3}   {'E[a] channel':>13} {'E[phi] persist':>15}")
alloc_rows = []
for corner in CORNERS:
    for m in (2, 5, 20):
        ea, ep, _ = allocation(surf[(corner, m)])
        alloc_rows.append(dict(truth=corner, m=m, Ea=float(ea), Ephi=float(ep)))
        print(f"{corner:>6} {m:>3}   {ea:>13.3f} {ep:>15.3f}")

json.dump(dict(alloc=alloc_rows,
               gram={str(m): gram(m)[0].tolist() for m in (2, 5, 20)}),
          open("figures/theory005.json", "w"), indent=1)


# ------------------------------------------------------------------- figures
div = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "div", ["#0d366b", "#3987e5", "#9ec5f4", "#f0efec", "#f0a3a3", "#d03b3b", "#7d1f1f"])
seq = plt.matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)

# Fig 14: the unit square, its corners, and the posterior ridges
fig, axes = plt.subplots(4, 3, figsize=(9.6, 11.4), sharex=True, sharey=True)
for r, corner in enumerate(CORNERS):
    for c, m in enumerate((2, 5, 20)):
        ax = axes[r, c]
        Z = surf[(corner, m)]
        W = np.exp(Z - Z.max())
        ax.imshow(W, origin="lower", aspect="auto", cmap=seq, vmin=0, vmax=1,
                  extent=[AG[0], AG[-1], PG[0], PG[-1]])
        ax.grid(False)
        for k, (a, p) in CORNERS.items():
            ax.plot([a], [min(p, PG[-1])], marker="o", ms=6, mfc="none",
                    mec="#d03b3b" if k == corner else "#52514e", mew=1.6)
            ax.annotate(k, (a, min(p, PG[-1])), textcoords="offset points",
                        xytext=(7 if a < 0.5 else -20, 6 if p < 0.5 else -14),
                        fontsize=8, color="#d03b3b" if k == corner else "#52514e")
        ea, ep, _ = allocation(Z)
        ax.plot([ea], [ep], marker="x", ms=9, mew=2.2, color="#eb6834")
        if r == 0:
            ax.set_title(f"m = {m}", fontsize=10)
        if c == 0:
            ax.set_ylabel(f"truth: {corner}\n$\\varphi$  (persistence)", fontsize=9)
        if r == 3:
            ax.set_xlabel("$a$  (channel: 0 = measurement, 1 = process)")
fig.suptitle("The four modes are corners of one smooth square, not four hypotheses.\n"
             "Shading = exact expected posterior; orange x = the allocation (its mean).",
             fontsize=11.5, color="#0b0b0b", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig(f"{OUT}/fig14-deviation-square.png", bbox_inches="tight")
plt.close(fig)
print("wrote figures/fig14-deviation-square.png")

# Fig 15: identifiability landscape -- where allocation must stay spread
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
for ax, m in zip(axes, (5, 20)):
    tidy(ax)
    im = ax.imshow(land[m], origin="lower", aspect="auto", cmap=seq,
                   extent=[AG[0], AG[-1], PG[0], PG[-1]])
    ax.grid(False)
    for k, (a, p) in CORNERS.items():
        ax.plot([a], [min(p, PG[-1])], marker="o", ms=6, mfc="none", mec="#d03b3b", mew=1.6)
        ax.annotate(k, (a, min(p, PG[-1])), textcoords="offset points",
                    xytext=(7 if a < 0.5 else -20, 6 if p < 0.5 else -14),
                    fontsize=8, color="#d03b3b")
    ax.set_title(f"m = {m}")
    ax.set_xlabel("$a$  (channel)")
    fig.colorbar(im, ax=ax, label="$\\log_{10}$ weakest Fisher eigenvalue")
axes[0].set_ylabel("$\\varphi$  (persistence)")
fig.suptitle("Identifiability is a landscape, not a yes/no. Dark = the data pins the\n"
             "deviation; light = it does not, and the allocation should stay spread.",
             fontsize=11, color="#0b0b0b", y=1.10)
fig.savefig(f"{OUT}/fig15-identifiability-landscape.png", bbox_inches="tight")
plt.close(fig)
print("wrote figures/fig15-identifiability-landscape.png")
