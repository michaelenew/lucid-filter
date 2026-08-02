"""THEORY-006: allocation as a smooth field, with no detection anywhere.

Two claims to make concrete.

  (1) "The two-point rule isn't a rule, it's happenstance past a threshold."
      Correct.  The underlying object is a SMOOTH SURFACE in (event size, m).
      99:1-at-m=2 is one contour drawn on it.  Shown in fig18: the contours are
      a smooth family and nothing switches on anywhere.

  (2) "The right approach differentiates the modes along a gradient and handles
      all modes at once, allocating change optimally among them."
      Built here.  At each step the allocator returns
        - E[level change | data]   -- how much of what was seen is real
        - E[a | data]              -- which channel it came from
        - E[phi | data]            -- how persistent it is
      all as continuous fields over the observed increments.  No test, no gate,
      no event time.  The Gaussian baseline is a flat plane (a constant fraction
      regardless of magnitude, which is exactly why a plain Kalman filter cannot
      react to a jump); the mixture is a smooth surface that bends.
"""
import json
import numpy as np
from theory_style import plt, tidy, save, SERIES, SEQ

OUT = "figures"
S2, Q = 1.0, 0.05
VINC = Q + 2 * S2
SD = np.sqrt(VINC)
CORNERS = {"PA": (1.0, 0.0), "MA": (0.0, 0.0), "PR": (1.0, 1.0), "MR": (0.0, 1.0)}


def sigma0(m):
    return (np.diag(np.full(m, VINC)) + np.diag(np.full(m - 1, -S2), 1)
            + np.diag(np.full(m - 1, -S2), -1))


def M(a, phi, m):
    out = np.zeros((m, m))
    for t in range(m):
        w = 1.0 if t == 0 else phi ** t
        if w == 0.0:
            continue
        out[t, t] += w * a
        u = np.zeros(m); u[t] = 1.0
        if t + 1 < m:
            u[t + 1] = -1.0
        out += w * (1 - a) * np.outer(u, u)
    return out


# ------------------------------------------------ the per-step allocator (m=2)
M2 = 2
AG = np.linspace(0.0, 1.0, 13)
PG = np.linspace(0.0, 0.96, 13)
AMPS = np.concatenate([[0.0], np.logspace(-2, 2.0, 28)])   # 0 = "nothing happened"
# log-uniform prior over amplitude, with a 50% point mass on "nothing"
PRI_A = np.concatenate([[0.5], np.full(28, 0.5 / 28)])

_cols = {k: [] for k in
         ("a", "p", "A", "s00", "s01", "s11", "ld", "gs0", "gs1", "lw")}
_S0 = sigma0(M2)
for a_ in AG:
    for ph in PG:
        Mm = M(a_, ph, M2)
        for iA, A in enumerate(AMPS):
            S = _S0 + A * Mm
            Si = np.linalg.inv(S)
            gs = np.array([Q + A * a_, Q + A * a_ * ph]) @ Si   # Cov(w_0+w_1, d) Si
            _cols["a"].append(a_); _cols["p"].append(ph); _cols["A"].append(A)
            _cols["s00"].append(Si[0, 0]); _cols["s01"].append(Si[0, 1])
            _cols["s11"].append(Si[1, 1])
            _cols["ld"].append(np.linalg.slogdet(S)[1])
            _cols["gs0"].append(gs[0]); _cols["gs1"].append(gs[1])
            _cols["lw"].append(np.log(PRI_A[iA] / (len(AG) * len(PG))))
H = {k: np.asarray(v) for k, v in _cols.items()}
NZ = H["A"] > 0
print(f"allocator mixture: {H['a'].size} components over (a, phi, A)", flush=True)


def allocate(d0, d1):
    """Vectorised over arrays of (d0, d1). Returns E[level change], E[a], E[phi]."""
    d0 = np.atleast_1d(np.asarray(d0, float)).ravel()
    d1 = np.atleast_1d(np.asarray(d1, float)).ravel()
    lev = np.empty_like(d0); ea = np.empty_like(d0); ep = np.empty_like(d0)
    CH = 256
    for s0 in range(0, d0.size, CH):
        x0, x1 = d0[s0:s0 + CH][None, :], d1[s0:s0 + CH][None, :]
        quad = (H["s00"][:, None] * x0 ** 2 + 2 * H["s01"][:, None] * x0 * x1
                + H["s11"][:, None] * x1 ** 2)
        lg = H["lw"][:, None] - 0.5 * (H["ld"][:, None] + quad)
        lg -= lg.max(0, keepdims=True)
        W = np.exp(lg, out=lg)
        W /= W.sum(0, keepdims=True)
        lev[s0:s0 + CH] = (W * (H["gs0"][:, None] * x0 + H["gs1"][:, None] * x1)).sum(0)
        # (a, phi) are only defined where something actually happened, so the
        # channel/persistence allocation conditions on the A>0 components.
        Wd = W[NZ]
        Wd = Wd / np.maximum(Wd.sum(0, keepdims=True), 1e-300)
        ea[s0:s0 + CH] = Wd.T @ H["a"][NZ]
        ep[s0:s0 + CH] = Wd.T @ H["p"][NZ]
    return lev, ea, ep


# Gaussian baseline: the plain filter, one component, A = 0
Si0 = np.linalg.inv(sigma0(M2))
g0 = np.array([Q, Q])
gs_g = g0 @ Si0

GRID = np.linspace(-6, 6, 97) * SD
X0, X1 = np.meshgrid(GRID, GRID)
_lev, _ea, _ep = allocate(X0.ravel(), X1.ravel())
LEV, EA, EP = (_lev.reshape(X0.shape), _ea.reshape(X0.shape), _ep.reshape(X0.shape))
GAU = gs_g[0] * X0 + gs_g[1] * X1

print("\nallocator behaviour on the two diagnostic rays "
      "(level change / observed d0)", flush=True)
print(f"{'d0/SD':>6} {'no reversal':>12} {'full reversal':>14} {'Gaussian':>10}")
for sv in (0.5, 1, 2, 3, 4, 6, 8):
    d = sv * SD
    lev_j = allocate(d, 0.0)[0][0]
    lev_o = allocate(d, -d)[0][0]
    print(f"{sv:>6.1f} {lev_j / d:>12.3f} {lev_o / d:>14.3f} {gs_g[0]:>10.3f}",
          flush=True)

# ------------------------------- the "two-point rule" as a contour on a surface
def profile_evidence(St, Mm, m):
    """min_A KL(St || Sigma0 + A M), fast: one generalised eig, then O(m) per A."""
    S0 = sigma0(m)
    w, V = np.linalg.eigh(S0)
    R = V @ np.diag(w ** -0.5) @ V.T            # Sigma0^{-1/2}
    lam, U = np.linalg.eigh(R @ Mm @ R)
    B = np.diag(U.T @ R @ St @ R @ U)
    ld0 = np.linalg.slogdet(S0)[1]
    ldt = np.linalg.slogdet(St)[1]
    best = np.inf
    for A in AMPS:
        den = 1.0 + A * lam
        if np.any(den <= 0):
            continue
        val = 0.5 * ((B / den).sum() - m + ld0 + np.log(den).sum() - ldt)
        best = min(best, val)
    return best


SGRID = np.linspace(0.5, 8.0, 22)               # jump size, in increment-SD
MGRID = [2, 3, 4, 6, 8, 12, 16, 24, 32]
SURF = np.zeros((len(SGRID), len(MGRID)))
for i, s in enumerate(SGRID):
    for j, m in enumerate(MGRID):
        St = sigma0(m) + (s * SD) ** 2 * M(1.0, 0.0, m)
        Ms = {k: M(*v, m) for k, v in CORNERS.items()}
        # worst-case evidence for the process-anomaly corner against the others
        k0 = profile_evidence(St, Ms["PA"], m)
        SURF[i, j] = min(profile_evidence(St, Ms[k], m) - k0
                         for k in ("MA", "PR", "MR"))
print(f"\nevidence surface: min {SURF.min():.2f}, max {SURF.max():.2f} nats")
print("the 4.6-nat (99:1) contour is one level set on this surface, nothing more")

# -------------------- detection vs attribution: the three questions separate
DECOMP_M = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 26, 32]
decomp = {k: [] for k in ("H0", "MA", "PR", "MR")}
for m in DECOMP_M:
    St = sigma0(m) + (4 * SD) ** 2 * M(1.0, 0.0, m)
    k0 = profile_evidence(St, M(1.0, 0.0, m), m)
    S0 = sigma0(m)
    decomp["H0"].append(0.5 * (np.trace(np.linalg.solve(S0, St)) - m
                               + np.linalg.slogdet(S0)[1]
                               - np.linalg.slogdet(St)[1]) - k0)
    for k in ("MA", "PR", "MR"):
        decomp[k].append(profile_evidence(St, M(*CORNERS[k], m), m) - k0)
print("\nfor a 4-SD jump: evidence against each rival, nats "
      "(non-oracle, amplitude profiled)")
print(f"{'m':>4} " + " ".join(f"{k:>8}" for k in decomp))
for i, m in enumerate(DECOMP_M):
    print(f"{m:>4} " + " ".join(f"{decomp[k][i]:>8.2f}" for k in decomp))

json.dump(dict(grid_sd=(GRID / SD).tolist(), decomp=decomp, decomp_m=DECOMP_M,
               surf=SURF.tolist(), sgrid=SGRID.tolist(), mgrid=MGRID),
          open("figures/theory006.json", "w"), indent=1)


# ------------------------------------------------------------------- figures
div = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "div", ["#0d366b", "#3987e5", "#9ec5f4", "#f0efec", "#f0a3a3", "#d03b3b", "#7d1f1f"])
seq = plt.matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
ext = [GRID[0] / SD, GRID[-1] / SD, GRID[0] / SD, GRID[-1] / SD]

# Fig 16: the allocation fields
fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9))
for ax, Z, ttl, cm, kw in [
        (axes[0], GAU / SD, "Gaussian filter: a flat plane\n(constant fraction, cannot react)",
         div, dict(vmin=-6, vmax=6)),
        (axes[1], LEV / SD, "Mixture allocator: a smooth surface\n$E[\\Delta\\theta \\mid d]$, no threshold",
         div, dict(vmin=-6, vmax=6)),
        (axes[2], EA, "Channel allocation $E[a\\mid d,\\,\\mathrm{deviation}]$\n0 = measurement, 1 = process",
         seq, dict(vmin=0, vmax=1))]:
    tidy(ax)
    im = ax.imshow(Z, origin="lower", extent=ext, cmap=cm, aspect="equal", **kw)
    ax.grid(False)
    ax.plot([-6, 6], [0, 0], color="#52514e", lw=0.8, ls=":")
    ax.plot([-6, 6], [6, -6], color="#52514e", lw=0.8, ls=":")
    ax.set_xlabel("$d_0$ / increment SD")
    ax.set_title(ttl, fontsize=9.5)
    fig.colorbar(im, ax=ax, shrink=0.86)
axes[0].set_ylabel("$d_1$ / increment SD")
axes[1].text(1.6, 0.45, "no reversal\n= jump", fontsize=8, color="#0b0b0b")
axes[1].text(-5.6, 4.0, "reversal\n= outlier", fontsize=8, color="#0b0b0b")
fig.suptitle("Allocation is a field over what was observed, not a decision about it",
             fontsize=11.5, color="#0b0b0b", y=1.07)
fig.savefig(f"{OUT}/fig16-allocation-field.png", bbox_inches="tight")
plt.close(fig)
print("wrote figures/fig16-allocation-field.png")

# Fig 17: the gradient, sliced -- smooth in magnitude, smooth in reversal
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
ax = tidy(axes[0])
ss = np.linspace(0.1, 8, 80)
for i, rev in enumerate([0.0, -0.25, -0.5, -0.75, -1.0]):
    y = allocate(ss * SD, rev * ss * SD)[0] / (ss * SD)
    ax.plot(ss, y, color=SEQ[i + 1], label=f"$d_1/d_0$ = {rev:g}")
ax.axhline(gs_g[0], color="#8a8880", ls="--", lw=1.4,
           label="Gaussian filter (flat)")
ax.set_xlabel("$|d_0|$ / increment SD")
ax.set_ylabel("fraction of $d_0$ allocated to the level")
ax.set_title("Absorption: smooth in magnitude and in reversal", fontsize=10)
ax.legend(loc="center right", fontsize=8)

ax = tidy(axes[1])
im = ax.imshow(SURF, origin="lower", aspect="auto", cmap=seq,
               extent=[0, len(MGRID) - 1, SGRID[0], SGRID[-1]], vmin=0, vmax=20)
cs = ax.contour(np.linspace(0, len(MGRID) - 1, len(MGRID)), SGRID, SURF,
                levels=[1.0, 2.2, 4.6, 9.2, 13.8], colors="#fcfcfb", linewidths=1.2)
ax.clabel(cs, fmt={1.0: "1", 2.2: "2.2 (90%)", 4.6: "4.6 (99%)",
                   9.2: "9.2", 13.8: "13.8"}, fontsize=7.5)
ax.set_xticks(range(len(MGRID)), [str(m) for m in MGRID])
ax.grid(False)
ax.set_xlabel("m  (points observed)")
ax.set_ylabel("jump size / increment SD")
ax.set_title("The '2 points' result is one contour on a surface", fontsize=10)
fig.colorbar(im, ax=ax, label="nats")
fig.suptitle("Worst-case evidence for the process-anomaly corner. Nothing switches on anywhere.",
             fontsize=11, color="#0b0b0b", y=1.04)
fig.savefig(f"{OUT}/fig17-gradient-slices.png", bbox_inches="tight")
plt.close(fig)
print("wrote figures/fig17-gradient-slices.png")


# Fig 18: detection is fast, channel attribution is fast, persistence is not
fig, ax = plt.subplots(figsize=(6.8, 4.2))
tidy(ax)
NAMES = {"H0": "vs $H_0$  -- did anything happen?",
         "MA": "vs MA  -- which channel? ($a$)",
         "PR": "vs PR  -- how persistent? ($\\varphi$)",
         "MR": "vs MR  -- channel + persistence"}
for i, k in enumerate(("H0", "MA", "PR", "MR")):
    ax.plot(DECOMP_M, decomp[k], color=SERIES[i], marker="o", ms=4, label=NAMES[k])
ax.axhline(4.6, color="#d03b3b", lw=1.2, ls="--")
ax.text(2.3, 4.9, "99:1", fontsize=8.5, color="#d03b3b")
ax.set_xlabel("m  (points observed)")
ax.set_ylabel("evidence, nats")
ax.set_title("A 4-SD jump, non-oracle: detection and channel resolve at once,\n"
             "persistence never does")
ax.legend(loc="lower right", fontsize=8.5)
save(fig, f"{OUT}/fig18-detection-vs-attribution.png")
