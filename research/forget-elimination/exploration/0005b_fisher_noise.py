"""0005b -- the Whittle table with the Fisher-equivalent noise level.

0005 treated xi = log z^2 as Gaussian with its VARIANCE pi^2/2 = 4.93 and came out 2-7x too
small, worst at small s.  The information a single draw carries about its log-variance is
I = 1/2 (resolvable-regime ledger), i.e. xi acts like Gaussian noise of variance 1/I = 2,
not 4.93.  Here the same formula is evaluated with 2 in place of pi^2/2.

Run:  python 0005b_fisher_noise.py   (seconds; needs 0001's _cache/partA.pkl)
"""
import os, pickle
import numpy as np
from rig import M, MEMBERS, CACHE, neighbours

W = (np.arange(8192) + 0.5) * np.pi / 8192
def table(noise):
    def spec(phi, s): return s * s * (1 - phi * phi) / (1 + phi * phi - 2 * phi * np.cos(W)) + noise
    out = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            if i != j:
                r = spec(*MEMBERS[i]) / spec(*MEMBERS[j]); out[i, j] = 0.5 * np.mean(r - 1 - np.log(r))
    return out
KLM = pickle.load(open(os.path.join(CACHE, "partA.pkl"), "rb"))[0]
nb = np.zeros((M, M), bool)
for i in range(M): nb[i, neighbours(i)] = True
off = ~np.eye(M, dtype=bool)
for label, noise in (("variance  pi^2/2 = 4.93", np.pi ** 2 / 2), ("Fisher    1/I    = 2.00", 2.0)):
    R = table(noise) / np.where(off, KLM, 1.0)
    print(f"noise = {label}:  theory/measured on the 44 neighbour pairs: median {np.median(R[nb]):.2f}, range {R[nb].min():.2f}-{R[nb].max():.2f};"
          f"  all pairs median {np.median(R[off]):.2f}, 10-90% {np.percentile(R[off], 10):.2f}-{np.percentile(R[off], 90):.2f}")
    for s0 in (0.2, 0.4, 0.8, 1.6, 3.2):
        sel = [R[i, j] for i in range(M) for j in neighbours(i) if min(MEMBERS[i][1], MEMBERS[j][1]) == s0]
        print(f"     neighbours with smaller s = {s0}: min {min(sel):.2f}  median {np.median(sel):.2f}  max {max(sel):.2f}")
print(f"small-signal expectation for the variance version: (I_gauss / I_true)^2 = ((2/pi^2) / (1/2))^2 = {((2 / np.pi ** 2) / 0.5) ** 2:.3f}")

# figure: both versions against the measured table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rig import FIG
INK, MUTED, GRID, BLUE, ORANGE = "#191919", "#666666", "#e6e6e6", "#257de0", "#f47230"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 200})
fig, ax = plt.subplots(figsize=(6.6, 5.4))
ax.plot(KLM[nb], table(np.pi ** 2 / 2)[nb], ls="none", marker="o", ms=5, color=BLUE, label="formula, noise set by its variance (4.93)")
ax.plot(KLM[nb], table(2.0)[nb], ls="none", marker="D", ms=5, color=ORANGE, label="formula, noise set by its information (2)")
xx = np.array([3e-5, 0.5]); ax.plot(xx, xx, color=INK, lw=1, ls="--"); ax.text(6e-5, 4.2e-5, "formula = simulation", fontsize=8, color=MUTED, rotation=40)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(3e-5, 0.5); ax.set_ylim(1e-5, 0.5)
ax.set_xlabel("measured by simulation (nats per step)"); ax.set_ylabel("from the closed formula (nats per step)")
ax.set_title("How separable two neighbouring bank members are:\nthe closed formula against the simulated table (44 pairs)")
ax.legend(fontsize=8.5, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "0005-theory-vs-measured.png"), facecolor="white"); plt.close(fig)
