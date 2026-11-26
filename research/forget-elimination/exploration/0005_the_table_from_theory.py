"""0005 -- the rate table from its definition instead of from simulation.

KL[i, j] is the divergence rate between bank members i and j.  In y = log e^2 a member is
y_t = lam_t + xi_t, xi = log z^2 (variance pi^2/2); treating xi as Gaussian gives Whittle's

    D(A || B) = (1/4pi) int_{-pi}^{pi} [ f_A/f_B - 1 - ln(f_A/f_B) ] dw
    f(w) = s^2 (1 - phi^2) / (1 + phi^2 - 2 phi cos w) + pi^2/2

a function of (phi, s) alone.  Compared with the measured table of 0001, then the walking
star of 0004 is re-run with the formula's rates on the same cached likelihoods.

Run:  python 0005_the_table_from_theory.py   (~6 min; needs the caches of 0001 and 0004)
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rig import (M, MEMBERS, C_RES, CACHE, FIG, cached, lse, neighbours, ceiling_rates, scenarios,
                 scenario_ll, lab)

W = (np.arange(8192) + 0.5) * np.pi / 8192                     # midpoint rule on (0, pi)
def spec(phi, s):
    return s * s * (1 - phi * phi) / (1 + phi * phi - 2 * phi * np.cos(W)) + np.pi ** 2 / 2
def whittle(a, b):
    r = spec(*a) / spec(*b)
    return 0.5 * np.mean(r - 1 - np.log(r))                    # (1/4pi) * 2 * int_0^pi = (1/2) mean

KLT = np.array([[whittle(MEMBERS[i], MEMBERS[j]) if i != j else 0.0 for j in range(M)] for i in range(M)])
KLM, SE, static_ll = pickle.load(open(os.path.join(CACHE, "partA.pkl"), "rb"))
off = ~np.eye(M, dtype=bool)
nb = np.zeros((M, M), bool)
for i in range(M): nb[i, neighbours(i)] = True

def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]

ratio = KLT / np.where(off, KLM, 1.0)
print("Whittle (theory) / measured divergence rate")
print(f"  all 210 ordered pairs : median {np.median(ratio[off]):.2f}  10-90% {np.percentile(ratio[off], 10):.3f}-{np.percentile(ratio[off], 90):.2f}"
      f"   Spearman {spearman(KLT[off], KLM[off]):.3f}")
print(f"  44 neighbour pairs    : median {np.median(ratio[nb]):.2f}  range {ratio[nb].min():.2f}-{ratio[nb].max():.2f}"
      f"   Spearman {spearman(KLT[nb], KLM[nb]):.3f}")
print("  neighbour pairs by the smaller s of the pair (theory/measured, min-median-max):")
for s0 in (0.2, 0.4, 0.8, 1.6, 3.2):
    sel = [ratio[i, j] for i in range(M) for j in neighbours(i) if min(MEMBERS[i][1], MEMBERS[j][1]) == s0]
    kind = [("phi" if MEMBERS[i][1] == MEMBERS[j][1] else "s") for i in range(M) for j in neighbours(i) if min(MEMBERS[i][1], MEMBERS[j][1]) == s0]
    sp = [r for r, k in zip(sel, kind) if k == "phi"]; ss_ = [r for r, k in zip(sel, kind) if k == "s"]
    print(f"    s = {s0}:  phi-neighbours " + (f"{min(sp):.2f} {np.median(sp):.2f} {max(sp):.2f}" if sp else "-")
          + "   s-neighbours " + (f"{min(ss_):.2f} {np.median(ss_):.2f} {max(ss_):.2f}" if ss_ else "-"))
z = np.abs(KLT - KLM)[nb] / np.maximum(SE[nb], 1e-12)
print(f"  measured table's own SE on neighbour pairs: median relative SE {np.median((SE / np.where(off, KLM, 1))[nb]):.2f}")

def rates_from(KL):
    KLs = 0.5 * (KL + KL.T); H = np.zeros((M, M))
    for i in range(M):
        js = neighbours(i); H[i, js] = KLs[i, js] / C_RES
        if H[i].sum() > 0.5: H[i] *= 0.5 / H[i].sum()
    return H
H_M, H_T = ceiling_rates(), rates_from(KLT)
print(f"  ceiling leave-rate per member, theory / measured: median {np.median(H_T.sum(1) / H_M.sum(1)):.2f}"
      f"  range {np.min(H_T.sum(1) / H_M.sum(1)):.2f}-{np.max(H_T.sum(1) / H_M.sum(1)):.2f}")

def star(ll, H, g=1.0, c=C_RES):
    """0004's walking star, with the rate matrix supplied."""
    HD = H - np.diag(H.sum(axis=1))
    T, R, _ = ll.shape
    u = np.zeros(R); w = np.full((R, 3, M), 1.0 / M); Wd = np.zeros((R, 2))
    mix = np.empty((T, R)); path = np.empty((T, R)); off3 = np.array([-g, 0.0, g])
    for t in range(T):
        path[t] = u
        mx = ll[t].max(axis=1)[:, None, None]
        j = np.maximum(w, 1e-300) * np.exp(ll[t][:, None, :] - mx)
        s = np.maximum(j.sum(axis=2), 1e-300); lp = np.log(s) + mx[:, :, 0]
        lq = np.stack([Wd[:, 0] - c, np.zeros(R), Wd[:, 1] - c], axis=1); lq -= lse(lq, keepdims=True)
        mix[t] = lse(lq + lp); post = j / s[..., None]
        Wd = np.maximum(Wd + lp[:, [0, 2]] - lp[:, [1]], 0.0)
        up = (Wd[:, 1] >= c) & (u + g <= 1e-9); dn = (Wd[:, 0] >= c) & (u - g >= -24.0) & ~up
        if up.any(): post[up] = post[up][:, [1, 2, 2]]; u[up] += g; Wd[up] = 0.0
        if dn.any(): post[dn] = post[dn][:, [0, 0, 1]]; u[dn] -= g; Wd[dn] = 0.0
        a = 2.0 ** np.minimum(u[:, None] + off3[None, :], 0.0)
        w = post + a[..., None] * (post @ HD)
    return mix, path

def evaluate(idx, ll, start):
    true_ll = np.take_along_axis(ll, idx[:, :, None], axis=2)[:, :, 0]
    out = {}
    for g in (1.0, 2.0):
        for nm, H in (("measured", H_M), ("theory", H_T)):
            mix, path = star(ll, H, g=g)
            r = (true_ll - mix)[start:].mean(axis=0)
            out[(g, nm)] = (r.mean(), r.std(ddof=1) / np.sqrt(len(r)), float(2 ** np.median(path[start:])))
    return out

res = {}
def run_static():
    acc = {}
    for i in range(M):
        ll = static_ll[i].astype(float)
        for k, v in evaluate(np.full(ll.shape[:2], i), ll, 2000).items(): acc.setdefault(k, []).append(v)
    return {k: (np.mean([x[0] for x in v]), np.std([x[0] for x in v], ddof=1) / np.sqrt(M), float(np.median([x[2] for x in v]))) for k, v in acc.items()}
res["calm"] = cached("0005_static", run_static)
for name, (gen, start) in scenarios(H_M).items():
    def run(name=name, gen=gen, start=start):
        idx, ll = scenario_ll(name, gen); return evaluate(idx, ll, start)
    res[name] = cached("0005_" + name.replace(" ", "_").replace(">", "").replace("(", "").replace(")", ""), run)

print("\nwalking star (c = 1.125): regret nats/step x 1000 with measured vs theory rates; the fraction a it sits at")
print(f"{'scenario':28s}{'g=1 measured':>14s}{'g=1 theory':>12s}{'ratio':>7s}{'a meas':>9s}{'a theory':>9s}   {'g=2 measured':>13s}{'g=2 theory':>12s}{'ratio':>7s}")
for sc, r in res.items():
    a, b, c_, d = r[(1.0, "measured")], r[(1.0, "theory")], r[(2.0, "measured")], r[(2.0, "theory")]
    print(f"{sc:28s}{1e3 * a[0]:14.2f}{1e3 * b[0]:12.2f}{b[0] / a[0]:7.2f}{a[2]:9.4f}{b[2]:9.4f}   {1e3 * c_[0]:13.2f}{1e3 * d[0]:12.2f}{d[0] / c_[0]:7.2f}")

# ---------------------------------------------------------------- figure
INK, MUTED, GRID, BLUE, ORANGE = "#191919", "#666666", "#e6e6e6", "#257de0", "#f47230"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 200})
fig, ax = plt.subplots(figsize=(6.4, 5.4))
far = off & ~nb
ax.plot(KLM[far], KLT[far], ls="none", marker="o", ms=3.5, color=BLUE, alpha=0.45, label="other pairs")
ax.plot(KLM[nb], KLT[nb], ls="none", marker="D", ms=5, color=ORANGE, label="grid neighbours (the ones the rule uses)")
xx = np.array([3e-5, 60]); ax.plot(xx, xx, color=INK, lw=1, ls="--")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(3e-5, 60); ax.set_ylim(3e-5, 60)
ax.set_xlabel("measured by simulation (nats per step)"); ax.set_ylabel("from the formula (nats per step)")
ax.set_title("How separable two bank members are:\nthe closed formula against the simulated table")
ax.legend(fontsize=8.5, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "0005-theory-vs-measured.png"), facecolor="white"); plt.close(fig)
