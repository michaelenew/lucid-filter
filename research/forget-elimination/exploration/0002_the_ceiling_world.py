"""0002 -- the resolvability ceiling as the model: (phi, s) move as fast as can be resolved.

The truth is taken to be a time-homogeneous chain over the bank's members with rate
`h_ij = KL_ij / c` from i to j (one over the pair's resolution time; KL symmetrised).  The
weight rule is the exact Bayes filter for that chain:

    w <- (w * likelihood) normalised;   w <- w @ P,   P_ij = h_ij,  P_ii = 1 - sum_j h_ij

  local : j ranges over grid neighbours of i only        (the parameters drift)
  full  : j ranges over all members, rows capped at 1/2  (any pair may swap)
  c     : 1.125 nats (spacing 1.5 blur widths)  or  8.08 nats (ln 1/eps0)

Scored against `forget = 0.999` and clamp-8.08 from 0001 on: a static truth; hops to any
member; hops to a neighbour; and a truth DRAWN FROM the local chain (the world at the
ceiling), where the matching local rule is exactly Bayes-optimal.

Run:  python 0002_the_ceiling_world.py    (~8 min cold; needs 0001's _cache/partA.pkl)
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rig import (M, MEMBERS, C0, CACHE, FIG, cached, lse, member_loglik, simulate,
                 neighbours, draws_from_path, lab)

C_RES = 1.5 ** 2 / 2                                   # 1.125 nats
KL, SE, static_ll = pickle.load(open(os.path.join(CACHE, "partA.pkl"), "rb"))
KLs = 0.5 * (KL + KL.T)
T_A, R_A = static_ll[0].shape[:2]

def chain(c, local):
    P = np.zeros((M, M))
    for i in range(M):
        js = neighbours(i) if local else [j for j in range(M) if j != i]
        P[i, js] = KLs[i, js] / c
        tot = P[i].sum()
        if tot > 0.5: P[i] *= 0.5 / tot
        P[i, i] = 1 - P[i].sum()
    return P

CHAINS = {(c, loc): chain(c, loc) for c in (C_RES, C0) for loc in (True, False)}

def replay(ll, rule, **kw):
    T, R, _ = ll.shape
    logw = np.full((R, M), -np.log(M)); mix = np.empty((T, R))
    for t in range(T):
        mix[t] = lse(logw + ll[t])
        if rule == "forget":
            logw = kw["f"] * logw + ll[t]
        elif rule == "clamp":
            logw = logw + ll[t]; logw = np.maximum(logw, logw.max(axis=1, keepdims=True) - kw["c"])
        elif rule == "chain":
            logw = logw + ll[t]; logw -= lse(logw, keepdims=True)
            logw = np.log(np.exp(logw) @ kw["P"] + 1e-300)
        elif rule == "share":
            logw = logw + ll[t]; logw -= lse(logw, keepdims=True)
            logw = np.logaddexp(np.log1p(-kw["a"]) + logw, np.log(kw["a"] / M))
        logw = logw - lse(logw, keepdims=True)
    return mix

RULES = [("forget 0.999 (shipped)", "forget", dict(f=0.999)),
         (f"clamp {C0:.2f}", "clamp", dict(c=C0)),
         (f"ceiling local, c = {C_RES:.3f}", "chain", dict(P=CHAINS[(C_RES, True)])),
         (f"ceiling local, c = {C0:.2f}", "chain", dict(P=CHAINS[(C0, True)])),
         (f"ceiling full,  c = {C_RES:.3f}", "chain", dict(P=CHAINS[(C_RES, False)])),
         (f"ceiling full,  c = {C0:.2f}", "chain", dict(P=CHAINS[(C0, False)]))]

def score(idx, ll, burn):
    true_ll = np.take_along_axis(ll, idx[:, :, None], axis=2)[:, :, 0]
    out = {}
    for name, rule, kw in RULES:
        reg = (true_ll - replay(ll, rule, **kw))[burn:].mean(axis=0)
        out[name] = (reg.mean(), reg.std(ddof=1) / np.sqrt(len(reg)))
    return out

def ceiling_world(T, R, c, seed):
    """Truth drawn from the local chain at level c."""
    rng = np.random.default_rng(seed); P = CHAINS[(c, True)]
    idx = np.empty((T, R), int); cur = rng.integers(M, size=R)
    cum = np.cumsum(P, axis=1)
    for t in range(T):
        u = rng.random(R)
        cur = np.array([np.searchsorted(cum[i], x) for i, x in zip(cur, u)]).clip(0, M - 1)
        idx[t] = cur
    return idx, draws_from_path(idx, rng)

R_B = 8
results, dwell = {}, {}
def run_static():
    acc = {}
    for i in range(M):
        for name, (reg, _) in score(np.full((T_A, R_A), i), static_ll[i], 2000).items():
            acc.setdefault(name, []).append(reg)
    return {k: (np.mean(v), np.std(v, ddof=1) / np.sqrt(M), np.array(v)) for k, v in acc.items()}
results["static"] = cached("0002_static", run_static)

for hop, D, T in [("any", 100, 60000), ("any", 1000, 60000), ("any", 10000, 120000),
                  ("nbr", 100, 60000), ("nbr", 1000, 60000), ("nbr", 10000, 120000)]:
    def run(hop=hop, D=D, T=T):
        idx, e = simulate(T, R_B, D, hop, 5000 + D + (0 if hop == "any" else 7))
        return score(idx, member_loglik(e), D)
    results[f"{hop} {D}"] = cached(f"0002_{hop}_{D}", run)

for c in (C0, C_RES):
    def run(c=c):
        idx, e = ceiling_world(60000, R_B, c, 900 + int(10 * c))
        d = 1.0 / max((idx[1:] != idx[:-1]).mean(), 1e-9)
        return score(idx, member_loglik(e), 2000), d
    results[f"world c={c:.2f}"], dwell[c] = cached(f"0002_world_{c:.3f}", run)

cols = list(results)
print("regret, nats/step x 1000, against the currently-true member (mean; SE in the .pkl)")
print(f"{'rule':30s}" + "".join(f"{c:>14s}" for c in cols))
for name, _, _ in RULES:
    print(f"{name:30s}" + "".join(f"{1e3 * results[c][name][0]:14.2f}" for c in cols))
print("SE x 1000")
for name, _, _ in RULES:
    print(f"{name:30s}" + "".join(f"{1e3 * results[c][name][1]:14.2f}" for c in cols))
print(f"\nmean dwell in the ceiling worlds: c = {C0:.2f}: {dwell[C0]:.0f} steps;  c = {C_RES:.3f}: {dwell[C_RES]:.0f} steps")
for c, loc in CHAINS:
    P = CHAINS[(c, loc)]; leave = 1 - np.diag(P)
    print(f"chain c={c:.3f} {'local' if loc else 'full '}: leave-rate per step min {leave.min():.4f} "
          f"median {np.median(leave):.4f} max {leave.max():.4f}")
nm = f"ceiling local, c = {C0:.2f}"
st = results["static"][nm][2]; leave = 1 - np.diag(CHAINS[(C0, True)])
print(f"\nQ2: static regret of '{nm}' per member vs that member's total neighbour rate:")
print("   " + "  ".join(f"{lab(i)} {1e3 * st[i]:.2f}/{1e3 * leave[i]:.2f}" for i in range(M)))
print(f"   ratio regret / rate: median {np.median(st / leave):.2f}, range {np.min(st / leave):.2f}-{np.max(st / leave):.2f}")

# ---------------------------------------------------------------- figure
INK, MUTED, GRID = "#191919", "#666666", "#e6e6e6"
BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 200})
show = [("forget 0.999 (shipped)", BLUE), (f"clamp {C0:.2f}", ORANGE), (f"ceiling local, c = {C0:.2f}", AQUA)]
groups = ["static", "nbr 10000", "nbr 1000", "nbr 100", f"world c={C0:.2f}", f"world c={C_RES:.2f}"]
labels = ["calm\n(no change)", "neighbour hop\nevery 10,000", "neighbour hop\nevery 1,000", "neighbour hop\nevery 100",
          f"at the ceiling\n(resolved = {C0:.1f} nats)", f"at the ceiling\n(resolved = {C_RES:.1f} nats)"]
fig, ax = plt.subplots(figsize=(11, 4.6))
w = 0.26
for k, (name, col) in enumerate(show):
    y = [1e3 * results[g][name][0] for g in groups]
    xs = np.arange(len(groups)) + (k - 1) * w
    ax.bar(xs, y, width=w - 0.03, color=col, label=name)
    for x, v in zip(xs, y):
        ax.text(x, v * 1.12, f"{v:.2g}" if v < 10 else f"{v:.0f}", ha="center", fontsize=7.5, color=INK)
ax.set_yscale("log"); ax.set_xticks(range(len(groups))); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("extra code length, nats per step x 1000\n(lower is better)")
ax.set_title("Designing for the fastest resolvable change: what it costs when the world is calmer, what it buys at the ceiling")
ax.legend(fontsize=8.5, loc="upper left"); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "0002-ceiling-world.png"), facecolor="white"); plt.close(fig)
