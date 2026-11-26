"""0004 -- the drift rate tracked by a walking star instead of a fixed ladder.

    u = log2(a),  a = fraction of the resolvability ceiling,  u <= 0
    star: nodes at u - g, u, u + g (arms clipped to the ceiling); each filters the bank
          under P(2^node) = I + 2^node (H - diag)
    arm lead:  W <- max(0, W + lp_arm - lp_centre)          (never behind the centre)
    recentre:  when W >= c the star moves one node that way and both leads restart at 0
    predict :  mixture of the three nodes, arm weight exp(W - c) against the centre's 1

Starts at the ceiling (u = 0).  Compared with 0003's ladder on identical likelihoods.

Run:  python 0004_the_walking_star.py   (~12 min cold; needs 0001's _cache/partA.pkl)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rig import (M, C0, C_RES, CACHE, FIG, HALF, cached, lse, ceiling_rates, scenarios, scenario_ll,
                 ladder_rungs, ladder_mix, simple_rule, ALPHAS)
import pickle

H = ceiling_rates()
HD = H - np.diag(H.sum(axis=1))
U_MIN = -24.0

def star(ll, g=1.0, c=C_RES, u0=0.0):
    """Returns (mixture log-predictive (T, R), centre path u (T, R), number of moves (R,))."""
    T, R, _ = ll.shape
    u = np.full(R, u0)
    w = np.full((R, 3, M), 1.0 / M)                       # member weights at nodes [-, 0, +]
    W = np.zeros((R, 2))                                  # arm leads [-, +]
    mix = np.empty((T, R)); path = np.empty((T, R)); moves = np.zeros(R, int)
    off = np.array([-g, 0.0, g]); rr = np.arange(R)
    for t in range(T):
        path[t] = u
        mx = ll[t].max(axis=1)[:, None, None]
        j = np.maximum(w, 1e-300) * np.exp(ll[t][:, None, :] - mx)
        s = np.maximum(j.sum(axis=2), 1e-300)
        lp = np.log(s) + mx[:, :, 0]                      # (R, 3)
        lq = np.stack([W[:, 0] - c, np.zeros(R), W[:, 1] - c], axis=1)
        lq -= lse(lq, keepdims=True)
        mix[t] = lse(lq + lp)
        post = j / s[..., None]
        W = np.maximum(W + lp[:, [0, 2]] - lp[:, [1]], 0.0)
        up = (W[:, 1] >= c) & (u + g <= 1e-9); dn = (W[:, 0] >= c) & (u - g >= U_MIN) & ~up
        if up.any():                                      # new [-,0,+] = old [0,+,+]
            post[up] = post[up][:, [1, 2, 2]]; u[up] += g; W[up] = 0.0; moves[up] += 1
        if dn.any():                                      # new [-,0,+] = old [-,-,0]
            post[dn] = post[dn][:, [0, 0, 1]]; u[dn] -= g; W[dn] = 0.0; moves[dn] += 1
        a = 2.0 ** np.minimum(u[:, None] + off[None, :], 0.0)          # arms clipped at the ceiling
        w = post + a[..., None] * (post @ HD)
    return mix, path, moves

VARIANTS = [("star, c = 1.125", dict(c=C_RES)), ("star, c = 8.08", dict(c=C0)),
            ("star, c = 1.125, g = 2", dict(c=C_RES, g=2.0))]

def evaluate(idx, ll, start):
    true_ll = np.take_along_axis(ll, idx[:, :, None], axis=2)[:, :, 0]
    lp = ladder_rungs(ll, H)
    preds = {"forget 0.999 (shipped)": simple_rule(ll, "forget", 0.999), f"clamp {C0:.2f}": simple_rule(ll, "clamp", C0),
             "ladder, pure Bayes": ladder_mix(lp)[0], "ladder, rungs clamped": ladder_mix(lp, C0)[0]}
    info = {}
    for name, kw in VARIANTS:
        mix, path, moves = star(ll, **kw)
        preds[name] = mix
        info[name] = dict(a_med=float(2 ** np.median(path[start:])), moves=float(moves.mean()),
                          path=np.median(path, axis=1)[::50])
    reg = {k: (true_ll - v)[start:].mean(axis=0) for k, v in preds.items()}
    rung = (true_ll[:, :, None] - lp)[start:].mean(axis=(0, 1))
    out = {k: (v.mean(), v.std(ddof=1) / np.sqrt(len(v))) for k, v in reg.items()}
    out["best single rung"] = (rung.min(), 0.0)
    return dict(reg=out, info=info, best_a=float(ALPHAS[rung.argmin()]))

res = {}
def run_static():
    KL, SE, static_ll = pickle.load(open(os.path.join(CACHE, "partA.pkl"), "rb"))
    regs, infos = {}, {}
    for i in range(M):
        ll = static_ll[i].astype(float)
        ev = evaluate(np.full(ll.shape[:2], i), ll, 2000)
        for k, (m, _) in ev["reg"].items(): regs.setdefault(k, []).append(m)
        for k, d in ev["info"].items(): infos.setdefault(k, []).append((d["a_med"], d["moves"]))
    return dict(reg={k: (np.mean(v), np.std(v, ddof=1) / np.sqrt(M)) for k, v in regs.items()},
                info={k: dict(a_med=float(np.median([x[0] for x in v])), moves=float(np.mean([x[1] for x in v])), path=None)
                      for k, v in infos.items()}, best_a=0.0)
res["calm"] = cached("0004_static", run_static)
for name, (gen, start) in scenarios(H).items():
    def run(name=name, gen=gen, start=start):
        idx, ll = scenario_ll(name, gen)
        return evaluate(idx, ll, start)
    res[name] = cached("0004_" + name.replace(" ", "_").replace(">", "").replace("(", "").replace(")", ""), run)

names = list(res["calm"]["reg"])
print("regret, nats/step x 1000, against the currently-true member")
print(f"{'scenario':28s}" + "".join(f"{n[:21]:>23s}" for n in names))
for sc, r in res.items():
    print(f"{sc:28s}" + "".join(f"{1e3 * r['reg'][n][0]:17.2f}+-{1e3 * r['reg'][n][1]:<4.2f}" for n in names))
print("\nstar: median fraction a it sits at (best single rung's a in brackets), and moves per record")
for sc, r in res.items():
    print(f"{sc:28s} [{r['best_a']:.4f}]  " + "   ".join(f"{k}: a={d['a_med']:.4f}, {d['moves']:.0f} moves" for k, d in r["info"].items()))
ref = "ladder, rungs clamped"
print("\nstar (c = 1.125) / clamped ladder:  " + "  ".join(
    f"{sc}: {r['reg']['star, c = 1.125'][0] / r['reg'][ref][0]:.2f}" for sc, r in res.items()))

# ---------------------------------------------------------------- figures
INK, MUTED, GRID = "#191919", "#666666", "#e6e6e6"
BLUE, ORANGE, AQUA, VIOLET = "#257de0", "#f47230", "#00b894", "#6a4ce2"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 200})
show = [("forget 0.999 (shipped)", BLUE), ("ladder, rungs clamped", AQUA), ("star, c = 1.125", ORANGE)]
groups = ["calm", "nbr 10000", "nbr 1000", "nbr 100", "drift a=0.01", "drift a=0.14", "drift a=1.0",
          "calm -> ceiling (2nd half)", "ceiling -> calm (2nd half)", "any 10000", "any 1000"]
labels = ["calm", "neighbour\nhop / 10,000", "neighbour\nhop / 1,000", "neighbour\nhop / 100", "drift at 1%\nof ceiling",
          "drift at 14%\nof ceiling", "drift at\nthe ceiling", "calm, then\nceiling", "ceiling,\nthen calm",
          "far hop\n/ 10,000", "far hop\n/ 1,000"]
fig, ax = plt.subplots(figsize=(13.5, 4.8)); wd = 0.27
for k, (name, col) in enumerate(show):
    y = [1e3 * res[g_]["reg"][name][0] for g_ in groups]; xs = np.arange(len(groups)) + (k - 1) * wd
    ax.bar(xs, y, width=wd - 0.03, color=col, label=name, zorder=3)
    for x, v in zip(xs, y):
        ax.text(x, v * 1.12, f"{v:.2g}" if v < 10 else f"{v:.0f}", ha="center", fontsize=7, color=INK)
ax.set_yscale("log"); ax.set_xticks(range(len(groups))); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("extra code length, nats per step x 1000\n(lower is better)")
ax.set_title("A 3-node walking star on the drift rate, against the 14-rung ladder and forget")
ax.legend(fontsize=8.5, loc="upper left"); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "0004-walking-star.png"), facecolor="white"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9), sharey=True)
for a_, key, title in ((ax[0], "calm -> ceiling (2nd half)", "Calm, then drifting at the ceiling"),
                       (ax[1], "ceiling -> calm (2nd half)", "Drifting at the ceiling, then calm")):
    for nm, col in (("star, c = 1.125", ORANGE), ("star, c = 8.08", VIOLET)):
        p = res[key]["info"][nm]["path"]
        a_.plot(np.arange(len(p)) * 50, 2.0 ** p, color=col, lw=2, label=nm)
    a_.axvline(HALF, color=MUTED, lw=1, ls="--"); a_.set_yscale("log"); a_.set_title(title); a_.set_xlabel("step")
ax[0].set_ylabel("fraction of the ceiling rate at the\nstar's centre (median of 8 records)"); ax[0].legend(fontsize=8.5, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "0004-star-path.png"), facecolor="white"); plt.close(fig)
