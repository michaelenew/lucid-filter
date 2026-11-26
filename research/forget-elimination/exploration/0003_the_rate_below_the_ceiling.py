"""0003 -- the fastest resolvable drift as a ceiling; the actual rate found beneath it.

    H      = local chain rates at c = 1.125 nats (0002): the fastest resolvable drift
    P(a)   = I + a H,   a in [0, 1]      the world drifts at a fraction a of the ceiling
    ladder = {0, 2^-12, ..., 2^0}        14 rungs; a = 0 is pure Bayes

Each rung filters the bank under its own chain; rung r's one-step predictive is
`p_r = sum_m w_{r,m} lik_m`.  The rungs are mixed by evidence, either pure Bayes (a is a
constant of the world) or with the rungs' log-odds clamped at 8.08 nats.

Run:  python 0003_the_rate_below_the_ceiling.py   (~10 min cold; needs 0001's _cache/partA.pkl)
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rig import (M, C0, CACHE, FIG, cached, lse, member_loglik, simulate, neighbours,
                 draws_from_path)

C_RES = 1.5 ** 2 / 2
KL, SE, static_ll = pickle.load(open(os.path.join(CACHE, "partA.pkl"), "rb"))
KLs = 0.5 * (KL + KL.T)
T_A, R_A = static_ll[0].shape[:2]

H = np.zeros((M, M))
for i in range(M):
    js = neighbours(i); H[i, js] = KLs[i, js] / C_RES
    if H[i].sum() > 0.5: H[i] *= 0.5 / H[i].sum()
ALPHAS = np.array([0.0] + [2.0 ** -k for k in range(12, -1, -1)])
NR = len(ALPHAS)
PS = np.stack([np.eye(M) + a * (H - np.diag(H.sum(axis=1))) for a in ALPHAS])     # (NR, M, M)

def run_rungs(ll):
    """Per-rung one-step log-predictives, (T, R, NR).  Rung 0 (a = 0, pure Bayes) is kept in
    the log domain: its weights underflow a linear representation on long records."""
    T, R, _ = ll.shape
    w = np.full((R, NR - 1, M), 1.0 / M); l0 = np.full((R, M), -np.log(M))
    out = np.empty((T, R, NR))
    for t in range(T):
        out[t, :, 0] = lse(l0 + ll[t]); l0 = l0 + ll[t]; l0 -= lse(l0, keepdims=True)
        mx = ll[t].max(axis=1)[:, None, None]
        j = np.maximum(w, 1e-300) * np.exp(ll[t][:, None, :] - mx)
        s = np.maximum(j.sum(axis=2), 1e-300)
        out[t, :, 1:] = np.log(s) + mx[:, :, 0]
        w = np.einsum("rkm,kmn->rkn", j / s[..., None], PS[1:])
    return out

def mix_rungs(lp, clamp=None):
    """Mixture over rungs; returns (log-predictive (T, R), posterior-mean a (T, R))."""
    T, R, _ = lp.shape
    lw = np.full((R, NR), -np.log(NR)); mix = np.empty((T, R)); am = np.empty((T, R))
    for t in range(T):
        mix[t] = lse(lw + lp[t]); am[t] = np.exp(lw) @ ALPHAS
        lw = lw + lp[t]
        if clamp: lw = np.maximum(lw, lw.max(axis=1, keepdims=True) - clamp)
        lw -= lse(lw, keepdims=True)
    return mix, am

def simple(ll, rule, v):
    T, R, _ = ll.shape
    logw = np.full((R, M), -np.log(M)); mix = np.empty((T, R))
    for t in range(T):
        mix[t] = lse(logw + ll[t])
        logw = v * logw + ll[t] if rule == "forget" else np.maximum(logw + ll[t], (logw + ll[t]).max(axis=1, keepdims=True) - v)
        logw -= lse(logw, keepdims=True)
    return mix

def world(T, R, alpha_of_t, seed):
    """Truth drawn from the local chain with a (possibly time-varying) fraction a."""
    rng = np.random.default_rng(seed)
    idx = np.empty((T, R), int); cur = rng.integers(M, size=R)
    nb = [neighbours(i) for i in range(M)]
    for t in range(T):
        a = alpha_of_t(t)
        if a > 0:
            u = rng.random(R)
            for r in range(R):
                i = cur[r]; p = a * H[i, nb[i]]; c = np.cumsum(p)
                if u[r] < c[-1]: cur[r] = nb[i][np.searchsorted(c, u[r])]
        idx[t] = cur
    return idx, draws_from_path(idx, rng)

def evaluate(idx, ll, burn, split=None):
    true_ll = np.take_along_axis(ll, idx[:, :, None], axis=2)[:, :, 0]
    lp = run_rungs(ll)
    mb, ab = mix_rungs(lp); mc, ac = mix_rungs(lp, C0)
    preds = {"forget 0.999 (shipped)": simple(ll, "forget", 0.999), f"clamp {C0:.2f}": simple(ll, "clamp", C0),
             "ceiling only (a = 1)": lp[:, :, -1], "ladder, pure Bayes": mb, f"ladder, rungs clamped {C0:.2f}": mc}
    sl = slice(burn, None) if split is None else slice(split, None)
    out = {k: (true_ll - v)[sl].mean(axis=0) for k, v in preds.items()}
    rung = (true_ll[:, :, None] - lp)[sl].mean(axis=(0, 1))
    out["best single rung (hindsight)"] = np.full(idx.shape[1], rung.min())
    return dict(reg={k: (v.mean(), v.std(ddof=1) / np.sqrt(len(v))) for k, v in out.items()},
                best_a=ALPHAS[rung.argmin()], a_bayes=ab[sl].mean(), a_clamp=ac[sl].mean(),
                a_path=ab.mean(axis=1)[::50], a_path_c=ac.mean(axis=1)[::50])

R_B = 8
res = {}
def run_static():
    regs, a = {}, []
    for i in range(M):
        ev = evaluate(np.full((T_A, R_A), i), static_ll[i].astype(float), 2000)
        for k, (m, _) in ev["reg"].items(): regs.setdefault(k, []).append(m)
        a.append(ev["a_bayes"])
    return dict(reg={k: (np.mean(v), np.std(v, ddof=1) / np.sqrt(M)) for k, v in regs.items()},
                best_a=0.0, a_bayes=np.mean(a), a_clamp=np.nan)
res["calm"] = cached("0003_static", run_static)

for hop, D, T in [("nbr", 10000, 120000), ("nbr", 1000, 60000), ("nbr", 100, 60000),
                  ("any", 10000, 120000), ("any", 1000, 60000), ("any", 100, 60000)]:
    def run(hop=hop, D=D, T=T):
        idx, e = simulate(T, R_B, D, hop, 5000 + D + (0 if hop == "any" else 7))
        return evaluate(idx, member_loglik(e), D)
    res[f"{hop} {D}"] = cached(f"0003_{hop}_{D}", run)

for a in (0.01, 0.14, 1.0):
    def run(a=a):
        idx, e = world(60000, R_B, lambda t: a, 300 + int(1000 * a))
        return evaluate(idx, member_loglik(e), 2000)
    res[f"drift a={a}"] = cached(f"0003_world_{a}", run)

HALF = 30000
for name, f in (("calm -> ceiling", lambda t: 0.0 if t < HALF else 1.0),
                ("ceiling -> calm", lambda t: 1.0 if t < HALF else 0.0)):
    def run(f=f, name=name):
        idx, e = world(2 * HALF, R_B, f, 77 + len(name))
        return evaluate(idx, member_loglik(e), 0, split=HALF)
    res[name + " (2nd half)"] = cached("0003_" + name.replace(" ", "").replace(">", ""), run)

names = list(res["calm"]["reg"])
print("regret, nats/step x 1000, against the currently-true member")
print(f"{'scenario':28s}" + "".join(f"{n[:19]:>21s}" for n in names) + f"{'best a':>9s}{'learned a':>11s}")
for sc, r in res.items():
    print(f"{sc:28s}" + "".join(f"{1e3 * r['reg'][n][0]:15.2f}+-{1e3 * r['reg'][n][1]:<4.2f}" for n in names)
          + f"{r['best_a']:9.4f}{r['a_bayes']:11.4f}")
print("\nladder (pure Bayes) / best single rung:  " +
      "  ".join(f"{sc}: {r['reg']['ladder, pure Bayes'][0] / max(r['reg']['best single rung (hindsight)'][0], 1e-9):.2f}"
                for sc, r in res.items()))

# ---------------------------------------------------------------- figures
INK, MUTED, GRID = "#191919", "#666666", "#e6e6e6"
BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 200})
show = [("forget 0.999 (shipped)", BLUE), (f"clamp {C0:.2f}", ORANGE), ("ladder, pure Bayes", AQUA)]
groups = ["calm", "nbr 10000", "nbr 1000", "nbr 100", "drift a=0.01", "drift a=0.14", "drift a=1.0", "any 10000", "any 1000", "any 100"]
labels = ["calm", "neighbour\nhop / 10,000", "neighbour\nhop / 1,000", "neighbour\nhop / 100", "drift at 1%\nof ceiling",
          "drift at 14%\nof ceiling", "drift at\nthe ceiling", "far hop\n/ 10,000", "far hop\n/ 1,000", "far hop\n/ 100"]
fig, ax = plt.subplots(figsize=(13, 4.8)); w = 0.27
for k, (name, col) in enumerate(show):
    y = [1e3 * res[g]["reg"][name][0] for g in groups]; xs = np.arange(len(groups)) + (k - 1) * w
    ax.bar(xs, y, width=w - 0.03, color=col, label=name, zorder=3)
    for x, v in zip(xs, y):
        ax.text(x, v * 1.12, f"{v:.2g}" if v < 10 else f"{v:.0f}", ha="center", fontsize=7, color=INK)
ax.set_yscale("log"); ax.set_xticks(range(len(groups))); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("extra code length, nats per step x 1000\n(lower is better)")
ax.set_title("A ladder of rates beneath the resolvability ceiling, against forget and the clamp")
ax.legend(fontsize=8.5, loc="upper left"); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "0003-rate-ladder.png"), facecolor="white"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9), sharey=True)
for a_, key, title in ((ax[0], "calm -> ceiling (2nd half)", "Calm, then drifting at the ceiling"),
                       (ax[1], "ceiling -> calm (2nd half)", "Drifting at the ceiling, then calm")):
    tt = np.arange(len(res[key]["a_path"])) * 50
    a_.plot(tt, np.maximum(res[key]["a_path"], 1e-5), color=AQUA, lw=2, label="ladder, pure Bayes")
    a_.plot(tt, np.maximum(res[key]["a_path_c"], 1e-5), color=ORANGE, lw=2, label="ladder, rungs clamped")
    a_.axvline(HALF, color=MUTED, lw=1, ls="--"); a_.set_yscale("log"); a_.set_title(title); a_.set_xlabel("step")
ax[0].set_ylabel("learned fraction of the ceiling rate\n(posterior mean, 8 records)"); ax[0].legend(fontsize=8.5, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "0003-learned-rate.png"), facecolor="white"); plt.close(fig)
