"""0001 -- the per-pair resolution time of the (phi, s) bank, and weight rules built on it.

Rig (isolates the bank-weight channel; nothing else of the filter is here):

    e_t = exp(lam_t / 2) z_t,      lam_{t+1} = phi lam_t + s sqrt(1 - phi^2) eps_t

`lam` is the log-variance about a KNOWN, COMMON log-level 0 (median-normalised), so the only
thing that separates members is the spread `s` and memory `phi` of the level -- what is left
once the filter's level walk has absorbed the location.  The bank is the shipped box:
`_PHIS x _SS` = 3 x 5 = 15 members.  Each member is an exact-to-quadrature grid (HMM)
filter on `lam`; its one-step predictive log-likelihood `ll[t, m]` does not depend on the
weight rule, so it is computed once per record and every rule is replayed on it.

Part A  -- KL[i, j]: nats per step by which member i out-predicts member j when i is true.
           The resolution time at tolerance eps is n_ij = ln(1/eps) / KL[i, j].
Part B  -- weight rules replayed on switching truths (dwell D), scored by code length
           (regret in nats/step against the member that is currently true):
             forget f       logw <- f (logw - lse) + ll                     (the shipped rule)
             clamp c        logw <- logw + ll; logw <- max(logw, max - c)   (log-odds cap)
             ceiling        clamp with a per-pair cap c_ij solving  eps ln(1/eps) = KL_ij/(M-1)
             pairwise       L_ij <- (1 - 1/n_ij) L_ij + (ll_i - ll_j);  logw_i = mean_j L_ij
             share 1/t      w <- (1 - a_t) w + a_t / M,  a_t = 1/(t+1)      (no constant)
             share 1/D      the same with a = 1/D                           (knows the truth)
Part C  -- re-selection time after each switch i -> j, against n_ji.

Run:  python 0001_pairwise_resolution.py     (numpy + matplotlib; ~10 min cold, cached after)
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

PHIS = (0.70, 0.85, 0.95)
SS = (0.20, 0.40, 0.80, 1.60, 3.20)
MEMBERS = [(p, s) for p in PHIS for s in SS]
M = len(MEMBERS)
K, G = 65, 5.0                     # quadrature nodes per member, half-range in units of s
EPS0 = 3.1e-4                      # the repository's one aliasing tolerance (resolution-criterion 0002)
C0 = float(np.log(1 / EPS0))       # 8.08 nats

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("FORGET_CACHE", os.path.join(HERE, "_cache"))
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(CACHE, exist_ok=True); os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------- the bank
def build_bank(k=K):
    X = np.empty((M, k)); A = np.empty((M, k, k)); P0 = np.empty((M, k))
    for m, (phi, s) in enumerate(MEMBERS):
        x = np.linspace(-G * s, G * s, k)
        sd = s * np.sqrt(1 - phi * phi)
        a = np.exp(-0.5 * ((x[:, None] - phi * x[None, :]) / sd) ** 2)   # a[k', k]: to k' from k
        A[m] = a / a.sum(axis=0, keepdims=True)
        p0 = np.exp(-0.5 * (x / s) ** 2)
        X[m], P0[m] = x, p0 / p0.sum()
    return X, A, P0

def member_loglik(e, k=K):
    """e: (T, R) draws -> ll: (T, R, M) one-step predictive log-likelihoods."""
    X, A, P0 = build_bank(k)
    T, R = e.shape
    p = np.broadcast_to(P0, (R, M, k)).copy()
    ll = np.empty((T, R, M))
    half_log2pi = 0.5 * np.log(2 * np.pi)
    for t in range(T):
        pred = np.matmul(A[None], p[..., None])[..., 0]                  # (R, M, k)
        e2 = (e[t] ** 2)[:, None, None]
        lem = -half_log2pi - 0.5 * X[None] - 0.5 * e2 * np.exp(-X[None])
        top = lem.max(axis=2, keepdims=True)                             # log-domain: a huge draw
        joint = np.maximum(pred, 1e-300) * np.exp(lem - top)             # must not zero a narrow member
        lik = joint.sum(axis=2)
        p = joint / lik[..., None]
        ll[t] = np.log(lik) + top[..., 0]
    return ll

# ---------------------------------------------------------------- truths
def simulate(T, R, D, hop, seed):
    """Piecewise-constant member index (switching every D steps) and the draws it generates."""
    rng = np.random.default_rng(seed)
    idx = np.empty((T, R), int)
    cur = rng.integers(M, size=R)
    for t in range(T):
        if D and t and t % D == 0:
            for r in range(R):
                if hop == "any":
                    cur[r] = (cur[r] + rng.integers(1, M)) % M
                else:                                                    # neighbour on the grid
                    ip, js = divmod(cur[r], len(SS))
                    opts = [(ip + d, js) for d in (-1, 1) if 0 <= ip + d < len(PHIS)] + \
                           [(ip, js + d) for d in (-1, 1) if 0 <= js + d < len(SS)]
                    a, b = opts[rng.integers(len(opts))]
                    cur[r] = a * len(SS) + b
        idx[t] = cur
    phi = np.array([m[0] for m in MEMBERS])[idx]; s = np.array([m[1] for m in MEMBERS])[idx]
    z = np.empty((T, R)); z[0] = rng.standard_normal(R)                  # standardised level
    innov = rng.standard_normal((T, R))
    for t in range(1, T):
        z[t] = phi[t] * z[t - 1] + np.sqrt(1 - phi[t] ** 2) * innov[t]
    e = np.exp(0.5 * s * z) * rng.standard_normal((T, R))
    return idx, e

def static_truth(T, R, m, seed):
    rng = np.random.default_rng(seed)
    phi, s = MEMBERS[m]
    z = np.empty((T, R)); z[0] = rng.standard_normal(R)
    innov = rng.standard_normal((T, R))
    for t in range(1, T):
        z[t] = phi * z[t - 1] + np.sqrt(1 - phi * phi) * innov[t]
    return np.full((T, R), m), np.exp(0.5 * s * z) * rng.standard_normal((T, R))

def cached(name, fn):
    path = os.path.join(CACHE, name + ".pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    out = fn(); pickle.dump(out, open(path, "wb"), protocol=4)
    return out

# ---------------------------------------------------------------- weight rules
def lse(a, axis=-1, keepdims=False):
    m = a.max(axis=axis, keepdims=True)
    out = m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis)

def replay(ll, rule, **kw):
    """Replay a weight rule.  Returns (mixture log-predictive (T, R), normalised log-weights
    BEFORE each observation (T, R, M))."""
    T, R, _ = ll.shape
    logw = np.full((R, M), -np.log(M))
    mix = np.empty((T, R)); W = np.empty((T, R, M))
    if rule == "pairwise":
        F = kw["F"]; L = np.zeros((R, M, M))
    for t in range(T):
        W[t] = logw
        mix[t] = lse(logw + ll[t])
        if rule == "forget":
            logw = kw["f"] * logw + ll[t]
        elif rule == "clamp":
            logw = logw + ll[t]
            logw = np.maximum(logw, logw.max(axis=1, keepdims=True) - kw["c"])
        elif rule == "ceiling":
            logw = logw + ll[t]
            top = logw.argmax(axis=1)
            logw = np.maximum(logw, logw.max(axis=1, keepdims=True) - kw["cap"][top])
        elif rule == "share":
            logw = logw + ll[t]; logw -= lse(logw, keepdims=True)
            a = 1.0 / (t + 2) if kw["a"] is None else kw["a"]
            logw = np.logaddexp(np.log1p(-a) + logw, np.log(a / M))
        elif rule == "pairwise":
            L = F[None] * L + (ll[t][:, :, None] - ll[t][:, None, :])
            logw = L.mean(axis=2)
        logw = logw - lse(logw, keepdims=True)
    return mix, W

def ceiling_caps(KLs):
    """Per-pair cap c_ij = ln(1/eps_ij), eps ln(1/eps) = KL_ij / (M - 1)."""
    x = np.clip(KLs / (M - 1), 1e-12, 0.36)
    eps = x.copy()
    for _ in range(200):
        eps = x / np.log(1 / np.minimum(eps, 0.999999))
        eps = np.clip(eps, 1e-300, 0.36)
    return np.log(1 / eps)

# ================================================================ Part A: the KL table
T_A, R_A = 30000, 4
def part_a():
    KL = np.zeros((M, M)); SE = np.zeros((M, M)); lls = {}
    for i in range(M):
        idx, e = static_truth(T_A, R_A, i, 1000 + i)
        ll = member_loglik(e)
        d = ll[2000:, :, i][:, :, None] - ll[2000:]                      # drop the start-up
        KL[i] = d.mean(axis=(0, 1))
        SE[i] = d.mean(axis=0).std(axis=0, ddof=1) / np.sqrt(R_A)
        lls[i] = ll
    return KL, SE, lls

KL, SE, static_ll = cached("partA", part_a)
KLs = 0.5 * (KL + KL.T)                                                 # symmetrised, for pair rules
Nres = C0 / np.where(KL > 0, KL, np.nan)
np.fill_diagonal(Nres, np.nan)
CAP = ceiling_caps(KLs); np.fill_diagonal(CAP, 0.0)
with np.errstate(divide="ignore"):
    F_PAIR = 1 - np.clip(KLs / C0, 0, 1)                                # 1 - 1/n_ij
np.fill_diagonal(F_PAIR, 0.0)

def lab(m): return f"({MEMBERS[m][0]:.2f},{MEMBERS[m][1]:.1f})"

print(f"bank: {M} members, K = {K} nodes, tolerance eps0 = {EPS0} -> c0 = {C0:.2f} nats")
print("\nPart A.  KL[i, j] in nats/step (row = truth), then resolution time n_ij = c0 / KL")
off = ~np.eye(M, dtype=bool)
print(f"  KL   min {KL[off].min():.5f}  median {np.median(KL[off]):.4f}  max {KL[off].max():.3f}")
print(f"  KL negative entries (wrong member out-predicts the truth): {(KL[off] < 0).sum()} of {off.sum()}"
      f"   worst {KL[off].min():.5f}  (its SE {SE[off][KL[off].argmin()]:.5f})")
q = np.nanpercentile(Nres[off], [0, 10, 25, 50, 75, 90, 100])
print("  n_ij percentiles 0/10/25/50/75/90/100: " + "  ".join(f"{v:.0f}" for v in q))
print("  n_ij between grid neighbours:")
for i in range(M):
    ip, js = divmod(i, len(SS))
    row = []
    if js + 1 < len(SS): row.append(f"s-up {Nres[i, i + 1]:7.0f} / back {Nres[i + 1, i]:7.0f}")
    if ip + 1 < len(PHIS): row.append(f"phi-up {Nres[i, i + len(SS)]:7.0f} / back {Nres[i + len(SS), i]:7.0f}")
    print(f"    {lab(i)}  " + "   ".join(row))
print(f"  pairs resolvable within 1000 steps: {(Nres[off] <= 1000).mean():.2f};"
      f" within 100: {(Nres[off] <= 100).mean():.2f}")
print(f"  ceiling caps c_ij: min {CAP[off].min():.2f}  median {np.median(CAP[off]):.2f}  max {CAP[off].max():.2f} nats")

# quadrature check: one member pair at K = 129
def quad_check():
    idx, e = static_truth(8000, 2, 7, 77)
    out = []
    for k in (K, 129):
        ll = member_loglik(e, k)
        out.append((ll[1000:, :, 7][:, :, None] - ll[1000:]).mean(axis=(0, 1)))
    return out
qa, qb = cached("quad", quad_check)
print(f"  quadrature: KL row of member {lab(7)} at K={K} vs K=129, max abs diff {np.abs(qa - qb).max():.2e}"
      f" (row median {np.median(np.abs(qa[np.arange(M) != 7])):.4f})")

# ================================================================ Part B: the race
RULES = [("forget 1 (pure Bayes)", "forget", dict(f=1.0)),
         ("forget 0.9999", "forget", dict(f=0.9999)),
         ("forget 0.999 (shipped)", "forget", dict(f=0.999)),
         ("forget 0.99", "forget", dict(f=0.99)),
         ("clamp 2.0", "clamp", dict(c=2.0)),
         ("clamp 4.5", "clamp", dict(c=4.5)),
         (f"clamp {C0:.2f} (eps0)", "clamp", dict(c=C0)),
         ("clamp 12", "clamp", dict(c=12.0)),
         ("ceiling (no constant)", "ceiling", dict(cap=CAP)),
         ("pairwise forgetting (eps0)", "pairwise", dict(F=F_PAIR)),
         ("share 1/t (no constant)", "share", dict(a=None)),
         ("share 1/D (knows D)", "share", dict(a="D"))]

def regret_and_resel(idx, ll, D):
    T, R, _ = ll.shape
    true_ll = np.take_along_axis(ll, idx[:, :, None], axis=2)[:, :, 0]
    out = {}
    burn = D if D else 2000
    for name, rule, kw in RULES:
        kw = dict(kw)
        if kw.get("a") == "D":
            if not D: continue
            kw["a"] = 1.0 / D
        mix, W = replay(ll, rule, **kw)
        reg = (true_ll - mix)[burn:].mean(axis=0)
        resel = []
        if D:
            wt = np.take_along_axis(W, idx[:, :, None], axis=2)[:, :, 0]   # weight on the true member
            for t0 in range(D, T - D + 1, D):
                for r in range(R):
                    hit = np.nonzero(wt[t0:t0 + D, r] > np.log(0.5))[0]
                    resel.append((idx[t0 - 1, r], idx[t0, r], hit[0] if len(hit) else D))
        out[name] = (reg.mean(), reg.std(ddof=1) / np.sqrt(R), np.array(resel))
    return out

SCEN = [("any", 100, 60000), ("any", 300, 60000), ("any", 1000, 60000), ("any", 3000, 120000),
        ("any", 10000, 120000), ("nbr", 1000, 60000), ("nbr", 10000, 120000)]
R_B = 8
results = {}
for hop, D, T in SCEN:
    def run(hop=hop, D=D, T=T):
        idx, e = simulate(T, R_B, D, hop, 5000 + D + (0 if hop == "any" else 7))
        return regret_and_resel(idx, member_loglik(e), D)
    results[(hop, D)] = cached(f"partB_{hop}_{D}_v1", run)

def run_static():
    acc = {}
    for i in range(M):
        idx = np.full((T_A, R_A), i)
        for name, (reg, se, _) in regret_and_resel(idx, static_ll[i], 0).items():
            acc.setdefault(name, []).append(reg)
    return {k: (np.mean(v), np.std(v, ddof=1) / np.sqrt(M), None) for k, v in acc.items()}
results[("static", 0)] = cached("partB_static_v1", run_static)

print("\nPart B.  regret, nats/step x 1000, against the currently-true member (mean over records; +- SE)")
cols = [("static", 0)] + [(h, D) for h, D, _ in SCEN]
print(f"{'rule':30s}" + "".join(f"{(h + ' ' + (str(D) if D else '')):>13s}" for h, D in cols))
for name, _, _ in RULES:
    row = f"{name:30s}"
    for c in cols:
        row += f"{1e3 * results[c][name][0]:8.2f}+-{1e3 * results[c][name][1]:<4.2f}"[:13].rjust(13) if name in results[c] else f"{'-':>13s}"
    print(row)

print("\nPart C.  re-selection time after a switch i -> j (steps until the new member holds > 1/2), median")
print(f"{'rule':30s}" + "".join(f"{(h + ' ' + str(D)):>12s}" for h, D, _ in SCEN))
for name, _, _ in RULES:
    row = f"{name:30s}"
    for h, D, _ in SCEN:
        row += f"{np.median(results[(h, D)][name][2][:, 2]):12.0f}" if name in results[(h, D)] else f"{'-':>12s}"
    print(row)

# predicted vs measured re-selection, any-hops at D = 3000
ev_c = results[("any", 3000)][f"clamp {C0:.2f} (eps0)"][2]
ev_f = results[("any", 3000)]["forget 0.999 (shipped)"][2]
ev_p = results[("any", 3000)]["pairwise forgetting (eps0)"][2]
pred = np.array([Nres[j, i] for i, j, _ in ev_c])                        # n_ji: j true, against old i
ok = np.isfinite(pred) & (ev_c[:, 2] < 3000)
print(f"\n  clamp: log10(measured / n_ji) over {ok.sum()} switches: median {np.median(np.log10(ev_c[ok, 2].clip(1) / pred[ok])):+.2f},"
      f" IQR {np.subtract(*np.percentile(np.log10(ev_c[ok, 2].clip(1) / pred[ok]), [75, 25])):.2f} decades")
print(f"  forget 0.999: measured re-selection 10/50/90 percentile: "
      + " / ".join(f"{v:.0f}" for v in np.percentile(ev_f[:, 2], [10, 50, 90]))
      + "   (prediction 1000 ln 2 = 693, pair-independent)")
for nm, ev in (("clamp", ev_c), ("forget 0.999", ev_f), ("pairwise", ev_p)):
    r = np.corrcoef(np.log(pred[ok]), np.log(ev[ok, 2].clip(1)))[0, 1]
    print(f"  {nm:13s} correlation of log re-selection time with log n_ji: {r:+.2f}")

# ================================================================ figures
INK, MUTED, GRID = "#191919", "#666666", "#e6e6e6"
BLUE, ORANGE, AQUA, YELLOW = "#257de0", "#f47230", "#00b894", "#f2a600"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 200})

# fig 1: the resolution-time table
fig, ax = plt.subplots(figsize=(8.6, 7.2))
im = ax.imshow(np.log10(Nres), cmap="Blues", vmin=0.5, vmax=4.5)
ax.set_xticks(range(M)); ax.set_yticks(range(M))
ax.set_xticklabels([lab(m) for m in range(M)], rotation=90, fontsize=7.5)
ax.set_yticklabels([lab(m) for m in range(M)], fontsize=7.5)
for i in range(M):
    for j in range(M):
        if i != j and np.isfinite(Nres[i, j]):
            v = Nres[i, j]
            ax.text(j, i, f"{v:.0f}" if v < 1e4 else f"{v / 1e3:.0f}k", ha="center", va="center",
                    fontsize=6, color="white" if np.log10(v) > 2.8 else INK)
ax.grid(False)
ax.set_xlabel("alternative member (phi, s)"); ax.set_ylabel("true member (phi, s)")
ax.set_title(f"Steps to tell the true member from an alternative, at tolerance {EPS0}\n"
             f"n = {C0:.2f} nats / (nats per step by which the truth out-predicts the alternative)")
cb = fig.colorbar(im, ax=ax, shrink=0.8, ticks=[1, 2, 3, 4]); cb.ax.set_yticklabels(["10", "100", "1k", "10k"])
cb.outline.set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "0001-resolution-times.png"), facecolor="white"); plt.close(fig)

# fig 2: regret against dwell
show = [("forget 1 (pure Bayes)", MUTED, "-", "o"), ("forget 0.999 (shipped)", BLUE, "-", "o"),
        (f"clamp {C0:.2f} (eps0)", ORANGE, "-", "D"), ("share 1/t (no constant)", AQUA, "-", "s"),
        ("share 1/D (knows D)", INK, ":", "x")]
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
for a, hop, title in ((ax[0], "any", "Truth hops to any other member"),
                      (ax[1], "nbr", "Truth hops to a neighbouring member (hard)")):
    Ds = [D for h, D, _ in SCEN if h == hop]
    for name, col, ls, mk in show:
        y = [results[(hop, D)][name][0] for D in Ds]
        se = [results[(hop, D)][name][1] for D in Ds]
        a.errorbar(Ds, y, yerr=se, color=col, ls=ls, marker=mk, lw=2, ms=5, label=name)
    a.set_xscale("log"); a.set_yscale("log"); a.set_title(title)
    a.set_xlabel("dwell D: steps between changes of (phi, s)")
    a.set_xticks(Ds); a.set_xticklabels([str(D) for D in Ds]); a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax[0].set_ylabel("extra code length, nats per step\n(lower is better)")
ax[0].legend(fontsize=8.5, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "0001-regret-vs-dwell.png"), facecolor="white"); plt.close(fig)

# fig 3: re-selection time against the pair's resolution time
fig, ax = plt.subplots(figsize=(6.6, 5.2))
rng = np.random.default_rng(0)
jit = np.exp(rng.normal(0, 0.03, len(pred)))
ax.plot(pred * jit, ev_f[:, 2].clip(1), ls="none", marker="o", ms=3.5, color=BLUE, alpha=0.55, label="forget 0.999 (shipped)")
ax.plot(pred * jit, ev_c[:, 2].clip(1), ls="none", marker="D", ms=3.5, color=ORANGE, alpha=0.55, label=f"clamp {C0:.2f} (eps0)")
xx = np.array([3, 3000]); ax.plot(xx, xx, color=INK, lw=1, ls="--"); ax.text(4.2, 3.2, "measured = predicted", fontsize=8, color=MUTED, rotation=38)
ax.axhline(693, color=BLUE, lw=1, ls=":"); ax.text(3.2, 760, "1000 ln 2", fontsize=8, color=MUTED)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(2.5, 4000); ax.set_ylim(0.8, 4000)
ax.set_xlabel("the pair's resolution time n (steps), from the table")
ax.set_ylabel("measured steps until the new member is re-selected")
ax.set_title("After a change: the clamp's delay scales with the pair's\nresolution time; forget's delay ignores it")
ax.legend(fontsize=8.5, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "0001-reselection.png"), facecolor="white"); plt.close(fig)
