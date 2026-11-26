"""0006 -- the rate star's spacing and node count from the filter's own resolution rules.

1. Shape.  Per-rung regret of the 0003 ladder against v = ln(a / a*), fitted to
   r (e^v - 1 - v): the Poisson log-rate (= log-scale) shelf-and-cliff.
2. Spacing.  The walk grid's absolute bound (resolution-criterion 0002) re-derived for the
   one-event rate likelihood g(v) = exp(v - e^v), |g^(w)| = |Gamma(1 + i w)| =
   sqrt(pi w / sinh(pi w)):   score alias(D) = (2 pi / D) * 2 |g^(2 pi / D)| = eps0.
3. Window.  Nodes k = -K..K at spacing D, window prior log-weight -(1.5^2/2) k^2 (spacing
   1.5 blur widths), K = ceil(3 / 1.5) = 2.  An arm's lead never falls below zero; the star
   recentres when the posterior mode leaves the centre.
4. Race on the cached likelihoods against 0004's hand-set stars.

Run:  python 0006_the_star_from_resolution.py   (~8 min; needs the caches of 0001 and 0004)
"""
import math, os, pickle
import numpy as np
from rig import (M, C0, C_RES, CACHE, cached, lse, ceiling_rates, scenarios, scenario_ll,
                 ladder_rungs, ALPHAS)

EPS0 = 2 * math.exp(-2 * math.pi ** 2 / 1.5 ** 2)
H = ceiling_rates(); HD = H - np.diag(H.sum(axis=1))
SC = scenarios(H)

# ---------------------------------------------------------------- 2. spacing
def ghat(w, shape):
    return math.sqrt(math.pi * w / math.sinh(math.pi * w)) if shape == 1 else 1.0 / math.sqrt(math.cosh(math.pi * w))
def solve(f, target, lo=0.2, hi=3.0):
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return 0.5 * (lo + hi)
print(f"eps0 = {EPS0:.3e}")
for shape, name in ((0.5, "log-variance, one observation (0002)"), (1, "log-rate, one event")):
    dn = solve(lambda D: 2 * ghat(2 * math.pi / D, shape), EPS0)
    ds = solve(lambda D: (2 * math.pi / D) * 2 * ghat(2 * math.pi / D, shape), EPS0)
    print(f"  {name:38s} normalisation alias = eps0 at D = {dn:.3f} nats;  score alias = eps0 at D = {ds:.3f} nats ({ds / math.log(2):.2f} octaves)")
G = solve(lambda D: (2 * math.pi / D) * 2 * ghat(2 * math.pi / D, 1), EPS0)
K = int(math.ceil(3.0 / 1.5))
print(f"derived star: spacing {G:.3f} nats, K = {K} ({2 * K + 1} nodes), window prior log-weights {[round(-C_RES * k * k, 3) for k in range(-K, K + 1)]}")

# ---------------------------------------------------------------- 1. shape
def shape_fit(name):
    idx, ll = scenario_ll(name, SC[name][0]); start = SC[name][1]
    true_ll = np.take_along_axis(ll, idx[:, :, None], axis=2)[:, :, 0]
    lp = ladder_rungs(ll, H)
    return (true_ll[:, :, None] - lp)[start:].mean(axis=(0, 1))
print("\nshape: per-rung regret against v = ln(a / a*), fitted to  floor + r (e^v - 1 - v)")
for name in ("drift a=0.14", "drift a=0.01", "nbr 1000", "nbr 100"):
    reg = cached("0006_shape_" + name.replace(" ", "_"), lambda name=name: shape_fit(name))
    la = np.log(ALPHAS[1:]); y = reg[1:]
    best = None
    for v0 in np.linspace(la.min(), la.max(), 400):          # a* on a fine grid; r and floor by least squares
        v = la - v0; X = np.stack([np.ones_like(v), np.exp(v) - 1 - v], axis=1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None); res = ((X @ coef - y) ** 2).sum()
        if best is None or res < best[0]: best = (res, v0, coef)
    res, v0, coef = best; r2 = 1 - res / ((y - y.mean()) ** 2).sum()
    # the same fit with a symmetric bowl, for contrast
    bq = None
    for v0q in np.linspace(la.min(), la.max(), 400):
        v = la - v0q; X = np.stack([np.ones_like(v), v * v], axis=1)
        c2, *_ = np.linalg.lstsq(X, y, rcond=None); rs = ((X @ c2 - y) ** 2).sum()
        if bq is None or rs < bq: bq = rs
    print(f"  {name:14s} a* = {math.exp(v0):.4f}  r = {1e3 * coef[1]:.2f}e-3/step  floor = {1e3 * coef[0]:.2f}e-3   R2 = {r2:.3f}"
          f"   (symmetric bowl R2 = {1 - bq / ((y - y.mean()) ** 2).sum():.3f})")

# ---------------------------------------------------------------- 3-4. the star
def star(ll, g, K, c=C_RES, mode="prior"):
    """Walking star in v = ln a (nats).  mode 'prior': node log-weight -c k^2, recentre when the
    posterior mode moves.  mode 'flat' (0004): every arm at -c, recentre when a lead reaches c."""
    T, R, _ = ll.shape; n = 2 * K + 1; ks = np.arange(-K, K + 1)
    pri = -c * ks ** 2 if mode == "prior" else np.where(ks == 0, 0.0, -c)
    u = np.zeros(R); w = np.full((R, n, M), 1.0 / M); W = np.zeros((R, n))
    mix = np.empty((T, R)); path = np.empty((T, R)); rr = np.arange(R)
    for t in range(T):
        path[t] = u
        mx = ll[t].max(axis=1)[:, None, None]
        j = np.maximum(w, 1e-300) * np.exp(ll[t][:, None, :] - mx)
        s = np.maximum(j.sum(axis=2), 1e-300); lp = np.log(s) + mx[:, :, 0]
        lq = W + pri[None]; lq = lq - lse(lq, keepdims=True)
        mix[t] = lse(lq + lp); post = j / s[..., None]
        W = np.maximum(W + lp - lp[:, [K]], 0.0)
        best = (W + pri[None]).argmax(axis=1); d = best - K
        d = np.where((W + pri[None])[rr, best] > 1e-12, d, 0)
        d = np.minimum(d, np.floor((1e-9 - u) / g).astype(int))        # the ceiling: u <= 0
        d = np.where(u + d * g < -17.0, 0, d)
        mv = np.flatnonzero(d)
        for r_ in mv:
            src = np.clip(np.arange(n) + d[r_], 0, n - 1)
            post[r_] = post[r_][src]; u[r_] += d[r_] * g; W[r_] = 0.0
        a = np.exp(np.minimum(u[:, None] + ks[None, :] * g, 0.0))
        w = post + a[..., None] * (post @ HD)
    return mix, path

LN2 = math.log(2)
VARIANTS = [("0004 star: 3 nodes, 1 octave", dict(g=LN2, K=1, mode="flat")),
            ("0004 star: 3 nodes, 2 octaves", dict(g=2 * LN2, K=1, mode="flat")),
            (f"derived: 5 nodes, {G:.2f} nats", dict(g=G, K=2, mode="prior")),
            (f"derived spacing, 3 nodes", dict(g=G, K=1, mode="prior"))]
def evaluate(idx, ll, start):
    true_ll = np.take_along_axis(ll, idx[:, :, None], axis=2)[:, :, 0]
    out = {}
    for nm, kw in VARIANTS:
        mix, path = star(ll, **kw); r = (true_ll - mix)[start:].mean(axis=0)
        out[nm] = (r.mean(), r.std(ddof=1) / np.sqrt(len(r)), float(np.exp(np.median(path[start:]))))
    return out
res = {}
def run_static():
    static_ll = pickle.load(open(os.path.join(CACHE, "partA.pkl"), "rb"))[2]; acc = {}
    for i in range(M):
        ll = static_ll[i].astype(float)
        for k, v in evaluate(np.full(ll.shape[:2], i), ll, 2000).items(): acc.setdefault(k, []).append(v)
    return {k: (np.mean([x[0] for x in v]), np.std([x[0] for x in v], ddof=1) / np.sqrt(M), float(np.median([x[2] for x in v]))) for k, v in acc.items()}
res["calm"] = cached("0006_static", run_static)
for name, (gen, start) in SC.items():
    def run(name=name, gen=gen, start=start):
        idx, ll = scenario_ll(name, gen); return evaluate(idx, ll, start)
    res[name] = cached("0006_" + name.replace(" ", "_").replace(">", "").replace("(", "").replace(")", ""), run)
names = [v[0] for v in VARIANTS]
print("\nregret, nats/step x 1000")
print(f"{'scenario':28s}" + "".join(f"{n:>32s}" for n in names))
for sc, r in res.items():
    print(f"{sc:28s}" + "".join(f"{1e3 * r[n][0]:26.2f}+-{1e3 * r[n][1]:<4.2f}" for n in names))
d = names[2]
print("\nderived star / the better of 0004's two:  " + "  ".join(
    f"{sc}: {r[d][0] / min(r[names[0]][0], r[names[1]][0]):.2f}" for sc, r in res.items()))
