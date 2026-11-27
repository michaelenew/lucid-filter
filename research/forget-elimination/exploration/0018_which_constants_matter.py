"""0018 -- sensitivity of the walking star to its recentring level, scalar rig.

The level c is the node prior's scale (log-weight -c k^2) and the lead a node needs to take
the centre: 1.125 = _GAP_FACTOR^2 / 2 in the filter.  Here c in {0.5625, 1.125, 2.25} with
the spacing held at 0.768 nats.  Cached likelihoods, the scenarios of 0004/0006.

Every scenario is checkpointed to `_cache/`.
Run:  python 0018_which_constants_matter.py
"""
import os, pickle
import numpy as np
from rig import M, CACHE, cached, lse, ceiling_rates, scenarios, scenario_ll
H = ceiling_rates(); HD = H - np.diag(H.sum(axis=1)); SC = scenarios(H)
G, K = 0.7684307886942956, 2
LEVELS = (0.5625, 1.125, 2.25)

def star(ll, c):
    T, R, _ = ll.shape; n = 2 * K + 1; ks = np.arange(-K, K + 1); pri = -c * ks ** 2.0
    u = np.zeros(R); w = np.full((R, n, M), 1.0 / M); W = np.zeros((R, n))
    mix = np.empty((T, R)); rr = np.arange(R); moves = np.zeros(R); path = np.empty((T, R))
    for t in range(T):
        path[t] = u
        mx = ll[t].max(axis=1)[:, None, None]
        j = np.maximum(w, 1e-300) * np.exp(ll[t][:, None, :] - mx)
        s = np.maximum(j.sum(axis=2), 1e-300); lp = np.log(s) + mx[:, :, 0]
        lq = W + pri[None]; lq = lq - lse(lq, keepdims=True)
        mix[t] = lse(lq + lp); post = j / s[..., None]
        W = np.maximum(W + lp - lp[:, [K]], 0.0)
        sc = W + pri[None]; best = sc.argmax(axis=1); d = best - K
        d = np.where(sc[rr, best] > 1e-12, d, 0)
        d = np.minimum(d, np.floor((1e-9 - u) / G).astype(int)); d = np.where(u + d * G < -17.0, 0, d)
        for r_ in np.flatnonzero(d):
            post[r_] = post[r_][np.clip(np.arange(n) + d[r_], 0, n - 1)]; u[r_] += d[r_] * G; W[r_] = 0.0; moves[r_] += 1
        a = np.exp(np.minimum(u[:, None] + ks[None, :] * G, 0.0))
        w = post + a[..., None] * (post @ HD)
    return mix, moves, path

def evaluate(idx, ll, start):
    true_ll = np.take_along_axis(ll, idx[:, :, None], axis=2)[:, :, 0]; out = {}
    for c in LEVELS:
        mix, moves, path = star(ll, c); r = (true_ll - mix)[start:].mean(axis=0)
        out[c] = (r.mean(), r.std(ddof=1) / np.sqrt(len(r)), float(moves.mean()), float(np.exp(np.median(path[start:]))))
    return out
res = {}
def run_static():
    static_ll = pickle.load(open(os.path.join(CACHE, "partA.pkl"), "rb"))[2]; acc = {}
    for i in range(M):
        ll = static_ll[i].astype(float)
        for k, v in evaluate(np.full(ll.shape[:2], i), ll, 2000).items(): acc.setdefault(k, []).append(v)
    return {k: tuple(np.mean([x[q] for x in v]) for q in range(4)) for k, v in acc.items()}
res["calm"] = cached("0018_static", run_static)
for name, (gen, start) in SC.items():
    def run(name=name, gen=gen, start=start):
        idx, ll = scenario_ll(name, gen); return evaluate(idx, ll, start)
    res[name] = cached("0018_" + name.replace(" ", "_").replace(">", "").replace("(", "").replace(")", ""), run)
print("regret, nats/step x 1000, by recentring level c;  ratio to c = 1.125;  moves per record")
print(f"{'scenario':28s}" + "".join(f"{'c=' + str(c):>10s}" for c in LEVELS) + f"{'0.56/1.125':>12s}{'2.25/1.125':>12s}   moves: " + " / ".join(str(c) for c in LEVELS))
for sc, r in res.items():
    print(f"{sc:28s}" + "".join(f"{1e3 * r[c][0]:10.2f}" for c in LEVELS) + f"{r[0.5625][0] / r[1.125][0]:12.2f}{r[2.25][0] / r[1.125][0]:12.2f}   "
          + " / ".join(f"{r[c][2]:.0f}" for c in LEVELS), flush=True)
