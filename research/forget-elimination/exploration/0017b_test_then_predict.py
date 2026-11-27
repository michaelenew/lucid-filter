"""0017b -- the floored lead doing one job instead of two.

The star (0006) floors each node's lead at zero and uses that one number both to decide when
to recentre and as the node's weight in the prediction.  Variant "test, then predict": the
floored lead still moves the star; the prediction weights are each node's evidence since the
star last moved, unfloored (the exact posterior over the nodes if the rate has been constant
since then).  Scalar rig, cached likelihoods, the scenarios of 0004/0006.

Every scenario is checkpointed to `_cache/`.
Run:  python 0017b_test_then_predict.py   (needs the caches of 0001 and 0004)
"""
import math, os, pickle
import numpy as np
from rig import M, C_RES, CACHE, cached, lse, ceiling_rates, scenarios, scenario_ll

H = ceiling_rates(); HD = H - np.diag(H.sum(axis=1)); SC = scenarios(H)
G, K = 0.7684307886942956, 2

def star(ll, split, u0=0.0):
    """split=False: the rule in the filter.  split=True: test by the floored lead, predict by
    the unfloored evidence since the last recentre."""
    T, R, _ = ll.shape; n = 2 * K + 1; ks = np.arange(-K, K + 1); pri = -C_RES * ks ** 2.0
    u = np.full(R, float(u0)); w = np.full((R, n, M), 1.0 / M); W = np.zeros((R, n)); E = np.zeros((R, n))
    mix = np.empty((T, R)); path = np.empty((T, R)); rr = np.arange(R); moves = np.zeros(R)
    for t in range(T):
        path[t] = u
        mx = ll[t].max(axis=1)[:, None, None]
        j = np.maximum(w, 1e-300) * np.exp(ll[t][:, None, :] - mx)
        s = np.maximum(j.sum(axis=2), 1e-300); lp = np.log(s) + mx[:, :, 0]
        lq = (E if split else W) + pri[None]; lq = lq - lse(lq, keepdims=True)
        mix[t] = lse(lq + lp); post = j / s[..., None]
        d_lp = lp - lp[:, [K]]
        W = np.maximum(W + d_lp, 0.0); E = E + d_lp
        sc = W + pri[None]; best = sc.argmax(axis=1); d = best - K
        d = np.where(sc[rr, best] > 1e-12, d, 0)
        d = np.minimum(d, np.floor((1e-9 - u) / G).astype(int)); d = np.where(u + d * G < -17.0, 0, d)
        for r_ in np.flatnonzero(d):
            src = np.clip(np.arange(n) + d[r_], 0, n - 1)
            post[r_] = post[r_][src]; u[r_] += d[r_] * G; W[r_] = 0.0; E[r_] = 0.0; moves[r_] += 1
        a = np.exp(np.minimum(u[:, None] + ks[None, :] * G, 0.0))
        w = post + a[..., None] * (post @ HD)
    return mix, path, moves

def evaluate(idx, ll, start):
    true_ll = np.take_along_axis(ll, idx[:, :, None], axis=2)[:, :, 0]; out = {}
    for nm, split in (("in the filter", False), ("test, then predict", True)):
        mix, path, moves = star(ll, split); r = (true_ll - mix)[start:].mean(axis=0)
        out[nm] = (r.mean(), r.std(ddof=1) / np.sqrt(len(r)), float(np.exp(np.median(path[start:]))), float(moves.mean()))
    return out
res = {}
def run_static():
    static_ll = pickle.load(open(os.path.join(CACHE, "partA.pkl"), "rb"))[2]; acc = {}
    for i in range(M):
        ll = static_ll[i].astype(float)
        for k, v in evaluate(np.full(ll.shape[:2], i), ll, 2000).items(): acc.setdefault(k, []).append(v)
    return {k: (np.mean([x[0] for x in v]), np.std([x[0] for x in v], ddof=1) / np.sqrt(M), float(np.median([x[2] for x in v])), float(np.mean([x[3] for x in v]))) for k, v in acc.items()}
res["calm"] = cached("0017b_static", run_static)
for name, (gen, start) in SC.items():
    def run(name=name, gen=gen, start=start):
        idx, ll = scenario_ll(name, gen); return evaluate(idx, ll, start)
    res[name] = cached("0017b_" + name.replace(" ", "_").replace(">", "").replace("(", "").replace(")", ""), run)
a, b = "in the filter", "test, then predict"
print(f"regret, nats/step x 1000;  the rate fraction each sits at;  moves per record")
print(f"{'scenario':28s} {a:>20s} {b:>22s} {'ratio':>7s}   {'rate a / b':>18s}   {'moves a / b':>14s}")
for sc, r in res.items():
    print(f"{sc:28s} {1e3 * r[a][0]:14.2f}+-{1e3 * r[a][1]:<4.2f} {1e3 * r[b][0]:16.2f}+-{1e3 * r[b][1]:<4.2f} {r[b][0] / r[a][0]:7.2f}   "
          f"{r[a][2]:8.4f} / {r[b][2]:<8.4f}   {r[a][3]:6.0f} / {r[b][3]:<6.0f}", flush=True)

# where the calm-world cost comes from: the same rule started low instead of at the ceiling,
# and the cost by thirds of the record
def calm_split():
    static_ll = pickle.load(open(os.path.join(CACHE, "partA.pkl"), "rb"))[2]; out = {}
    for nm, u0 in (("started at the ceiling", 0.0), ("started at e^-8 of it", -8.0)):
        acc = []
        for i in range(M):
            ll = static_ll[i].astype(float); mix, path, _ = star(ll, False, u0)
            reg = ll[:, :, i] - mix; T = len(reg)
            acc.append([reg[2000:].mean(), reg[2000:T // 3].mean(), reg[T // 3:2 * T // 3].mean(), reg[2 * T // 3:].mean(), np.exp(np.median(path[2 * T // 3:]))])
        out[nm] = np.mean(acc, axis=0)
    return out
cs = cached("0017b_calm_split", calm_split)
print("\ncalm, regret x 1000:  whole (after 2000) | first third | middle third | last third | rate fraction in the last third")
for nm, v in cs.items():
    print(f"  {nm:24s} {1e3 * v[0]:6.2f} | {1e3 * v[1]:6.2f} | {1e3 * v[2]:6.2f} | {1e3 * v[3]:6.2f} | {v[4]:.5f}")
