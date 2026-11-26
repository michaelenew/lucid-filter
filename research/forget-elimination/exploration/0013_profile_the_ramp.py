"""0013 -- why `forget=None` loses on a slow ramp of sensor noise (0012's `ramp` mode).

Arms: A default; B forget = 1; S forget=None; S0 the same with the state step off; and
single-class filters at the classes the default and the drift favour.  Per phase of the
ramp: RMSE over the oracle's.  Per arm: where the bank's weight sits and how well the
reported measurement scale follows the truth.

Every (seed, arm) run is checkpointed to `_cache/`.
Run from the repository root:  python research/forget-elimination/exploration/0013_profile_the_ramp.py
"""
import importlib.util, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from lucid.filter import lucid as L
from lucid.filter.lucid import LucidFilter, _logsumexp
from rig import cached
spec = importlib.util.spec_from_file_location("m12", os.path.join(HERE, "0012_new_stress_modes.py"))
src = open(os.path.join(HERE, "0012_new_stress_modes.py")).read().split("MODES = [")[0]      # generators only
m12 = {"__file__": os.path.join(HERE, "0012_new_stress_modes.py"), "__name__": "m12"}; exec(compile(src, "0012_gen", "exec"), m12)
make, kalman = m12["make"], m12["kalman"]
TAG = "v3"; SEEDS = list(range(8))
PHASES = (("flat before", 100, 1000), ("rising", 1000, 3000), ("falling", 3000, 5000), ("flat after", 5000, 6000))

def run(d, kw, carry=True, rest=None):
    """`rest`: a forget factor applied to every node's weights on top of the chain -- i.e. the
    escape kept for the bank dimensions the chain does not cover (the split ladder)."""
    keep, keep_post = L._ClassDrift._carry, L._ClassDrift.post
    if not carry: L._ClassDrift._carry = lambda self, flt, post, rate: None
    if rest is not None:
        def post_rest(self, flt, llw, a, finite):
            out = keep_post(self, flt, llw, a, finite)
            if finite: self.lw = (rest ** a) * (self.lw - llw[None, :]) + llw[None, :]
            return out
        L._ClassDrift.post = post_rest
    try:
        f = LucidFilter(**kw); T = len(d["y"])
        est = np.empty(T); ms = np.empty(T); wcls = np.zeros((T, 15)); v = np.full(T, np.nan); lls = np.empty(T)
        key = np.round(f.phi_arr, 9) * 1000 + np.round(f.s_arr, 9); uq = list(dict.fromkeys(key))
        cid = np.array([uq.index(k) for k in key]); classes = [(float(f.phi_arr[list(key).index(k)]), float(f.s_arr[list(key).index(k)])) for k in uq]
        for t in range(T):
            st = f.update(np.array([d["y"][t]])); est[t] = st.mean[0]; ms[t] = st.measurement_scale[0]; lls[t] = st.loglik
            if f._drift is None:
                post = np.exp(f._logw - _logsumexp(f._logw))
            else:
                dr = f._drift; lq = dr.lead + dr.node_prior; lq = lq - _logsumexp(lq)
                post = sum(math.exp(lq[i]) * np.exp(dr.lw[i] - _logsumexp(dr.lw[i])) for i in range(len(lq))); v[t] = math.exp(dr.v)
            wcls[t, :len(uq)] = np.bincount(np.tile(cid, f._ndw), weights=post, minlength=len(uq))
        return dict(est=est, ms=ms, w=wcls[:, :len(uq)], v=v, ll=f.loglik, lls=lls, classes=classes)
    finally:
        L._ClassDrift._carry, L._ClassDrift.post = keep, keep_post

ARMS = [("A default", {}, True), ("B forget=1", dict(forget=1.0), True), ("S forget=None", dict(forget=None), True),
        ("S0 no state step", dict(forget=None), False), ("S + 0.999 on the rest", dict(forget=None), True),
        ("one class (0.95, 0.2)", dict(phis=(0.95,), ss=(0.2,)), True), ("one class (0.95, 0.8)", dict(phis=(0.95,), ss=(0.8,)), True),
        ("one class (0.70, 0.4)", dict(phis=(0.70,), ss=(0.4,)), True)]
rm = lambda e, a, b: float(np.sqrt(np.mean(e[a:b] ** 2)))
acc = {nm: dict(ph={p[0]: [] for p in PHASES}, pl={p[0]: [] for p in PHASES}, all=[], ll=[], lag=[], rough=[], w=[], v=[]) for nm, _, _ in ARMS}
for seed in SEEDS:
    d = make("ramp", seed); ko, _ = kalman(d["y"], d["q"], d["r"], d["dt"]); eo = ko - d["x"]
    lr = np.log(d["r"])
    for nm, kw, carry in ARMS:
        slug = "".join(ch if ch.isalnum() else "_" for ch in nm)
        r = cached(f"0013{TAG}_{seed}_{slug}", lambda: run(d, kw, carry, rest=0.999 if "on the rest" in nm else None))
        e = r["est"] - d["x"]; a = acc[nm]
        for p, lo, hi in PHASES: a["ph"][p].append(rm(e, lo, hi) / rm(eo, lo, hi)); a["pl"][p].append(float(np.mean(r["lls"][lo:hi])))
        a["all"].append(rm(e, 100, None) / rm(eo, 100, None)); a["ll"].append(r["ll"])
        off = np.median((r["ms"] - lr)[100:1000])                      # the scale's own unit offset, from the flat start
        dev = r["ms"] - lr - off
        a["lag"].append((float(np.mean(dev[1000:3000])), float(np.mean(dev[3000:5000]))))     # <0 rising = behind; >0 falling = behind
        a["rough"].append(float(np.std(np.diff(r["ms"][100:1000]))))
        a["w"].append(r["w"][100:].mean(0)); a["v"].append(np.nanmedian(r["v"][100:]) if np.isfinite(r["v"]).any() else np.nan)
        if len(r["classes"]) == 15: classes = r["classes"]
base = acc["A default"]
print(f"ramp mode, {len(SEEDS)} seeds.  RMSE over the oracle's, by phase; log-likelihood minus the default's; paired t against the default on the whole record")
print(f"{'arm':24s} {'whole':>7s} {'t':>7s}  " + "  ".join(f"{p[0]:>11s}" for p in PHASES) + f"  {'dloglik':>8s}  {'scale lag: rising / falling (nats)':>36s}  {'scale jitter':>12s}")
for nm, _, _ in ARMS:
    a = acc[nm]; dl = np.log(a["all"]) - np.log(base["all"])
    t = dl.mean() / (dl.std(ddof=1) / math.sqrt(len(dl))) if nm != "A default" else 0.0
    lag = np.mean(a["lag"], axis=0)
    print(f"{nm:24s} {np.mean(a['all']):7.3f} {t:+7.2f}  " + "  ".join(f"{np.mean(a['ph'][p[0]]):11.3f}" for p in PHASES)
          + f"  {np.mean(a['ll']) - np.mean(base['ll']):+8.1f}  {lag[0]:+17.3f} / {lag[1]:+.3f}{'':10s}  {np.mean(a['rough']):12.4f}")
print("\nlog predictive per step x 1000, minus the default's, by phase (and the first 100 steps):")
for nm, _, _ in ARMS:
    a = acc[nm]
    print(f"  {nm:24s} " + "  ".join(f"{p[0]} {1e3 * (np.mean(a['pl'][p[0]]) - np.mean(base['pl'][p[0]])):+7.2f}" for p in PHASES))
print("\nmean bank weight by class (phi, s), whole record:")
full = acc["A default"]["w"][0].shape[0]
for nm in ("A default", "B forget=1", "S forget=None", "S0 no state step", "S + 0.999 on the rest"):
    w = np.mean(acc[nm]["w"], axis=0); top = np.argsort(w)[::-1][:5]
    v = np.nanmean(acc[nm]["v"]) if np.isfinite(acc[nm]["v"]).any() else float("nan")
    print(f"  {nm:18s} " + "  ".join(f"({classes[i][0]:.2f},{classes[i][1]:g}) {w[i]:.2f}" for i in top)
          + (f"   rate fraction {v:.3f}" if np.isfinite(v) else ""))
svals = sorted({c[1] for c in classes})
print("\nmean bank weight by s (summed over phi), per phase:")
for nm in ("A default", "B forget=1", "S forget=None", "S + 0.999 on the rest"):
    for p, lo, hi in PHASES:
        ws = []
        for seed in SEEDS:
            slug = "".join(ch if ch.isalnum() else "_" for ch in nm)
            r = cached(f"0013{TAG}_{seed}_{slug}", lambda: None)
            ws.append([r["w"][lo:hi][:, [i for i, c in enumerate(classes) if c[1] == sv]].sum(1).mean() for sv in svals])
        print(f"  {nm:14s} {p:12s} " + "  ".join(f"s={sv:g}: {x:.2f}" for sv, x in zip(svals, np.mean(ws, axis=0))))
