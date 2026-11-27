"""0017c -- the star started low (pure Bayes until the data asks for drift) instead of at the ceiling.

0017b found the scalar rig's calm-world cost is the walk DOWN from the ceiling start (0.41e-3
nats/step from the ceiling, 0.04e-3 from e^-8 of it).  Here: what a low start costs where the
world is restless from the beginning.  Hero rig (24 seeds) and four stress modes (8 seeds),
against the default and against the ceiling start (cached, 0010/0012 v4).  A probe-side
override of `_ClassDrift.reset`; nothing under `lucid/` is changed.

Every (rig, seed) run is checkpointed to `_cache/`.
Run from the repository root:  python research/forget-elimination/exploration/0017c_start_low.py
"""
import math, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from lucid.filter import lucid as L
from lucid.filter.lucid import LucidFilter
from rig import cached, CACHE
V0 = -8.0
_reset0 = L._ClassDrift.reset
def _reset_low(self): _reset0(self); self.v = V0
def low(fn):
    L._ClassDrift.reset = _reset_low
    try: return fn()
    finally: L._ClassDrift.reset = _reset0
def load(name): return pickle.load(open(os.path.join(CACHE, name + ".pkl"), "rb"))
def paired(x, y):
    d = np.asarray(y) - np.asarray(x); sem = d.std(ddof=1) / math.sqrt(len(d)); return d.mean(), d.mean() / sem

# ---- hero
N, JA, JU, NA = 900, 380, 9.0, 600
W = (("jump", slice(JA, JA + 40)), ("steady", slice(80, JA)), ("loud", slice(NA + 40, N)))
rows = {a: dict(ll=[], **{k: [] for k, _ in W}) for a in ("default", "ceiling start", "low start")}; fr = []
for seed in range(11, 35):
    rng = np.random.default_rng(seed); th = np.cumsum(rng.normal(0, math.sqrt(0.02), N)); th[JA:] += JU
    y = th + rng.normal(0, np.where(np.arange(N) < NA, 1.0, 3.0))
    def one():
        f = LucidFilter(forget=None); m = np.asarray(f.filter(y.reshape(-1, 1)).mean).reshape(-1)
        return dict(mse={k: float(np.mean((m[sl] - th[sl]) ** 2)) for k, sl in W}, ll=f.loglik, v=f._drift.v)
    r = {"default": load(f"0010v4_scalar_{seed}_A"), "ceiling start": load(f"0010v4_scalar_{seed}_S"),
         "low start": cached(f"0017c_scalar_{seed}", lambda: low(one))}
    for a in rows:
        rows[a]["ll"].append(r[a]["ll"])
        for k, _ in W: rows[a][k].append(r[a]["mse"][k])
    fr.append(math.exp(r["low start"]["v"]))
print("HERO, 24 seeds: RMSE per window, log-likelihood")
for a in rows:
    print(f"  {a:14s} " + "  ".join(f"{k} {math.sqrt(np.mean(rows[a][k])):.4f}" for k, _ in W) + f"   loglik {np.mean(rows[a]['ll']):.2f}")
for a, b in (("default", "low start"), ("ceiling start", "low start")):
    print(f"  {b} minus {a}: " + "  ".join("%s dMSE %+.4f (t=%+.2f)" % ((k,) + paired(rows[a][k], rows[b][k])) for k, _ in W)
          + "   dloglik %+.2f (t=%+.2f)" % paired(rows[a]["ll"], rows[b]["ll"]))
print(f"  low start: final rate fraction median {np.median(fr):.4f}, range {min(fr):.5f}-{max(fr):.3f}", flush=True)

# ---- stress modes
src = open(os.path.join(HERE, "0012_new_stress_modes.py")).read().split("MODES = [")[0]
m12 = {"__file__": os.path.join(HERE, "0012_new_stress_modes.py"), "__name__": "m12"}; exec(compile(src, "0012_gen", "exec"), m12)
make, kalman, run_filter = m12["make"], m12["kalman"], m12["run_filter"]
rm = lambda e: float(np.sqrt(np.mean(e[100:] ** 2)))
print("\nSTRESS, 8 seeds: RMSE over the oracle's; log-likelihood per step minus the default's (x1000)")
print(f"  {'mode':12s} {'default':>8s} {'ceiling':>8s} {'low':>8s}   {'low/ceiling':>11s} {'t':>6s}   {'dll ceiling':>11s} {'dll low':>8s}   {'rate: ceiling / low':>20s}")
for mode in ("calm", "ramp", "flicker 25", "flicker 100", "stale"):
    acc = dict(A=[], S=[], Lo=[], dS=[], dL=[], vS=[], vL=[])
    for seed in range(8):
        if mode == "calm":
            rng = np.random.default_rng(4000 + seed); T = 6000
            x = np.cumsum(rng.normal(0, math.sqrt(0.02), T)); d = dict(y=x + rng.normal(0, 1, T), x=x, q=np.full(T, 0.02), r=np.ones(T), dt=np.ones(T))
            A = cached(f"0017c_calm_{seed}_A", lambda: run_filter(d, {})); S = cached(f"0017c_calm_{seed}_S", lambda: run_filter(d, dict(forget=None)))
        else:
            d = make(mode, seed); key = mode.replace(" ", "")
            A = load(f"0012v4_{key}_{seed}_A"); S = load(f"0012v4_{key}_{seed}_S")
        Lo = cached(f"0017c_{mode.replace(' ', '')}_{seed}_low", lambda: low(lambda: run_filter(d, dict(forget=None))))
        eo = kalman(d["y"], d["q"], d["r"], d["dt"])[0] - d["x"]; T = len(eo)
        acc["A"].append(rm(A["est"] - d["x"]) / rm(eo)); acc["S"].append(rm(S["est"] - d["x"]) / rm(eo)); acc["Lo"].append(rm(Lo["est"] - d["x"]) / rm(eo))
        acc["dS"].append(1e3 * (S["ll"] - A["ll"]) / T); acc["dL"].append(1e3 * (Lo["ll"] - A["ll"]) / T)
        acc["vS"].append(math.exp(S["v"])); acc["vL"].append(math.exp(Lo["v"]))
    dl = np.log(acc["Lo"]) - np.log(acc["S"]); t = dl.mean() / (dl.std(ddof=1) / math.sqrt(8))
    print(f"  {mode:12s} {np.mean(acc['A']):8.3f} {np.mean(acc['S']):8.3f} {np.mean(acc['Lo']):8.3f}   {math.exp(dl.mean()):11.3f} {t:+6.2f}   "
          f"{np.mean(acc['dS']):+11.2f} {np.mean(acc['dL']):+8.2f}   {np.median(acc['vS']):9.4f} / {np.median(acc['vL']):.4f}", flush=True)
