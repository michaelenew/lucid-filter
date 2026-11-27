"""0021 -- the split's drift read off the class each cell already asserts.

`drift3.DriftImplied`: per cell, the split's log-odds diffuses with D = 2 s^2 (1 - phi^2),
hop rate D / dlo^2 between neighbouring rungs.  No ceiling, no star, no constant.  Same rigs
and comparisons as 0020 (default; the filter as it stands; 0020's star per axis).

Every (rig, seed) run is checkpointed to `_cache/`.
Run from the repository root:  python research/forget-elimination/exploration/0021_the_split_the_class_implies.py
"""
import math, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from lucid.filter import lucid as L
from lucid.filter.lucid import LucidFilter
from rig import cached, CACHE
import drift3
src = open(os.path.join(HERE, "0012_new_stress_modes.py")).read().split("MODES = [")[0]
m12 = {"__file__": os.path.join(HERE, "0012_new_stress_modes.py"), "__name__": "m12"}; exec(compile(src, "0012_gen", "exec"), m12)
make, kalman = m12["make"], m12["kalman"]
def load(name): return pickle.load(open(os.path.join(CACHE, name + ".pkl"), "rb"))
rm = lambda e: float(np.sqrt(np.mean(e[100:] ** 2)))
def run(y, mem=None):
    keep = L._LADDER_MEM
    if mem: L._LADDER_MEM = float(mem)
    try:
        with drift3.implied():
            f = LucidFilter(forget=None)
            est = np.asarray(f.filter(np.asarray(y).reshape(-1, 1)).mean).reshape(-1)
            return dict(est=est, ll=f.loglik, v=f._drift.v)
    finally:
        L._LADDER_MEM = keep

print("RAMP, 6 seeds: RMSE over the oracle's")
print(f"  {'_LADDER_MEM':>11s} {'default':>8s} {'as it stands':>13s} {'star per axis':>14s} {'class-implied':>14s}")
for mem in (250, 1000, 4000):
    A, S, D3, I = [], [], [], []
    for seed in range(6):
        d = make("ramp", seed); eo = kalman(d["y"], d["q"], d["r"], d["dt"])[0] - d["x"]
        A.append(rm(load(f"0018b_{mem}_{seed}_A")["est"] - d["x"]) / rm(eo)); S.append(rm(load(f"0018b_{mem}_{seed}_S")["est"] - d["x"]) / rm(eo))
        if mem != 250: D3.append(rm(load(f"0020v1_ramp_{mem}_{seed}")["est"] - d["x"]) / rm(eo))
        I.append(rm(cached(f"0021_ramp_{mem}_{seed}", lambda: run(d["y"], mem))["est"] - d["x"]) / rm(eo))
    print(f"  {mem:11d} {np.mean(A):8.3f} {np.mean(S):13.3f} {np.mean(D3) if D3 else float('nan'):14.3f} {np.mean(I):14.3f}", flush=True)

N, JA, JU, NA = 900, 380, 9.0, 600
W = (("jump", slice(JA, JA + 40)), ("steady", slice(80, JA)), ("loud", slice(NA + 40, N)))
rows = {a: dict(ll=[], **{k: [] for k, _ in W}) for a in ("default", "as it stands", "star per axis", "class-implied")}
for seed in range(11, 35):
    rng = np.random.default_rng(seed); th = np.cumsum(rng.normal(0, math.sqrt(0.02), N)); th[JA:] += JU
    y = th + rng.normal(0, np.where(np.arange(N) < NA, 1.0, 3.0))
    def one():
        r = run(y); return dict(mse={k: float(np.mean((r["est"][sl] - th[sl]) ** 2)) for k, sl in W}, ll=r["ll"])
    r = {"default": load(f"0010v4_scalar_{seed}_A"), "as it stands": load(f"0010v4_scalar_{seed}_S"), "star per axis": load(f"0020v1_hero_{seed}"),
         "class-implied": cached(f"0021_hero_{seed}", one)}
    for a in rows:
        rows[a]["ll"].append(r[a]["ll"])
        for k, _ in W: rows[a][k].append(r[a]["mse"][k])
print("\nHERO, 24 seeds: RMSE per window, log-likelihood, paired t of the log-likelihood against the default")
for a in rows:
    d = np.array(rows[a]["ll"]) - np.array(rows["default"]["ll"]); t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d))) if a != "default" else 0.0
    print(f"  {a:14s} " + "  ".join(f"{k} {math.sqrt(np.mean(rows[a][k])):.4f}" for k, _ in W) + f"   loglik {np.mean(rows[a]['ll']):.2f}   t={t:+.2f}", flush=True)

print("\nSTRESS, 8 seeds: RMSE over the oracle's  |  log-lik per step minus the default's, x1000")
print(f"  {'mode':12s} {'default':>8s} {'as it stands':>13s} {'star per axis':>14s} {'class-implied':>14s}  |  {'as it stands':>13s} {'star per axis':>14s} {'class-implied':>14s}   t(implied vs as-it-stands, RMSE)")
for mode in ("calm", "ramp", "stale", "flicker 25", "flicker 100", "flicker 400", "manoeuvre", "outliers"):
    A, S, D3, I, dS, dD, dI = [], [], [], [], [], [], []
    for seed in range(8):
        if mode == "calm":
            rng = np.random.default_rng(4000 + seed); T = 6000
            x = np.cumsum(rng.normal(0, math.sqrt(0.02), T)); d = dict(y=x + rng.normal(0, 1, T), x=x, q=np.full(T, 0.02), r=np.ones(T), dt=np.ones(T))
            a = load(f"0017c_calm_{seed}_A"); s = load(f"0017c_calm_{seed}_S")
        else:
            d = make(mode, seed); key = mode.replace(" ", ""); a = load(f"0012v4_{key}_{seed}_A"); s = load(f"0012v4_{key}_{seed}_S")
        d3 = load(f"0020v1_{mode.replace(' ', '')}_{seed}") if mode in ("calm", "stale", "flicker 25", "flicker 100") else None
        r = cached(f"0021_{mode.replace(' ', '')}_{seed}", lambda: run(d["y"]))
        eo = kalman(d["y"], d["q"], d["r"], d["dt"])[0] - d["x"]; T = len(eo)
        A.append(rm(a["est"] - d["x"]) / rm(eo)); S.append(rm(s["est"] - d["x"]) / rm(eo)); I.append(rm(r["est"] - d["x"]) / rm(eo))
        dS.append(1e3 * (s["ll"] - a["ll"]) / T); dI.append(1e3 * (r["ll"] - a["ll"]) / T)
        if d3 is not None: D3.append(rm(d3["est"] - d["x"]) / rm(eo)); dD.append(1e3 * (d3["ll"] - a["ll"]) / T)
    dl = np.log(I) - np.log(S); t = dl.mean() / (dl.std(ddof=1) / math.sqrt(len(dl)))
    f3 = lambda v: f"{np.mean(v):14.3f}" if v else f"{'-':>14s}"; g3 = lambda v: f"{np.mean(v):+14.2f}" if v else f"{'-':>14s}"
    print(f"  {mode:12s} {np.mean(A):8.3f} {np.mean(S):13.3f} {f3(D3)} {np.mean(I):14.3f}  |  {np.mean(dS):+13.2f} {g3(dD)} {np.mean(dI):+14.2f}   {t:+.2f}", flush=True)
