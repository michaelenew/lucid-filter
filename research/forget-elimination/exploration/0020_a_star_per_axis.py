"""0020 -- a star per axis beneath grid-free interval ceilings (`drift3.py`).

Against the default and the filter as it stands (`forget=None`: one star, ladder ceilings
tied to the grid, hazard at its ceiling):

  ramp   (0012's mode; _LADDER_MEM 1000 and 4000)      -- the split must move; grid independence
  faults (0014's rig; calm 600 and 4000)               -- the hazard belief
  hero   (24 seeds, 900 steps)                         -- a short record
  stress (calm, stale, flicker 25, flicker 100)        -- the walk down, and change

Every (rig, seed, arm) run is checkpointed to `_cache/`.
Run from the repository root:  python research/forget-elimination/exploration/0020_a_star_per_axis.py [ramp] [faults] [hero] [stress]
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
only = set(sys.argv[1:]) or {"ramp", "faults", "hero", "stress"}
TAG = "v1"
def load(name): return pickle.load(open(os.path.join(CACHE, name + ".pkl"), "rb"))
rm = lambda e: float(np.sqrt(np.mean(e[100:] ** 2)))
def frac(d): return np.exp(np.array(d.path)) if len(d.path) else np.ones((1, 3))

def scalar3(y, mem=None):
    keep = L._LADDER_MEM
    if mem: L._LADDER_MEM = float(mem)
    try:
        with drift3.installed():
            f = LucidFilter(forget=None); T = len(y); est = np.empty(T); ll = np.empty(T)
            for t in range(T):
                st = f.update(np.array([y[t]])); est[t] = st.mean[0]; ll[t] = st.loglik
            return dict(est=est, lls=ll, ll=f.loglik, path=frac(f._drift)[::20], moves=f._drift.moves.copy(), rungs=f.split_arr.shape[0])
    finally:
        L._LADDER_MEM = keep

if "ramp" in only:
    print("RAMP, 6 seeds: RMSE over the oracle's")
    print(f"  {'_LADDER_MEM':>11s} {'default':>8s} {'as it stands':>13s} {'star per axis':>14s}   split fraction of its ceiling: at 1000 / 3000 / end      moves (class, split)")
    for mem in (1000, 4000):
        A, S, D3, fr, mv = [], [], [], [], []
        for seed in range(6):
            d = make("ramp", seed); eo = kalman(d["y"], d["q"], d["r"], d["dt"])[0] - d["x"]
            a = load(f"0018b_{mem}_{seed}_A"); s = load(f"0018b_{mem}_{seed}_S")
            r = cached(f"0020{TAG}_ramp_{mem}_{seed}", lambda: scalar3(d["y"], mem))
            A.append(rm(a["est"] - d["x"]) / rm(eo)); S.append(rm(s["est"] - d["x"]) / rm(eo)); D3.append(rm(r["est"] - d["x"]) / rm(eo))
            p = r["path"]; fr.append([p[50][1], p[150][1], p[-1][1]]); mv.append(r["moves"][:2])
        f_ = np.exp(np.mean(np.log(fr), axis=0))
        print(f"  {mem:11d} {np.mean(A):8.3f} {np.mean(S):13.3f} {np.mean(D3):14.3f}   {f_[0]:.2e} / {f_[1]:.2e} / {f_[2]:.2e}      {np.mean(mv, axis=0).round(1)}", flush=True)

if "faults" in only:
    POST = 900
    def mk(C, seed):
        r = np.random.default_rng(700 + seed); T = C + POST; x = np.zeros(T)
        for t in range(1, T): x[t] = (0.9 if t < C else 0.3) * x[t - 1] + 0.3 * r.standard_normal()
        return x, x[:, None] + 0.7 * r.standard_normal((T, 2))
    def run3(C, seed):
        x, Y = mk(C, seed)
        with drift3.installed():
            f = LucidFilter(dynamics=[[0.9]], H=[[1.0], [1.0]], process=[[0.09]], measurement=[0.49, 0.49], faults=True, forget=None)
            T = len(Y); fault = np.empty(T); hz = np.empty(T); err = np.empty(T)
            for t in range(T):
                st = f.update(Y[t]); fault[t] = st.fault; hz[t] = st.hazard; err[t] = st.mean[0] - x[t]
            return dict(fault=fault, hazard=hz, err=err, path=frac(f._drift)[::20], moves=f._drift.moves.copy())
    print("\nFAULTS, 4 seeds")
    print(f"  {'calm':>5s} {'arm':14s} {'hazard before':>13s} {'false alarms: first 500 / rest':>31s} {'first > 1/2':>11s} {'held 50':>8s} {'RMSE +300':>9s}   hazard fraction of its ceiling at 500 / end of calm")
    for C in (600, 4000):
        for nm in ("default", "as it stands", "star per axis"):
            hz, fa0, fa1, d1, d50, rmse, fr = [], [], [], [], [], [], []
            for seed in range(4):
                r = load(f"0014_{C}_{seed}_default") if nm == "default" else load(f"0014_{C}_{seed}_forget=None_ownceiling") if nm == "as it stands" \
                    else cached(f"0020{TAG}_faults_{C}_{seed}", lambda: run3(C, seed))
                f = r["fault"]; post = f[C:] > 0.5
                hz.append(float(np.exp(np.mean(np.log(r["hazard"][C - 50:C]))))); fa0.append(float(np.mean(f[150:500] > 0.5))); fa1.append(float(np.mean(f[500:C] > 0.5)) if C > 500 else float("nan"))
                d1.append(int(np.argmax(post)) if post.any() else POST)
                run50 = np.convolve(post.astype(int), np.ones(50, int), "valid") == 50; d50.append(int(np.argmax(run50)) if run50.any() else POST)
                rmse.append(float(np.sqrt(np.mean(r["err"][C:C + 300] ** 2))))
                if "path" in r: fr.append([r["path"][25][2], r["path"][C // 20 - 1][2]])
            extra = f"   {np.exp(np.mean(np.log(fr), axis=0))[0]:.2e} / {np.exp(np.mean(np.log(fr), axis=0))[1]:.2e}" if fr else ""
            print(f"  {C:5d} {nm:14s} {np.mean(hz):13.2e} {np.mean(fa0):15.3f} / {np.nanmean(fa1):<13.3f} {np.median(d1):11.0f} {np.median(d50):8.0f} {np.mean(rmse):9.4f}{extra}    per seed {d1}", flush=True)

if "hero" in only:
    N, JA, JU, NA = 900, 380, 9.0, 600
    W = (("jump", slice(JA, JA + 40)), ("steady", slice(80, JA)), ("loud", slice(NA + 40, N)))
    rows = {a: dict(ll=[], **{k: [] for k, _ in W}) for a in ("default", "as it stands", "star per axis")}; fr = []
    for seed in range(11, 35):
        rng = np.random.default_rng(seed); th = np.cumsum(rng.normal(0, math.sqrt(0.02), N)); th[JA:] += JU
        y = th + rng.normal(0, np.where(np.arange(N) < NA, 1.0, 3.0))
        def one():
            r = scalar3(y); return dict(mse={k: float(np.mean((r["est"][sl] - th[sl]) ** 2)) for k, sl in W}, ll=r["ll"], end=r["path"][-1])
        r = {"default": load(f"0010v4_scalar_{seed}_A"), "as it stands": load(f"0010v4_scalar_{seed}_S"), "star per axis": cached(f"0020{TAG}_hero_{seed}", one)}
        for a in rows:
            rows[a]["ll"].append(r[a]["ll"])
            for k, _ in W: rows[a][k].append(r[a]["mse"][k])
        fr.append(r["star per axis"]["end"])
    print("\nHERO, 24 seeds: RMSE per window, log-likelihood")
    for a in rows:
        print(f"  {a:14s} " + "  ".join(f"{k} {math.sqrt(np.mean(rows[a][k])):.4f}" for k, _ in W) + f"   loglik {np.mean(rows[a]['ll']):.2f}")
    d = np.array(rows["star per axis"]["ll"]) - np.array(rows["default"]["ll"])
    print(f"  star per axis minus default: dloglik {d.mean():+.2f} (t={d.mean() / (d.std(ddof=1) / math.sqrt(len(d))):+.2f});  fractions at the end (class, split): {np.exp(np.mean(np.log(fr), axis=0))[:2].round(4)}", flush=True)

if "stress" in only:
    print("\nSTRESS, 8 seeds: RMSE over the oracle's; log-lik per step minus the default's, x1000 (whole record | first third | last third)")
    print(f"  {'mode':12s} {'default':>8s} {'as it stands':>13s} {'star per axis':>14s}   {'as it stands':>26s}   {'star per axis':>26s}   split fraction at the end")
    for mode in ("calm", "stale", "flicker 25", "flicker 100"):
        A, S, D3, dS, dD, fr = [], [], [], [], [], []
        for seed in range(8):
            if mode == "calm":
                rng = np.random.default_rng(4000 + seed); T = 6000
                x = np.cumsum(rng.normal(0, math.sqrt(0.02), T)); d = dict(y=x + rng.normal(0, 1, T), x=x, q=np.full(T, 0.02), r=np.ones(T), dt=np.ones(T))
                a = load(f"0017c_calm_{seed}_A"); s = load(f"0017c_calm_{seed}_S")
            else:
                d = make(mode, seed); key = mode.replace(" ", ""); a = load(f"0012v4_{key}_{seed}_A"); s = load(f"0012v4_{key}_{seed}_S")
            r = cached(f"0020{TAG}_{mode.replace(' ', '')}_{seed}", lambda: scalar3(d["y"]))
            eo = kalman(d["y"], d["q"], d["r"], d["dt"])[0] - d["x"]; T = len(eo)
            A.append(rm(a["est"] - d["x"]) / rm(eo)); S.append(rm(s["est"] - d["x"]) / rm(eo)); D3.append(rm(r["est"] - d["x"]) / rm(eo))
            dS.append(1e3 * (s["ll"] - a["ll"]) / T); dD.append(1e3 * (r["ll"] - a["ll"]) / T); fr.append(r["path"][-1][1])
        print(f"  {mode:12s} {np.mean(A):8.3f} {np.mean(S):13.3f} {np.mean(D3):14.3f}   {np.mean(dS):+26.2f}   {np.mean(dD):+26.2f}   {np.exp(np.mean(np.log(fr))):.2e}", flush=True)
