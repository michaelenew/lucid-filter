"""The memory ladder against every fixed forget, scalar hero rig (12 seeds, three windows).
    python probe_mem.py [mems]"""
import math, os, sys, types
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib; memladder = importlib.import_module('0063_memory_ladder_patch')
MEMS = sys.argv[1] if len(sys.argv) > 1 else "10,32,100,316,1000,inf"
src = memladder.patched_source(MEMS)
mod = types.ModuleType("lmem"); mod.__file__ = "lucid/filter/lucid.py"; sys.modules["lmem"] = mod
exec(compile(src, "lucid/filter/lucid.py", "exec"), mod.__dict__)
np.seterr(all="ignore")
N, JA, JU, NA = 900, 380, 9.0, 600
rows = {k: [] for k in ("jump", "steady", "C")}; mws = []
for seed in range(11, 23):
    rng = np.random.default_rng(seed); th = np.cumsum(rng.normal(0, math.sqrt(0.02), N)); th[JA:] += JU
    sd = np.where(np.arange(N) < NA, 1.0, 3.0); y = th + rng.normal(0, sd)
    f = mod.LucidFilter(); est = np.empty(N); mw = np.empty((N, f._Jm))
    for t in range(N):
        est[t] = f.update([y[t]]).mean[0]; mw[t] = f._mw
    mws.append(mw)
    for k, sl in (("jump", slice(JA, JA + 40)), ("steady", slice(80, JA)), ("C", slice(NA + 40, N))):
        rows[k].append(np.mean((est[sl] - th[sl]) ** 2))
print("MEMORY LADDER rungs", MEMS, "| RMSE:", "  ".join(f"{k} {math.sqrt(np.mean(v)):.4f}" for k, v in rows.items()), flush=True)
print("   best fixed:        jump 1.3538 (T=10)  steady 0.3832 (T>=1000)  C 0.8072 (T=100);  shipped 0.999: 1.7474 / 0.3833 / 0.8890", flush=True)
mw = np.mean(mws, 0)
for nm, a, b in (("early calm", 80, 380), ("jump", 380, 420), ("post-jump", 420, 600), ("regime C", 640, 900)):
    print(f"   rung weights {nm:>11} {a:3d}-{b:3d}: " + " ".join(f"{m}:{w:.2f}" for m, w in zip(MEMS.split(','), mw[a:b].mean(0))), flush=True)
