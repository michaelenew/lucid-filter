"""0018b -- is `_LADDER_MEM` still only a node budget under the class drift?

The ramp stress mode of 0012 (it needs the split ladder to move), with `_LADDER_MEM` set to
250, 1000 (shipped) and 4000, for the default and for `forget=None`.  The constant is
patched on the module before each filter is built; nothing under `lucid/` is changed.

Every (mem, seed, arm) run is checkpointed to `_cache/`.
Run from the repository root:  python research/forget-elimination/exploration/0018b_ladder_mem.py
"""
import math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from lucid.filter import lucid as L
from lucid.filter.lucid import LucidFilter
from rig import cached
src = open(os.path.join(HERE, "0012_new_stress_modes.py")).read().split("MODES = [")[0]
m12 = {"__file__": os.path.join(HERE, "0012_new_stress_modes.py"), "__name__": "m12"}; exec(compile(src, "0012_gen", "exec"), m12)
make, kalman, run_filter = m12["make"], m12["kalman"], m12["run_filter"]
SEEDS = range(6); rm = lambda e: float(np.sqrt(np.mean(e[100:] ** 2)))
def run(d, kw, mem):
    keep = L._LADDER_MEM; L._LADDER_MEM = float(mem)
    try:
        f = LucidFilter(**kw); nb = f.split_arr.shape[0]
        est = np.asarray(f.filter(d["y"].reshape(-1, 1)).mean).reshape(-1)
        return dict(est=est, ll=f.loglik, rungs=nb)
    finally:
        L._LADDER_MEM = keep
print(f"{'_LADDER_MEM':>11s} {'split rungs':>11s} {'default':>8s} {'forget=None':>12s} {'drift/default':>14s}   loglik/step: default, drift (x1000, minus default at 1000)")
base = None
for mem in (250, 1000, 4000):
    A, S, lA, lS, nr = [], [], [], [], 0
    for seed in SEEDS:
        d = make("ramp", seed); eo = kalman(d["y"], d["q"], d["r"], d["dt"])[0] - d["x"]; T = len(eo)
        a = cached(f"0018b_{mem}_{seed}_A", lambda: run(d, {}, mem)); s = cached(f"0018b_{mem}_{seed}_S", lambda: run(d, dict(forget=None), mem))
        A.append(rm(a["est"] - d["x"]) / rm(eo)); S.append(rm(s["est"] - d["x"]) / rm(eo)); lA.append(a["ll"] / T); lS.append(s["ll"] / T); nr = a["rungs"]
    if mem == 1000: base = np.mean(lA)
    print(f"{mem:11d} {nr:11d} {np.mean(A):8.3f} {np.mean(S):12.3f} {np.mean(S) / np.mean(A):14.3f}   {np.mean(lA):.5f}  {np.mean(lS):.5f}", flush=True)
