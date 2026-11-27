"""0019 -- the ladder chains made independent of how finely the ladder is cut.

In the filter the hop rate between neighbouring rungs is (their divergence) / _DRIFT_LEVEL,
which goes as g^2 for spacing g, so the belief diffuses as g^4 (0018).  Variants, both
anchored to agree with the filter at _LADDER_MEM = 1000:

    literal : evidence per hop proportional to the move  -> hop rate ~ g     (x (mem/1000)^0.5)
    exact   : diffusion independent of g                 -> hop rate ~ 1/g^2 (x (mem/1000)^2)

The ramp stress mode of 0012, _LADDER_MEM in {250, 1000, 4000}.  Probe-side: the split and
hazard generators' eigenvalues are rescaled after `_ClassDrift.__init__`; nothing under
`lucid/` is changed.

Every (mem, seed, arm) run is checkpointed to `_cache/`.
Run from the repository root:  python research/forget-elimination/exploration/0019_grid_free_ladder.py
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
make, kalman = m12["make"], m12["kalman"]
SEEDS = range(6); rm = lambda e: float(np.sqrt(np.mean(e[100:] ** 2)))
_init0 = L._ClassDrift.__init__

def run(d, mem, power):
    """power: exponent p in  hop rate x (mem/1000)^p ;  None = the filter as it stands."""
    keep = L._LADDER_MEM; L._LADDER_MEM = float(mem)
    if power is not None:
        fac = (mem / 1000.0) ** power
        def init(self, *a, **k): _init0(self, *a, **k); self._slam = self._slam * fac; self._jlam = self._jlam * fac
        L._ClassDrift.__init__ = init
    try:
        f = LucidFilter(forget=None)
        est = np.asarray(f.filter(d["y"].reshape(-1, 1)).mean).reshape(-1)
        return dict(est=est, ll=f.loglik, rungs=f.split_arr.shape[0], v=f._drift.v)
    finally:
        L._LADDER_MEM = keep; L._ClassDrift.__init__ = _init0

ARMS = [("in the filter (rate ~ g^2)", None), ("literal (rate ~ g)", 0.5), ("exact (rate ~ 1/g^2)", 2.0)]
print(f"{'arm':30s}" + "".join(f"{'mem ' + str(m):>12s}" for m in (250, 1000, 4000)) + "   split rungs    spread 4000/250")
for nm, p in ARMS:
    row, rungs = [], []
    for mem in (250, 1000, 4000):
        vals = []
        for seed in SEEDS:
            d = make("ramp", seed); eo = kalman(d["y"], d["q"], d["r"], d["dt"])[0] - d["x"]
            if p is None:
                r = cached(f"0018b_{mem}_{seed}_S", lambda: run(d, mem, None)); n = None
            else:
                r = cached(f"0019_{mem}_{seed}_{p}", lambda: run(d, mem, p)); n = r["rungs"]
            vals.append(rm(r["est"] - d["x"]) / rm(eo))
        row.append(np.mean(vals)); rungs.append(n)
    print(f"{nm:30s}" + "".join(f"{v:12.3f}" for v in row) + f"   {rungs}    {row[2] / row[0]:.3f}", flush=True)
