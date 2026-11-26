"""0007b -- the arm rig on further seeds, shipped (A) against the star (S).  See 0007.
Each (seed, arm) run is checkpointed to `_cache/` and skipped on re-run.

Run from the repository root:  python research/forget-elimination/exploration/0007b_arm_seeds.py 1 2
"""
import importlib.util, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import star_filter as SF
from rig import cached
A = SF.load("shipped"); S = SF.load("star")
sys.path.insert(0, "research/multivariate-statfilter/scripts"); import arm5dof as AR
spec = importlib.util.spec_from_file_location("p54", "research/multivariate-statfilter/exploration/0054_physical_sensors.py")
m54 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m54)
np.seterr(all="ignore")
jstd, pot, acc = m54.schedule(); Wn = m54.windows()
for seed in [int(x) for x in sys.argv[1:]] or [1, 2]:
    U, St, Y = AR.simulate(seed, jstd, pot, acc)
    orc = AR.kalman(U, Y, [j ** 2 * (AR.B @ AR.B.T) for j in jstd],
                    [np.concatenate([[pot[k, j] ** 2, acc[k, j] ** 2, acc[k, j] ** 2] for j in range(AR.NJ)]) for k in range(m54.T)])
    Pt = m54.tip(St); Po = m54.tip(orc)
    for nm, mod, kw in (("A shipped", A, {}), ("S star", S, dict(forget=1.0))):
        def one(mod=mod, kw=kw, nm=nm):
            f = mod.LucidFilter(dynamics=AR.F, control=AR.B, H=AR.measure, process=AR.Q0, measurement=AR.R0, **kw)
            est = np.asarray(f.filter(Y, U).mean)
            if not np.all(np.isfinite(est)): return None
            Pl = m54.tip(est)
            return dict(ll=f.loglik, win={k: m54.rms(Pl, Pt, m54.mask(sp)) / m54.rms(Po, Pt, m54.mask(sp)) for k, sp in Wn.items()},
                        star=(math.exp(f._star["v"]), f._star["moves"]) if nm.startswith("S") else None)
        r = cached(f"0007_arm_{seed}_{nm[0]}", one)
        if r is None:
            print(f"seed {seed}  {nm:10s}: NaN", flush=True); continue
        extra = f"  rate fraction {r['star'][0]:.4f}, {r['star'][1]} moves" if r["star"] else ""
        print(f"seed {seed}  {nm:10s}: " + "  ".join(f"{k} {v:.3f}" for k, v in r["win"].items()) + f"   loglik {r['ll']:.1f}" + extra, flush=True)
