"""0007c -- where the star loses the arm's calm window on seed 2, and whether the start matters.

Tip error in 100-step blocks for: shipped; the star started at the ceiling (as derived); the
star started low (v0 = -6, a rate fraction of 0.0025).  One filter in memory at a time; each variant is checkpointed to `_cache/`.
The low start is applied through `star_filter.V0` (an override on the filter object is
erased by the filter's own reset -- the first version of this probe did that and its
'low start' arm was the plain star).

Run from the repository root:  python research/forget-elimination/exploration/0007c_arm_seed2_trace.py [seed]
"""
import gc, importlib.util, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import star_filter as SF
from rig import cached
sys.path.insert(0, "research/multivariate-statfilter/scripts"); import arm5dof as AR
spec = importlib.util.spec_from_file_location("p54", "research/multivariate-statfilter/exploration/0054_physical_sensors.py")
m54 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m54)
np.seterr(all="ignore")
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2
jstd, pot, acc = m54.schedule(); Wn = m54.windows()
print("windows:", {k: sp for k, sp in Wn.items()}, flush=True)
U, St, Y = AR.simulate(seed, jstd, pot, acc); Pt = m54.tip(St)
orc = AR.kalman(U, Y, [j ** 2 * (AR.B @ AR.B.T) for j in jstd],
                [np.concatenate([[pot[k, j] ** 2, acc[k, j] ** 2, acc[k, j] ** 2] for j in range(AR.NJ)]) for k in range(m54.T)])
Po = m54.tip(orc)
def err(P): d = np.asarray(P) - np.asarray(Pt); return np.sqrt((d.reshape(len(d), -1) ** 2).sum(axis=1))
out = {"oracle": (err(Po), None, None)}
for nm, variant, kw, v0 in (("shipped", "shipped", {}, None), ("star", "star", dict(forget=1.0), None),
                            ("star, low start", "star", dict(forget=1.0), -6.0)):
    def one(variant=variant, kw=kw, v0=v0):
        mod = SF.load(variant); SF.V0 = 0.0 if v0 is None else v0
        f = mod.LucidFilter(dynamics=AR.F, control=AR.B, H=AR.measure, process=AR.Q0, measurement=AR.R0, **kw)
        P = m54.tip(np.asarray(f.filter(Y, U).mean)); SF.V0 = 0.0
        return dict(err=err(P), ll=f.loglik, path=np.exp(np.array(f._star["path"])) if variant == "star" else None,
                    win={k: m54.rms(P, Pt, m54.mask(sp)) / m54.rms(Po, Pt, m54.mask(sp)) for k, sp in Wn.items()})
    r = cached(f"0007c_{seed}_{nm.replace(', ', '_').replace(' ', '_')}_v2", one)
    out[nm] = (r["err"], r["ll"], r["path"])
    print(f"{nm:16s} loglik {r['ll']:.1f}  " + "  ".join(f"{k} {v:.3f}" for k, v in r["win"].items()), flush=True)
    gc.collect()
print("\ntip error (rms over 100-step blocks); the star's rate fraction at the block's end")
for lo in range(0, m54.T, 100):
    sl = slice(lo, lo + 100)
    print(f"  {lo:4d}-{lo + 100:4d}  " + "  ".join(f"{nm} {np.sqrt(np.mean(out[nm][0][sl] ** 2)):.4f}" for nm in out)
          + f"   rate: star {out['star'][2][min(lo + 99, len(out['star'][2]) - 1)]:.3f}, low start {out['star, low start'][2][min(lo + 99, len(out['star, low start'][2]) - 1)]:.4f}")
