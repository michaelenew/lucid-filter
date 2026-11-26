"""0007b -- the arm rig on further seeds, shipped (A) against the star (S).  See 0007.

Run from the repository root:  python research/forget-elimination/exploration/0007b_arm_seeds.py 1 2
"""
import importlib.util, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import star_filter as SF
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
        f = mod.LucidFilter(dynamics=AR.F, control=AR.B, H=AR.measure, process=AR.Q0, measurement=AR.R0, **kw)
        est = np.asarray(f.filter(Y, U).mean)
        if np.all(np.isfinite(est)):
            Pl = m54.tip(est)
            extra = f"  rate fraction {math.exp(f._star['v']):.4f}, {f._star['moves']} moves" if nm.startswith("S") else ""
            print(f"seed {seed}  {nm:10s}: " + "  ".join(f"{k} {m54.rms(Pl, Pt, m54.mask(sp)) / m54.rms(Po, Pt, m54.mask(sp)):.3f}" for k, sp in Wn.items())
                  + f"   loglik {f.loglik:.1f}" + extra, flush=True)
        else:
            print(f"seed {seed}  {nm:10s}: NaN", flush=True)
