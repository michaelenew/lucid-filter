"""0009 -- the chain carries member state with the weight it moves (IMM step), on the arm.

Variants of `star_filter.MIX`: () weights only (0007); ("m",) state mean+covariance;
("mu",) scale-walk centres; ("m", "mu") both.  One filter in memory at a time; every run is
checkpointed to `_cache/`.

Run from the repository root:
    python research/forget-elimination/exploration/0009_state_moves_with_weight.py [seed ...] [--only both]
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
args = [a for a in sys.argv[1:] if not a.startswith("--")]
only_both = "--only" in sys.argv
seeds = [int(a) for a in args if a.lstrip("-").isdigit()] or [2]
jstd, pot, acc = m54.schedule(); Wn = m54.windows()
VARIANTS = [("shipped", "shipped", {}, ()), ("star, weights only", "star", dict(forget=1.0), ()),
            ("star + m", "star", dict(forget=1.0), ("m",)), ("star + mu", "star", dict(forget=1.0), ("mu",)),
            ("star + m + mu", "star", dict(forget=1.0), ("m", "mu"))]
if only_both: VARIANTS = [VARIANTS[0], VARIANTS[1], VARIANTS[4]]
for seed in seeds:
    U, St, Y = AR.simulate(seed, jstd, pot, acc); Pt = m54.tip(St)
    orc = AR.kalman(U, Y, [j ** 2 * (AR.B @ AR.B.T) for j in jstd],
                    [np.concatenate([[pot[k, j] ** 2, acc[k, j] ** 2, acc[k, j] ** 2] for j in range(AR.NJ)]) for k in range(m54.T)])
    Po = m54.tip(orc)
    print(f"seed {seed}:  tip RMSE over the oracle's per window; log-likelihood; worst 100-step block (x oracle)", flush=True)
    for nm, variant, kw, mix in VARIANTS:
        def one(variant=variant, kw=kw, mix=mix):
            mod = SF.load(variant); SF.MIX = mix
            f = mod.LucidFilter(dynamics=AR.F, control=AR.B, H=AR.measure, process=AR.Q0, measurement=AR.R0, **kw)
            est = np.asarray(f.filter(Y, U).mean); SF.MIX = ()
            if not np.all(np.isfinite(est)): return None
            P = m54.tip(est)
            return dict(err=np.sqrt(((P - Pt) ** 2).sum(1)), ll=f.loglik,
                        win={k: m54.rms(P, Pt, m54.mask(sp)) / m54.rms(Po, Pt, m54.mask(sp)) for k, sp in Wn.items()})
        key = f"0009_{seed}_{variant}_{'_'.join(mix) or 'none'}"
        if variant == "shipped" or not mix:                    # reuse 0007/0008's runs where they exist
            key = f"0009_{seed}_{variant}_none"
        r = cached(key, one); gc.collect()
        if r is None:
            print(f"  {nm:20s} NaN", flush=True); continue
        eo = np.sqrt(((Po - Pt) ** 2).sum(1))
        blocks = [(np.sqrt(np.mean(r["err"][lo:lo + 100] ** 2)) / np.sqrt(np.mean(eo[lo:lo + 100] ** 2)), lo) for lo in range(0, m54.T, 100)]
        wb = max(blocks)
        print(f"  {nm:20s} " + "  ".join(f"{k} {v:.3f}" for k, v in r["win"].items()) + f"   loglik {r['ll']:.0f}   worst block {wb[0]:.1f}x at {wb[1]}", flush=True)
