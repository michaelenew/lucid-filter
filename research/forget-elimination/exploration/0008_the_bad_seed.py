"""0008 -- arm seed 2: why the star's state estimate spikes after a burst.

Step 1: per step, the bank's weight on each (phi, s) class and the tip error of each class's
        own state estimate, for shipped and for the star.
Step 2: the star with its ceiling rates scaled by 0.3, 0.1, 0.03.

One filter in memory at a time; every run is checkpointed to `_cache/`.
Run from the repository root:  python research/forget-elimination/exploration/0008_the_bad_seed.py [seed]
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
U, St, Y = AR.simulate(seed, jstd, pot, acc); Pt = m54.tip(St)
orc = AR.kalman(U, Y, [j ** 2 * (AR.B @ AR.B.T) for j in jstd],
                [np.concatenate([[pot[k, j] ** 2, acc[k, j] ** 2, acc[k, j] ** 2] for j in range(AR.NJ)]) for k in range(m54.T)])
Po = m54.tip(orc)
def tip1(x): return m54.tip(np.asarray(x)[None, :AR.NJ * AR.ORDER])[0]

def run(variant, kw, scale=1.0, trace=True):
    mod = SF.load(variant, debug=trace); SF.RATE_SCALE = scale
    f = mod.LucidFilter(dynamics=AR.F, control=AR.B, H=AR.measure, process=AR.Q0, measurement=AR.R0, **kw)
    rec = dict(w=[], e=[])
    if trace:
        cls = None
        def dbg(self, post, mn):
            nonlocal cls
            if cls is None:                                    # class id of every weight row
                key = np.round(self.phi_arr, 9) * 1000 + np.round(self.s_arr, 9)
                uq = list(dict.fromkeys(key)); cell_cls = np.array([uq.index(k) for k in key])
                cls = np.tile(cell_cls, self._ndw); rec["classes"] = [(float(self.phi_arr[list(key).index(k)]), float(self.s_arr[list(key).index(k)])) for k in uq]
            t = len(rec["w"]); rows = mn[self._wm][:, :self.n]
            w = np.bincount(cls, weights=post, minlength=len(rec["classes"]))
            e = np.empty(len(w))
            for c in range(len(w)):
                sel = cls == c
                e[c] = np.linalg.norm(tip1((post[sel] @ rows[sel]) / max(w[c], 1e-300)) - Pt[t]) if w[c] > 1e-12 else np.nan
            rec["w"].append(w); rec["e"].append(e)
        f._dbg = dbg
    est = np.asarray(f.filter(Y, U).mean); SF.RATE_SCALE = 1.0
    P = m54.tip(est)
    out = dict(err=np.sqrt(((P - Pt) ** 2).sum(1)), ll=f.loglik,
               win={k: m54.rms(P, Pt, m54.mask(sp)) / m54.rms(Po, Pt, m54.mask(sp)) for k, sp in Wn.items()},
               path=np.exp(np.array(f._star["path"])) if variant == "star" else None)
    if trace: out.update(w=np.array(rec["w"]), e=np.array(rec["e"]), classes=rec["classes"])
    del f, mod; gc.collect()
    return out

A = cached(f"0008_{seed}_shipped", lambda: run("shipped", {}))
S = cached(f"0008_{seed}_star", lambda: run("star", dict(forget=1.0)))
classes = A["classes"]; svals = sorted({c[1] for c in classes})
def by_s(x, how):
    return np.stack([getattr(np, how)(x[:, [i for i, c in enumerate(classes) if c[1] == s]], axis=1) for s in svals], axis=1)
print(f"seed {seed}.  windows: " + "  ".join(f"{k} A {A['win'][k]:.3f} / S {S['win'][k]:.3f}" for k in Wn))
print("\nStep 1.  per 50-step block: tip error; weight on each s-class (summed over phi); tip error of each s-class's own estimate (weight-averaged over phi)")
print("         s classes: " + "  ".join(f"{s:g}" for s in svals))
for lo in list(range(400, 700, 50)) + list(range(1300, 1550, 50)):
    sl = slice(lo, lo + 50)
    for nm, R in (("shipped", A), ("star   ", S)):
        w = by_s(R["w"][sl], "sum").mean(axis=0)
        ew = np.nansum(R["e"][sl] * R["w"][sl], axis=0); ww = R["w"][sl].sum(axis=0)
        e = [np.sum([ew[i] for i, c in enumerate(classes) if c[1] == s]) / max(np.sum([ww[i] for i, c in enumerate(classes) if c[1] == s]), 1e-300) for s in svals]
        rate = f"  rate {S['path'][min(lo + 49, len(S['path']) - 1)]:.2f}" if nm.startswith("star") else ""
        print(f"  {lo:4d}-{lo + 50:4d} {nm} err {np.sqrt(np.mean(R['err'][sl] ** 2)):.4f}   w " + " ".join(f"{x:.3f}" for x in w)
              + "   class err " + " ".join(f"{x:.4f}" for x in e) + rate)
for lo in (500, 1400):
    sl = slice(lo, lo + 100); t = lo + int(np.argmax(S["err"][sl]))
    print(f"  worst star step in {lo}-{lo + 100}: t = {t}, err {S['err'][t]:.3f} (shipped {A['err'][t]:.4f});  star weight by s " + " ".join(f"{x:.3f}" for x in by_s(S["w"][t:t + 1], "sum")[0])
          + ";  class err by s (min over phi) " + " ".join(f"{x:.3f}" for x in by_s(np.nan_to_num(S["e"][t:t + 1], nan=np.inf), "min")[0]))

print("\nStep 2.  the star with its ceiling rates scaled")
print(f"  {'scale':>6s}  " + "  ".join(f"{k:>8s}" for k in Wn) + "   loglik   err 500-600  err 1400-1500")
for nm, R in (("shipped", A), ("1", S)):
    print(f"  {nm:>6s}  " + "  ".join(f"{R['win'][k]:8.3f}" for k in Wn) + f"  {R['ll']:8.0f}   {np.sqrt(np.mean(R['err'][500:600] ** 2)):.4f}      {np.sqrt(np.mean(R['err'][1400:1500] ** 2)):.4f}")
for sc in (0.3, 0.1, 0.03):
    R = cached(f"0008_{seed}_star_scale{sc}", lambda sc=sc: run("star", dict(forget=1.0), scale=sc, trace=False))
    print(f"  {sc:>6g}  " + "  ".join(f"{R['win'][k]:8.3f}" for k in Wn) + f"  {R['ll']:8.0f}   {np.sqrt(np.mean(R['err'][500:600] ** 2)):.4f}      {np.sqrt(np.mean(R['err'][1400:1500] ** 2)):.4f}", flush=True)
