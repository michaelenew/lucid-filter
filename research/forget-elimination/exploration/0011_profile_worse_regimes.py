"""0011 -- profile the arm windows where `forget=None` is worse than the default.

Per step, for the default and for the class drift: the tip error, the bank's weight on each
(phi, s) class, the tip error of each class's own estimate, the drift's rate fraction, and the
reported measurement/process scales.  Then, per 50-step block of a named window: where the
loss is, where the weight is, and whether a better member was available.

Every (seed, arm) trace is checkpointed to `_cache/`; one filter in memory at a time.
Run from the repository root:
    python research/forget-elimination/exploration/0011_profile_worse_regimes.py SEED:WINDOW [SEED:WINDOW ...]
"""
import gc, importlib.util, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from lucid.filter.lucid import LucidFilter, _logsumexp
from rig import cached
sys.path.insert(0, "research/multivariate-statfilter/scripts"); import arm5dof as AR
spec = importlib.util.spec_from_file_location("p54", "research/multivariate-statfilter/exploration/0054_physical_sensors.py")
m54 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m54)
np.seterr(all="ignore")
TAG = "v3"
jstd, pot, acc = m54.schedule(); Wn = m54.windows()
NST = AR.NJ * AR.ORDER
def tips(X): return m54.tip(np.asarray(X)[:, :NST])

def trace(seed, kw):
    U, St, Y = AR.simulate(seed, jstd, pot, acc); Pt = m54.tip(St)
    f = LucidFilter(dynamics=AR.F, control=AR.B, H=AR.measure, process=AR.Q0, measurement=AR.R0, **kw)
    bank = f._banks[0]; T = len(Y); nc = f._nc
    out = dict(err=np.empty(T), w=np.empty((T, nc)), ce=np.empty((T, nc)), v=np.full(T, np.nan),
               ms=np.empty((T, f.m)), ps=np.empty((T, f.n)), classes=list(zip(f.phi_arr.tolist(), f.s_arr.tolist())))
    for t in range(T):
        st = f.update(Y[t], u=U[t])
        if f._drift is None:
            post = np.exp(f._logw - _logsumexp(f._logw))
        else:
            d = f._drift; lq = d.lead + d.node_prior; lq = lq - _logsumexp(lq)
            post = sum(math.exp(lq[i]) * np.exp(d.lw[i] - _logsumexp(d.lw[i])) for i in range(len(lq)))
            out["v"][t] = math.exp(d.v)
        out["w"][t] = post
        out["ce"][t] = np.sqrt(((tips(bank._m) - Pt[t]) ** 2).sum(1))
        out["err"][t] = float(np.sqrt(((m54.tip(np.asarray(st.mean)[None]) - Pt[t]) ** 2).sum()))
        out["ms"][t] = st.measurement_scale; out["ps"][t] = st.process_scale
    out["ll"] = f.loglik
    del f; gc.collect()
    return out

def oracle_err(seed):
    U, St, Y = AR.simulate(seed, jstd, pot, acc)
    orc = AR.kalman(U, Y, [j ** 2 * (AR.B @ AR.B.T) for j in jstd],
                    [np.concatenate([[pot[k, j] ** 2, acc[k, j] ** 2, acc[k, j] ** 2] for j in range(AR.NJ)]) for k in range(m54.T)])
    return np.sqrt(((m54.tip(orc) - m54.tip(St)) ** 2).sum(1))

for arg in sys.argv[1:] or ["0:PROCESS"]:
    seed, win = arg.split(":"); seed = int(seed)
    A = cached(f"0011{TAG}_{seed}_A", lambda: trace(seed, {}))
    S = cached(f"0011{TAG}_{seed}_S", lambda: trace(seed, dict(forget=None)))
    eo = cached(f"0011{TAG}_{seed}_oracle", lambda: oracle_err(seed))
    cls = A["classes"]; svals = sorted({c[1] for c in cls})
    def by_s(w): return np.array([w[:, [i for i, c in enumerate(cls) if c[1] == s]].sum(1) for s in svals]).T
    rm = lambda e, sl: float(np.sqrt(np.mean(e[sl] ** 2)))
    for a, b in Wn[win]:
        full = slice(a, b)
        print(f"\nseed {seed}  {win} [{a}, {b}):  x oracle  default {rm(A['err'], full) / rm(eo, full):.3f}   drift {rm(S['err'], full) / rm(eo, full):.3f}")
        print(f"  {'block':>11s} {'oracle':>7s} {'default':>8s} {'drift':>7s} {'ratio':>6s} | drift weight on s = " + "/".join(f"{s:g}" for s in svals)
              + " | best class err (drift)  its s | default's class | rate | mean log meas-scale: default / drift")
        for lo in range(a - (a % 50), b, 50):
            sl = slice(max(lo, a), min(lo + 50, b))
            if sl.stop - sl.start < 10: continue
            w = by_s(S["w"][sl]).mean(0)
            ce = np.sqrt(np.mean(S["ce"][sl] ** 2, axis=0)); bi = int(np.argmin(ce))
            wa = by_s(A["w"][sl]).mean(0)
            print(f"  {sl.start:5d}-{sl.stop:5d} {rm(eo, sl):7.4f} {rm(A['err'], sl):8.4f} {rm(S['err'], sl):7.4f} {rm(S['err'], sl) / rm(A['err'], sl):6.2f} | "
                  + " ".join(f"{x:.2f}" for x in w) + f" | {ce[bi]:.4f} {cls[bi][1]:g} | s={svals[int(np.argmax(wa))]:g} ({wa.max():.2f}) | {np.nanmean(S['v'][sl]):.2f} | "
                  f"{A['ms'][sl].mean():+.2f} / {S['ms'][sl].mean():+.2f}")
        # where the loss sits within the window
        e2a, e2s = A["err"][full] ** 2, S["err"][full] ** 2
        excess = e2s - e2a; order = np.argsort(excess)[::-1]; top = order[: max(1, len(order) // 20)]
        print(f"  the worst 5% of steps carry {excess[top].sum() / max(excess.sum(), 1e-300):.0%} of the excess squared error"
              f" (excess total {excess.sum():.4g}; drift's oracle-class regret: best single class would give {rm(np.nanmin(S['ce'][full], axis=1), full) / rm(eo, full):.3f} x oracle)")
