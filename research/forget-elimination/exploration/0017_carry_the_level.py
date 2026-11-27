"""0017 -- carry the noise LEVEL with the weight, in each member's own representation.

The drift carries the state mean and covariance with the weight it moves.  Here the receiving
member's scale-walk centre is also shifted so that its posterior-mean log-scale (centre + the
window's mean offset) equals that of the members its weight came from; its window posterior
and walk variance are left alone.  A probe-side override of `_ClassDrift._mix`; nothing under
`lucid/` is changed.  Arm, the seeds and windows of 0010.

Every (seed, arm) run is checkpointed to `_cache/`; one filter in memory at a time.
Run from the repository root:  python research/forget-elimination/exploration/0017_carry_the_level.py [seed ...]
"""
import gc, importlib.util, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from lucid.filter import lucid as L
from lucid.filter.lucid import LucidFilter
from rig import cached
sys.path.insert(0, "research/multivariate-statfilter/scripts"); import arm5dof as AR
spec = importlib.util.spec_from_file_location("p54", "research/multivariate-statfilter/exploration/0054_physical_sensors.py")
m54 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m54)
np.seterr(all="ignore")
jstd, pot, acc = m54.schedule(); Wn = m54.windows()
seeds = [int(a) for a in sys.argv[1:]] or [0, 1, 2]

_mix0 = L._ClassDrift._mix
LEVEL = "all"          # which axes' level is carried: "all", or "sensor" (the measurement scales only)
def _mix_level(self, flt, specs, pm, Pj, Pc, Ps):
    plan = self._group(flt, specs)
    lev = None
    if plan is not None and all(isinstance(b, L._EngineBank) for b, _, _ in plan[1]):
        gids, parts = plan
        acts = [tuple(np.asarray(b._act).tolist()) for b, _, _ in parts]
        if len(set(acts)) == 1 and len(acts[0]):
            act = np.array(acts[0]); N = len(gids); S = len(specs)
            pos = np.arange(len(act)) if LEVEL == "all" else np.flatnonzero(act >= parts[0][0].n)
            lev = np.empty((N, len(act)))
            for bank, rows, slots in parts:
                lev[slots] = bank.mu[rows][:, act] + (bank._pi[rows] * bank._off[rows][:, act]).sum(-1)
            w = pm[gids].reshape(S, self.nps, self.nb)
            def push(x):
                x = np.einsum("jpb...,jk->kpb...", x, Pj); x = np.einsum("jpb...,pq->jqb...", x, Pc)
                return np.einsum("jpb...,bc->jpc...", x, Ps)
            tot = push(w); dead = tot <= 1e-300
            l4 = lev.reshape(S, self.nps, self.nb, len(act))
            new = np.where(dead[..., None], l4, push(w[..., None] * l4) / np.where(dead, 1.0, tot)[..., None]).reshape(N, len(act))
            for bank, rows, slots in parts:
                mu = bank.mu[rows]; mu[:, act[pos]] += (new[slots] - lev[slots])[:, pos]; bank.mu[rows] = mu
    _mix0(self, flt, specs, pm, Pj, Pc, Ps)

def run(seed, level):
    global LEVEL
    LEVEL = level if isinstance(level, str) else "all"
    U, St, Y = AR.simulate(seed, jstd, pot, acc); Pt = m54.tip(St)
    orc = AR.kalman(U, Y, [j ** 2 * (AR.B @ AR.B.T) for j in jstd],
                    [np.concatenate([[pot[k, j] ** 2, acc[k, j] ** 2, acc[k, j] ** 2] for j in range(AR.NJ)]) for k in range(m54.T)])
    Po = m54.tip(orc)
    L._ClassDrift._mix = _mix_level if level else _mix0
    try:
        f = LucidFilter(dynamics=AR.F, control=AR.B, H=AR.measure, process=AR.Q0, measurement=AR.R0, forget=None)
        est = np.asarray(f.filter(Y, U).mean)
    finally:
        L._ClassDrift._mix = _mix0
    if not np.all(np.isfinite(est)): return None
    P = m54.tip(est); e = np.sqrt(((P - Pt) ** 2).sum(1)); eo = np.sqrt(((Po - Pt) ** 2).sum(1))
    return dict(ll=f.loglik, win={k: m54.rms(P, Pt, m54.mask(sp)) / m54.rms(Po, Pt, m54.mask(sp)) for k, sp in Wn.items()},
                blocks=[float(np.sqrt(np.mean(e[lo:lo + 100] ** 2)) / np.sqrt(np.mean(eo[lo:lo + 100] ** 2))) for lo in range(0, m54.T, 100)])

tab = {a: {k: [] for k in Wn} for a in ("state only", "state + level", "state + sensor level")}; lls = {a: [] for a in tab}; wb = {a: [] for a in tab}
for seed in seeds:
    for nm, level in (("state only", False), ("state + level", True), ("state + sensor level", "sensor")):
        key = "sensorlevel" if level == "sensor" else ("level" if level else "state")
        r = cached(f"0017_{seed}_{key}", lambda: run(seed, level)); gc.collect()
        if r is None:
            print(f"seed {seed} {nm:20s} NaN", flush=True); continue
        for k in Wn: tab[nm][k].append(r["win"][k])
        lls[nm].append(r["ll"]); wb[nm].append(max(r["blocks"]))
        print(f"seed {seed} {nm:20s} " + "  ".join(f"{k} {v:.3f}" for k, v in r["win"].items()) + f"   loglik {r['ll']:.0f}   worst block {max(r['blocks']):.1f}x", flush=True)
for nm in tab:
    if lls[nm]: print(f"mean   {nm:20s} " + "  ".join(f"{k} {np.mean(v):.3f}" for k, v in tab[nm].items()) + f"   loglik {np.mean(lls[nm]):.0f}   worst block {np.max(wb[nm]):.1f}x")
