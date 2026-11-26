"""0016 -- the hazard rungs drifting at their own ceiling instead of the class star's fraction.

Same rig as 0014 (AR(1), two sensors, dynamics 0.9 -> 0.3 after a calm of C steps,
`faults=True`).  Arms: default; forget=None as it stands (rungs scaled by the class star,
0015); forget=None with the rungs' chain at its own ceiling (a probe-side override of
`_ClassDrift._P`; nothing in `lucid/` is changed by this probe).

Every (calm, seed, arm) run is checkpointed to `_cache/`.
Run from the repository root:  python research/forget-elimination/exploration/0016_rungs_at_their_own_rate.py
"""
import math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from lucid.filter import lucid as L
from lucid.filter.lucid import LucidFilter
from rig import cached
POST = 900; SEEDS = (0, 1, 2, 3)

def own_ceiling(on):
    """Probe-side override: P_j = exp(a G_j) with a the elapsed nominal steps."""
    C = L._ClassDrift
    if not hasattr(C, "_orig"): C._orig = (C._P, C.prior, C.post)
    P0, prior0, post0 = C._orig
    if not on:
        C._P, C.prior, C.post = P0, prior0, post0; return
    def prior(self, flt, a): self._a = a; return prior0(self, flt, a)
    def post(self, flt, llw, a, finite): self._a = a; return post0(self, flt, llw, a, finite)
    def P(self, tau):
        Pc, Ps, _ = P0(self, tau)
        return Pc, Ps, np.maximum((self._jV * np.exp(getattr(self, "_a", 1.0) * self._jlam)) @ self._jV.T, 0.0)
    C._P, C.prior, C.post = P, prior, post

def make(C, seed):
    r = np.random.default_rng(700 + seed); T = C + POST; x = np.zeros(T)
    for t in range(1, T): x[t] = (0.9 if t < C else 0.3) * x[t - 1] + 0.3 * r.standard_normal()
    return x, x[:, None] + 0.7 * r.standard_normal((T, 2))

def run(C, seed, kw, own):
    x, Y = make(C, seed); own_ceiling(own)
    try:
        f = LucidFilter(dynamics=[[0.9]], H=[[1.0], [1.0]], process=[[0.09]], measurement=[0.49, 0.49], faults=True, **kw)
        T = len(Y); fault = np.empty(T); hz = np.empty(T); err = np.empty(T); ll = np.empty(T); v = np.full(T, np.nan)
        for t in range(T):
            st = f.update(Y[t]); fault[t] = st.fault; hz[t] = st.hazard; err[t] = st.mean[0] - x[t]; ll[t] = st.loglik
            if f._drift is not None: v[t] = math.exp(f._drift.v)
        return dict(fault=fault, hazard=hz, err=err, ll=ll, v=v)
    finally:
        own_ceiling(False)

ARMS = [("default", {}, False), ("None, rungs x star", dict(forget=None), False), ("None, rungs own ceiling", dict(forget=None), True)]
print(f"{'calm':>6s} {'arm':24s} {'hazard before':>13s} {'false alarms':>12s} {'first > 1/2':>11s} {'held 50':>8s} {'RMSE +300':>9s} {'calm loglik/step - default (x1000)':>34s} {'star frac (calm)':>16s}")
for C in (600, 4000):
    base = None
    for nm, kw, own in ARMS:
        hz, fa, d1, d50, rm, lc, vf = [], [], [], [], [], [], []
        for seed in SEEDS:
            r = cached(f"0016_{C}_{seed}_{''.join(ch if ch.isalnum() else '_' for ch in nm)}", lambda: run(C, seed, kw, own))
            f = r["fault"]; post = f[C:] > 0.5
            hz.append(float(np.exp(np.mean(np.log(r["hazard"][C - 50:C]))))); fa.append(float(np.mean(f[150:C] > 0.5)))
            d1.append(int(np.argmax(post)) if post.any() else POST)
            run50 = np.convolve(post.astype(int), np.ones(50, int), "valid") == 50
            d50.append(int(np.argmax(run50)) if run50.any() else POST)
            rm.append(float(np.sqrt(np.mean(r["err"][C:C + 300] ** 2)))); lc.append(float(np.mean(r["ll"][150:C])))
            vf.append(float(np.nanmedian(r["v"][150:C])) if np.isfinite(r["v"]).any() else float("nan"))
        if base is None: base = np.array(lc)
        d = 1e3 * (np.array(lc) - base)
        print(f"{C:6d} {nm:24s} {np.mean(hz):13.2e} {np.mean(fa):12.3f} {np.median(d1):11.0f} {np.median(d50):8.0f} {np.mean(rm):9.4f} "
              f"{d.mean():+20.3f} +- {d.std(ddof=1) / 2:.3f}{'':6s} {np.mean(vf):16.3f}    first>1/2 per seed {d1}", flush=True)
