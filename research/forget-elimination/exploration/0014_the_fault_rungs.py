"""0014 -- the hazard rungs under `forget=None`: do they freeze after a long calm?

AR(1) state, two sensors (no split ladder), dynamics 0.9 -> 0.3 after a calm of C steps,
`faults=True` (the 16-rung hazard ladder).  Arms: default; forget = 1; forget=None.
(The `forget=None` arm is the filter as it now stands, with the hazard rungs drifting at their
own ceiling (0016); its first run, without that chain, is in `0014_output.txt`.)
Read-outs: the reported hazard just before the switch; steps until the fault read-out
first exceeds 1/2 and until it stays above for 50 steps; state RMSE in the 300 steps after.

Every (calm, seed, arm) run is checkpointed to `_cache/`.
Run from the repository root:  python research/forget-elimination/exploration/0014_the_fault_rungs.py
"""
import math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from lucid.filter.lucid import LucidFilter
from rig import cached
POST = 900; SEEDS = (0, 1, 2, 3)
ARMS = [("default", {}), ("forget = 1", dict(forget=1.0)), ("forget=None", dict(forget=None))]

def make(C, seed):
    r = np.random.default_rng(700 + seed); T = C + POST; x = np.zeros(T)
    for t in range(1, T): x[t] = (0.9 if t < C else 0.3) * x[t - 1] + 0.3 * r.standard_normal()
    return x, x[:, None] + 0.7 * r.standard_normal((T, 2))

def run(C, seed, kw):
    x, Y = make(C, seed)
    res = LucidFilter(dynamics=[[0.9]], H=[[1.0], [1.0]], process=[[0.09]], measurement=[0.49, 0.49], faults=True, **kw).filter(Y)
    return dict(fault=np.asarray(res.fault), hazard=np.asarray(res.hazard), err=np.asarray(res.mean)[:, 0] - x,
                F=np.asarray(res.dynamics)[:, 0, 0])

print(f"{'calm':>6s} {'arm':12s} {'hazard before':>13s} {'false alarms':>12s} {'first > 1/2':>11s} {'held 50':>8s} {'RMSE +300':>9s} {'F at end':>8s}")
for C in (600, 4000):
    for nm, kw in ARMS:
        hz, fa, d1, d50, rm, Fe = [], [], [], [], [], []
        for seed in SEEDS:
            tag = "_ownceiling" if kw.get("forget", 0) is None else ""        # the drift arm, re-run after 0015
            r = cached(f"0014_{C}_{seed}_{nm.replace(' ', '')}{tag}", lambda: run(C, seed, kw))
            f = r["fault"]; post = f[C:] > 0.5
            hz.append(float(np.exp(np.mean(np.log(r["hazard"][C - 50:C])))))
            fa.append(float(np.mean(f[150:C] > 0.5)))
            d1.append(int(np.argmax(post)) if post.any() else POST)
            run50 = np.convolve(post.astype(int), np.ones(50, int), "valid") == 50
            d50.append(int(np.argmax(run50)) if run50.any() else POST)
            rm.append(float(np.sqrt(np.mean(r["err"][C:C + 300] ** 2)))); Fe.append(float(r["F"][-1]))
        print(f"{C:6d} {nm:12s} {np.mean(hz):13.2e} {np.mean(fa):12.3f} {np.median(d1):11.0f} {np.median(d50):8.0f} {np.mean(rm):9.4f} {np.mean(Fe):8.3f}"
              f"    per seed first>1/2: {d1}  held: {d50}", flush=True)
