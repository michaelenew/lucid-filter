"""0022 -- the dynamics departure as a class with a zero corner (`continuum_fault.py`).

0014's rig: AR(1) state, two sensors, dynamics 0.9 -> 0.3 after a calm of C steps.
Arms: default; the filter as it stands (forget=None); the continuum departure (no switch, no
restart, a 5 x 5 box of (rho, s_g) walkers, the zero corner as the nominal).

The continuum has no event to flag, so every arm is scored on what its dynamics estimate
does: how long until F-hat crosses the midpoint 0.6; how much F-hat wanders in the calm; the
state error in the calm and after; where F-hat ends.  The `fault` read-out is reported too.

Every (calm, seed) run is checkpointed to `_cache/`.
Run from the repository root:  python research/forget-elimination/exploration/0022_the_departure_as_a_class.py
"""
import math, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from rig import cached, CACHE
import continuum_fault as CF
POST = 900; SEEDS = (0, 1, 2, 3)
def load(name): return pickle.load(open(os.path.join(CACHE, name + ".pkl"), "rb"))
def mk(C, seed):
    r = np.random.default_rng(700 + seed); T = C + POST; x = np.zeros(T)
    for t in range(1, T): x[t] = (0.9 if t < C else 0.3) * x[t - 1] + 0.3 * r.standard_normal()
    return x, x[:, None] + 0.7 * r.standard_normal((T, 2))
def run(C, seed):
    x, Y = mk(C, seed); mod = CF.load()
    f = mod.LucidFilter(dynamics=[[0.9]], H=[[1.0], [1.0]], process=[[0.09]], measurement=[0.49, 0.49], faults=True, forget=None)
    T = len(Y); fault = np.empty(T); err = np.empty(T); F = np.empty(T); wz = np.empty(T); frac = np.empty(T)
    nr, ns = len(CF.DEP_RHOS), len(CF.DEP_SS); cellw = np.zeros((T, nr * ns))
    for t in range(T):
        st = f.update(Y[t]); fault[t] = st.fault; err[t] = st.mean[0] - x[t]; F[t] = st.dynamics[0, 0]
        d = f._drift; lq = d.lead + d.node_prior; lq = lq - mod._logsumexp(lq)
        post = sum(math.exp(lq[i]) * np.exp(d.lw[i] - mod._logsumexp(d.lw[i])) for i in range(len(lq))).reshape(d.J, 2, -1)
        wz[t] = post[:, 0].sum(); cellw[t] = post[:, 1].sum(axis=1); frac[t] = math.exp(d.v)
    return dict(fault=fault, err=err, F=F, wz=wz, cellw_calm=cellw[150:C].mean(0), cellw_end=cellw[-150:].mean(0), frac=frac[::20])

print(f"{'calm':>5s} {'arm':20s} {'steps to F-hat < 0.6':>20s} {'F-hat wander in calm':>21s} {'RMSE calm':>10s} {'RMSE +300':>10s} {'F-hat at end (0.3)':>19s} {'fault read-out: calm mean / >1/2':>33s}")
keep = {}
for C in (600, 4000):
    for nm in ("default", "as it stands", "continuum"):
        cross, wander, rc, rp, fe, fm, fh, per = [], [], [], [], [], [], [], []
        for seed in SEEDS:
            r = load(f"0014_{C}_{seed}_default") if nm == "default" else load(f"0014_{C}_{seed}_forget=None_ownceiling") if nm == "as it stands" \
                else cached(f"0022_{C}_{seed}", lambda: run(C, seed))
            F = np.asarray(r["F"]); below = F[C:] < 0.6
            run20 = np.convolve(below.astype(int), np.ones(20, int), "valid") == 20        # crossed and stayed 20 steps
            cross.append(int(np.argmax(run20)) if run20.any() else POST); per.append(cross[-1])
            wander.append(float(np.sqrt(np.mean((F[150:C] - 0.9) ** 2)))); rc.append(float(np.sqrt(np.mean(r["err"][150:C] ** 2))))
            rp.append(float(np.sqrt(np.mean(r["err"][C:C + 300] ** 2)))); fe.append(float(np.mean(F[-100:])))
            fm.append(float(np.mean(r["fault"][150:C]))); fh.append(float(np.mean(r["fault"][150:C] > 0.5)))
            if nm == "continuum": keep[(C, seed)] = r
        print(f"{C:5d} {nm:20s} {np.median(cross):20.0f} {np.mean(wander):21.4f} {np.mean(rc):10.4f} {np.mean(rp):10.4f} {np.mean(fe):19.3f} {np.mean(fm):22.3f} / {np.mean(fh):.3f}    per seed {per}", flush=True)

nr, ns = len(CF.DEP_RHOS), len(CF.DEP_SS)
for C in (600, 4000):
    wc = np.mean([keep[(C, s)]["cellw_calm"] for s in SEEDS], axis=0).reshape(ns, nr); we = np.mean([keep[(C, s)]["cellw_end"] for s in SEEDS], axis=0).reshape(ns, nr)
    z = np.mean([keep[(C, s)]["wz"][150:C].mean() for s in SEEDS]); ze = np.mean([keep[(C, s)]["wz"][-150:].mean() for s in SEEDS])
    fr = np.exp(np.mean([np.log(keep[(C, s)]["frac"][C // 20 - 1]) for s in SEEDS]))
    print(f"\ncontinuum, calm {C}: weight on the zero corner {z:.3f} in calm, {ze:.3f} at the end;  star's fraction at the end of the calm {fr:.3f}")
    print("   weight by spread s_g (rows) x rate rho (columns " + ", ".join(f"{r:g}" for r in CF.DEP_RHOS) + ")   calm  |  after the fault")
    for k in range(ns):
        print(f"   s_g = {CF.DEP_SS[k]:<5g} " + " ".join(f"{v:.3f}" for v in wc[k]) + "   |   " + " ".join(f"{v:.3f}" for v in we[k]))
