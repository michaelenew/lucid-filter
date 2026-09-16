"""Chain repro: ONE joint of the arm, linear -- position/velocity/acceleration with jerk noise, a pot on position
and an accelerometer on acceleration (the arm's own single-joint model, constant H).  PROCESS burst (jerk x20),
calm, POTFAIL (pot x15), calm.  Ladders J = 2 and 5; RMSE per phase vs the oracle; each rung's own position error
and likelihood.    python probe_repro2.py [nseeds]"""
import math, os, sys, types
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "research/multivariate-statfilter/scripts")
import arm5dof as AR
def load(J):
    src = open("lucid/filter/lucid.py").read()
    o = "        self.rates = (1.0, 1.0 / _mem)\n"; assert o in src
    src = src.replace(o, "        self.rates = tuple(float(r) for r in np.geomspace(1.0, 1.0 / _mem, %d))\n" % J)
    o = "        post, patience, att_rate = self._att_marginal(post)       # over the attribution rungs\n"; assert o in src
    src = src.replace(o, o + "        self._pm_full = post.copy(); self._llv = llv.copy()\n")
    mod = types.ModuleType("lr2_%d" % J); mod.__file__ = "lucid/filter/lucid.py"; sys.modules[mod.__name__] = mod
    exec(compile(src, "lucid/filter/lucid.py", "exec"), mod.__dict__); return mod
F = AR.F[:3, :3]; G = AR.G; H = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]); Q0 = AR.JERK ** 2 * np.outer(G, G) + 1e-12 * np.eye(3); R0 = np.array([AR.POT ** 2, AR.ACC ** 2])
T = 1500; PH = [("calm", 0, 300), ("PROCESS", 300, 600), ("calm", 600, 800), ("POTFAIL", 800, 1100), ("calm", 1100, 1500)]
def sched(t): return (AR.JERK * (20.0 if 300 <= t < 600 else 1.0), AR.POT * (15.0 if 800 <= t < 1100 else 1.0))
def sim(seed):
    g = np.random.default_rng(seed); x = np.zeros(3); X = np.empty((T, 3)); Y = np.empty((T, 2))
    for t in range(T):
        j, p = sched(t); x = F @ x + G * j * g.standard_normal(); X[t] = x
        Y[t] = H @ x + np.array([p, AR.ACC]) * g.standard_normal(2)
    return X, Y
def oracle(Y):
    m = np.zeros(3); P = np.eye(3); out = np.empty((T, 3))
    for t in range(T):
        j, p = sched(t); mp = F @ m; Pp = F @ P @ F.T + j ** 2 * np.outer(G, G); Rv = np.diag([p ** 2, AR.ACC ** 2])
        S = H @ Pp @ H.T + Rv; K = Pp @ H.T @ np.linalg.inv(S); m = mp + K @ (Y[t] - H @ mp); P = (np.eye(3) - K @ H) @ Pp; out[t] = m
    return out
ns = int(sys.argv[1]) if len(sys.argv) > 1 else 3
for J in (2, 5):
    mod = load(J); res = {nm: [] for nm, _, _ in PH}; orc = {nm: [] for nm, _, _ in PH}; percopy = []
    for seed in range(ns):
        X, Y = sim(seed); o = oracle(Y)
        f = mod.LucidFilter(dynamics=F, H=H, process=Q0, measurement=R0)
        M = len(f._members); nc0 = M // J; rung = np.arange(M) // nc0; b = f._banks[0]
        est = np.empty((T, 3)); rows = []
        for t in range(T):
            st = f.update(Y[t]); est[t] = st.mean; pm = f._pm_full; llv = f._llv; rec = []
            for r in range(J):
                sel = rung == r; w = pm[sel]; w = w / max(w.sum(), 1e-300)
                rec += [float(pm[sel].sum()), float(np.abs(b._m[sel, 0] - X[t, 0]) @ w), float(np.log(np.exp(llv[sel] - llv[sel].max()) @ w) + llv[sel].max()), float(w @ b.mu[sel, 2])]
            rows.append(rec)
        percopy.append(np.array(rows))
        for nm, a, bb in PH:
            res[nm].append(np.sqrt(np.mean((est[a + 40:bb, 0] - X[a + 40:bb, 0]) ** 2))); orc[nm].append(np.sqrt(np.mean((o[a + 40:bb, 0] - X[a + 40:bb, 0]) ** 2)))
    print(f"J={J} members {M} (position RMSE / oracle, {ns} seeds; held axes of a memory copy: {np.flatnonzero(b._held[M-1]).tolist()}):  " + "  ".join(f"{nm} {np.mean(res[nm]) / np.mean(orc[nm]):.3f}" for nm, _, _ in PH), flush=True)
    pc = np.mean(percopy, 0)
    for nm, a, bb in PH:
        seg = pc[a + 40:bb]
        print(f"    {nm:>8}: " + " | ".join(f"r{r} w {seg[:, 4*r].mean():.2f} err {seg[:, 4*r+1].mean():.4f} ll {seg[:, 4*r+2].mean():+.2f} jerk-scale {seg[:, 4*r+3].mean():+.2f}" for r in range(J)), flush=True)
