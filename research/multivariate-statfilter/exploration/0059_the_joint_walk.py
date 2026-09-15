"""THE JOINT WALK: multivariate Fisher scoring on the scales.  The shipped walk takes a Newton step PER AXIS
(each axis explains the whole surprise on its own); this is its diagonal approximation, and the dropped cross
terms are the attribution.  With the joint scale-Fisher I_jk = 1/2 tr(S^-1 dS_j S^-1 dS_k) at the centre node
(diagonal: each axis's window-averaged Fisher, as shipped), the step is (Pmu^-1 + I)^-1 grad -- the same
estimator, with its off-diagonals restored; for one active axis it is the shipped step exactly.  No copies, no
ladder, no constant.    python probe_joint.py x {scalar|async|arm|dyn} [seed] [swap]"""
import math, os, sys, time, types, importlib.util
import numpy as np
sys.path.insert(0, ".")
import subprocess
SRC = "lucid/filter/lucid.py"; RIG = sys.argv[2]
_PRE = subprocess.run(["git", "show", "08b925b:lucid/filter/lucid.py"], capture_output=True, text=True).stdout   # the pre-grid source both variants patch
NOSTAR = False
def load(variant):
    src = _PRE
    if variant == "C":
        old = src[src.index("        for ax, k in enumerate(self._act):       # NB: not `a` -- that is the elapsed time\n"):src.index("        self._pi[:] = pi\n        self._m[:] = m_new\n")]
        new = '''        r_act = len(self._act)
        gradv = np.zeros((M, r_act)); infov = np.zeros((M, r_act)); dSc = [None] * r_act; live = np.zeros(r_act, bool)
        Sic = Si[:, 0]                                       # the centre node's S^-1, shared by every axis
        for ax, k in enumerate(self._act):       # NB: not `a` -- that is the elapsed time
            idx = self._axwin[k]
            dpk = self._dS_axis(k, obs, aQ if k < n else a, Hout)
            if not dpk.any():                    # no evidence here: drift, never freeze
                self._Pmu[:, k] = np.minimum(self._Pmu[:, k] + self._qmu[:, k] * a,
                                             self._Pmu_cap[:, k])
                continue
            Sik = Si[:, idx]
            Sie = np.einsum("bgij,bj->bgi", Sik, e)
            score = 0.5 * (np.einsum("bgi,bgij,bgj->bg", Sie, dpk, Sie)
                           - np.einsum("bgij,bgji->bg", Sik, dpk))
            SidS = np.einsum("bgij,bgjk->bgik", Sik, dpk)
            info_g = 0.5 * np.einsum("bgij,bgji->bg", SidS, SidS)
            infov[:, ax] = np.einsum("bg,bg->b", pi[:, ax], info_g) + _RIDGE
            gradv[:, ax] = np.einsum("bg,bg->b", pi[:, ax], score)
            dSc[ax] = np.einsum("bij,bjk->bik", Sic, dpk[:, self._c])
            live[ax] = True
        L = np.flatnonzero(live)
        if L.size:
            I = np.zeros((M, L.size, L.size))
            for a_i, ax in enumerate(L):
                I[:, a_i, a_i] = infov[:, ax]
                for b_i in range(a_i + 1, L.size):
                    bx = L[b_i]
                    v = 0.5 * np.einsum("bij,bji->b", dSc[ax], dSc[bx])
                    I[:, a_i, b_i] = v; I[:, b_i, a_i] = v
            ks = np.array([self._act[ax] for ax in L])
            Pm = self._Pmu[:, ks]
            A_ = I + np.einsum("br,rs->brs", 1.0 / Pm, np.eye(L.size))
            Post = np.linalg.inv(A_)
            step = np.einsum("brs,bs->br", Post, gradv[:, L])
            for a_i, ax in enumerate(L):
                k = int(ks[a_i])
                self.mu[:, k] += np.clip(step[:, a_i], -budget[:, k], budget[:, k])
                self._Pmu[:, k] = np.minimum(Post[:, a_i, a_i] + self._qmu[:, k] * a, self._Pmu_cap[:, k])
'''
        src = src.replace(old, new)
    mod = types.ModuleType("ljoint_" + variant); mod.__file__ = "lucid/filter/lucid.py"; sys.modules[mod.__name__] = mod
    exec(compile(src, "lucid/filter/lucid.py", "exec"), mod.__dict__); return mod
A = load("A"); C = load("C")
np.seterr(all="ignore")
print("joint walk vs shipped", flush=True)
if RIG == "scalar":
    N, JA, JU, NA = 900, 380, 9.0, 600
    rows = {k: [] for k in ("jump", "steady", "C")}; tA = tC = 0.0
    for seed in range(11, 23):
        rng = np.random.default_rng(seed); th = np.cumsum(rng.normal(0, math.sqrt(0.02), N)); th[JA:] += JU
        sd = np.where(np.arange(N) < NA, 1.0, 3.0); y = th + rng.normal(0, sd)
        t0 = time.perf_counter(); ma = np.asarray(A.LucidFilter().filter(y.reshape(-1, 1)).mean).reshape(-1); tA += time.perf_counter() - t0
        t0 = time.perf_counter(); mc = np.asarray(C.LucidFilter().filter(y.reshape(-1, 1)).mean).reshape(-1); tC += time.perf_counter() - t0
        for k, sl in (("jump", slice(JA, JA + 40)), ("steady", slice(80, JA)), ("C", slice(NA + 40, N))):
            rows[k].append((np.mean((ma[sl] - th[sl]) ** 2), np.mean((mc[sl] - th[sl]) ** 2)))
    print("SCALAR paired (12 seeds), RMSE A -> C, paired diff in MSE +- sem, t:")
    for k, v in rows.items():
        a = np.array([x[0] for x in v]); c = np.array([x[1] for x in v]); d = c - a; sem = d.std(ddof=1) / math.sqrt(len(d))
        print(f"  {k:7s} RMSE {math.sqrt(a.mean()):.4f} -> {math.sqrt(c.mean()):.4f}   dMSE {d.mean():+.4f} +- {sem:.4f}  t={d.mean()/sem:+.2f}", flush=True)
    print(f"  ms/step: A {1e3*tA/(12*N):.2f}  C {1e3*tC/(12*N):.2f}  (x{tC/tA:.1f})", flush=True)
elif RIG == "dyn":
    def rng(seed): return np.random.default_rng(seed)
    def ar1(T=1200, a=0.6, q=0.09, r=0.25, seed=0):
        g = rng(seed); x = np.zeros(T)
        for t in range(1, T): x[t] = a * x[t - 1] + math.sqrt(q) * g.standard_normal()
        return (x + math.sqrt(r) * g.standard_normal(T))[:, None], x
    def rmse(e, x, lo=300): return float(np.sqrt(np.mean((e[lo:] - x[lo:]) ** 2)))
    for nm, mod in (("A", A), ("C", C)):
        Y, x = ar1(T=600, a=0.3, seed=2)
        L = mod.LucidFilter(dynamics=None, process=[[0.09]], measurement=[0.25]).filter(Y); W = mod.LucidFilter(dynamics=[[1.0]], process=[[0.09]], measurement=[0.25]).filter(Y)
        print(f"{nm}: test1 learned {rmse(L.mean[:,0],x):.4f} vs walk {rmse(W.mean[:,0],x):.4f} (must be <); fault[-1] {L.fault[-1]:.3f} (>0.5); F[-1] {L.dynamics[-1,0,0]:.3f} (|.|<0.8)", flush=True)
        Y, x = ar1(T=1200, a=0.6, seed=1)
        L = mod.LucidFilter(dynamics=None, process=[[0.09]], measurement=[0.25]).filter(Y); O = mod.LucidFilter(dynamics=[[0.6]], process=[[0.09]], measurement=[0.25]).filter(Y)
        print(f"{nm}: test2 learned/oracle {rmse(L.mean[:,0],x)/rmse(O.mean[:,0],x):.4f} (gate 1.15)", flush=True)
elif RIG == "async":
    spec = importlib.util.spec_from_file_location("rig", "research/pointwise-streaming/exploration/0005_the_asynchronous_rig.py")
    rig = importlib.util.module_from_spec(spec); spec.loader.exec_module(rig)
    for nm, mod in (("A", A), ("C", C)):
        whole = []; hot = []
        for seed in (0, 10, 20, 30, 40):
            stream = rig.simulate(seed); truth = np.array([s[3] for s in stream]); times = np.array([s[1] for s in stream])
            F, Q = rig.nominal_model(); f = mod.LucidFilter(dynamics=F, H=rig.H, process=Q, measurement=rig.SIGMA ** 2, timestep=rig.NOMINAL)
            est = np.empty((len(stream), 2))
            for k, (i, t, val, _tr, _sd) in enumerate(stream): est[k] = f.observe(i, val, t=t).mean
            err = est[:, 0] - truth[:, 0]; j, t0, t1, fac = rig.FAIL
            hw = (times >= t0 + 1) & (times < t1); ww = (times >= 1) & (times < rig.DURATION)
            whole.append(np.sqrt(np.mean(err[ww] ** 2))); hot.append(np.sqrt(np.mean(err[hw] ** 2)))
        print(f"ASYNC {nm}: whole {np.mean(whole)/0.0345:.2f}x  hot {np.mean(hot)/0.0241:.2f}x  (worst hot {max(hot)/0.0241:.2f}x)", flush=True)
else:
    seed = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 0
    sys.path.insert(0, "research/multivariate-statfilter/scripts"); import arm5dof as AR
    spec = importlib.util.spec_from_file_location("p54", "research/multivariate-statfilter/exploration/0054_physical_sensors.py")
    m54 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m54)
    if "swap" in sys.argv[3:]:
        m54.PHASES = [("calm", 0, 250), ("PROCESS", 250, 500), ("calm", 500, 650), ("SENSOR", 650, 900), ("calm", 900, 1050),
                      ("POTFAIL", 1050, 1300), ("calm", 1300, 1450), ("BOTH", 1450, 1700), ("calm", 1700, 1900)]
    jstd, pot, acc = m54.schedule(); U, S_, Y = AR.simulate(seed, jstd, pot, acc)
    orc = AR.kalman(U, Y, [j ** 2 * (AR.B @ AR.B.T) for j in jstd], [np.concatenate([[pot[k, j] ** 2, acc[k, j] ** 2, acc[k, j] ** 2] for j in range(AR.NJ)]) for k in range(m54.T)])
    Pt = m54.tip(S_); Po = m54.tip(orc); W = m54.windows()
    for nm, mod in (("A", A), ("C", C)):
        f = mod.LucidFilter(dynamics=AR.F, control=AR.B, H=AR.measure, process=AR.Q0, measurement=AR.R0)
        ests = []; xis = []; cw = []
        t0 = time.perf_counter()
        for t in range(m54.T):
            st = f.update(Y[t], U[t]); ests.append(st.mean.copy()); xis.append(st.process_scale.copy())
        dt = time.perf_counter() - t0; est = np.array(ests); xi = np.array(xis)
        if np.all(np.isfinite(est)):
            Pl = m54.tip(est)
            print(f"ARM seed{seed} {nm} ({1e3*dt/m54.T:.0f} ms/step, {len(f._members)} members): " + "  ".join(f"{k} {m54.rms(Pl,Pt,m54.mask(sp))/m54.rms(Po,Pt,m54.mask(sp)):.2f}" for k, sp in W.items())
                  + f"   | jerk-mode scale during SENSOR {xi[[a for n_,a,b in m54.PHASES if n_=='SENSOR'][0]+100:[b for n_,a,b in m54.PHASES if n_=='SENSOR'][0], 10:].mean():+.2f}", flush=True)
            if cw:
                cw = np.array(cw)
                print("    copy weights per phase (centre, process-mem, sensor-mem): " + "  ".join(f"{pnm} " + "/".join(f"{v:.2f}" for v in cw[a:b].mean(0)) for pnm, a, b in m54.PHASES) + f" | switch at end {getattr(st, 'switch', float('nan')):.3g}", flush=True)
        else:
            print(f"ARM seed{seed} {nm}: NaN", flush=True)
