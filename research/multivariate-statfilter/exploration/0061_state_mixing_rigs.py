"""v5 = the branch's floor copy + IMM state mixing at the switching rate (v5_patch), J rungs.
    python probe_v5.py J {scalar|async|arm|dyn} [seed] [swap]"""
import math, os, sys, time, types, importlib.util, subprocess
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib; v5_patch = importlib.import_module('0061_state_mixing_patch')
JR = int(sys.argv[1]); RIG = sys.argv[2]
def load(variant):
    if variant == "A":
        src = subprocess.run(["git", "show", "origin/main:lucid/filter/lucid.py"], capture_output=True, text=True).stdout
    else:
        src = v5_patch.patched_source(JR if JR != 2 else None)
    mod = types.ModuleType("lv5_" + variant); mod.__file__ = "lucid/filter/lucid.py"; sys.modules[mod.__name__] = mod
    exec(compile(src, "lucid/filter/lucid.py", "exec"), mod.__dict__); return mod
A = load("A"); C = load("C")
np.seterr(all="ignore")
print("v5: floor copy + IMM state mixing at the switching rate, J =", JR, "vs shipped (origin/main)", flush=True)
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
        ests = []; xis = []; pats = []
        t0 = time.perf_counter()
        for t in range(m54.T):
            st = f.update(Y[t], U[t]); ests.append(st.mean.copy()); xis.append(st.process_scale.copy()); pats.append(getattr(f, '_last_patience', np.nan))
        dt = time.perf_counter() - t0; est = np.array(ests); xi = np.array(xis)
        if np.all(np.isfinite(est)):
            Pl = m54.tip(est)
            err_all = np.sqrt(((Pl - Pt) ** 2).sum(1)); print(f"    all-steps tip RMSE {np.sqrt((err_all**2).mean()):.5f} m, worst step {err_all.max():.3f} m at {int(err_all.argmax())}", flush=True)
            print(f"ARM seed{seed} {nm} ({1e3*dt/m54.T:.0f} ms/step, {len(f._members)} members): " + "  ".join(f"{k} {m54.rms(Pl,Pt,m54.mask(sp))/m54.rms(Po,Pt,m54.mask(sp)):.2f}" for k, sp in W.items())
                  + f"   | jerk-mode scale during SENSOR {xi[[a for n_,a,b in m54.PHASES if n_=='SENSOR'][0]+100:[b for n_,a,b in m54.PHASES if n_=='SENSOR'][0], 10:].mean():+.2f}"
                  + (f" | held axes (member 0): {np.flatnonzero(f._banks[0]._held[len(f._members)//2]).tolist()}" if nm == "C" else ""), flush=True)
            if nm == "C":
                print("    patience per phase: " + "  ".join(f"{pnm} {np.mean(pats[a:b]):.2f}" for pnm, a, b in m54.PHASES) + f" | switch at end {f._last_switch:.3g}", flush=True)
        else:
            print(f"ARM seed{seed} {nm}: NaN", flush=True)
