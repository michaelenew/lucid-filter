"""What does the escape still buy?  The SAME filter at forget = 0.999 and at the nominal forget = 1
(pure Bayes), on every rig.  Structure is forget-independent now, so this isolates the weight decay.
    python probe_forget.py {arm|scalar|async|dyn} [swap]"""
import math, os, sys, time, types, importlib.util
import numpy as np
sys.path.insert(0, ".")
RIG = sys.argv[1]
from lucid.filter.lucid import LucidFilter
np.seterr(all="ignore")
FG = tuple(float(x) for x in os.environ.get('FGS', '0.999,1.0').split(','))
if RIG == "scalar":
    N, JA, JU, NA = 900, 380, 9.0, 600
    rows = {fg: {k: [] for k in ("jump", "steady", "C")} for fg in FG}
    for seed in range(11, 23):
        rng = np.random.default_rng(seed); th = np.cumsum(rng.normal(0, math.sqrt(0.02), N)); th[JA:] += JU
        sd = np.where(np.arange(N) < NA, 1.0, 3.0); y = th + rng.normal(0, sd)
        for fg in FG:
            m = np.asarray(LucidFilter(forget=fg).filter(y.reshape(-1, 1)).mean).reshape(-1)
            for k, sl in (("jump", slice(JA, JA + 40)), ("steady", slice(80, JA)), ("C", slice(NA + 40, N))):
                rows[fg][k].append(np.mean((m[sl] - th[sl]) ** 2))
    for fg in FG:
        print(f"SCALAR forget={fg}: " + "  ".join(f"{k} RMSE {math.sqrt(np.mean(v)):.4f}" for k, v in rows[fg].items()), flush=True)
elif RIG == "async":
    spec = importlib.util.spec_from_file_location("rig", "research/pointwise-streaming/exploration/0005_the_asynchronous_rig.py")
    rig = importlib.util.module_from_spec(spec); spec.loader.exec_module(rig)
    for fg in FG:
        whole = []; hot = []
        for seed in (0, 10, 20, 30, 40):
            stream = rig.simulate(seed); truth = np.array([s[3] for s in stream]); times = np.array([s[1] for s in stream])
            F, Q = rig.nominal_model(); f = LucidFilter(dynamics=F, H=rig.H, process=Q, measurement=rig.SIGMA ** 2, timestep=rig.NOMINAL, forget=fg)
            est = np.empty((len(stream), 2))
            for k, (i, t, val, _tr, _sd) in enumerate(stream): est[k] = f.observe(i, val, t=t).mean
            err = est[:, 0] - truth[:, 0]; j, t0, t1, fac = rig.FAIL
            hw = (times >= t0 + 1) & (times < t1); ww = (times >= 1) & (times < rig.DURATION)
            whole.append(np.sqrt(np.mean(err[ww] ** 2))); hot.append(np.sqrt(np.mean(err[hw] ** 2)))
        print(f"ASYNC forget={fg}: whole {np.mean(whole)/0.0345:.2f}x  hot {np.mean(hot)/0.0241:.2f}x", flush=True)
elif RIG == "dyn":
    def ar1(T=1200, a=0.6, q=0.09, r=0.25, seed=0):
        g = np.random.default_rng(seed); x = np.zeros(T)
        for t in range(1, T): x[t] = a * x[t - 1] + math.sqrt(q) * g.standard_normal()
        return (x + math.sqrt(r) * g.standard_normal(T))[:, None], x
    def rmse(e, x, lo=300): return float(np.sqrt(np.mean((e[lo:] - x[lo:]) ** 2)))
    for fg in FG:
        Y, x = ar1(T=600, a=0.3, seed=2)
        L = LucidFilter(dynamics=None, process=[[0.09]], measurement=[0.25], forget=fg).filter(Y)
        W = LucidFilter(dynamics=[[1.0]], process=[[0.09]], measurement=[0.25], forget=fg).filter(Y)
        Y2, x2 = ar1(T=1200, a=0.6, seed=1)
        L2 = LucidFilter(dynamics=None, process=[[0.09]], measurement=[0.25], forget=fg).filter(Y2)
        O2 = LucidFilter(dynamics=[[0.6]], process=[[0.09]], measurement=[0.25], forget=fg).filter(Y2)
        print(f"DYN forget={fg}: test1 learned {rmse(L.mean[:,0],x):.4f} vs walk {rmse(W.mean[:,0],x):.4f} (must be <); "
              f"fault {L.fault[-1]:.3f} (>0.5); F {L.dynamics[-1,0,0]:.3f} (|.|<0.8); test2 ratio {rmse(L2.mean[:,0],x2)/rmse(O2.mean[:,0],x2):.4f} (<1.15)", flush=True)
else:
    sys.path.insert(0, "research/multivariate-statfilter/scripts"); import arm5dof as AR
    spec = importlib.util.spec_from_file_location("p54", "research/multivariate-statfilter/exploration/0054_physical_sensors.py")
    m54 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m54)
    if "swap" in sys.argv[2:]:
        m54.PHASES = [("calm", 0, 250), ("PROCESS", 250, 500), ("calm", 500, 650), ("SENSOR", 650, 900), ("calm", 900, 1050),
                      ("POTFAIL", 1050, 1300), ("calm", 1300, 1450), ("BOTH", 1450, 1700), ("calm", 1700, 1900)]
    jstd, pot, acc = m54.schedule(); U, S_, Y = AR.simulate(0, jstd, pot, acc)
    orc = AR.kalman(U, Y, [j ** 2 * (AR.B @ AR.B.T) for j in jstd], [np.concatenate([[pot[k, j] ** 2, acc[k, j] ** 2, acc[k, j] ** 2] for j in range(AR.NJ)]) for k in range(m54.T)])
    Pt = m54.tip(S_); Po = m54.tip(orc); W = m54.windows()
    for fg in FG:
        f = LucidFilter(dynamics=AR.F, control=AR.B, H=AR.measure, process=AR.Q0, measurement=AR.R0, forget=fg)
        r = f.filter(Y, U); est = np.asarray(r.mean)
        if not np.all(np.isfinite(est)):
            print(f"ARM forget={fg}: NaN", flush=True); continue
        Pl = m54.tip(est); xi = np.asarray(r.process_scale)
        sa = [a for n_, a, b in m54.PHASES if n_ == "SENSOR"][0]; sb = [b for n_, a, b in m54.PHASES if n_ == "SENSOR"][0]
        print(f"ARM forget={fg}: " + "  ".join(f"{k} {m54.rms(Pl,Pt,m54.mask(sp))/m54.rms(Po,Pt,m54.mask(sp)):.3f}" for k, sp in W.items())
              + f" | jerk scale in SENSOR {xi[sa+100:sb, 10:].mean():+.2f} | patience {np.nanmean(r.patience):.2f}", flush=True)
