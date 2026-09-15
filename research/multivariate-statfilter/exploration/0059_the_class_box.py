"""THE DERIVED FORM: per-axis-type classes.  A member's process axes and sensor axes carry their own (phi, s)
class; the process-scale class box gains the memory timescale phi_mem = 1 - 1/mem (the longest persistence the
bank can resolve within its memory -- a class closer to 1 is indistinguishable from constant), and so does the
sensor-scale box.  The product is taken as a caltrop STAR (the repo's answer to product explosion): the shared
box as the centre, one arm with the process class at phi_mem, one arm with the sensor class at phi_mem.
Copies switch under `_switch_rungs`.  No rate hack.  python probe_classes.py x {scalar|async|arm|dyn} [seed] [swap] [nostar]"""
import math, os, sys, time, types, importlib.util
import numpy as np
sys.path.insert(0, ".")
SRC = "lucid/filter/lucid.py"; RIG = sys.argv[2]
NOSTAR = "nostar" in sys.argv[3:]
def load(variant):
    src = open(SRC).read()
    if variant == "A":     # the pre-port filter (git main)
        src = __import__("subprocess").run(["git", "show", "origin/main:lucid/filter/lucid.py"], capture_output=True, text=True).stdout
    else:
        # 1. per-axis-type classes on the engine
        o = "    def __init__(self, Q0, R0, H, F, B, phi, s, walk_axes=None, cap=None, group_class=None,"
        assert o in src
        src = src.replace(o, "    def __init__(self, Q0, R0, H, F, B, phi, s, walk_axes=None, cap=None, group_class=None, phi_process=None, phi_sensor=None,")
        o = "        self.phi_ax = np.full(self.D, self.phi)\n        self.s_ax = np.full(self.D, self.s)\n"
        assert o in src
        src = src.replace(o, o + "        if phi_process is not None:\n            self.phi_ax[:n] = float(phi_process)\n        if phi_sensor is not None:\n            self.phi_ax[n:] = float(phi_sensor)\n")
        # 2. the class star instead of rate copies
        o = "        self.rates = (1.0, 1.0 / _mem)\n        cells = [(ph, sv, bq, br, rt) for rt in self.rates for (ph, sv, bq, br) in cells]\n"
        assert o in src
        src = src.replace(o, "        _phimem = 1.0 - 1.0 / _mem\n        self.rates = (1.0, 1.0, 1.0)%s\n        self.star = ((None, None), (_phimem, None), (None, _phimem))%s\n        cells = [(ph, sv, bq, br, pp, ps) for (pp, ps) in self.star for (ph, sv, bq, br) in cells]\n" % (("[:1]" if NOSTAR else ""), ("[:1]" if NOSTAR else "")))
        o = "                for ph, sv, bq, br, rt in cells:\n                    e = _WalkEngine(bq, br, Hm, Fs, Bs, ph, sv, fisher_Si=Si_c, prop=pr)\n                    e._rate = rt\n"
        assert o in src
        src = src.replace(o, "                for ph, sv, bq, br, pp, ps in cells:\n                    e = _WalkEngine(bq, br, Hm, Fs, Bs, ph, sv, fisher_Si=Si_c, prop=pr, phi_process=pp, phi_sensor=ps)\n")
        o = "            for ph, sv, bq, br, rt in cells:   # the split rides into the augmentation with the base\n"
        assert o in src
        src = src.replace(o, "            for ph, sv, bq, br, pp, ps in cells:   # the split rides into the augmentation with the base\n")
        o = "                e = _WalkEngine(Qa, Ra, Ha, Fa, Ba, ph, sv, walk_axes=walk, cap=dep.cap,\n                                fisher_Si=Si_c)\n"
        assert o in src
        src = src.replace(o, "                e = _WalkEngine(Qa, Ra, Ha, Fa, Ba, ph, sv, walk_axes=walk, cap=dep.cap,\n                                fisher_Si=Si_c, phi_process=pp, phi_sensor=ps)\n")
        o = "                e._rate = 1.0\n"
        assert o in src
        src = src.replace(o, "")
        # phi_process on the AUGMENTED engine must not touch the departure coefficients' scale axes (excluded from the walk anyway)
        src = src.replace("        self.phi_arr = np.tile(self.phi_arr, len(self.rates))\n        self.s_arr = np.tile(self.s_arr, len(self.rates))\n",
                          "        self.phi_arr = np.tile(self.phi_arr, len(self.star))\n        self.s_arr = np.tile(self.s_arr, len(self.star))\n")
        src = src.replace("        ka = len(self.rates)\n", "        ka = len(self.star)\n")
        src = src.replace("        nr = len(self.rates)\n        W = np.exp(logw - float(logw.max())).reshape(self._Ja, self._ndw, nr, self._nc // nr)\n",
                          "        nr = len(self.star)\n        W = np.exp(logw - float(logw.max())).reshape(self._Ja, self._ndw, nr, self._nc // nr)\n")
        src = src.replace("        P = post.reshape(self._Ja, self._ndw, len(self.rates), -1)\n        return P.sum(0).ravel(), float(P[:, :, 1:, :].sum()), float(P.sum(axis=(1, 2, 3)) @ self._att)\n",
                          "        P = post.reshape(self._Ja, self._ndw, len(self.star), -1)\n        self._copy_w = P.sum(axis=(0, 1, 3))\n        return P.sum(0).ravel(), float(P[:, :, 1:2, :].sum()), float(P.sum(axis=(1, 2, 3)) @ self._att)\n")
    mod = types.ModuleType("lcls_" + variant); mod.__file__ = SRC; sys.modules[mod.__name__] = mod
    exec(compile(src, SRC, "exec"), mod.__dict__); return mod
A = load("A"); C = load("C")
np.seterr(all="ignore")
print("class star:", "centre only" if NOSTAR else "centre + process-memory arm + sensor-memory arm", flush=True)
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
            if hasattr(f, "_copy_w"): cw.append(f._copy_w.copy())
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
