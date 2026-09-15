"""A grid in the LAG of process-scale attribution: copies of every (phi, s) member whose process-axis walk
applies its step L steps late (L = 0 eager, L = inf never), combined by the bank like every other grid.
Sensor axes stay eager.  python probe_lag.py "0,10,30,100,inf" {scalar|async|arm} [seed]"""
import math, os, sys, time, types, importlib.util
import numpy as np
sys.path.insert(0, ".")
SRC = "lucid/filter/lucid.py"
LAGS = tuple(float(x) for x in sys.argv[1].split(",")); RIG = sys.argv[2]
SHARE = "share" in sys.argv[4:]
def load(variant):
    src = open(SRC).read()
    if variant == "C":
        old = "        cells = [(ph, sv, bq, br) for ph in phis for sv in ss for (bq, br) in bases]\n"
        assert old in src
        src = src.replace(old, old + "        self.lags = (%s,)\n        cells = [(ph, sv, bq, br, L) for L in self.lags for (ph, sv, bq, br) in cells]\n        self.phi_arr = np.tile(self.phi_arr, len(self.lags)); self.s_arr = np.tile(self.s_arr, len(self.lags))\n" % ", ".join("float('inf')" if not np.isfinite(L) else repr(L) for L in LAGS))
        o1 = "                for ph, sv, bq, br in cells:\n                    e = _WalkEngine(bq, br, Hm, Fs, Bs, ph, sv, fisher_Si=Si_c, prop=pr)\n"
        assert o1 in src
        src = src.replace(o1, "                for ph, sv, bq, br, L in cells:\n                    e = _WalkEngine(bq, br, Hm, Fs, Bs, ph, sv, fisher_Si=Si_c, prop=pr)\n                    e._lag = L\n")
        o2 = "            for ph, sv, bq, br in cells:   # the split rides into the augmentation with the base\n"
        assert o2 in src
        src = src.replace(o2, "            for ph, sv, bq, br, L in cells:   # the split rides into the augmentation with the base\n")
        o3 = "                e._dyn = dep.callable_for()\n                e._dep = dep\n"
        assert o3 in src
        src = src.replace(o3, o3 + "                e._lag = L\n")
        # the bank: per-member lag, a ring buffer of proposed process-axis steps
        o4 = "        self.mu, self._Pmu = st(\"mu\"), st(\"_Pmu\")\n"
        assert o4 in src
        src = src.replace(o4, o4 + "        self._lagv = np.array([getattr(f, '_lag', 0.0) for f in members], float)\n        fin = self._lagv[np.isfinite(self._lagv)]\n        self._Lmax = int(fin.max()) if fin.size else 0\n        self._lagbuf = np.zeros((self._Lmax + 1, M, self.n))\n        self._lagt = 0\n")
        o5 = "            Kmu = self._Pmu[:, k] / (self._Pmu[:, k] + 1.0 / info)\n            self.mu[:, k] += np.clip(Kmu * (grad / info), -budget[:, k], budget[:, k])\n"
        assert o5 in src
        src = src.replace(o5, '''            Kmu = self._Pmu[:, k] / (self._Pmu[:, k] + 1.0 / info)
            step = np.clip(Kmu * (grad / info), -budget[:, k], budget[:, k])
            if k < n:                                   # a process axis: apply each member's step at its own lag
                W = self._Lmax + 1; t_ = self._lagt
                self._lagbuf[t_ % W, :, k] = step
                applied = np.zeros(M)
                for L in np.unique(self._lagv):
                    if not np.isfinite(L):
                        continue
                    Li = int(L); sel = self._lagv == L
                    if t_ >= Li:
                        applied[sel] = self._lagbuf[(t_ - Li) % W, sel, k]
                self.mu[:, k] += applied
            else:
                self.mu[:, k] += step
''')
        o6 = "        self._pi[:] = pi\n        self._m[:] = m_new\n        self._P[:] = P_new\n        if held is not None:                    # hold each pair's split, keep its total\n"
        assert o6 in src
        src = src.replace(o6, "        self._lagt += 1\n" + o6)
        if SHARE:   # the lag copies are hypotheses about attribution, not about the state: collapse the state across them
            o7 = "        pm = np.bincount(self._wm, weights=post, minlength=M)      # member marginals\n"
            assert o7 in src
            src = src.replace(o7, o7 + '''        if getattr(self, "lags", None) is not None and len(self.lags) > 1 and self._nspec == 1:
            nl = len(self.lags); nc0 = M // nl
            for c0 in range(nc0):
                idx = [c0 + l * nc0 for l in range(nl)]
                w = pm[idx]; sw = float(w.sum())
                if sw <= 0.0:
                    continue
                w = w / sw
                mc = w @ mn[idx]
                d = mn[idx] - mc
                Pc = np.einsum("b,bij->ij", w, vr[idx]) + np.einsum("b,bi,bj->ij", w, d, d)
                for j in idx:
                    self._members[j]._m[:] = mc
                    self._members[j]._P[:] = Pc
                    mn[j] = mc; vr[j] = Pc
''')
    mod = types.ModuleType("llag_" + variant); mod.__file__ = SRC; sys.modules[mod.__name__] = mod
    exec(compile(src, SRC, "exec"), mod.__dict__); return mod
A = load("A"); C = load("C")
np.seterr(all="ignore")
print("lags", LAGS, "share" if SHARE else "separate states", flush=True)
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
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    sys.path.insert(0, "research/multivariate-statfilter/scripts"); import arm5dof as AR
    spec = importlib.util.spec_from_file_location("p54", "research/multivariate-statfilter/exploration/0054_physical_sensors.py")
    m54 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m54)
    jstd, pot, acc = m54.schedule(); U, S_, Y = AR.simulate(seed, jstd, pot, acc)
    orc = AR.kalman(U, Y, [j ** 2 * (AR.B @ AR.B.T) for j in jstd], [np.concatenate([[pot[k, j] ** 2, acc[k, j] ** 2, acc[k, j] ** 2] for j in range(AR.NJ)]) for k in range(m54.T)])
    Pt = m54.tip(S_); Po = m54.tip(orc); W = m54.windows()
    for nm, mod in (("A", A), ("C", C)):
        f = mod.LucidFilter(dynamics=AR.F, control=AR.B, H=AR.measure, process=AR.Q0, measurement=AR.R0)
        t0 = time.perf_counter(); r = f.filter(Y, U); dt = time.perf_counter() - t0; est = np.asarray(r.mean)
        if np.all(np.isfinite(est)):
            Pl = m54.tip(est); xi = np.asarray(r.process_scale)
            print(f"ARM seed{seed} {nm} ({1e3*dt/m54.T:.0f} ms/step, {len(f._members)} members): " + "  ".join(f"{k} {m54.rms(Pl,Pt,m54.mask(sp))/m54.rms(Po,Pt,m54.mask(sp)):.2f}" for k, sp in W.items())
                  + f"   | jerk-mode scale during SENSOR {xi[350:500, 10:].mean():+.2f}, PROCESS {xi[750:900, 10:].mean():+.2f} (truth {2*math.log(m54.JERK_MULT):+.2f})", flush=True)
        else:
            print(f"ARM seed{seed} {nm}: NaN", flush=True)
