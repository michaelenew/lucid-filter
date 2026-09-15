"""The PATIENT walk: copies of every (phi, s) member with patience L.  L = 0 is the shipped walk; L = inf never
moves its process scale; a finite L never uses the one-step score on a process axis -- it accumulates the eager
sibling's predictive-likelihood advantage over a memory of L steps and relaxes toward the sibling's process scale
at rate 1/L, weighted by the posterior that the sibling is right.  Own state per copy.  Sensor axes stay eager.
    python probe_patient.py "0,30,100,300,inf" {scalar|async|arm} [seed]"""
import math, os, sys, time, types, importlib.util
import numpy as np
sys.path.insert(0, ".")
SRC = "lucid/filter/lucid.py"
LAGS = tuple(float(x) for x in sys.argv[1].split(",")); RIG = sys.argv[2]
TWO = "two" in sys.argv[4:]
LEAK = next((float(x[5:]) for x in sys.argv[4:] if x.startswith("leak=")), 0.0)
def load(variant):
    src = open(SRC).read()
    if variant == "C":
        old = "        cells = [(ph, sv, bq, br) for ph in phis for sv in ss for (bq, br) in bases]\n"
        assert old in src
        src = src.replace(old, old + "        self.lags = (%s,)\n        cells = [(ph, sv, bq, br, L, c0) for L in self.lags for c0, (ph, sv, bq, br) in enumerate(cells)]\n        self.phi_arr = np.tile(self.phi_arr, len(self.lags)); self.s_arr = np.tile(self.s_arr, len(self.lags))\n" % ", ".join("float('inf')" if not np.isfinite(L) else repr(L) for L in LAGS))
        o1 = "                for ph, sv, bq, br in cells:\n                    e = _WalkEngine(bq, br, Hm, Fs, Bs, ph, sv, fisher_Si=Si_c, prop=pr)\n"
        assert o1 in src
        src = src.replace(o1, "                for ph, sv, bq, br, L, c0 in cells:\n                    e = _WalkEngine(bq, br, Hm, Fs, Bs, ph, sv, fisher_Si=Si_c, prop=pr)\n                    e._lag = L; e._lagc0 = c0\n")
        o2 = "            for ph, sv, bq, br in cells:   # the split rides into the augmentation with the base\n"
        assert o2 in src
        src = src.replace(o2, "            for ph, sv, bq, br, L, c0 in cells:   # the split rides into the augmentation with the base\n")
        o3 = "                e._dyn = dep.callable_for()\n                e._dep = dep\n"
        assert o3 in src
        src = src.replace(o3, o3 + "                e._lag = L; e._lagc0 = c0\n")
        # the bank: per-member patience.  L = 0 eager (the shipped walk); L = inf never moves its process
        # scale; finite L accumulates the eager sibling's predictive-likelihood advantage over a memory of L
        # steps and relaxes toward the sibling's process scale at rate 1/L, weighted by the posterior that the
        # sibling is right.  No copy but the eager one uses the one-step score on a process axis.
        o4 = "        self.mu, self._Pmu = st(\"mu\"), st(\"_Pmu\")\n"
        assert o4 in src
        src = src.replace(o4, o4 + '''        self._lagv = np.array([getattr(f, '_lag', 0.0) for f in members], float)
        self._lagc0 = np.array([getattr(f, '_lagc0', j) for j, f in enumerate(members)], int)
        eag = {int(self._lagc0[j]): j for j in range(M) if self._lagv[j] == 0.0}
        self._lagsib = np.array([eag.get(int(self._lagc0[j]), j) for j in range(M)], int)
        self._lagE = np.zeros(M)
''')
        o5 = "            Kmu = self._Pmu[:, k] / (self._Pmu[:, k] + 1.0 / info)\n            self.mu[:, k] += np.clip(Kmu * (grad / info), -budget[:, k], budget[:, k])\n"
        assert o5 in src
        src = src.replace(o5, '''            Kmu = self._Pmu[:, k] / (self._Pmu[:, k] + 1.0 / info)
            step = np.clip(Kmu * (grad / info), -budget[:, k], budget[:, k])
            if k < n:
                step = np.where(self._lagv == 0.0, step, 0.0)     # only the eager copy walks a process axis on the score
            self.mu[:, k] += step
''')
        o6 = "        self._pi[:] = pi\n        self._m[:] = m_new\n        self._P[:] = P_new\n        if held is not None:                    # hold each pair's split, keep its total\n"
        assert o6 in src
        src = src.replace(o6, '''        pat = np.flatnonzero(np.isfinite(self._lagv) & (self._lagv > 0.0))
        if pat.size:
            L = self._lagv[pat]; sib = self._lagsib[pat]
            self._lagE[pat] = (1.0 - 1.0 / L) * self._lagE[pat] + (ll[sib] - ll[pat])
            sig = 1.0 / (1.0 + np.exp(-np.clip(self._lagE[pat], -50.0, 50.0)))
            if _TWO_SIDED:   # relax toward the evidence-weighted point between the eager sibling and the base (mu = 0)
                target = sig[:, None] * self.mu[sib, :n]
            else:
                target = self.mu[pat, :n] + sig[:, None] * (self.mu[sib, :n] - self.mu[pat, :n])
            self.mu[pat, :n] += (1.0 / L)[:, None] * (target - self.mu[pat, :n])
''' + o6)
    src = src.replace("_TWO_SIDED", repr(TWO))
    if variant == "C" and LEAK > 0.0:   # a switching prior across the patience copies of each class cell: uniform leak at rate LEAK per step
        o8 = "        prior = self._logw - _logsumexp(self._logw)\n        if self._ndbase > 1:\n"
        assert o8 in src
        src = src.replace(o8, '''        prior = self._logw - _logsumexp(self._logw)
        if getattr(self, "lags", None) is not None and len(self.lags) > 1 and self._nspec == 1:
            p = np.exp(prior); nl = len(self.lags); nc0 = M // nl
            g = p.reshape(nl, nc0)
            g = (1.0 - _LEAK) * g + _LEAK * g.mean(0, keepdims=True)
            prior = np.log(np.maximum(g.reshape(-1), 1e-300))
        if self._ndbase > 1:
''')
        src = src.replace("_LEAK", repr(LEAK))
    mod = types.ModuleType("llag_" + variant); mod.__file__ = SRC; sys.modules[mod.__name__] = mod
    exec(compile(src, SRC, "exec"), mod.__dict__); return mod
A = load("A"); C = load("C")
np.seterr(all="ignore")
print("patience grid", LAGS, "two-sided" if TWO else "toward eager only", "leak", LEAK, flush=True)
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
        lagv = np.array([getattr(e, "_lag", 0.0) for e in f._members]); Ls = np.unique(lagv); wtr = []; ests = []; xis = []
        t0 = time.perf_counter()
        for t in range(m54.T):
            st = f.update(Y[t], U[t]); lw = f._logw; w = np.exp(lw - lw.max()); w /= w.sum()
            wtr.append([w[lagv == L].sum() for L in Ls]); ests.append(st.mean.copy()); xis.append(st.process_scale.copy())
        dt = time.perf_counter() - t0; est = np.array(ests); wtr = np.array(wtr)
        class _R: pass
        r = _R(); r.process_scale = np.array(xis)
        if np.all(np.isfinite(est)):
            Pl = m54.tip(est); xi = np.asarray(r.process_scale)
            print(f"ARM seed{seed} {nm} ({1e3*dt/m54.T:.0f} ms/step, {len(f._members)} members): " + "  ".join(f"{k} {m54.rms(Pl,Pt,m54.mask(sp))/m54.rms(Po,Pt,m54.mask(sp)):.2f}" for k, sp in W.items())
                  + f"   | jerk-mode scale during SENSOR {xi[350:500, 10:].mean():+.2f}, PROCESS {xi[750:900, 10:].mean():+.2f} (truth {2*math.log(m54.JERK_MULT):+.2f})", flush=True)
        else:
            print(f"ARM seed{seed} {nm}: NaN", flush=True)
        if nm == "C":
            for pnm, a, b in m54.PHASES:
                print(f"    {pnm:>8} {a:4d}-{b:4d}: bank weight per patience " + "  ".join(f"L={L:g}: {wtr[a:b, i].mean():.2f}" for i, L in enumerate(Ls)), flush=True)
