"""PRODUCTION CANDIDATE: two copies per class cell walking their process scales at the step's timescale (rate 1) and
at the bank's memory (rate 1/mem); the switching rate between them a fixed leak or the hazard LADDER over weight rows
sharing the filters.
    python probe_prod.py "1,mem" {scalar|async|arm} [seed] [leak=r | ladder] [swap]

Superseded docstring of the PATIENT walk: copies of every (phi, s) member with patience L.  L = 0 is the shipped walk; L = inf never
moves its process scale; a finite L never uses the one-step score on a process axis -- it accumulates the eager
sibling's predictive-likelihood advantage over a memory of L steps and relaxes toward the sibling's process scale
at rate 1/L, weighted by the posterior that the sibling is right.  Own state per copy.  Sensor axes stay eager.
    python probe_patient.py "0,30,100,300,inf" {scalar|async|arm} [seed]"""
import math, os, sys, time, types, importlib.util
import numpy as np
sys.path.insert(0, ".")
SRC = "lucid/filter/lucid.py"
RATES = tuple(sys.argv[1].split(",")); RIG = sys.argv[2]
LADDER = "ladder" in sys.argv[4:]
LEAK = next((float(x[5:]) for x in sys.argv[4:] if x.startswith("leak=")), 0.0)
def load(variant):
    src = open(SRC).read()
    if variant == "C":
        old = "        cells = [(ph, sv, bq, br) for ph in phis for sv in ss for (bq, br) in bases]\n"
        assert old in src
        src = src.replace(old, old + "        _mem = min(1.0 / (1.0 - self.forget), _LADDER_MEM) if self.forget < 1.0 else _LADDER_MEM\n        self.lags = (%s,)\n        cells = [(ph, sv, bq, br, L, c0) for L in self.lags for c0, (ph, sv, bq, br) in enumerate(cells)]\n        self.phi_arr = np.tile(self.phi_arr, len(self.lags)); self.s_arr = np.tile(self.s_arr, len(self.lags))\n" % ", ".join("1.0 / _mem" if r == "mem" else "1.0" for r in RATES))
        o1 = "                for ph, sv, bq, br in cells:\n                    e = _WalkEngine(bq, br, Hm, Fs, Bs, ph, sv, fisher_Si=Si_c, prop=pr)\n"
        assert o1 in src
        src = src.replace(o1, "                for ph, sv, bq, br, L, c0 in cells:\n                    e = _WalkEngine(bq, br, Hm, Fs, Bs, ph, sv, fisher_Si=Si_c, prop=pr)\n                    e._rate = L; e._lagc0 = c0\n")
        o2 = "            for ph, sv, bq, br in cells:   # the split rides into the augmentation with the base\n"
        assert o2 in src
        src = src.replace(o2, "            for ph, sv, bq, br, L, c0 in cells:   # the split rides into the augmentation with the base\n")
        o3 = "                e._dyn = dep.callable_for()\n                e._dep = dep\n"
        assert o3 in src
        src = src.replace(o3, o3 + "                e._rate = L; e._lagc0 = c0\n")
        # the bank: each copy walks its process scales at its own RATE -- 1 for the step's timescale (the
        # shipped walk), 1/mem for the bank's memory (mem = min(1/(1 - forget), _LADDER_MEM), the same read
        # as `_rung_odds`).  Sensor axes walk at the step's timescale on every copy.  Own state per copy.
        o4 = "        self.mu, self._Pmu = st(\"mu\"), st(\"_Pmu\")\n"
        assert o4 in src
        src = src.replace(o4, o4 + "        self._rate = np.array([getattr(f, '_rate', 1.0) for f in members], float)\n")
        o5 = "            Kmu = self._Pmu[:, k] / (self._Pmu[:, k] + 1.0 / info)\n            self.mu[:, k] += np.clip(Kmu * (grad / info), -budget[:, k], budget[:, k])\n"
        assert o5 in src
        src = src.replace(o5, '''            Kmu = self._Pmu[:, k] / (self._Pmu[:, k] + 1.0 / info)
            step = np.clip(Kmu * (grad / info), -budget[:, k], budget[:, k])
            if k < n:
                step = step * self._rate
            self.mu[:, k] += step
''')
    if variant == "C" and LEAK > 0.0 and not LADDER:   # a fixed switching rate across the copies of each class cell
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
    if variant == "C" and LADDER:   # the switching rate as a LADDER (the hazard ladder's rungs), weight rows sharing the filters
        o9 = "        self._logw = np.zeros(self._ndw * self._nc)\n"
        assert o9 in src
        src = src.replace(o9, "        self._logw = np.zeros(self._ndw * self._nc * (len(_HAZARDS) if getattr(self, 'lags', None) is not None and len(self.lags) > 1 and self._nspec == 1 else 1))\n")
        o8 = "        prior = self._logw - _logsumexp(self._logw)\n        if self._ndbase > 1:\n"
        assert o8 in src
        src = src.replace(o8, '''        prior = self._logw - _logsumexp(self._logw)
        _att = getattr(self, "lags", None) is not None and len(self.lags) > 1 and self._nspec == 1
        if _att:
            Jl = len(_HAZARDS); nl = len(self.lags); nc0 = M // nl
            W = np.exp(prior).reshape(Jl, nl, nc0)
            lam = np.clip(1.0 - np.asarray(_HAZARDS) * nl / (nl - 1.0), 0.0, 1.0) ** a      # exact chain power of the uniform-leak chain
            W = lam[:, None, None] * W + (1.0 - lam)[:, None, None] * W.mean(1, keepdims=True)
            prior = np.log(np.maximum(W.reshape(-1), 1e-300))
        if self._ndbase > 1:
''')
        o10 = "        llw = llv[self._wm]\n"
        assert o10 in src
        src = src.replace(o10, "        llw = llv[self._wm]\n        if _att:\n            llw = np.tile(llw, Jl)\n")
        o11 = "        pm = np.bincount(self._wm, weights=post, minlength=M)      # member marginals\n"
        assert o11 in src
        src = src.replace(o11, "        if _att:\n            self._att_post = post.reshape(Jl, -1).sum(1)\n            post = post.reshape(Jl, -1).sum(0)\n" + o11)
    mod = types.ModuleType("llag_" + variant); mod.__file__ = SRC; sys.modules[mod.__name__] = mod
    exec(compile(src, SRC, "exec"), mod.__dict__); return mod
A = load("A"); C = load("C")
np.seterr(all="ignore")
print("rates", RATES, "ladder" if LADDER else f"leak {LEAK}", flush=True)
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
    if "swap" in sys.argv[4:]:   # the sensor burst AFTER the process burst: must the never copy come back?
        m54.PHASES = [("calm", 0, 250), ("PROCESS", 250, 500), ("calm", 500, 650), ("SENSOR", 650, 900), ("calm", 900, 1050),
                      ("POTFAIL", 1050, 1300), ("calm", 1300, 1450), ("BOTH", 1450, 1700), ("calm", 1700, 1900)]
    jstd, pot, acc = m54.schedule(); U, S_, Y = AR.simulate(seed, jstd, pot, acc)
    orc = AR.kalman(U, Y, [j ** 2 * (AR.B @ AR.B.T) for j in jstd], [np.concatenate([[pot[k, j] ** 2, acc[k, j] ** 2, acc[k, j] ** 2] for j in range(AR.NJ)]) for k in range(m54.T)])
    Pt = m54.tip(S_); Po = m54.tip(orc); W = m54.windows()
    for nm, mod in (("A", A), ("C", C)):
        f = mod.LucidFilter(dynamics=AR.F, control=AR.B, H=AR.measure, process=AR.Q0, measurement=AR.R0)
        lagv = np.array([getattr(e, "_rate", 1.0) for e in f._members]); Ls = np.unique(lagv)[::-1]; wtr = []; ests = []; xis = []
        t0 = time.perf_counter()
        for t in range(m54.T):
            st = f.update(Y[t], U[t]); lw = f._logw; w = np.exp(lw - lw.max()); w /= w.sum(); w = w.reshape(-1, len(lagv)).sum(0)
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
                print(f"    {pnm:>8} {a:4d}-{b:4d}: bank weight per rate " + "  ".join(f"{L:.3g}: {wtr[a:b, i].mean():.2f}" for i, L in enumerate(Ls)), flush=True)
            if hasattr(f, "_att_post"):
                print("    attribution-switch ladder posterior mean rate at end: %.3g" % float(np.asarray(f._att_post) @ np.asarray(mod._HAZARDS)), flush=True)
