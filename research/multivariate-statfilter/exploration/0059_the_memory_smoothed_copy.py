"""ATTRIBUTION GRID v3 -- derived pieces only.  Two copies per class cell (a budget).  The MEMORY copy holds
every axis at its FLOOR (base share below 1/(phi+1) = 0.38, i.e. SNR < 1: process noise below its reading noise,
where the per-step Fisher vanishes and the per-step score is not the statistic -- resolution-criterion 0006) at the
eager copy's scale smoothed over the bank's own memory (exponential, as the bank's forget is); on every other axis
it walks as the eager copy does.  Copies switch under `_switch_rungs` (arcsine ladder, Jeffreys prior).  Own state
per copy.    python probe_v3.py x {scalar|async|arm|dyn} [seed] [swap]"""
import math, os, sys, time, types, importlib.util, subprocess
import numpy as np
sys.path.insert(0, ".")
RIG = sys.argv[2]
NOSTAR = False
FLOOR_SHARE = 1.0 / (0.5 * (1.0 + math.sqrt(5.0)) + 1.0)     # g_q at x = 1: 1/(golden ratio + 1) = 0.382
def load(variant):
    src = subprocess.run(["git", "show", "origin/main:lucid/filter/lucid.py"], capture_output=True, text=True).stdout
    if variant == "C":
        cur = open("lucid/filter/lucid.py").read()
        # (1) the switching ladder, from the branch
        a = cur.index("# AUDIT[derived+budget] the attribution grid's switching ladder"); b = cur.index("_SWITCH = _switch_rungs()")
        b = cur.index("\n", b) + 1
        src = src.replace("_HAZARDS = _hazard_rungs()           # the default hazard box: 16 rungs, 0.42 down to 2.9e-7, complete to 0\n",
                          "_HAZARDS = _hazard_rungs()           # the default hazard box: 16 rungs, 0.42 down to 2.9e-7, complete to 0\n\n\n" + cur[a:b])
        # (2) copies as cells
        o = "        cells = [(ph, sv, bq, br) for ph in phis for sv in ss for (bq, br) in bases]\n"
        assert o in src
        src = src.replace(o, o + "        _mem = min(1.0 / (1.0 - self.forget), _LADDER_MEM) if self.forget < 1.0 else _LADDER_MEM\n        self.rates = (1.0, 1.0 / _mem)\n        cells = [(ph, sv, bq, br, rt, c0) for rt in self.rates for c0, (ph, sv, bq, br) in enumerate(cells)]\n        self.phi_arr = np.tile(self.phi_arr, len(self.rates)); self.s_arr = np.tile(self.s_arr, len(self.rates))\n")
        o = "                for ph, sv, bq, br in cells:\n                    e = _WalkEngine(bq, br, Hm, Fs, Bs, ph, sv, fisher_Si=Si_c, prop=pr)\n"
        assert o in src
        src = src.replace(o, "                for ph, sv, bq, br, rt, c0 in cells:\n                    e = _WalkEngine(bq, br, Hm, Fs, Bs, ph, sv, fisher_Si=Si_c, prop=pr)\n                    e._rate = rt; e._lagc0 = c0\n")
        o = "            for ph, sv, bq, br in cells:   # the split rides into the augmentation with the base\n"
        assert o in src
        src = src.replace(o, "            for ph, sv, bq, br, rt, c0 in cells:   # the split rides into the augmentation with the base\n")
        o = "                e._dyn = dep.callable_for()\n                e._dep = dep\n"
        assert o in src
        src = src.replace(o, o + "                e._rate = 1.0; e._lagc0 = c0\n")
        # (3) the bank: floor mask per member, memory copy holds floor axes at the eager sibling's EMA
        o = "        self.mu, self._Pmu = st(\"mu\"), st(\"_Pmu\")\n"
        assert o in src
        src = src.replace(o, o + '''        self._rate = np.array([getattr(f, '_rate', 1.0) for f in members], float)
        self._lagc0 = np.array([getattr(f, '_lagc0', j) for j, f in enumerate(members)], int)
        eag = {int(self._lagc0[j]): j for j in range(M) if self._rate[j] == 1.0}
        self._sib = np.array([eag.get(int(self._lagc0[j]), j) for j in range(M)], int)
        share = np.sqrt(np.maximum(2.0 * (st("_Ichar") - _RIDGE), 0.0))          # (M, D) base share per axis
        self._floor = (share < %r) & (self._rate[:, None] < 1.0)                   # held axes: floor axes of memory copies
        for (k, i, _h) in self._groups:                                              # a confounded pair is the split ladder's, not the copy's
            self._floor[:, k] = False; self._floor[:, self.n + i] = False
        self._mem = 1.0 / float(self._rate[self._rate < 1.0].min()) if np.any(self._rate < 1.0) else 1.0
''' % FLOOR_SHARE)
        o = "            Kmu = self._Pmu[:, k] / (self._Pmu[:, k] + 1.0 / info)\n            self.mu[:, k] += np.clip(Kmu * (grad / info), -budget[:, k], budget[:, k])\n"
        assert o in src
        src = src.replace(o, '''            Kmu = self._Pmu[:, k] / (self._Pmu[:, k] + 1.0 / info)
            step = np.clip(Kmu * (grad / info), -budget[:, k], budget[:, k])
            self.mu[:, k] += np.where(self._floor[:, k], 0.0, step)     # a held axis does not walk on the per-step score
''')
        o = "        self._pi[:] = pi\n        self._m[:] = m_new\n        self._P[:] = P_new\n        if held is not None:                    # hold each pair's split, keep its total\n"
        assert o in src
        src = src.replace(o, '''        if np.any(self._floor):                 # held axes: the eager sibling's scale as the memory sees it
            tgt = self.mu[self._sib]
            self.mu += np.where(self._floor, (1.0 / self._mem) * (tgt - self.mu), 0.0)
''' + o)
        # (4) the weight rows over the switching ladder, from the branch
        for o_, n_ in (("        self._logw = np.zeros(self._ndw * self._nc)\n", "        self._logw = np.zeros(self._Ja * self._ndw * self._nc)\n"),
                       ("            self._lam2 = np.ones(1)\n            self._Md = np.ones((1, 1, 1))\n",
                        "            self._lam2 = np.ones(1)\n            self._Md = np.ones((1, 1, 1))\n        self._att = np.asarray(_SWITCH, float)\n        self._Ja = self._att.size\n        ka = len(self.rates)\n        self._att_lam = np.clip(1.0 - self._att * ka / (ka - 1.0), 0.0, 1.0)\n"),
                       ("        prior = self._logw - _logsumexp(self._logw)\n        if self._ndbase > 1:\n            prior = self._hazard_mix(prior, a)\n",
                        "        prior = self._logw - _logsumexp(self._logw)\n        prior = self._attribution_mix(prior, a)\n        if self._ndbase > 1:\n            prior = self._hazard_mix(prior, a)\n"),
                       ("        llw = llv[self._wm]\n        if np.any(np.isfinite(yv)):", "        llw = np.tile(llv[self._wm], self._Ja)\n        if np.any(np.isfinite(yv)):"),
                       ("        post = np.exp(self._logw - _logsumexp(self._logw))\n        pm = np.bincount(self._wm, weights=post, minlength=M)      # member marginals\n",
                        "        post = np.exp(self._logw - _logsumexp(self._logw))\n        post, patience, att_rate = self._att_marginal(post)\n        self._last_patience, self._last_switch = patience, att_rate\n        pm = np.bincount(self._wm, weights=post, minlength=M)      # member marginals\n")):
            assert o_ in src, o_[:60]; src = src.replace(o_, n_)
        a = cur.index("    def _hazard_mix(self, logw, a=1.0):"); b = cur.index("    def _spec_g(self, d):")
        a0 = src.index("    def _hazard_mix(self, logw, a=1.0):"); b0 = src.index("    def _spec_g(self, d):")
        src = src[:a0] + cur[a:b] + src[b0:]
    mod = types.ModuleType("lv3_" + variant); mod.__file__ = "lucid/filter/lucid.py"; sys.modules[mod.__name__] = mod
    exec(compile(src, "lucid/filter/lucid.py", "exec"), mod.__dict__); return mod
A = load("A"); C = load("C")
np.seterr(all="ignore")
print("attribution grid v3 (floor axes at the memory) vs shipped", flush=True)
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
            print(f"ARM seed{seed} {nm} ({1e3*dt/m54.T:.0f} ms/step, {len(f._members)} members): " + "  ".join(f"{k} {m54.rms(Pl,Pt,m54.mask(sp))/m54.rms(Po,Pt,m54.mask(sp)):.2f}" for k, sp in W.items())
                  + f"   | jerk-mode scale during SENSOR {xi[[a for n_,a,b in m54.PHASES if n_=='SENSOR'][0]+100:[b for n_,a,b in m54.PHASES if n_=='SENSOR'][0], 10:].mean():+.2f}"
                  + (f" | held axes (member 0): {np.flatnonzero(f._banks[0]._floor[len(f._members)//2]).tolist()}" if nm == "C" else ""), flush=True)
            if nm == "C":
                print("    patience per phase: " + "  ".join(f"{pnm} {np.mean(pats[a:b]):.2f}" for pnm, a, b in m54.PHASES) + f" | switch at end {f._last_switch:.3g}", flush=True)
        else:
            print(f"ARM seed{seed} {nm}: NaN", flush=True)
