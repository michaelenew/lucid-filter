"""Per-rung trace of the memory ladder on the arm: each rung's OWN tip error, its weight, its per-step
log-score against the pure-Bayes rung (whole window and its first 25 steps), its posterior-mean class cell and
scales.   python 0064_memory_trace.py [swap]   (env MEMS=10,32,100,316,1000,inf | arc | tau; CLASSONLY=1)"""
import math, os, sys, types, importlib.util
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util as _iu; _sp = _iu.spec_from_file_location("memladder", "research/multivariate-statfilter/exploration/0063_memory_ladder_patch.py"); memladder = _iu.module_from_spec(_sp); _sp.loader.exec_module(memladder)
MEMS = os.environ.get("MEMS", "10,32,100,316,1000,inf")
src = memladder.patched_source(MEMS, classonly=bool(int(os.environ.get("CLASSONLY", "0"))))
src = src.replace('        post = np.einsum("j,jr->r", mw, np.stack([np.exp(r - _logsumexp(r)) for r in self._logw]))',
                  '        self._postj = np.stack([np.exp(r - _logsumexp(r)) for r in self._logw])\n'
                  '        post = np.einsum("j,jr->r", mw, self._postj)')
src = src.replace("            rung_ll = np.array([_logsumexp(prior[j] + llw) for j in range(self._Jm)])\n",
                  "            rung_ll = np.array([_logsumexp(prior[j] + llw) for j in range(self._Jm)])\n            self._rung_ll = rung_ll\n", 1)
src = src.replace("        mean = pm @ mn\n", "        mean = pm @ mn\n        self._mn_last = mn; self._psc = psc; self._msc = msc\n", 1)
mod = types.ModuleType("lmemt2"); mod.__file__ = "lucid/filter/lucid.py"; sys.modules["lmemt2"] = mod
exec(compile(src, "lucid/filter/lucid.py", "exec"), mod.__dict__)
np.seterr(all="ignore")
sys.path.insert(0, "research/multivariate-statfilter/scripts"); import arm5dof as AR
spec = importlib.util.spec_from_file_location("p54", "research/multivariate-statfilter/exploration/0054_physical_sensors.py")
m54 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m54)
if "swap" in sys.argv[1:]:
    m54.PHASES = [("calm", 0, 250), ("PROCESS", 250, 500), ("calm", 500, 650), ("SENSOR", 650, 900), ("calm", 900, 1050),
                  ("POTFAIL", 1050, 1300), ("calm", 1300, 1450), ("BOTH", 1450, 1700), ("calm", 1700, 1900)]
jstd, pot, acc = m54.schedule(); U, S_, Y = AR.simulate(0, jstd, pot, acc)
orc = AR.kalman(U, Y, [j ** 2 * (AR.B @ AR.B.T) for j in jstd], [np.concatenate([[pot[k, j] ** 2, acc[k, j] ** 2, acc[k, j] ** 2] for j in range(AR.NJ)]) for k in range(m54.T)])
Pt = m54.tip(S_); Po = m54.tip(orc)
f = mod.LucidFilter(dynamics=AR.F, control=AR.B, H=AR.measure, process=AR.Q0, measurement=AR.R0)
T = m54.T; Jm = f._Jm; nr = len(f.rates); ncell = f._nc // nr
M = len(f._members)
phi_c = f.phi_arr[:ncell]; s_c = f.s_arr[:ncell]
nsplit = len(f.split_arr); split_c = np.tile(np.arange(nsplit), ncell // nsplit)   # cells: for ph for s for base
estj = np.empty((Jm, T, AR.N)); mw = np.empty((T, Jm)); rll = np.empty((T, Jm)); patj = np.empty((T, Jm))
phij = np.empty((T, Jm)); lsj = np.empty((T, Jm)); spj = np.empty((T, Jm)); psj = np.empty((T, Jm, f._psc_dim if hasattr(f, "_psc_dim") else 1)); 
ps_list = []; ms_list = []
for t in range(T):
    f.update(Y[t], u=U[t])
    mw[t] = f._mw; rll[t] = f._rung_ll
    ps_t = []; ms_t = []
    for j in range(Jm):
        pj, pat, _ = f._att_marginal(f._postj[j]); patj[t, j] = pat
        pmj = np.bincount(f._wm, weights=pj, minlength=M)
        estj[j, t] = pmj @ f._mn_last
        ps_t.append(pmj @ f._psc); ms_t.append(pmj @ f._msc)
        cellw = pj.reshape(f._ndw, nr, ncell).sum(axis=(0, 1))
        phij[t, j] = cellw @ phi_c; lsj[t, j] = cellw @ np.log(s_c); spj[t, j] = cellw @ split_c
    ps_list.append(ps_t); ms_list.append(ms_t)
PS = np.array(ps_list); MS = np.array(ms_list)   # (T, Jm, dim)
Pj = [m54.tip(e) for e in estj]
names = MEMS.split(",")
print("rungs", MEMS, "| schedule", "swap" if "swap" in sys.argv[1:] else "normal", "| classonly", os.environ.get("CLASSONLY", "0"), flush=True)
print("cells", ncell, "phi", sorted(set(np.round(phi_c, 3))), "s", sorted(set(np.round(s_c, 4))), "splits", nsplit, "| members", M, "| ps dim", PS.shape[2], "ms dim", MS.shape[2])
for nm, a, b in m54.PHASES:
    sl = m54.mask([(a + m54.SKIP, b)]); o = m54.rms(Po, Pt, sl)
    d = rll[a:b] - rll[a:b, -1:]
    print(f"{nm:>8} {a:4d}-{b:4d}: own-rung x-oracle " + " ".join(f"{m}:{m54.rms(P,Pt,sl)/o:.2f}" for m, P in zip(names, Pj))
          + " | rung w " + " ".join(f"{w:.2f}" for w in mw[a:b].mean(0))
          + " | LL/step vs inf " + " ".join(f"{w:+.3f}" for w in d.mean(0))
          + " | LL first 25 steps " + " ".join(f"{w:+.1f}" for w in d[:25].sum(0)), flush=True)
    print(f"{'':>19} phi " + " ".join(f"{w:.2f}" for w in phij[a:b].mean(0)) + " | log s " + " ".join(f"{w:+.2f}" for w in lsj[a:b].mean(0))
          + " | split " + " ".join(f"{w:.1f}" for w in spj[a:b].mean(0)) + " | patience " + " ".join(f"{w:.2f}" for w in patj[a:b].mean(0)), flush=True)
    print(f"{'':>19} proc-scale theta " + " ".join(f"{w:+.2f}" for w in PS[a:b, :, 0::3].mean(axis=(0, 2))) + " | omega " + " ".join(f"{w:+.2f}" for w in PS[a:b, :, 1::3].mean(axis=(0, 2))) + " | alpha " + " ".join(f"{w:+.2f}" for w in PS[a:b, :, 2::3].mean(axis=(0, 2)))
          + " | sens-scale pot " + " ".join(f"{w:+.2f}" for w in MS[a:b, :, 0::3].mean(axis=(0, 2)))
          + " | sens-scale acc " + " ".join(f"{w:+.2f}" for w in MS[a:b, :, 1::3].mean(axis=(0, 2))), flush=True)
