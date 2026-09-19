"""Isolate the switching RATE: the branch's derived floor copy, with the switching ladder replaced by a single
pinned rung r (so only the rate differs from the ladder run).  Also prints the ladder's posterior OVER RUNGS.
    python probe_pin.py {r|ladder} [swap]"""
import math, os, sys, time, types, importlib.util
import numpy as np
sys.path.insert(0, ".")
ARG = sys.argv[1]
TOPFN = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '0062_topfn.txt')).read()
def load():
    src = open("lucid/filter/lucid.py").read()
    if ARG.startswith("top="):
        o = "        self._att = np.asarray(_SWITCH, float)\n"; assert o in src
        src = src.replace(o, "        self._att = np.asarray(_switch_rungs_top(%s), float)\n" % ARG[4:])
        src = src.replace("def _switch_rungs(mem=None):", TOPFN)
    elif ARG != "ladder":
        o = "        self._att = np.asarray(_SWITCH, float)\n"; assert o in src
        src = src.replace(o, "        self._att = np.asarray([%s], float)\n" % ARG)
    o = "        post, patience, att_rate = self._att_marginal(post)       # over the attribution rungs\n"; assert o in src
    src = src.replace(o, o + "        self._last_patience, self._last_switch = patience, att_rate\n        self._rungpost = np.exp(self._logw - _logsumexp(self._logw)).reshape(self._Ja, -1).sum(1)\n")
    mod = types.ModuleType("lpin"); mod.__file__ = "lucid/filter/lucid.py"; sys.modules["lpin"] = mod
    exec(compile(src, "lucid/filter/lucid.py", "exec"), mod.__dict__); return mod
C = load(); np.seterr(all="ignore")
sys.path.insert(0, "research/multivariate-statfilter/scripts"); import arm5dof as AR
spec = importlib.util.spec_from_file_location("p54", "research/multivariate-statfilter/exploration/0054_physical_sensors.py")
m54 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m54)
if "swap" in sys.argv[2:]:
    m54.PHASES = [("calm", 0, 250), ("PROCESS", 250, 500), ("calm", 500, 650), ("SENSOR", 650, 900), ("calm", 900, 1050),
                  ("POTFAIL", 1050, 1300), ("calm", 1300, 1450), ("BOTH", 1450, 1700), ("calm", 1700, 1900)]
jstd, pot, acc = m54.schedule(); U, S_, Y = AR.simulate(0, jstd, pot, acc)
orc = AR.kalman(U, Y, [j ** 2 * (AR.B @ AR.B.T) for j in jstd], [np.concatenate([[pot[k, j] ** 2, acc[k, j] ** 2, acc[k, j] ** 2] for j in range(AR.NJ)]) for k in range(m54.T)])
Pt = m54.tip(S_); Po = m54.tip(orc); W = m54.windows()
f = C.LucidFilter(dynamics=AR.F, control=AR.B, H=AR.measure, process=AR.Q0, measurement=AR.R0)
ests = []; pats = []; rp = []
t0 = time.perf_counter()
for t in range(m54.T):
    st = f.update(Y[t], U[t]); ests.append(st.mean.copy()); pats.append(f._last_patience); rp.append(f._rungpost.copy())
dt = time.perf_counter() - t0; est = np.array(ests); pats = np.array(pats); rp = np.array(rp)
Pl = m54.tip(est); xi = np.asarray([0])
print(f"rate = {ARG} ({1e3*dt/m54.T:.0f} ms/step): " + "  ".join(f"{k} {m54.rms(Pl,Pt,m54.mask(sp))/m54.rms(Po,Pt,m54.mask(sp)):.2f}" for k, sp in W.items()), flush=True)
print("    patience: " + "  ".join(f"{pnm} {pats[a:b].mean():.2f}" for pnm, a, b in m54.PHASES), flush=True)
if ARG == "ladder" or ARG.startswith("top="):
    rates = np.asarray(C._SWITCH if ARG == "ladder" else C._switch_rungs_top(float(ARG[4:])))
    end = rp[-1]; i = np.argsort(end)[::-1][:5]
    print("    ladder posterior at end, top rungs: " + "  ".join(f"rho={rates[j]:.4g}: {end[j]:.2f}" for j in i), flush=True)
    for pnm, a, b in m54.PHASES:
        mw = rp[a:b].mean(0); j = np.argsort(mw)[::-1][:3]
        print(f"      {pnm:>8}: mean rate {float(mw @ rates):.4f}; top " + " ".join(f"{rates[k]:.3g}:{mw[k]:.2f}" for k in j), flush=True)
