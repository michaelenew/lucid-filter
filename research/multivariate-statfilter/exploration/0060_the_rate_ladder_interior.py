"""The rate ladder's interior: J rungs log-uniform between the step (1) and the memory's floor (1/mem), on the
arm.  A budget must be monotone: more rungs must not cost code length.    python probe_ladder.py J [seed] [swap]"""
import math, sys, time, types, importlib.util
import numpy as np
sys.path.insert(0, ".")
J = int(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 0
def load(J):
    src = open("lucid/filter/lucid.py").read()
    o = "        self.rates = (1.0, 1.0 / _mem)\n"; assert o in src
    src = src.replace(o, "        self.rates = tuple(float(r) for r in np.geomspace(1.0, 1.0 / _mem, %d))\n" % J)
    o = "        return P.sum(0).ravel(), float(P[:, :, 1:2, :].sum()), float(P.sum(axis=(1, 2, 3)) @ self._att)\n"
    if o in src:
        src = src.replace(o, "        self._copy_w = P.sum(axis=(0, 1, 3))\n        return P.sum(0).ravel(), float(P[:, :, 1:, :].sum()), float(P.sum(axis=(1, 2, 3)) @ self._att)\n")
    else:
        o = "        return P.sum(0).ravel(), float(P[:, :, 1:, :].sum()), float(P.sum(axis=(1, 2, 3)) @ self._att)\n"; assert o in src
        src = src.replace(o, "        self._copy_w = P.sum(axis=(0, 1, 3))\n" + o)
    mod = types.ModuleType("lladder_%d" % J); mod.__file__ = "lucid/filter/lucid.py"; sys.modules[mod.__name__] = mod
    exec(compile(src, "lucid/filter/lucid.py", "exec"), mod.__dict__); return mod
C = load(J); np.seterr(all="ignore")
sys.path.insert(0, "research/multivariate-statfilter/scripts"); import arm5dof as AR
spec = importlib.util.spec_from_file_location("p54", "research/multivariate-statfilter/exploration/0054_physical_sensors.py")
m54 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m54)
if "swap" in sys.argv[2:]:
    m54.PHASES = [("calm", 0, 250), ("PROCESS", 250, 500), ("calm", 500, 650), ("SENSOR", 650, 900), ("calm", 900, 1050),
                  ("POTFAIL", 1050, 1300), ("calm", 1300, 1450), ("BOTH", 1450, 1700), ("calm", 1700, 1900)]
jstd, pot, acc = m54.schedule(); U, S_, Y = AR.simulate(seed, jstd, pot, acc)
orc = AR.kalman(U, Y, [j ** 2 * (AR.B @ AR.B.T) for j in jstd], [np.concatenate([[pot[k, j] ** 2, acc[k, j] ** 2, acc[k, j] ** 2] for j in range(AR.NJ)]) for k in range(m54.T)])
Pt = m54.tip(S_); Po = m54.tip(orc); W = m54.windows()
f = C.LucidFilter(dynamics=AR.F, control=AR.B, H=AR.measure, process=AR.Q0, measurement=AR.R0)
print("J =", J, "rates", ["%.3g" % r for r in f.rates], "members", len(f._members), flush=True)
ests = []; cw = []; ll = 0.0; t0 = time.perf_counter()
for t in range(m54.T):
    st = f.update(Y[t], U[t]); ests.append(st.mean.copy()); cw.append(f._banks[0]._copy_w.copy() if hasattr(f._banks[0], "_copy_w") else f._copy_w.copy()); ll += st.loglik
dt = time.perf_counter() - t0; est = np.array(ests); cw = np.array(cw)
Pl = m54.tip(est)
print(f"J={J} ({1e3*dt/m54.T:.0f} ms/step): " + "  ".join(f"{k} {m54.rms(Pl,Pt,m54.mask(sp))/m54.rms(Po,Pt,m54.mask(sp)):.3f}" for k, sp in W.items())
      + f" | code length (total loglik) {ll:.1f} | tip RMSE all {np.sqrt(((Pl-Pt)**2).sum(1).mean()):.5f} m", flush=True)
err = np.sqrt(((Pl - Pt) ** 2).sum(1)); pass
i = int(err.argmax()); print(f"    max tip error {err[i]:.4f} m at step {i}; steps with error > 0.05 m: {int((err > 0.05).sum())} (first {int(np.flatnonzero(err > 0.05)[0]) if (err > 0.05).any() else -1})", flush=True)
for pnm, a, b in m54.PHASES:
    print(f"    {pnm:>8} {a:4d}-{b:4d}: weight per rung " + " ".join(f"{v:.2f}" for v in cw[a:b].mean(0)), flush=True)
