"""Per-copy diagnostics through POTFAIL on the arm: each rung's own tip error (from its members' states, weighted
within the rung), its per-step predictive log-likelihood, and its scales on the jerk modes, the regularisation
modes and the failed pot.    python probe_percopy.py J"""
import math, os, sys, time, types, importlib.util
import numpy as np
sys.path.insert(0, ".")
J = int(sys.argv[1])
src = open("lucid/filter/lucid.py").read()
o = "        self.rates = (1.0, 1.0 / _mem)\n"; assert o in src
src = src.replace(o, "        self.rates = tuple(float(r) for r in np.geomspace(1.0, 1.0 / _mem, %d))\n" % J)
o = "        post, patience, att_rate = self._att_marginal(post)       # over the attribution rungs\n"; assert o in src
src = src.replace(o, o + "        self._pm_full = post.copy(); self._llv = llv.copy()\n")
mod = types.ModuleType("lpc"); mod.__file__ = "lucid/filter/lucid.py"; sys.modules["lpc"] = mod
exec(compile(src, "lucid/filter/lucid.py", "exec"), mod.__dict__); C = mod
np.seterr(all="ignore")
sys.path.insert(0, "research/multivariate-statfilter/scripts"); import arm5dof as AR
spec = importlib.util.spec_from_file_location("p54", "research/multivariate-statfilter/exploration/0054_physical_sensors.py")
m54 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m54)
jstd, pot, acc = m54.schedule(); U, S_, Y = AR.simulate(0, jstd, pot, acc)
Pt = m54.tip(S_)
f = C.LucidFilter(dynamics=AR.F, control=AR.B, H=AR.measure, process=AR.Q0, measurement=AR.R0)
M = len(f._members); nc0 = M // J; rung = np.arange(M) // nc0; b = f._banks[0]; n = 15
POT2 = n + 3 * m54.FAIL_J          # the failed pot's sensor axis (joint 2's pot is sensor index 3*2)
rows = []
for t in range(m54.T):
    st = f.update(Y[t], U[t])
    pm = f._pm_full; llv = f._llv
    tips = np.array([AR.joints3d(b._m[j].reshape(AR.NJ, AR.ORDER)[:, 0])[-1] for j in range(M)])
    rec = [t, float(np.sqrt(((m54.tip(st.mean[None])[0] - Pt[t]) ** 2).sum()))]
    for r in range(J):
        sel = rung == r; w = pm[sel]; w = w / max(w.sum(), 1e-300)
        err_r = float(np.sqrt(((tips[sel] - Pt[t]) ** 2).sum(1)) @ w)          # weighted within the rung
        ll_r = float(np.log(np.exp(llv[sel] - llv[sel].max()) @ w) + llv[sel].max())
        mu = b.mu[sel]; wm = w
        rec += [float(pm[sel].sum()), err_r, ll_r, float(wm @ mu[:, 10:n].mean(1)), float(wm @ mu[:, :10].mean(1)), float(wm @ mu[:, POT2])]
    rows.append(rec)
R = np.array(rows)
pass
rates = f.rates
print("J=%d rates %s" % (J, ["%.3g" % r for r in rates]))
def show(a, bb, label):
    seg = R[a:bb]
    print(f"  {label:>14} {a}-{bb}: mixture tip RMSE {np.sqrt((seg[:,1]**2).mean()):.4f}")
    for r in range(J):
        c = 2 + 6 * r
        print(f"      rung {r} (rate {rates[r]:.3g}): weight {seg[:,c].mean():.2f}  own tip RMSE {np.sqrt((seg[:,c+1]**2).mean()):.4f}  mean ll {seg[:,c+2].mean():+.2f}  jerk-mode scale {seg[:,c+3].mean():+.2f}  reg-mode scale {seg[:,c+4].mean():+.2f}  failed-pot scale {seg[:,c+5].mean():+.2f}")
for a, bb, lab in ((900, 1050, "calm before"), (1050, 1090, "POTFAIL onset"), (1090, 1150, "POTFAIL early"), (1150, 1300, "POTFAIL late"), (1300, 1450, "calm after")):
    show(a, bb, lab)
