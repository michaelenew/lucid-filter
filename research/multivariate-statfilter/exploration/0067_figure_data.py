"""Per-step data for the figure: arm seed 1, t < 340, mixture tip error, oracle tip error, and every member's
joint-1 position state, for a given filter source.   python fig_data.py <src.py> <tag>"""
import sys, types, numpy as np, importlib.util
sys.path.insert(0, "."); sys.path.insert(0, "research/multivariate-statfilter/scripts")
S="/tmp/claude-0/-home-user-lucid-filter/56bbf4e5-3fe2-5e9c-93f7-d50a0493cfbc/scratchpad/"
src = open(sys.argv[1]).read()
o = "        pm = np.bincount(self._wm, weights=post, minlength=M)      # member marginals\n"; assert o in src
src = src.replace(o, "        self._mn_last = mn.copy(); self._post_last = post.copy()\n" + o, 1)
mod = types.ModuleType("lfig"); mod.__file__ = "lucid/filter/lucid.py"; sys.modules["lfig"] = mod
exec(compile(src, "lucid/filter/lucid.py", "exec"), mod.__dict__)
np.seterr(all="ignore")
import arm5dof as AR
spec = importlib.util.spec_from_file_location("p54", "research/multivariate-statfilter/exploration/0054_physical_sensors.py")
m54 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m54)
jstd, pot, acc = m54.schedule(); U, S_, Y = AR.simulate(1, jstd, pot, acc)
orc = AR.kalman(U, Y, [j ** 2 * (AR.B @ AR.B.T) for j in jstd], [np.concatenate([[pot[k, j] ** 2, acc[k, j] ** 2, acc[k, j] ** 2] for j in range(AR.NJ)]) for k in range(m54.T)])
Pt = m54.tip(S_); Po = m54.tip(orc)
f = mod.LucidFilter(dynamics=AR.F, control=AR.B, H=AR.measure, process=AR.Q0, measurement=AR.R0)
T = 340; err = np.zeros(T); oerr = np.zeros(T); th1 = []; wts = []
for t in range(T):
    st = f.update(Y[t], U[t])
    err[t] = np.linalg.norm(m54.tip(st.mean[None])[0] - Pt[t]); oerr[t] = np.linalg.norm(Po[t] - Pt[t])
    th1.append(f._mn_last[:, 0].copy()); wts.append(f._post_last.copy())
np.savez(S + f"fig_{sys.argv[2]}.npz", err=err, oerr=oerr, th1=np.array(th1), wts=np.array(wts), truth1=S_[:T, 0])
print("saved", sys.argv[2])
