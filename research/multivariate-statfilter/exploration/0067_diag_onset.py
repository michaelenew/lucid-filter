"""The arm's first onset (seed 1, t = 250) step by step, on any filter source (env DSRC): every member's position
state and the weights across the spike step -- which member the bank hands the weight to, and what its state
did inside that one update.   DSRC=<lucid.py source> python 0067_diag_onset.py"""
import sys, types, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "research/multivariate-statfilter/scripts")
S="/tmp/claude-0/-home-user-lucid-filter/56bbf4e5-3fe2-5e9c-93f7-d50a0493cfbc/scratchpad/"
import os
src = open(S + os.environ.get("DSRC", "lucid_grid.py")).read()
o = "        pm = np.bincount(self._wm, weights=post, minlength=M)      # member marginals\n"; assert o in src
src = src.replace(o, "        self._llv_last = llv.copy(); self._mn_last = mn.copy(); self._inn_last = inn.copy(); self._post_last = post.copy(); self._msc_last = msc.copy(); self._psc_last = psc.copy()\n" + o, 1)
mod = types.ModuleType("lgo"); mod.__file__ = "lucid/filter/lucid.py"; sys.modules["lgo"] = mod
exec(compile(src, "lucid/filter/lucid.py", "exec"), mod.__dict__)
np.seterr(all="ignore")
import arm5dof as AR, importlib.util
spec = importlib.util.spec_from_file_location("p54", "research/multivariate-statfilter/exploration/0054_physical_sensors.py")
m54 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m54)
jstd, pot, acc = m54.schedule(); U, S_, Y = AR.simulate(1, jstd, pot, acc)
f = mod.LucidFilter(dynamics=AR.F, control=AR.B, H=AR.measure, process=AR.Q0, measurement=AR.R0)
b = f._banks[0]; n = f.n; M = b.M; nr = len(getattr(f, "rates", (1.0,))); ncell = M // nr
print("bank members", M, "rates", getattr(f, "rates", None), "| cells per copy", ncell, "| held axes any:", np.flatnonzero(b._held.any(0)) if hasattr(b, "_held") else None)
prev = None
for t in range(0, 252):
    st = f.update(Y[t], U[t])
    if t in (249, 250, 251):
        pe = f._post_last.reshape(f._ndw, nr, ncell); i = int(np.argmax(pe[0, 0])); j = (i + ncell) if nr > 1 else i
        e_all = f._inn_last; mn = f._mn_last
        print(f"   t={t}: weights over cells (copy 0) {np.round(pe[0, 0], 2)} | theta1 of every member {np.round(mn[:, 0], 2)} | acc-axis window-read scale per member {np.round(f._msc_last[:, 1] if hasattr(f, '_msc_last') else np.zeros(M), 1)}", flush=True)
        yhat_i = Y[t] - e_all[i]; yhat_j = Y[t] - e_all[j]
        print(f"   t={t} member {i} vs {j}: y[:6] {np.round(Y[t][:6], 3)} | yhat_i[:6] {np.round(yhat_i[:6], 3)} | yhat_j[:6] {np.round(yhat_j[:6], 3)} | post state max|diff| {np.abs(mn[i]-mn[j]).max():.4f} | post state i[:3] {np.round(mn[i][:3], 3)} j[:3] {np.round(mn[j][:3], 3)}", flush=True)
        if prev is not None:
            print(f"      predicted from previous post states: H(F m_i + B u)[:6] {np.round(AR.measure(AR.F @ prev[0] + AR.B @ U[t])[1][:6], 3)}  H(F m_j + B u)[:6] {np.round(AR.measure(AR.F @ prev[1] + AR.B @ U[t])[1][:6], 3)}", flush=True)
        prev = (mn[i].copy(), mn[j].copy())
    continue
    llv = b._ll if False else f._llv_last
    e_all = f._inn_last; mn = f._mn_last
    # the cell with the most weight in the eager copy at this step
    pe = f._post_last.reshape(f._ndw, nr, ncell)
    i = int(np.argmax(pe[0, 0])); j = i + ncell          # eager member i, its patient twin j (same cell)
    dmu = b.mu[j] - b.mu[i]; dm = mn[j] - mn[i]
    print(f"t={t}: LL eager {llv[i]:+9.1f} patient {llv[j]:+9.1f} | max|dmu| {np.abs(dmu).max():.3f} (axis {int(np.argmax(np.abs(dmu)))}) | max|dstate| {np.abs(dm).max():.4f} | |e| eager max {np.abs(e_all[i]).max():.2f} patient {np.abs(e_all[j]).max():.2f} | eager mu acc {b.mu[i, n+1::3].mean():+.2f} proc {b.mu[i, :n].mean():+.2f} | patient mu acc {b.mu[j, n+1::3].mean():+.2f} proc {b.mu[j, :n].mean():+.2f} | weight eager copy {pe[:, 0].sum():.3f} | best LL over bank {llv.max():+.1f} (member {int(np.argmax(llv))}, copy {int(np.argmax(llv)) // ncell})", flush=True)
