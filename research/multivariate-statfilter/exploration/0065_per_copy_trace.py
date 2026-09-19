"""Per-copy trace (eager copy vs patient copy) on the drone no-crate mission or the arm, from any filter source
(a single-copy source is traced as one copy): each copy's weight, own error, scales, per-axis mu and per-step score by phase.
    python 0065_per_copy_trace.py <lucid.py source|tree> {drone|arm} <seed>"""
import sys, types, importlib.util, os, time
import numpy as np
sys.path.insert(0, ".")
src, rig, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
code = open(src).read()
o = "        post, patience, att_rate = self._att_marginal(post)\n"
if o in code:
    code = code.replace(o, "        self._post_full = post.copy()\n" + o, 1)
else:
    o = "        pm = np.bincount(self._wm, weights=post, minlength=M)"; assert o in code
    code = code.replace(o, "        self._post_full = post.copy()\n" + o, 1)
o2 = "        mean = pm @ mn\n"; assert o2 in code
code = code.replace(o2, o2 + "        self._mn_last = mn; self._msc_last = msc; self._psc_last = psc; self._llv_last = llv\n", 1)
import lucid.filter, lucid as _top
mod = types.ModuleType("lucid.filter.lucid"); mod.__file__ = "lucid/filter/lucid.py"; mod.__package__ = "lucid.filter"
sys.modules["lucid.filter.lucid"] = mod
exec(compile(code, "lucid/filter/lucid.py", "exec"), mod.__dict__)
lucid.filter.lucid = mod
for name in ("LucidFilter", "LucidResult", "LucidStep"):
    if hasattr(mod, name): setattr(lucid.filter, name, getattr(mod, name)); setattr(_top, name, getattr(mod, name))
np.seterr(all="ignore")
def copy_split(f):
    nr = len(getattr(f, "rates", (1.0,)))
    if nr == 1:
        w = f._post_full; pm = np.bincount(f._wm, weights=w, minlength=len(f._mn_last))
        return [(1.0, pm @ f._mn_last, pm @ f._msc_last, pm @ f._psc_last)]
    P = f._post_full.reshape(f._Ja, f._ndw, nr, -1)
    out = []
    for r in range(nr):
        w = np.zeros_like(f._post_full).reshape(f._Ja, f._ndw, nr, -1); w[:, :, r, :] = P[:, :, r, :]; w = w.ravel()
        tot = w.sum(); pm = np.bincount(f._wm, weights=w, minlength=len(f._mn_last)) / max(tot, 1e-300)
        out.append((tot, pm @ f._mn_last, pm @ f._msc_last, pm @ f._psc_last))
    return out
if rig == "drone":
    sys.path.insert(0, "research/dynamics-learning/scripts"); import drone3d as R
    U, X, Y, hold = R.simulate(seed, carry=False); sv, sw = R.schedule()
    orc = R.kalman(U, Y, sv, sw, hold)
    f = R.make_filter(); T = R.T; POS = np.arange(3); phases = R.PHASES; SKIP = 120
    truth = X[:, POS]; oe = np.linalg.norm(orc[:, POS] - truth, axis=1)
    def est_of(m): return m[POS]
    sens_ax = slice(0, 3)    # GPS position channels
    print("drone no-crate seed", seed, "| rates", getattr(f, "rates", None), "| copies' sensor axes = GPS position 0-2; process axes = position modes 0-2", flush=True)
else:
    sys.path.insert(0, "research/multivariate-statfilter/scripts"); import arm5dof as A
    spec = importlib.util.spec_from_file_location("p54", "research/multivariate-statfilter/exploration/0054_physical_sensors.py")
    m54 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m54)
    jstd, pot, acc = m54.schedule(); U, S_, Y = A.simulate(seed, jstd, pot, acc)
    orc = A.kalman(U, Y, [j ** 2 * (A.B @ A.B.T) for j in jstd], [np.concatenate([[pot[k, j] ** 2, acc[k, j] ** 2, acc[k, j] ** 2] for j in range(A.NJ)]) for k in range(m54.T)])
    f = A.make_filter(); T = m54.T; phases = m54.PHASES; SKIP = m54.SKIP
    truth = m54.tip(S_); oe = np.linalg.norm(m54.tip(orc) - truth, axis=1)
    def est_of(m): return m54.tip(m[None])[0]
    sens_ax = slice(1, None, 3)   # accelerometer channels
    print("arm seed", seed, "| rates", getattr(f, "rates", None), "| copies' sensor axes = accelerometers; process axes = all modes", flush=True)
RATES = getattr(f, "rates", (1.0,)); nr = len(RATES)
E = np.zeros((T, 1 + nr)); Wt = np.zeros((T, nr)); MS = np.zeros((T, nr)); PS = np.zeros((T, nr)); MIX = np.zeros(T)
PSax = np.zeros((T, nr, f.n)); MSax = np.zeros((T, nr, f.m)); LLc = np.zeros((T, nr))
t0 = time.time()
for t in range(T):
    st = f.update(Y[t], U[t])
    E[t, 0] = np.linalg.norm(est_of(st.mean) - truth[t])
    for r, (w, m, ms, ps) in enumerate(copy_split(f)):
        Wt[t, r] = w; E[t, 1 + r] = np.linalg.norm(est_of(m) - truth[t]); MS[t, r] = ms[sens_ax].mean(); PS[t, r] = ps[:3].mean() if rig == "drone" else ps.mean()
        PSax[t, r] = ps; MSax[t, r] = ms
    if nr > 1:
        Pf = f._post_full.reshape(f._Ja, f._ndw, nr, -1); llf = np.tile(f._llv_last[f._wm], f._Ja).reshape(f._Ja, f._ndw, nr, -1)
        for r in range(nr):
            w = Pf[:, :, r, :].ravel(); w = w / max(w.sum(), 1e-300); l = llf[:, :, r, :].ravel()
            LLc[t, r] = float(np.log(np.exp(l - l.max()) @ w) + l.max())
print(f"run {time.time()-t0:.0f}s", flush=True)
if rig == "drone" and nr > 1:
    b0 = f._banks[0]
    print("held axes (bank 0 rows x axes): any held per axis:", np.flatnonzero(getattr(b0, "_held", np.zeros((1,1), bool)).any(0)), "| n process axes", f.n, "| sensor axes", f.m, flush=True)
    for nm, a, bb in [("calm before", 1420, 1550), ("MULTIPATH", 1670, 1900), ("calm after", 2020, 2500)]:
        for r in range(nr):
            print(f"   {nm:>12} copy rate {RATES[r]:.3g}: proc mu per mode {np.round(PSax[a:bb, r].mean(0), 2)} | sens mu per channel {np.round(MSax[a:bb, r].mean(0), 2)} | mean LL/step {LLc[a:bb, r].mean():+.3f}", flush=True)
def show(a, b, label):
    sl = slice(a, b); rms = lambda e: float(np.sqrt(np.mean(e ** 2)))
    line = f"{label:>22} {a:4d}-{b:4d}: mixture {rms(E[sl,0])/rms(oe[sl]):5.2f}x oracle (worst {E[sl,0].max():.3f} m)"
    for r in range(nr):
        line += f" | copy rate {RATES[r]:.3g}: w {Wt[sl,r].mean():.2f} own {rms(E[sl,1+r])/rms(oe[sl]):5.2f}x (worst {E[sl,1+r].max():.3f}) sens {MS[sl,r].mean():+.2f} proc {PS[sl,r].mean():+.2f}"
    print(line, flush=True)
for nm, a, b in phases:
    if b - a > 60:
        show(a, min(a + 40, b), nm + " onset"); show(a + SKIP, b, nm)
