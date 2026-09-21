"""Why do the patient (mode >= 1) members freeze?  Trace one mode-0 and one mode-1 member of the same cell on
the arm, seed 1, through the SENSOR onset: mu on joint 0's accelerometer-x axis and on its alpha mode, the
reading, the kernel target, Kmu."""
import sys, types, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "research/multivariate-statfilter/scripts")
S="/tmp/claude-0/-home-user-lucid-filter/56bbf4e5-3fe2-5e9c-93f7-d50a0493cfbc/scratchpad/"
src = open(S + "lucid_k01.py").read()
old = "                target = np.einsum(\"bl,lb->b\", Wn, self._gbuf[ax, :nf])\n"
assert old in src
src = src.replace(old, old + "                self._dbg[ax] = (reading.copy(), target.copy(), Kmu.copy(), self.mu[:, k].copy(), (grad / info).copy())\n")
src = src.replace("        self._nfill = np.zeros(len(self._act), int)\n", "        self._nfill = np.zeros(len(self._act), int)\n        self._dbg = {}\n")
mod = types.ModuleType("lkd"); mod.__file__ = "lucid/filter/lucid.py"; sys.modules["lkd"] = mod
exec(compile(src, "lucid/filter/lucid.py", "exec"), mod.__dict__)
np.seterr(all="ignore")
import arm5dof as AR, importlib.util
spec = importlib.util.spec_from_file_location("p54", "research/multivariate-statfilter/exploration/0054_physical_sensors.py")
m54 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m54)
jstd, pot, acc = m54.schedule(); U, S_, Y = AR.simulate(1, jstd, pot, acc)
f = mod.LucidFilter(dynamics=AR.F, control=AR.B, H=AR.measure, process=AR.Q0, measurement=AR.R0)
b = f._banks[0]; n = f.n
modes = b._wmode; ncell = len(modes) // 2
i0, i1 = 0, ncell            # the same cell, mode 0 and mode 1
print("bank members", b.M, "modes", modes[:3], modes[ncell:ncell+3], "| act axes", len(b._act), "| Nk", b._Nk, "| kernel row of member i1:", np.round(b._WK[i1, :5], 3))
axes = {"acc0x (sensor 1)": n + 1, "alpha0 (process 2)": 2}
for t in range(0, 320):
    st = f.update(Y[t], U[t])
    if 246 <= t <= 262 or t in (280, 300, 319):
        line = f"t={t}"
        for nm, k in axes.items():
            ax = b._act.index(k) if k in b._act else None
            if ax is None or ax not in b._dbg: line += f" | {nm}: (inactive)"; continue
            r, tg, km, mu, ratio = b._dbg[ax]
            line += f" | {nm}: mu0 {mu[i0]:+.2f} mu1 {mu[i1]:+.2f} ratio0 {ratio[i0]:+.2f} ratio1 {ratio[i1]:+.2f} read1 {r[i1]:+.2f} target1 {tg[i1]:+.2f} Kmu0 {km[i0]:.3f} Kmu1 {km[i1]:.3f}"
        print(line, flush=True)
print("patience at 300:", f.patience)
