"""Arm rig: each engine's per-axis share g = sqrt(2 I_char) at the balanced base (the local Fisher's
share of the channel), and the shipped filter's walk excursions per axis over the run (seed 0)."""
import sys, numpy as np, importlib.util, math
sys.path.insert(0, "."); sys.path.insert(0, "research/multivariate-statfilter/scripts")
import arm5dof as AR
from lucid.filter.lucid import LucidFilter, _RIDGE
spec = importlib.util.spec_from_file_location("p54", "research/multivariate-statfilter/exploration/0054_physical_sensors.py")
m54 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m54)
np.seterr(all="ignore")
jstd, pot, acc = m54.schedule(); U, S_, Y = AR.simulate(0, jstd, pot, acc)
f = LucidFilter(dynamics=AR.F, control=AR.B, H=AR.measure, process=AR.Q0, measurement=AR.R0)
mem = f._members
e0 = mem[0]
n = e0.n
print("engine n=%d m=%d D=%d active=%d members=%d" % (e0.n, e0.m, e0.D, len(e0._act), len(mem)))
g = np.sqrt(np.maximum(2 * (e0._Ichar - _RIDGE), 0))
print("share g per axis (process modes then sensors), member 0 (phi=%.2f s=%.2f):" % (e0.phi, e0.s))
print("  process:", " ".join("%.3f" % v for v in g[:n]))
print("  sensor :", " ".join("%.3f" % v for v in g[n:]))
print("  gap per axis (1.5 s):", e0.gap[0], " outer node offset:", 2 * e0.gap[0])
# walk excursions: run the filter and record mu per member per axis at each step via the bank
mus = []
res = f.filter(Y, U)
# the bank keeps mu as (M, D); read the final and track max over the run by re-running stepwise
f2 = LucidFilter(dynamics=AR.F, control=AR.B, H=AR.measure, process=AR.Q0, measurement=AR.R0)
mx = None; mn = None
for t in range(len(Y)):
    f2.update(Y[t], U[t])
    mu = np.concatenate([b.mu for b in f2._banks], 0)
    mx = mu.copy() if mx is None else np.maximum(mx, mu); mn = mu.copy() if mn is None else np.minimum(mn, mu)
print("max mu over run, per member (rows) / axis (cols), process axes:")
for j in range(mx.shape[0]):
    print("  m%02d phi=%.2f s=%.2f  max %s | sensors max %s" % (j, mem[j].phi, mem[j].s, " ".join("%5.1f" % v for v in mx[j, :n]), " ".join("%5.1f" % v for v in mx[j, n:])))
print("min mu, process axes:")
for j in range(mn.shape[0]):
    print("  m%02d  min %s | sensors min %s" % (j, " ".join("%5.1f" % v for v in mn[j, :n]), " ".join("%5.1f" % v for v in mn[j, n:])))
