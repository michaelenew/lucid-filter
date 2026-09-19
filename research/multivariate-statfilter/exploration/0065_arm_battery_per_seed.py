"""Per-seed arm battery (the README rig, 0054): burst RMSE over every non-calm step (onsets included), the worst
tip error and where it happens, per-regime ratios.   python 0065_arm_battery_per_seed.py <lucid.py source|tree>"""
import sys, types, importlib.util, os, time
import numpy as np
sys.path.insert(0, ".")
src = sys.argv[1]
if src != "tree":
    import lucid.filter, lucid as _top
    mod = types.ModuleType("lucid.filter.lucid"); mod.__file__ = "lucid/filter/lucid.py"; mod.__package__ = "lucid.filter"
    sys.modules["lucid.filter.lucid"] = mod
    exec(compile(open(src).read(), "lucid/filter/lucid.py", "exec"), mod.__dict__)
    lucid.filter.lucid = mod
    for name in ("LucidFilter", "LucidResult", "LucidStep"):
        if hasattr(mod, name): setattr(lucid.filter, name, getattr(mod, name)); setattr(_top, name, getattr(mod, name))
sys.path.insert(0, "research/multivariate-statfilter/scripts"); import arm5dof as A
spec = importlib.util.spec_from_file_location("p54", "research/multivariate-statfilter/exploration/0054_physical_sensors.py")
m54 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m54)
np.seterr(all="ignore")
print("source", os.path.basename(src), "| memory ladder:", hasattr(sys.modules["lucid.filter.lucid"], "_memory_rungs"), flush=True)
W = m54.windows()
def phase_of(k):
    for nm, a, b in m54.PHASES:
        if a <= k < b: return f"{nm}@{k-a}"
    return "?"
for sd in range(m54.NS):
    t0 = time.time()
    jstd, pot, acc = m54.schedule(); U, S_, Y = A.simulate(sd, jstd, pot, acc)
    Qs = [j ** 2 * (A.B @ A.B.T) for j in jstd]
    Rs = [np.concatenate([[pot[k, j] ** 2, acc[k, j] ** 2, acc[k, j] ** 2] for j in range(A.NJ)]) for k in range(m54.T)]
    r = A.make_filter().filter(Y, U); est = np.asarray(r.mean)
    orc = A.kalman(U, Y, Qs, Rs)
    Pt, Pl, Po = m54.tip(S_), m54.tip(est), m54.tip(orc)
    on = np.zeros(m54.T, bool)
    for nm, a, b in m54.PHASES:
        if nm != "calm": on[a:b] = True
    err = np.linalg.norm(Pl - Pt, axis=1); k = int(np.argmax(err))
    reg = "  ".join(f"{key} {m54.rms(Pl,Pt,m54.mask(sp))/m54.rms(Po,Pt,m54.mask(sp)):.2f}" for key, sp in W.items())
    big = np.flatnonzero(err > 0.2)
    print(f"seed {sd}: burst RMSE lucid {m54.rms(Pl,Pt,on):.4f} oracle {m54.rms(Po,Pt,on):.4f} | worst tip error {err[k]:.3f} m at step {k} ({phase_of(k)}), oracle there {np.linalg.norm(Po[k]-Pt[k]):.3f} | steps with error > 0.2 m: {big.size}"
          + (f" (first {big[0]} {phase_of(big[0])}, last {big[-1]} {phase_of(big[-1])})" if big.size else "") + f" | {reg} | {time.time()-t0:.0f}s", flush=True)
