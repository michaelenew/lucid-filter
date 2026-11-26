"""0007 -- the derived rate star in the shipped filter: hero, asynchronous and arm rigs.

Arms:  A shipped (forget = 0.999);  B forget = 1;  S star on (phi, s), forget = 1 elsewhere.
Rigs and windows are those of resolution-criterion/0003 (paired, same seeds).  The patch is
`star_filter.py`; nothing under `lucid/` is modified.

Run from the repository root:  python research/forget-elimination/exploration/0007_the_star_in_the_filter.py
"""
import importlib.util, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import star_filter as SF
A = SF.load("shipped"); S = SF.load("star")
ARMS = [("A shipped", A, {}), ("B forget=1", A, dict(forget=1.0)), ("S star", S, dict(forget=1.0))]
only = set(sys.argv[1:]) or {"scalar", "async", "arm"}
print(f"star: spacing {SF.G:.3f} nats, {2 * SF.K + 1} nodes, recentre level {SF.C_RES} nats", flush=True)

def paired(x, y):
    d = np.asarray(y) - np.asarray(x); sem = d.std(ddof=1) / math.sqrt(len(d))
    return d.mean(), sem, d.mean() / sem if sem > 0 else float("nan")

if "scalar" in only:
    N, JA, JU, NA = 900, 380, 9.0, 600
    W = (("jump", slice(JA, JA + 40)), ("steady", slice(80, JA)), ("loud", slice(NA + 40, N)))
    mse = {a[0]: {k: [] for k, _ in W} for a in ARMS}; ll = {a[0]: [] for a in ARMS}; tm = {a[0]: 0.0 for a in ARMS}
    frac = []
    for seed in range(11, 23):
        rng = np.random.default_rng(seed); th = np.cumsum(rng.normal(0, math.sqrt(0.02), N)); th[JA:] += JU
        y = th + rng.normal(0, np.where(np.arange(N) < NA, 1.0, 3.0))
        for nm, mod, kw in ARMS:
            f = mod.LucidFilter(**kw); t0 = time.perf_counter()
            m = np.asarray(f.filter(y.reshape(-1, 1)).mean).reshape(-1); tm[nm] += time.perf_counter() - t0
            for k, sl in W: mse[nm][k].append(np.mean((m[sl] - th[sl]) ** 2))
            ll[nm].append(f.loglik)
            if nm.startswith("S"): frac.append((math.exp(f._star["v"]), f._star["moves"]))
    print("\nSCALAR hero rig, 12 seeds.  RMSE per window, total log-likelihood, ms/step")
    for nm, _, _ in ARMS:
        print(f"  {nm:12s} " + "  ".join(f"{k} {math.sqrt(np.mean(mse[nm][k])):.4f}" for k, _ in W)
              + f"   loglik {np.mean(ll[nm]):.2f}   {1e3 * tm[nm] / (12 * N):.2f} ms/step")
    for nm in ("B forget=1", "S star"):
        print(f"  {nm} minus A, paired:  " + "  ".join("%s dMSE %+.4f+-%.4f (t=%+.2f)" % ((k,) + paired(mse["A shipped"][k], mse[nm][k])) for k, _ in W)
              + "   dloglik %+.2f+-%.2f (t=%+.2f)" % paired(ll["A shipped"], ll[nm]))
    print(f"  star: final rate fraction median {np.median([x[0] for x in frac]):.3f} (range {min(x[0] for x in frac):.3f}-{max(x[0] for x in frac):.3f}), moves per record {np.mean([x[1] for x in frac]):.1f}", flush=True)

if "async" in only:
    spec = importlib.util.spec_from_file_location("rig_async", "research/pointwise-streaming/exploration/0005_the_asynchronous_rig.py")
    rig = importlib.util.module_from_spec(spec); spec.loader.exec_module(rig)
    res = {a[0]: dict(whole=[], hot=[], ll=[]) for a in ARMS}
    for seed in (0, 10, 20, 30, 40):
        stream = rig.simulate(seed); truth = np.array([s[3] for s in stream]); times = np.array([s[1] for s in stream])
        F, Q = rig.nominal_model()
        for nm, mod, kw in ARMS:
            f = mod.LucidFilter(dynamics=F, H=rig.H, process=Q, measurement=rig.SIGMA ** 2, timestep=rig.NOMINAL, **kw)
            est = np.empty((len(stream), 2))
            for k, (i, t, val, _tr, _sd) in enumerate(stream): est[k] = f.observe(i, val, t=t).mean
            err = est[:, 0] - truth[:, 0]; j, t0, t1, fac = rig.FAIL
            hw = (times >= t0 + 1) & (times < t1); ww = (times >= 1) & (times < rig.DURATION)
            res[nm]["whole"].append(np.sqrt(np.mean(err[ww] ** 2))); res[nm]["hot"].append(np.sqrt(np.mean(err[hw] ** 2)))
            res[nm]["ll"].append(f.loglik)
    print("\nASYNC rig, 5 seeds.  RMSE over the oracle's (whole 0.0345, hot 0.0241), total log-likelihood")
    for nm, _, _ in ARMS:
        r = res[nm]
        print(f"  {nm:12s} whole {np.mean(r['whole']) / 0.0345:.3f}x  hot {np.mean(r['hot']) / 0.0241:.3f}x  (worst hot {max(r['hot']) / 0.0241:.3f}x)   loglik {np.mean(r['ll']):.1f}")
    for nm in ("B forget=1", "S star"):
        print(f"  {nm} minus A, paired:  whole dRMSE %+.5f+-%.5f (t=%+.2f)" % paired(res["A shipped"]["whole"], res[nm]["whole"])
              + "  hot dRMSE %+.5f+-%.5f (t=%+.2f)" % paired(res["A shipped"]["hot"], res[nm]["hot"])
              + "  dloglik %+.2f+-%.2f (t=%+.2f)" % paired(res["A shipped"]["ll"], res[nm]["ll"]), flush=True)

if "arm" in only:
    sys.path.insert(0, "research/multivariate-statfilter/scripts"); import arm5dof as AR
    spec = importlib.util.spec_from_file_location("p54", "research/multivariate-statfilter/exploration/0054_physical_sensors.py")
    m54 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m54)
    np.seterr(all="ignore")
    jstd, pot, acc = m54.schedule(); U, St, Y = AR.simulate(0, jstd, pot, acc)
    orc = AR.kalman(U, Y, [j ** 2 * (AR.B @ AR.B.T) for j in jstd],
                    [np.concatenate([[pot[k, j] ** 2, acc[k, j] ** 2, acc[k, j] ** 2] for j in range(AR.NJ)]) for k in range(m54.T)])
    Pt = m54.tip(St); Po = m54.tip(orc); Wn = m54.windows()
    print(f"\nARM seed 0 ({m54.T} steps).  tip RMSE over the oracle's, per window; total log-likelihood")
    for nm, mod, kw in ARMS:
        t0 = time.perf_counter()
        f = mod.LucidFilter(dynamics=AR.F, control=AR.B, H=AR.measure, process=AR.Q0, measurement=AR.R0, **kw)
        est = np.asarray(f.filter(Y, U).mean); dt = time.perf_counter() - t0
        if np.all(np.isfinite(est)):
            Pl = m54.tip(est)
            extra = f"  star rate fraction {math.exp(f._star['v']):.4f}, {f._star['moves']} moves" if nm.startswith("S") else ""
            print(f"  {nm:12s} ({1e3 * dt / m54.T:.0f} ms/step): " + "  ".join(
                f"{k} {m54.rms(Pl, Pt, m54.mask(sp)) / m54.rms(Po, Pt, m54.mask(sp)):.3f}" for k, sp in Wn.items())
                + f"   loglik {f.loglik:.1f}" + extra, flush=True)
        else:
            print(f"  {nm:12s}: NaN", flush=True)
