"""0010 -- the acceptance battery for `LucidFilter(forget=None)` against the default.

The class drift now lives in `lucid/filter/lucid.py`; this probe uses the filter directly
(no load-time patch).  Rigs and windows are those of resolution-criterion/0003 and 0007.
Every (rig, seed, arm) run is checkpointed to `_cache/`; one filter in memory at a time.

Run from the repository root:  python research/forget-elimination/exploration/0010_the_battery.py [scalar] [async] [arm]
"""
import gc, importlib.util, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from lucid.filter.lucid import LucidFilter
from rig import cached
TAG = "v1"                                   # bump when the filter's drift code changes
ARMS = [("A default", {}), ("S forget=None", dict(forget=None))]
only = set(sys.argv[1:]) or {"scalar", "async", "arm"}

def paired(x, y):
    d = np.asarray(y) - np.asarray(x); sem = d.std(ddof=1) / math.sqrt(len(d))
    return d.mean(), sem, d.mean() / sem if sem > 0 else float("nan")

if "scalar" in only:
    N, JA, JU, NA = 900, 380, 9.0, 600
    W = (("jump", slice(JA, JA + 40)), ("steady", slice(80, JA)), ("loud", slice(NA + 40, N)))
    mse = {a[0]: {k: [] for k, _ in W} for a in ARMS}; ll = {a[0]: [] for a in ARMS}; tm = {a[0]: 0.0 for a in ARMS}; frac = []
    seeds = list(range(11, 35))
    for seed in seeds:
        rng = np.random.default_rng(seed); th = np.cumsum(rng.normal(0, math.sqrt(0.02), N)); th[JA:] += JU
        y = th + rng.normal(0, np.where(np.arange(N) < NA, 1.0, 3.0))
        for nm, kw in ARMS:
            def one(kw=kw):
                f = LucidFilter(**kw); t0 = time.perf_counter()
                m = np.asarray(f.filter(y.reshape(-1, 1)).mean).reshape(-1); dt = time.perf_counter() - t0
                return dict(mse={k: float(np.mean((m[sl] - th[sl]) ** 2)) for k, sl in W}, ll=f.loglik, dt=dt,
                            v=None if f._drift is None else f._drift.v)
            r = cached(f"0010{TAG}_scalar_{seed}_{nm[0]}", one)
            tm[nm] += r["dt"]; ll[nm].append(r["ll"])
            for k, _ in W: mse[nm][k].append(r["mse"][k])
            if r["v"] is not None: frac.append(math.exp(r["v"]))
    print(f"SCALAR hero rig, {len(seeds)} seeds.  RMSE per window, total log-likelihood, ms/step")
    for nm, _ in ARMS:
        print(f"  {nm:14s} " + "  ".join(f"{k} {math.sqrt(np.mean(mse[nm][k])):.4f}" for k, _ in W)
              + f"   loglik {np.mean(ll[nm]):.2f}   {1e3 * tm[nm] / (len(seeds) * N):.2f} ms/step")
    a, b = ARMS[0][0], ARMS[1][0]
    print("  S minus A, paired:  " + "  ".join("%s dMSE %+.4f+-%.4f (t=%+.2f)" % ((k,) + paired(mse[a][k], mse[b][k])) for k, _ in W)
          + "   dloglik %+.2f+-%.2f (t=%+.2f)" % paired(ll[a], ll[b]))
    print(f"  final rate fraction: median {np.median(frac):.3f}, range {min(frac):.3f}-{max(frac):.3f}", flush=True)

if "async" in only:
    spec = importlib.util.spec_from_file_location("rig_async", "research/pointwise-streaming/exploration/0005_the_asynchronous_rig.py")
    rig = importlib.util.module_from_spec(spec); spec.loader.exec_module(rig)
    res = {a[0]: dict(whole=[], hot=[], ll=[]) for a in ARMS}
    seeds = (0, 10, 20, 30, 40, 50, 60, 70)
    for seed in seeds:
        stream = rig.simulate(seed); truth = np.array([s[3] for s in stream]); times = np.array([s[1] for s in stream])
        F, Q = rig.nominal_model()
        for nm, kw in ARMS:
            def one(kw=kw):
                f = LucidFilter(dynamics=F, H=rig.H, process=Q, measurement=rig.SIGMA ** 2, timestep=rig.NOMINAL, **kw)
                est = np.empty((len(stream), 2))
                for k, (i, t, val, _tr, _sd) in enumerate(stream): est[k] = f.observe(i, val, t=t).mean
                err = est[:, 0] - truth[:, 0]; j, t0, t1, fac = rig.FAIL
                hw = (times >= t0 + 1) & (times < t1); ww = (times >= 1) & (times < rig.DURATION)
                return dict(whole=float(np.sqrt(np.mean(err[ww] ** 2))), hot=float(np.sqrt(np.mean(err[hw] ** 2))), ll=f.loglik)
            r = cached(f"0010{TAG}_async_{seed}_{nm[0]}", one)
            for key in ("whole", "hot", "ll"): res[nm][key].append(r[key])
    print(f"\nASYNC rig, {len(seeds)} seeds.  RMSE over the oracle's (whole 0.0345, hot 0.0241), total log-likelihood")
    for nm, _ in ARMS:
        r = res[nm]
        print(f"  {nm:14s} whole {np.mean(r['whole']) / 0.0345:.3f}x  hot {np.mean(r['hot']) / 0.0241:.3f}x  (worst hot {max(r['hot']) / 0.0241:.3f}x)   loglik {np.mean(r['ll']):.1f}")
    a, b = ARMS[0][0], ARMS[1][0]
    print("  S minus A, paired:  whole dRMSE %+.5f+-%.5f (t=%+.2f)" % paired(res[a]["whole"], res[b]["whole"])
          + "  hot dRMSE %+.5f+-%.5f (t=%+.2f)" % paired(res[a]["hot"], res[b]["hot"])
          + "  dloglik %+.2f+-%.2f (t=%+.2f)" % paired(res[a]["ll"], res[b]["ll"]), flush=True)

if "arm" in only:
    sys.path.insert(0, "research/multivariate-statfilter/scripts"); import arm5dof as AR
    spec = importlib.util.spec_from_file_location("p54", "research/multivariate-statfilter/exploration/0054_physical_sensors.py")
    m54 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m54)
    np.seterr(all="ignore")
    jstd, pot, acc = m54.schedule(); Wn = m54.windows()
    seeds = (0, 1, 2, 3, 4, 5)
    tab = {a[0]: {k: [] for k in Wn} for a in ARMS}; lls = {a[0]: [] for a in ARMS}; worst = {a[0]: [] for a in ARMS}
    print(f"\nARM, seeds {seeds}.  tip RMSE over the oracle's per window; log-likelihood; worst 100-step block (x oracle)")
    for seed in seeds:
        dat = {}
        def data():
            if not dat:
                U, St, Y = AR.simulate(seed, jstd, pot, acc)
                orc = AR.kalman(U, Y, [j ** 2 * (AR.B @ AR.B.T) for j in jstd],
                                [np.concatenate([[pot[k, j] ** 2, acc[k, j] ** 2, acc[k, j] ** 2] for j in range(AR.NJ)]) for k in range(m54.T)])
                dat.update(U=U, Y=Y, Pt=m54.tip(St), Po=m54.tip(orc))
            return dat
        for nm, kw in ARMS:
            def one(kw=kw):
                d = data(); t0 = time.perf_counter()
                f = LucidFilter(dynamics=AR.F, control=AR.B, H=AR.measure, process=AR.Q0, measurement=AR.R0, **kw)
                est = np.asarray(f.filter(d["Y"], d["U"]).mean); dt = time.perf_counter() - t0
                if not np.all(np.isfinite(est)): return None
                P = m54.tip(est); eo = np.sqrt(((d["Po"] - d["Pt"]) ** 2).sum(1)); e = np.sqrt(((P - d["Pt"]) ** 2).sum(1))
                blocks = [float(np.sqrt(np.mean(e[lo:lo + 100] ** 2)) / np.sqrt(np.mean(eo[lo:lo + 100] ** 2))) for lo in range(0, m54.T, 100)]
                return dict(dt=dt, ll=f.loglik, blocks=blocks, err=e, v=None if f._drift is None else f._drift.v,
                            win={k: m54.rms(P, d["Pt"], m54.mask(sp)) / m54.rms(d["Po"], d["Pt"], m54.mask(sp)) for k, sp in Wn.items()})
            r = cached(f"0010{TAG}_arm_{seed}_{nm[0]}", one); gc.collect()
            if r is None:
                print(f"  seed {seed} {nm:14s}: NaN", flush=True); continue
            for k in Wn: tab[nm][k].append(r["win"][k])
            lls[nm].append(r["ll"]); worst[nm].append(max(r["blocks"]))
            print(f"  seed {seed} {nm:14s} " + "  ".join(f"{k} {v:.3f}" for k, v in r["win"].items())
                  + f"   loglik {r['ll']:.0f}   worst block {max(r['blocks']):.1f}x   {1e3 * r['dt'] / m54.T:.0f} ms/step", flush=True)
    for nm, _ in ARMS:
        print(f"  mean   {nm:14s} " + "  ".join(f"{k} {np.mean(v):.3f}" for k, v in tab[nm].items()) + f"   loglik {np.mean(lls[nm]):.0f}   worst block {np.max(worst[nm]):.1f}x")
    a, b = ARMS[0][0], ARMS[1][0]
    print("  S/A per window, paired log-ratio:  " + "  ".join(
        "%s %+.3f+-%.3f (t=%+.2f)" % ((k,) + paired(np.log(tab[a][k]), np.log(tab[b][k]))) for k in Wn), flush=True)
