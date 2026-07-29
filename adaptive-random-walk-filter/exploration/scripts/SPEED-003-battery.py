"""SPEED-003: the accuracy/speed battery, run against any core.py.

Speed work is only worth anything if the estimate does not move, so every
change is judged on the same 9-probe x 4-seed battery THEORY-009 used, plus two
things THEORY-009 did not record and that a speed change could silently break:

    loglik   the objective value at the returned optimum.  fit() is a maximiser;
             a faster fit that lands lower has lost accuracy, whatever the MSE
             says on this particular battery.
    seconds  wall clock per fit.

Usage:
    python exploration/scripts/SPEED-003-battery.py baseline   # frozen copy
    python exploration/scripts/SPEED-003-battery.py current    # output/statfilter
    python exploration/scripts/SPEED-003-battery.py compare    # both, side by side
"""
import importlib.util
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BENCH = os.path.join(ROOT, "exploration", "speedbench")

PROBES = ["diffusion q=.005", "diffusion q=.05", "diffusion q=.5",
          "drift-rate regime", "pure step", "jump+drift",
          "outlier contam.", "hetero noise", "heavy-tail noise"]
SEEDS = [20260728, 101, 202, 303]


def load_core(which):
    """Import a core.py by path under a private module name."""
    path = (os.path.join(BENCH, "core_baseline.py") if which == "baseline"
            else os.path.join(ROOT, "output", "statfilter", "core.py"))
    spec = importlib.util.spec_from_file_location(f"_core_{which}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------- probes
def probe(name, rng, N=1200):
    """Verbatim from THEORY-007, with the rng passed in rather than global."""
    s2 = np.ones(N)
    q = np.full(N, 0.05)
    if name == "diffusion q=.005":
        q[:] = 0.005
    elif name == "diffusion q=.5":
        q[:] = 0.5
    elif name == "drift-rate regime":
        q[: N // 2] = 0.005
        q[N // 2:] = 0.5
    elif name == "pure step":
        q[:] = 1e-9
    elif name == "hetero noise":
        s2[N // 2:] = 9.0
    w = rng.standard_normal(N) * np.sqrt(q)
    th = np.cumsum(w)
    if name in ("pure step", "jump+drift"):
        for t in (N // 4, N // 2, 3 * N // 4):
            th[t:] += 6.0
    v = rng.standard_normal(N) * np.sqrt(s2)
    if name == "heavy-tail noise":
        v = rng.standard_t(3, N) / np.sqrt(3.0)
    x = th + v
    if name == "outlier contam.":
        idx = rng.choice(N, N // 100, replace=False)
        x[idx] += rng.standard_normal(len(idx)) * 8
    return x, th


def tuned_kalman_mse(x, th):
    """Constant-gain Kalman, gain chosen in hindsight to minimise this MSE."""
    K = np.linspace(0.005, 0.995, 199)[:, None]
    m = np.empty((K.size, x.size))
    cur = np.full(K.size, x[0])
    for t in range(x.size):
        cur = cur + K[:, 0] * (x[t] - cur)
        m[:, t] = cur
    return float(((m - th) ** 2).mean(1).min())


# --------------------------------------------------------------------- jobs
def job(args):
    which, sd, idx, name = args
    core = load_core(which)
    rng = np.random.default_rng([sd, idx])
    x, th = probe(name, rng)
    t0 = time.perf_counter()
    try:
        f = core.AdaptiveFilter.fit(x)
    except Exception as exc:
        # The baseline fit() can genuinely raise: Nelder-Mead is unbounded, and
        # once it drives logit(phi) past -709 the math.exp inside _expit
        # overflows.  Record it rather than lose the whole run.
        return dict(which=which, probe=name, seed=sd,
                    secs=time.perf_counter() - t0, failed=f"{type(exc).__name__}: {exc}")
    secs = time.perf_counter() - t0
    r = f.filter(x)
    mse = float(((r.mean - th) ** 2).mean())
    return dict(which=which, probe=name, seed=sd, secs=secs,
                ratio=mse / tuned_kalman_mse(x, th),
                loglik=float(r.loglik), par=f.params.to_dict())


def run(whichs):
    jobs = [(w, sd, i, nm) for w in whichs
            for sd in SEEDS for i, nm in enumerate(PROBES)]
    with Pool(min(cpu_count(), 4)) as pool:
        return pool.map(job, jobs)


def summarise(rows, label):
    bad = [r for r in rows if r.get("failed")]
    rows = [r for r in rows if not r.get("failed")]
    by = {p: [r for r in rows if r["probe"] == p] for p in PROBES}
    print(f"\n=== {label} ===")
    for r in bad:
        print(f"  FIT FAILED on {r['probe']} seed {r['seed']}: {r['failed']}")
    if any(len(v) != len(SEEDS) for v in by.values()):
        print("  (summary below is over the fits that completed)")
        by = {p: v for p, v in by.items() if len(v) == len(SEEDS)}
    print(f"{'probe':>18} {'ratio (per seed)':>32} {'median':>8} {'sec/fit':>9}")
    for p in PROBES:
        rs = [r["ratio"] for r in by[p]]
        print(f"{p:>18} " + " ".join(f"{v:>7.3f}" for v in rs)
              + f" {np.median(rs):>8.3f} {np.mean([r['secs'] for r in by[p]]):>9.1f}")
    geos = [float(np.exp(np.mean([np.log(by[p][i]["ratio"]) for p in PROBES])))
            for i in range(len(SEEDS))]
    worsts = [float(max(by[p][i]["ratio"] for p in PROBES)) for i in range(len(SEEDS))]
    secs = [r["secs"] for r in rows]
    print(f"{'geometric mean':>18} " + " ".join(f"{g:>7.3f}" for g in geos))
    print(f"{'worst case':>18} " + " ".join(f"{w:>7.3f}" for w in worsts))
    print(f"  across seeds: geo mean {np.mean(geos):.4f}, worst {np.mean(worsts):.4f}")
    print(f"  total fit time {np.sum(secs):.1f} s over {len(rows)} fits "
          f"({np.mean(secs):.1f} s/fit, max {np.max(secs):.1f})")
    return dict(rows=rows, geos=geos, worsts=worsts)


def compare(a, b):
    """a = baseline rows, b = current rows, aligned by (probe, seed)."""
    key = lambda r: (r["probe"], r["seed"])
    A = {key(r): r for r in a}
    B = {key(r): r for r in b}
    print(f"\n=== baseline vs current, per fit ===")
    print(f"{'probe':>18} {'seed':>9} {'d loglik':>10} {'ratio old':>10} "
          f"{'ratio new':>10} {'speedup':>8}")
    dlls, ups = [], []
    for k in sorted(A, key=lambda k: (PROBES.index(k[0]), k[1])):
        dll = B[k]["loglik"] - A[k]["loglik"]
        up = A[k]["secs"] / B[k]["secs"]
        dlls.append(dll)
        ups.append(up)
        flag = "  <-- worse" if dll < -0.5 else ""
        print(f"{k[0]:>18} {k[1]:>9} {dll:>10.3f} {A[k]['ratio']:>10.3f} "
              f"{B[k]['ratio']:>10.3f} {up:>7.1f}x{flag}")
    print(f"\n  d loglik (new - old): min {min(dlls):+.3f}  median "
          f"{np.median(dlls):+.3f}  max {max(dlls):+.3f}")
    print(f"  speedup: min {min(ups):.1f}x  median {np.median(ups):.1f}x  "
          f"max {max(ups):.1f}x")
    print(f"  total wall: {sum(r['secs'] for r in a):.0f} s -> "
          f"{sum(r['secs'] for r in b):.0f} s")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "current"
    os.makedirs(BENCH, exist_ok=True)
    if what == "compare":
        a = json.load(open(f"{BENCH}/battery_baseline.json"))["rows"]
        b = json.load(open(f"{BENCH}/battery_current.json"))["rows"]
        summarise(a, "baseline")
        summarise(b, "current")
        compare(a, b)
    elif what == "both":
        rows = run(["baseline", "current"])
        a = [r for r in rows if r["which"] == "baseline"]
        b = [r for r in rows if r["which"] == "current"]
        json.dump(summarise(a, "baseline"), open(f"{BENCH}/battery_baseline.json", "w"), indent=1)
        json.dump(summarise(b, "current"), open(f"{BENCH}/battery_current.json", "w"), indent=1)
        compare(a, b)
    else:
        rows = run([what])
        out = summarise(rows, what)
        json.dump(out, open(f"{BENCH}/battery_{what}.json", "w"), indent=1)
