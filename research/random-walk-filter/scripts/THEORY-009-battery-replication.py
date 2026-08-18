"""THEORY-009: replicate the probe battery across seeds.

THEORY-007 reports one realisation per probe.  An MSE ratio on a single draw is
noisy, and the headline (a geometric mean below 1, i.e. beating the
hindsight-tuned constant-gain Kalman) should not rest on one seed.  This runs
the whole battery on several independent draws and reports the spread.

The baseline is the same one used throughout the project: a CONSTANT-gain
Kalman filter whose gain is chosen in hindsight to minimise that series' own
MSE.  Beating it is expected wherever the truth is non-stationary -- one fixed
gain cannot serve two regimes -- so a ratio below 1 is not a claim of beating
the optimal filter, only this specific oracle.
"""
import json
import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.optimize import minimize          # used by fit(), exec'd in from THEORY-007
from theory_style import plt, tidy, save, SERIES

OUT = "figures"
src = open("scripts/THEORY-007-complete-allocator.py").read()
exec(src[src.index("def gh(n):"):src.index("PROBES = [")])

PROBES = ["diffusion q=.005", "diffusion q=.05", "diffusion q=.5",
          "drift-rate regime", "pure step", "jump+drift",
          "outlier contam.", "hetero noise", "heavy-tail noise"]
SEEDS = [20260728, 101, 202, 303]

# The 36 fits are independent, and each one is ~1300 sequential likelihood
# evaluations at 38 ms apiece -- 98% of which is numpy dispatch on 25-element
# arrays, not arithmetic.  Nothing about that vectorises (the recursion is
# sequential in t), but the fits themselves are embarrassingly parallel.
def job(args):
    sd, idx, name = args
    globals()["rng"] = np.random.default_rng([sd, idx])   # independent stream per job
    x, th = probe(name)
    o = run(x, fit(x), want_alloc=True)
    mse = float(((o["mean"] - th) ** 2).mean())
    return name, sd, mse / tuned_kalman_mse(x, th), o["par"]


if __name__ == "__main__":
    jobs = [(sd, i, nm) for sd in SEEDS for i, nm in enumerate(PROBES)]
    with Pool(min(cpu_count(), 4)) as pool:
        done = pool.map(job, jobs)
    res = {p: [] for p in PROBES}
    pars = {p: [] for p in PROBES}
    for sd in SEEDS:
        for name, s_, r, pr in done:
            if s_ == sd:
                res[name].append(r)
                pars[name].append(pr)

    print(f"\n{'probe':>18} " + " ".join(f"{s:>8}" for s in SEEDS) + f" {'median':>8}")
    for p in PROBES:
        print(f"{p:>18} " + " ".join(f"{v:>8.3f}" for v in res[p])
              + f" {np.median(res[p]):>8.3f}")
    geos = [float(np.exp(np.mean([np.log(res[p][i]) for p in PROBES])))
            for i in range(len(SEEDS))]
    worsts = [float(max(res[p][i] for p in PROBES)) for i in range(len(SEEDS))]
    print(f"{'geometric mean':>18} " + " ".join(f"{g:>8.3f}" for g in geos))
    print(f"{'worst case':>18} " + " ".join(f"{w:>8.3f}" for w in worsts))
    print(f"\nacross seeds: geo mean {np.mean(geos):.3f} "
          f"(range {min(geos):.3f}-{max(geos):.3f}), "
          f"worst case {np.mean(worsts):.3f} (range {min(worsts):.3f}-{max(worsts):.3f})")

    # how stable are the fitted mode parameters, channel by channel?
    print(f"\n{'probe':>18} {'phiM':>22} {'sM':>22}")
    for p in PROBES:
        pm = [q["phiM"] for q in pars[p]]
        sm = [q["sM"] for q in pars[p]]
        print(f"{p:>18} " + " ".join(f"{v:>5.2f}" for v in pm) + "   "
              + " ".join(f"{v:>5.2f}" for v in sm))

    json.dump(dict(ratios=res, geos=geos, worsts=worsts, seeds=SEEDS,
                   pars={p: pars[p] for p in PROBES}),
              open("figures/theory009.json", "w"), indent=1)

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    tidy(ax)
    for i, p in enumerate(PROBES):
        ax.plot(np.full(len(SEEDS), i), res[p], marker="o", ms=6, ls="none",
                color=SERIES[0], alpha=0.65, mec="#fcfcfb", mew=1.0)
        ax.plot([i - 0.22, i + 0.22], [np.median(res[p])] * 2, color=SERIES[1], lw=2.2)
    ax.axhline(1.0, color="#d03b3b", lw=1.3, ls="--")
    ax.text(-0.4, 1.03, "parity with the hindsight-tuned constant-gain Kalman",
            fontsize=8, color="#d03b3b")
    ax.set_xticks(range(len(PROBES)), PROBES, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("MSE ratio (lower is better)")
    ax.set_title(f"Probe battery across {len(SEEDS)} seeds; orange bar = median\n"
                 "six parameters, all learned, nothing tuned or thresholded")
    save(fig, f"{OUT}/fig23-battery-replication.png")
