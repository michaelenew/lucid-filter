"""THEORY-010: is the persistence coordinate phi_M actually learned?

Across four seeds (THEORY-009) the fitted phi_M behaves asymmetrically:

    hetero noise (persistent sigma^2 shift):  0.92, 0.50, 0.94, 0.93
    outlier contaminated (impulsive spikes):  0.50, 0.50, 0.50, 0.00

0.50 is exactly the optimiser's starting value (logit 0), so on the outlier
probe the fit is mostly *not moving* rather than concluding "impulsive".  The
single-seed contrast reported earlier (0.000 vs 0.931) was partly luck.

Two readings, and they need separating:
  (a) the optimiser is failing -- there is a peak at low phi_M and Nelder-Mead
      is not finding it;
  (b) the likelihood is genuinely flat in phi_M on impulsive data, so there is
      nothing to find.

(b) is what THEORY-003 predicts: anomaly evidence is a bounded budget, so an
impulsive channel offers almost nothing to estimate a persistence from, while
regime evidence accrues linearly and should be learnable.  Profiling decides it.
"""
import json
import numpy as np
from scipy.optimize import minimize
from theory_style import plt, tidy, save, SERIES

OUT = "figures"
src = open("scripts/THEORY-007-complete-allocator.py").read()
exec(src[src.index("def gh(n):"):src.index("PROBES = [")])

PHIS = np.linspace(0.0, 0.97, 25)
CASES = ["outlier contam.", "hetero noise", "diffusion q=.05"]
SEEDS = [20260728, 101, 202, 303]

prof = {c: [] for c in CASES}
for c in CASES:
    for sd in SEEDS:
        globals()["rng"] = np.random.default_rng([sd, CASES.index(c)])
        x, _ = probe(c)
        p = fit(x)
        row = []
        for ph in PHIS:
            q = p.copy()
            q[3] = np.log(max(ph, 1e-6) / max(1 - ph, 1e-6))     # set phi_M
            # re-optimise everything else at this phi_M so the profile is honest
            free = np.delete(q, 3)

            def obj(f):
                full = np.insert(f, 3, q[3])
                return -run(x, full) / len(x)

            r = minimize(obj, free, method="Nelder-Mead",
                         options=dict(maxiter=140, xatol=5e-3, fatol=1e-4))
            row.append(-r.fun * len(x))
        prof[c].append(np.array(row))
    A = np.array(prof[c])
    rng_nats = (A.max(1) - A.min(1))
    print(f"{c:>18}  argmax phi_M per seed: "
          + " ".join(f"{PHIS[i]:.2f}" for i in A.argmax(1))
          + f"   profile range: " + " ".join(f"{v:.2f}" for v in rng_nats) + " nats")

json.dump({c: [a.tolist() for a in prof[c]] for c in CASES} | {"phis": PHIS.tolist()},
          open("figures/theory010.json", "w"), indent=1)

fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), sharey=True)
for ax, c in zip(axes, CASES):
    tidy(ax)
    for i, a in enumerate(prof[c]):
        ax.plot(PHIS, a - a.max(), color=SERIES[i], lw=1.4, label=f"seed {SEEDS[i]}")
    ax.axhline(-1.353, color="#d03b3b", lw=1.1, ls="--")
    ax.set_ylim(-12, 1)
    ax.set_xlabel("$\\varphi_M$  (persistence of the measurement scale)")
    ax.set_title(c, fontsize=10)
axes[0].set_ylabel("profile log-likelihood (nats, rel. to max)")
axes[0].text(0.02, -2.4, "1.353 nats: the null\nfloor from THEORY-007A",
             fontsize=7.5, color="#d03b3b")
axes[0].legend(loc="lower left", fontsize=7.5)
fig.suptitle("Persistence is learnable where the theory says evidence accrues, and not where it saturates",
             fontsize=11, color="#0b0b0b", y=1.06)
save(fig, f"{OUT}/fig24-persistence-profile.png")
