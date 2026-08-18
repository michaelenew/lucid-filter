"""THEORY-008: is the fitted process-scale volatility s_P meaningful?

On a pure Gaussian diffusion (no heteroscedasticity at all) THEORY-007's ML fit
sometimes lands on a large s_P and sometimes on s_P = 0 exactly, with no effect
on tracking MSE.  Two possible explanations, and they need separating:

  (a) quadrature artifact -- at 5 nodes a large s is coarsely represented, and
      ML may be climbing discretisation error rather than a feature of the data;
  (b) a genuinely flat ridge -- maximum likelihood never penalises flexibility,
      so s_P > 0 weakly improves the fit for free and the optimiser wanders.

The test: profile the likelihood in s_P at several quadrature orders.  If the
preference for large s_P shrinks as nodes are added, it is (a).  If it survives
and is simply flat, it is (b) -- in which case the fitted number should not be
read as an estimate, and the honest report is the profile itself.
"""
import json
import numpy as np
from theory_style import plt, tidy, save, SERIES

OUT = "figures"
src = open("scripts/THEORY-007-complete-allocator.py").read()
exec(src[src.index("def gh(n):"):src.index("def fit(")])

rng = np.random.default_rng(11)
N, QTRUE, S2TRUE = 1200, 0.005, 1.0
th = np.cumsum(rng.standard_normal(N) * np.sqrt(QTRUE))
x = th + rng.standard_normal(N) * np.sqrt(S2TRUE)


def par(Q, S2, sP, sM=1e-3, phP=0.5, phM=0.5):
    return np.array([np.log(Q), np.log(S2), np.log(phP / (1 - phP)),
                     np.log(phM / (1 - phM)), np.log(max(sP, 1e-6)), np.log(sM)])


SPS = np.concatenate([[1e-6], np.linspace(0.25, 7.0, 28)])
ORDERS = [5, 9, 15, 25]
prof = {}
for ng in ORDERS:
    globals()["NG"] = ng
    row = []
    for sP in SPS:
        # profile: at each s_P take the best Q on a scan, sigma^2 pinned by the
        # variogram identity gamma_0 = Q + 2 sigma^2 so every candidate is valid
        d = np.diff(x)
        g0 = float(np.mean(d ** 2))
        grid = g0 * np.logspace(-6, -0.05, 22)
        row.append(max(run(x, par(Qc, (g0 - Qc) / 2, sP)) for Qc in grid))
    prof[ng] = np.array(row)
    i = int(np.argmax(prof[ng]))
    print(f"NG={ng:>3}  best s_P={SPS[i]:>5.2f}  "
          f"ll(best)-ll(s_P=0) = {prof[ng][i] - prof[ng][0]:>7.3f} nats  "
          f"(over {N} points)")

json.dump({str(k): v.tolist() for k, v in prof.items()} | {"sps": SPS.tolist()},
          open("figures/theory008.json", "w"), indent=1)

fig, ax = plt.subplots(figsize=(6.8, 4.2))
tidy(ax)
for i, ng in enumerate(ORDERS):
    ax.plot(SPS, prof[ng] - prof[ng][0], color=SERIES[i], label=f"{ng} quadrature nodes")
ax.axhline(0, color="#8a8880", lw=1.0)
ax.axhline(1.0, color="#c9c8c3", lw=1.0, ls=":")
ax.text(0.3, 1.15, "1 nat", fontsize=8, color="#52514e")
ax.set_xlabel("$s_P$  (log-SD of the process-noise scale)")
ax.set_ylabel("profile log-likelihood, relative to $s_P=0$  (nats)")
ax.set_title("Is the fitted process volatility real?\n"
             "Profile likelihood on a series with none, 1200 points")
ax.legend()
save(fig, f"{OUT}/fig22-volatility-identifiability.png")
