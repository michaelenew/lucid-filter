"""0018 -- how the moments of the N-step increment grow with N, one pure regime at a time.

Model (scalar, the filter's own class):

    x_{t+1} = x_t + w_t,   w_t | lam_q ~ N(0, Q exp(lam_q,t - s_q^2/2))
    y_t     = x_t + v_t,   v_t | lam_r ~ N(0, R exp(lam_r,t - s_r^2/2))
    lam_{t+1} = phi lam_t + s sqrt(1 - phi^2) eps_t          (stationary, Var lam = s^2)

`lam` is a log-VARIANCE; `s` is its stationary SD and `phi` its one-step persistence, the
shipped `(phi, s)` chart (`lucid.py`: innovation variance `s^2 (1 - phi^2)`).  The
`- s^2/2` keeps the mean variance at Q (or R) for every `(phi, s)`.

Observable: the lag-N increment  D_N = y_{t+N} - y_t.  Odd moments vanish.  Closed forms
(derivation in the .md):

    measurement, fixed scale :  Var = 2R          c4 = 0
    process,     fixed scale :  Var = N Q         c4 = 0
    measurement, wandering   :  Var = 2R          c4 = 6 R^2 [(e^{s^2}-1) + (e^{s^2 phi^N}-1)]
    process,     wandering   :  Var = N Q         c4 = 3 Q^2 sum_{i,j<=N} (e^{s^2 phi^|i-j|} - 1)

`c4` is the fourth cumulant, E D^4 - 3 Var^2; excess kurtosis is c4 / Var^2.  Cumulants of
independent parts add, so a mixed record's c4 is the sum of the two rows.

Two-time statistic: for the raw noise draw e (w, or v), Corr(e_t^2, e_{t+N}^2)
    = (e^{s^2 phi^N} - 1) / (3 e^{s^2} - 1).

Run:  python 0018_moments_by_lag.py     (numpy + matplotlib only; ~8 min, cached in 0018_cache.pkl)
Writes ../figures/0018-*.png and prints the theory-vs-simulation table.
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

LAGS = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128])
M, L = 2000, 4000            # independent paths x steps per path (8e6 samples per lag)
Q = R = 1.0

# ---------------------------------------------------------------- theory
def th_var(kind, N):
    return 2 * R * np.ones_like(N, float) if kind == "meas" else Q * N.astype(float)

def th_c4(kind, N, phi, s):
    """Fourth cumulant of D_N.  s = 0 is the fixed-scale regime."""
    g = s * s
    out = np.zeros(len(N))
    for a, n in enumerate(N):
        if kind == "meas":
            out[a] = 6 * R**2 * ((np.exp(g) - 1) + (np.exp(g * phi**n) - 1))
        else:
            k = np.arange(1, n)
            out[a] = 3 * Q**2 * (n * (np.exp(g) - 1)
                                 + 2 * np.sum((n - k) * (np.exp(g * phi**k) - 1)))
    return out

def th_exkurt(kind, N, phi, s):
    return th_c4(kind, N, phi, s) / th_var(kind, N) ** 2

def th_sqcorr(N, phi, s):
    g = s * s
    return (np.exp(g * phi**N.astype(float)) - 1) / (3 * np.exp(g) - 1)

# ---------------------------------------------------------------- simulation
def draws(phi, s, rng):
    """M x L noise draws with unit mean variance and an AR(1) log-variance."""
    e = rng.standard_normal((M, L))
    if s == 0:
        return e
    lam = np.empty((M, L))
    lam[:, 0] = s * rng.standard_normal(M)                  # exact stationary start
    innov = s * np.sqrt(1 - phi * phi) * rng.standard_normal((M, L))
    for t in range(1, L):
        lam[:, t] = phi * lam[:, t - 1] + innov[:, t]
    return e * np.exp(0.5 * (lam - s * s / 2))

def increments_stats(y):
    """Per-lag (variance, c4) of y[t+N] - y[t], with a path-level standard error on c4/var^2."""
    var, c4, ek, ek_se = [], [], [], []
    for n in LAGS:
        d = y[:, n:] - y[:, :-n]
        m2, m4 = np.mean(d**2), np.mean(d**4)
        var.append(m2); c4.append(m4 - 3 * m2**2); ek.append(m4 / m2**2 - 3)
        # delta-method SE from path-to-path scatter of the two moment means
        p2, p4 = np.mean(d**2, axis=1), np.mean(d**4, axis=1)
        infl = (p4 - m4) / m2**2 - 2 * m4 / m2**3 * (p2 - m2)
        ek_se.append(infl.std(ddof=1) / np.sqrt(M))
    return map(np.array, (var, c4, ek, ek_se))

_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "0018_cache.pkl")
_CACHE = pickle.load(open(_CACHE_PATH, "rb")) if os.path.exists(_CACHE_PATH) else {}

def simulate(kind, phi, s, seed):
    key = (kind, phi, s, seed, M, L)
    if key not in _CACHE:
        _CACHE[key] = _simulate(kind, phi, s, seed)
        pickle.dump(_CACHE, open(_CACHE_PATH, "wb"))
    return _CACHE[key]

def _simulate(kind, phi, s, seed):
    rng = np.random.default_rng(seed)
    e = draws(phi, s, rng)
    y = np.sqrt(R) * e if kind == "meas" else np.cumsum(np.sqrt(Q) * e, axis=1)
    var, c4, ek, ek_se = increments_stats(y)
    sq = e**2
    v0 = sq.var()
    sqc = np.array([np.mean((sq[:, n:] - sq.mean()) * (sq[:, :-n] - sq.mean())) / v0 for n in LAGS])
    return dict(var=var, c4=c4, ek=ek, ek_se=ek_se, sqc=sqc)

# ---------------------------------------------------------------- style
INK, MUTED, GRID = "#191919", "#666666", "#e6e6e6"
BLUE, ORANGE = "#257de0", "#f47230"
BLUES = ["#9dc8f7", "#6aabf1", "#338ced", "#1b64be", "#0e3d7e"]
ORANGES = ["#f9c4a4", "#f69d68", "#f47230", "#c2501a", "#7f3009"]
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 200,
})

def logx(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128"])
    ax.set_xlabel("N, steps ahead")
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG, exist_ok=True)
Nf = np.arange(1, 129)

# ================================================================ figure 1: the four regimes
PHI0, S0 = 0.9, 1.0
four = [  # (label, kind, phi, s, color, linestyle, marker)
    ("measurement, fixed level", "meas", 0.0, 0.0, BLUE, "-", "o"),
    ("process, fixed level", "proc", 0.0, 0.0, ORANGE, "-", "o"),
    (f"measurement, wandering level (phi={PHI0}, s={S0})", "meas", PHI0, S0, BLUE, "--", "D"),
    (f"process, wandering level (phi={PHI0}, s={S0})", "proc", PHI0, S0, ORANGE, "--", "D"),
]
res4 = [simulate(k, p, s, 10 + i) for i, (_, k, p, s, *_r) in enumerate(four)]

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.3))
for (lab, kind, phi, s, col, ls, mk), r in zip(four, res4):
    wander = s > 0
    kw = dict(color=col, ls=ls, lw=2)
    mkw = dict(color=col, marker=mk, ls="none", ms=5.5 if wander else 4,
               mfc="white" if wander else col, mew=1.4, zorder=3)
    ax[0].plot(Nf, th_var(kind, Nf), **kw); ax[0].plot(LAGS, r["var"], **mkw)
    if wander:
        ax[1].plot(Nf, th_c4(kind, Nf, phi, s), **kw); ax[1].plot(LAGS, r["c4"], **mkw)
    ax[2].plot(Nf, th_exkurt(kind, Nf, phi, s), label=lab, **kw); ax[2].plot(LAGS, r["ek"], **mkw)
ax[0].set_yscale("log"); ax[0].set_title("Variance: flat vs growing.\nWandering changes nothing")
ax[0].set_ylabel("variance of the N-step change")
ax[1].set_yscale("log")
ax[1].set_title("4th cumulant: settles (measurement) vs grows\n(process). Fixed level: exactly 0, not drawn")
ax[1].set_ylabel("4th cumulant  (E D^4 - 3 Var^2)")
ax[2].set_title("Excess kurtosis (4th cumulant / variance^2):\ntail-heaviness vs N")
ax[2].set_ylabel("excess kurtosis")
for a in ax: logx(a)
fig.legend(*ax[2].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=9,
           bbox_to_anchor=(0.5, -0.11))
fig.text(0.995, -0.1, "lines: closed form   markers: simulation (8e6 samples per lag)",
         ha="right", color=MUTED, fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "0018-four-regimes.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)

# ================================================================ figure 2: phi and s sweeps
PHIS = [0.0, 0.7, 0.9, 0.97, 0.995]
SS = [0.4, 0.6, 0.8, 1.0, 1.2]
fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.6), sharex=True)
table = []
for row, (kind, ramp, name) in enumerate([("proc", ORANGES, "Process noise"),
                                          ("meas", BLUES, "Measurement noise")]):
    for col_i, (sweep, vals) in enumerate([("phi", PHIS), ("s", SS)]):
        a = ax[row, col_i]
        for j, v in enumerate(vals):
            phi, s = (v, S0) if sweep == "phi" else (PHI0, v)
            r = simulate(kind, phi, s, 100 + 37 * row + 11 * col_i + j)
            th = th_exkurt(kind, Nf, phi, s)
            a.plot(Nf, th, color=ramp[j], lw=2)
            a.errorbar(LAGS, r["ek"], yerr=2 * r["ek_se"], color=ramp[j], marker="o", ls="none",
                       ms=3.8, elinewidth=1, zorder=3)
            if (kind, sweep) == ("meas", "phi"):
                a.plot([], [], color=ramp[j], lw=2, label=f"{sweep}={v}")
            else:
                a.annotate(f"{sweep}={v}", (Nf[-1], th[-1]), xytext=(5, 0),
                           textcoords="offset points", va="center", fontsize=8.5, color=INK)
            table.append((kind, phi, s, r, th_exkurt(kind, LAGS, phi, s), th_sqcorr(LAGS, phi, s)))
        fixed = f"s = {S0}" if sweep == "phi" else f"phi = {PHI0}"
        what = "memory phi" if sweep == "phi" else "spread s"
        a.set_title(f"{name}: sweep {what}  ({fixed})")
        a.set_ylabel("excess kurtosis of the N-step change")
        if (kind, sweep) == ("meas", "phi"):
            a.legend(loc="upper right", fontsize=8.5, ncol=2)
            a.set_ylim(2.0, 6.6)
        else:
            a.set_yscale("log")
        a.set_xlim(0.9, 128 * 2.3)
        logx(a)
        if row == 0: a.set_xlabel("")
fig.text(0.995, 0.0, "lines: closed form   markers: simulation, bars = 2 standard errors",
         ha="right", color=MUTED, fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "0018-phi-s-sweeps.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)

# ================================================================ figure 3: squared-draw correlation
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.1), sharey=True)
for col_i, (sweep, vals) in enumerate([("phi", PHIS), ("s", SS)]):
    a = ax[col_i]
    for j, v in enumerate(vals):
        phi, s = (v, S0) if sweep == "phi" else (PHI0, v)
        r = [t for t in table if t[0] == "proc" and t[1] == phi and t[2] == s][0][3]
        th = th_sqcorr(Nf, phi, s)
        if phi > 0:
            a.plot(Nf, th, color=BLUES[j], lw=2, label=f"{sweep}={v}")
        ok = (r["sqc"] > 2e-3) & (phi > 0)
        a.plot(LAGS[ok], r["sqc"][ok], color=BLUES[j], marker="o", ls="none", ms=3.8, zorder=3)
    fixed = f"s = {S0}" if sweep == "phi" else f"phi = {PHI0}"
    a.set_title(f"Sweep {'memory phi' if sweep == 'phi' else 'spread s'}  ({fixed})")
    a.set_yscale("log"); a.set_ylim(2e-3, 1); a.set_xlim(0.9, 140)
    a.legend(loc="lower left", fontsize=8.5)
    logx(a)
ax[0].set_ylabel("correlation of squared noise draws, N apart")
fig.text(0.995, -0.03, "lines: closed form   markers: simulation.  phi = 0 is exactly zero at every "
         "N >= 1 (not drawn); markers below 0.002 are sampling noise and are dropped.", ha="right", color=MUTED, fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "0018-squared-correlation.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)

# ================================================================ checks
print("theory vs simulation, excess kurtosis: worst |z| over lags (z = (sim - theory) / SE)")
print(f"{'kind':5s} {'phi':>6s} {'s':>5s}  {'th N=1':>8s} {'sim N=1':>8s}  {'th N=128':>9s} {'sim N=128':>9s}  {'max|z|':>6s}")
for kind, phi, s, r, th, _ in table:
    z = np.abs(r["ek"] - th) / r["ek_se"]
    print(f"{kind:5s} {phi:6.3f} {s:5.2f}  {th[0]:8.3f} {r['ek'][0]:8.3f}  {th[-1]:9.4f} {r['ek'][-1]:9.4f}  {z.max():6.2f}")

# variance: wandering leaves it untouched
for (lab, kind, phi, s, *_r), r in zip(four, res4):
    print(f"variance  {lab:52s} max rel err vs theory {np.max(np.abs(r['var'] / th_var(kind, LAGS) - 1)):.4f}")

# additivity of c4 in a mixed record (both scales wandering, different (phi, s))
rng = np.random.default_rng(7)
pq, sq_, pr, sr = 0.97, 0.8, 0.7, 1.0
y = np.cumsum(draws(pq, sq_, rng), axis=1) + draws(pr, sr, rng)
var, c4, ek, _ = increments_stats(y)
c4_th = th_c4("proc", LAGS, pq, sq_) + th_c4("meas", LAGS, pr, sr)
print("mixed record (process phi=0.97 s=0.8, measurement phi=0.7 s=1.0): c4 sim / (sum of the two closed forms)")
print("  " + "  ".join(f"N={n}: {a / b:.3f}" for n, a, b in zip(LAGS, c4, c4_th)))

# two lags pin (s, phi): invert the closed forms at N = 1, 2
k1, k2 = 3 + th_exkurt("proc", np.array([1]), 0.9, 1.0)[0], 3 + th_exkurt("proc", np.array([2]), 0.9, 1.0)[0]
g = np.log(k1 / 3); phi_hat = np.log(2 * k2 / 3 - k1 / 3) / g
print(f"process inversion from kurtosis at N=1,2: s^2 = {g:.4f} (true 1), phi = {phi_hat:.4f} (true 0.9)")
