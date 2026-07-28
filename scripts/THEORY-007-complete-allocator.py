"""THEORY-007: the computation, finished.  No thresholds, no tuned constants.

Two things close here.

A. THE SCAN COST, which 05 flagged as the largest open gap.
   Asking "did an event happen, and where?" costs a penalty that GROWS with
   series length: the max of n candidate LLRs under H0 drifts up like log n.
   Asking instead "how large is the deviation field?" -- one variance component
   over the whole series -- costs a boundary LRT whose null distribution is
   (1/2)chi2_0 + (1/2)chi2_1, INDEPENDENT of n.  The scan penalty does not have
   to be paid because the scan does not have to be performed.  Verified below.

B. THE COMPLETE ALLOCATOR.  Every "mode" is a property of the noise itself:

     theta_t = theta_{t-1} + w_t,   w_t ~ N(0, Q  exp(lamP_t))
     x_t     = theta_t     + v_t,   v_t ~ N(0, s2 exp(lamM_t))
     lam^c_t = phi_c lam^c_{t-1} + sqrt(nu_c) z_t,   c in {P, M}

   The four modes are not hypotheses, they are the two channels x the two ends
   of each channel's own autocorrelation.  Everything the filter needs is in
   six numbers -- Q, s2, phi_P, phi_M, s_P, s_M -- and all six are LEARNED by
   maximising the exact marginal likelihood.  There is no threshold anywhere,
   no gate, no event time, and no constant chosen by hand.  Q and s2 are the
   MEDIAN variances: (log-mean, log-SD) are the orthogonal coordinates of a
   log-normal, whereas (arithmetic mean, log-SD) confounds them completely.

   Two exact conservation laws come out per step, with no rule imposing them:

     amplitude:  e_t = (P/S) e_t + (Q_t/S) e_t + (R_t/S) e_t,  coefficients sum to 1
                 -> "where I already was" + "the level really moved" + "noise"
     scale:      E[lam^c_t | D] = phi_c E[lam^c_{t-1} | D]  +  (the rest)
                 -> "carried over"  (regime)  +  "new at t"  (anomaly)

   Four signed mode coordinates, exhaustive, continuous, always defined.

The one approximation: the level posterior is collapsed to a single Gaussian
each step (GPB1).  That is a numerical scheme, not a knob -- as is the
quadrature order for the log-scale grid.
"""
import json
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
from theory_style import plt, tidy, save, SERIES, SEQ

OUT = "figures"
rng = np.random.default_rng(20260728)


# =========================================================== A. the scan cost
def scan_vs_field(n, reps=4000):
    """Under H0, compare the two null penalties.

    scan:  max over t0 of the profile LLR for a single level shift at t0
    field: LLR for one variance component covering every t (boundary test)
    """
    scan, field = np.empty(reps), np.empty(reps)
    for r in range(reps):
        d = rng.standard_normal(n)                 # whitened increments under H0
        # a shift at t0 shows up as a mean on d_{t0}; profile LLR = d^2/2
        scan[r] = 0.5 * (d ** 2).max()
        # one common variance component s^2 on all n: profile LLR at the boundary
        S = (d ** 2).mean()
        field[r] = 0.5 * n * (S - 1 - np.log(S)) if S > 1 else 0.0
    return scan, field


print("A. scan penalty vs field penalty under H0 (nats, 95th percentile)")
print(f"{'n':>7} {'scan p95':>10} {'log n':>8} {'field p95':>11}")
scanrows = []
for n in (10, 30, 100, 300, 1000, 3000):
    s, f = scan_vs_field(n)
    scanrows.append(dict(n=n, scan=float(np.percentile(s, 95)),
                         field=float(np.percentile(f, 95))))
    print(f"{n:>7} {np.percentile(s, 95):>10.2f} {np.log(n):>8.2f} "
          f"{np.percentile(f, 95):>11.2f}")
print(f"  scan grows like log n; field is flat at "
      f"{chi2.ppf(0.90, 1) / 2:.3f} = the boundary-LRT 95th pct")


# ====================================================== B. the full allocator
def gh(n):
    z, w = np.polynomial.hermite_e.hermegauss(n)
    return z, w / w.sum()


def chain(phi, s, n):
    """Quadrature grid for a stationary AR(1) log-scale, plus its transition matrix.

    Nodes are the Gauss-Hermite abscissae of the stationary law; the transition
    is the exact Gaussian kernel reweighted onto those nodes.  Nothing chosen.
    """
    z, w = gh(n)
    lam = s * z
    nu = max(s * s * (1 - phi * phi), 1e-12)
    T = w[None, :] * np.exp(np.clip(
        -0.5 * (lam[None, :] - phi * lam[:, None]) ** 2 / nu
        + 0.5 * lam[None, :] ** 2 / max(s * s, 1e-12), -700.0, 700.0))
    T /= T.sum(1, keepdims=True)
    return lam, w, T


NG = 5


def run(x, par, want_alloc=False):
    """Exact HMM forward over the joint log-scale grid, GPB1 on the level.

    par = (log Q, log s2, logit phiP, logit phiM, log sP, log sM)
    """
    lq, ls2, tpP, tpM, lsP, lsM = par
    Q, S2 = np.exp(lq), np.exp(ls2)
    phP, phM = 1 / (1 + np.exp(-tpP)), 1 / (1 + np.exp(-tpM))
    sP, sM = np.exp(lsP), np.exp(lsM)
    lamP, wP, TP = chain(phP, sP, NG)
    lamM, wM, TM = chain(phM, sM, NG)
    LP = np.repeat(lamP, NG)                       # joint grid, P varies slowly
    LM = np.tile(lamM, NG)
    T = np.kron(TP, TM)
    pi = np.kron(wP, wM)

    # Parameterise each log-variance by its MEAN (log Q) and SD (s): those are
    # the orthogonal coordinates of a log-normal.  Centring by -s^2/2 instead --
    # so that E[multiplier]=1 and Q is the arithmetic-mean variance -- makes
    # log Q and s^2/2 perfectly confounded, and the fit runs away along that
    # ridge: on the pure-step probe it reached Q ~ 1e105 with s_P = 24.6 while
    # still fitting well.  Q and s2 are therefore MEDIAN variances here.
    Qg = Q * np.exp(np.clip(LP, -60.0, 60.0))
    Rg = S2 * np.exp(np.clip(LM, -60.0, 60.0))

    m, P = x[0], float(np.var(x))                 # diffuse start, taken from the data
    ll = 0.0
    N = len(x)
    if want_alloc:
        out = dict(mean=np.empty(N), lamP=np.empty(N), lamM=np.empty(N),
                   cP=np.empty(N), cQ=np.empty(N), cR=np.empty(N))
    for t in range(N):
        pi = pi @ T
        Pprev = P
        Pp = P + Qg
        S = Pp + Rg
        e = x[t] - m
        lg = -0.5 * (np.log(2 * np.pi * S) + e * e / S)
        mx = lg.max()                              # take it BEFORE shifting
        wgt = pi * np.exp(lg - mx)
        Z = wgt.sum()
        ll += np.log(Z) + mx
        pi = wgt / Z
        K = Pp / S
        mi = m + K * e
        Pi = (1 - K) * Pp
        mnew = pi @ mi
        P = pi @ (Pi + (mi - mnew) ** 2)
        m = mnew
        if want_alloc:
            out["mean"][t] = m
            out["lamP"][t] = pi @ LP
            out["lamM"][t] = pi @ LM
            out["cP"][t] = pi @ (Pprev / S)        # "I was already wrong about theta"
            out["cQ"][t] = pi @ (Qg / S)           # "the level really moved"
            out["cR"][t] = pi @ (Rg / S)           # "that was noise"
    if want_alloc:
        out["ll"] = ll
        out["phi"] = (phP, phM)
        out["par"] = dict(Q=Q, s2=S2, phiP=phP, phiM=phM, sP=sP, sM=sM)
        return out
    return ll


def fit(x):
    """Staged ML.  Nothing here is a tuning choice.

    Stage 0   -- a 1-D scan over Q with sigma^2 pinned by the variogram identity
        gamma_0 = Q + 2 sigma^2,  gamma_0 = E[d^2]
    so the pair is always admissible and the scan range is set by the data's own
    scale.  Stage 0.5 -- a coarse 5x5 scan over (phi_P, phi_M).  Stage 1 -- full
    6-D ML from that start, run twice (from a quiet and from a volatile
    log-scale) and the better likelihood kept.  All of this is numerical
    robustness: the answer is the ML estimate either way.
    """
    d = np.diff(x)
    g0 = float(np.mean(d ** 2))
    grid = g0 * np.logspace(-5, -0.05, 25)
    lls = []
    for Qc in grid:
        p = np.array([np.log(Qc), np.log((g0 - Qc) / 2), 0.0, 0.0,
                      np.log(1e-3), np.log(1e-3)])
        lls.append(run(x, p))
    Q0 = grid[int(np.argmax(lls))]
    # Stage 0.5 -- a coarse 5x5 scan over the two persistences.  THEORY-010
    # showed the profile in phi_M carries tens of nats of curvature on impulsive
    # data, and that 6-D Nelder-Mead from a single start was simply failing to
    # find it: the fit sat at its logit-0 start of phi = 0.5.  Twenty-five extra
    # likelihood evaluations against ~1300 for the search, so this is free.
    PH = np.array([0.02, 0.25, 0.5, 0.75, 0.95])
    lg = lambda z: float(np.log(z / (1 - z)))
    bph, bv = (0.0, 0.0), -np.inf
    for pp in PH:
        for pm in PH:
            v = run(x, np.array([np.log(Q0), np.log((g0 - Q0) / 2),
                                 lg(pp), lg(pm), np.log(0.6), np.log(0.6)]))
            if v > bv:
                bph, bv = (lg(pp), lg(pm)), v
    best, bestf = None, np.inf
    for s0 in (0.03, 0.6):
        p0 = np.array([np.log(Q0), np.log((g0 - Q0) / 2), bph[0], bph[1],
                       np.log(s0), np.log(s0)])
        r = minimize(lambda p: -run(x, p) / len(x), p0, method="Nelder-Mead",
                     options=dict(maxiter=500, xatol=2e-3, fatol=1e-5))
        if r.fun < bestf:
            best, bestf = r.x, r.fun
    return best


# --------------------------------------------------------------- the probes
def probe(name, N=1200):
    th = np.zeros(N)
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
    elif name == "jump+drift":
        pass
    elif name == "outlier contam.":
        pass
    elif name == "hetero noise":
        s2[N // 2:] = 9.0
    elif name == "heavy-tail noise":
        pass
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
    best = np.inf
    for K in np.linspace(0.005, 0.995, 199):
        m = x[0]
        e = np.empty(len(x))
        for t in range(len(x)):
            m = m + K * (x[t] - m)
            e[t] = m - th[t]
        best = min(best, float((e ** 2).mean()))
    return best


PROBES = ["diffusion q=.005", "diffusion q=.05", "diffusion q=.5",
          "drift-rate regime", "pure step", "jump+drift",
          "outlier contam.", "hetero noise", "heavy-tail noise"]

print("\nB. the allocator on the probe battery -- six parameters, all learned")
print(f"{'probe':>18} {'MSE':>8} {'tunedKF':>8} {'ratio':>7}   "
      f"{'Q':>7} {'s2':>6} {'phiP':>6} {'sP':>6} {'phiM':>6} {'sM':>6}")
rows, ratios, keep = [], [], {}
for name in PROBES:
    x, th = probe(name)
    p = fit(x)
    o = run(x, p, want_alloc=True)
    mse = float(((o["mean"] - th) ** 2).mean())
    ref = tuned_kalman_mse(x, th)
    ratios.append(mse / ref)
    pr = o["par"]
    rows.append(dict(probe=name, mse=mse, tuned=ref, ratio=mse / ref, **pr))
    print(f"{name:>18} {mse:>8.4f} {ref:>8.4f} {mse / ref:>7.3f}   "
          f"{pr['Q']:>7.4f} {pr['s2']:>6.3f} {pr['phiP']:>6.3f} {pr['sP']:>6.3f} "
          f"{pr['phiM']:>6.3f} {pr['sM']:>6.3f}")
    if name in ("pure step", "hetero noise", "outlier contam.", "diffusion q=.05"):
        keep[name] = (x, th, o)
g = float(np.exp(np.mean(np.log(ratios))))
print(f"{'geometric mean':>18} {'':>8} {'':>8} {g:>7.3f}")
print(f"{'worst case':>18} {'':>8} {'':>8} {max(ratios):>7.3f}")

json.dump(dict(scan=scanrows, probes=rows, geo=g, worst=float(max(ratios))),
          open("figures/theory007.json", "w"), indent=1)


# --------------------------------------------------------------------- figures
# Fig 19: the scan penalty grows, the field penalty does not
fig, ax = plt.subplots(figsize=(6.4, 4.0))
tidy(ax)
ns = [r["n"] for r in scanrows]
ax.semilogx(ns, [r["scan"] for r in scanrows], color=SERIES[0], marker="o",
            label="scan: max over $t_0$ of a single-event LLR")
ax.semilogx(ns, [r["field"] for r in scanrows], color=SERIES[2], marker="o",
            label="field: one variance component over all $t$")
ax.semilogx(ns, np.log(ns), color="#8a8880", ls="--", lw=1.2, label="$\\log n$")
BOUND = chi2.ppf(0.90, 1) / 2.0     # 95th pct of (1/2)delta_0 + (1/2)chi2_1, in nats
ax.axhline(BOUND, color="#c9c8c3", lw=1.0)
ax.text(12, BOUND + 0.15, f"{BOUND:.2f} nats: the boundary-LRT 95th pct, flat in $n$",
        fontsize=8, color="#52514e")
ax.set_xlabel("n  (series length)")
ax.set_ylabel("null penalty, 95th percentile (nats)")
ax.set_title("The scan cost is only paid if you scan.\n"
             "Asking 'how big is the deviation field' costs a constant.")
ax.legend(loc="upper left")
save(fig, f"{OUT}/fig19-scan-vs-field.png")

# Fig 20: the four mode coordinates over time, on three probes
names = ["pure step", "outlier contam.", "hetero noise"]
fig, axes = plt.subplots(3, 2, figsize=(11.2, 8.2))
for r, nm in enumerate(names):
    x, th, o = keep[nm]
    phP, phM = o["phi"]
    t = np.arange(len(x))
    ax = tidy(axes[r, 0])
    ax.plot(t, x, color="#c9c8c3", lw=0.6, label="observed")
    ax.plot(t, th, color="#0b0b0b", lw=1.4, label="truth")
    ax.plot(t, o["mean"], color=SERIES[1], lw=1.4, label="allocator")
    ax.set_ylabel(nm, fontsize=9.5)
    if r == 0:
        ax.set_title("tracking", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
    ax = tidy(axes[r, 1])
    PAc = o["lamP"] - phP * np.concatenate([[0], o["lamP"][:-1]])
    PRc = phP * np.concatenate([[0], o["lamP"][:-1]])
    MAc = o["lamM"] - phM * np.concatenate([[0], o["lamM"][:-1]])
    MRc = phM * np.concatenate([[0], o["lamM"][:-1]])
    for i, (c, lb) in enumerate([(PAc, "PA"), (PRc, "PR"), (MAc, "MA"), (MRc, "MR")]):
        ax.plot(t, c, color=SERIES[i], lw=1.0, label=lb)
    ax.axhline(0, color="#8a8880", lw=0.8)
    if r == 0:
        ax.set_title("the four signed mode coordinates (log-scale nats)", fontsize=10)
        ax.legend(loc="upper left", ncol=4, fontsize=8)
for ax in axes[2]:
    ax.set_xlabel("t")
fig.suptitle("One allocator, all four modes at once, nothing thresholded",
             fontsize=11.5, color="#0b0b0b", y=1.01)
save(fig, f"{OUT}/fig20-mode-coordinates.png")

# Fig 21: the innovation partition, per step, summing to 1
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
x, th, o = keep["pure step"]
t = np.arange(len(x))
ax = tidy(axes[0])
ax.stackplot(t, o["cP"], o["cQ"], o["cR"],
             colors=[SEQ[1], SERIES[1], SERIES[0]],
             labels=["prior level error", "real level move ($Q$)", "measurement noise ($R$)"])
ax.set_xlim(440, 560)
ax.set_ylim(0, 1)
ax.set_xlabel("t  (around the step at 500)")
ax.set_ylabel("share of the innovation")
ax.set_title("Amplitude conservation, per step: the innovation\n"
             "partitions three ways and the shares sum to 1")
ax.legend(loc="lower left", fontsize=8)
ax = tidy(axes[1])
x2, th2, o2 = keep["outlier contam."]
ax.stackplot(np.arange(len(x2)), o2["cP"], o2["cQ"], o2["cR"],
             colors=[SEQ[1], SERIES[1], SERIES[0]])
ax.set_xlim(440, 560)
ax.set_ylim(0, 1)
ax.set_xlabel("t  (outlier-contaminated series)")
ax.set_title("Same filter, same partition, different regime\n(no branch, no gate)")
save(fig, f"{OUT}/fig21-innovation-partition.png")
