"""0035 -- How few measurements can possibly reveal ODE behaviour?

The complaint this answers: the filter's learning looks laggier than it needs
to be.  Before that can be called a defect, there has to be a bound -- the
number of points at which *no* procedure could have known, so that the gap
between the bound and what the filter does is the real lag.

Three calculations, none of them a fit:

  A  THE INFORMATION BOUND, deterministic signal.  The cleanest form of "is
     there a velocity?": distinguish a constant from a constant-plus-ramp in
     white noise.  For a known-variance Gaussian the expected log-likelihood
     ratio is exactly half the non-centrality, so

         nats(n) = v^2 * sum_t (t - tbar)^2 / (2 s^2) = n(n^2-1) / (24 r^2)

     with r = s / v the noise-to-velocity-step ratio.  Cubic in n: the slope
     estimator's variance falls as n^-3.  Acceleration is a quadratic against a
     linear and goes as n^5, which is where the "2 to 3x more points" intuition
     comes from -- it is really a different power law, not a different constant.

  B  THE INFORMATION BOUND, the actual stochastic models.  Exact KL divergence
     between the Gaussian laws of y_1..y_n under two members of the SAME
     3-dimensional family:

         FLAT      alpha = (1, 0, 0)    roots {1,0,0}   -- this IS the parent
         VELOCITY  alpha = (2,-1, 0)    double unit root
         ACCEL     alpha = (3,-3, 1)    triple unit root
         ODE       the damped oscillator plus offset

     KL(N(0,S1) || N(0,S0)) = (tr(S0^-1 S1) - n + logdet S0 - logdet S1)/2.
     No simulation and no threshold until the last step.

     **The prior scale is part of the question, not a nuisance to be made
     diffuse.**  A first attempt used a diffuse P0 and got answers in the tens
     of millions of nats that moved with P0: correctly so, because if the
     initial velocity may be arbitrarily large then "is there a velocity?" is
     arbitrarily easy.  Conditioning on the first few points does not rescue it
     -- the divergence is in how the models extrapolate an unbounded initial
     velocity, not in the first points.  So the prior is specified in the
     DERIVATIVE basis, P0 = D diag(L^2, v^2, a^2) D^T, and parameterised by the
     same r = sigma / v as part A.  The level scale L stays diffuse and is
     provably harmless: every model here has a unit root, so alpha sums to 1 and
     a constant vector -- which is exactly D[:,0] = (1,1,1) -- is fixed by all
     four transitions alike.

  C  WHAT THE FILTER ACHIEVES.  From a cold start with the true parameters,
     how many steps until each derivative's posterior SD reaches its own
     steady-state value -- i.e. until the recursion has learned as much as it
     ever will.  (Comparing against the truth's own SD, which an earlier
     version did, is useless: the level has a unit root and wanders to an SD of
     ~550, so the filter "beats" it on the first point.)  The gap between C and
     B is implementation lag; the gap between B and zero is physics.

That FLAT is a member of the family is the point of the whole exercise: "no ODE
governance" is a hypothesis with a likelihood, not an absence of evidence.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ode-adaptive-filter", "output"))
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from odefilter import OdeFilter, Params, difference_matrix  # noqa: E402

RHO, TH = 0.9489, 0.3462
THRESH = (1.0, 3.0, 5.0)          # nats: ~3:1, ~20:1, ~150:1 odds


def alpha3(rho, th):
    c = 2.0 * rho * math.cos(th)
    return np.array([c + 1.0, -(rho * rho + c), rho * rho])


FAMILY = {
    "FLAT": np.array([1.0, 0.0, 0.0]),        # x_t = x_{t-1}: the parent
    "VELOCITY": np.array([2.0, -1.0, 0.0]),
    "ACCEL": np.array([3.0, -3.0, 1.0]),
    "ODE": alpha3(RHO, TH),
}


def companion(a):
    F = np.zeros((3, 3))
    F[0] = a
    F[1:, :-1] = np.eye(2)
    return F


# ----------------------------------------------------------------- part A
def nats_poly(n, deg, r):
    """Expected LLR for adding one polynomial degree, unit coefficient / r.

    deg = 1 tests a ramp against a constant; deg = 2 a quadratic against a
    ramp.  The added regressor is t^deg / deg!, so its coefficient is the
    deg-th derivative and `r` is noise SD per that derivative's per-step step.
    """
    if n <= deg + 1:
        return 0.0
    t = np.arange(n, dtype=float)
    X0 = np.column_stack([t ** k for k in range(deg)]) if deg else np.empty((n, 0))
    X0 = np.column_stack([np.ones(n), X0]) if deg else np.ones((n, 1))
    g = (t ** deg) / math.factorial(deg)
    resid = g - X0 @ np.linalg.lstsq(X0, g, rcond=None)[0]
    return float(np.dot(resid, resid) / (2.0 * r * r))


def first_n(fn, thresh, nmax=200):
    for n in range(2, nmax + 1):
        if fn(n) >= thresh:
            return n
    return None


# ----------------------------------------------------------------- part B
def prior_lag(L, v, acc):
    """P0 in lag coordinates from a prior stated on (level, velocity, accel).

    D is an involution, so deriv -> lag is D itself.
    """
    D = difference_matrix(3)
    return D @ np.diag([L * L, v * v, acc * acc]) @ D.T


def cov_y(a, n, Q, S2, P0):
    """Exact Cov(y_1..y_n) for z_t = F z_{t-1} + e1 w_t, y_t = z_t[0] + v_t."""
    F = companion(a)
    P = np.asarray(P0, dtype=float).copy()
    e1 = np.zeros(3)
    e1[0] = 1.0
    Pk = []
    for _ in range(n):                       # P at each time, after transition
        P = F @ P @ F.T
        P[0, 0] += Q
        Pk.append(P.copy())
    S = np.zeros((n, n))
    for s in range(n):
        Fp = np.eye(3)
        for t in range(s, n):
            S[s, t] = S[t, s] = float(e1 @ (Pk[s] @ Fp.T) @ e1)
            Fp = F @ Fp
    return S + S2 * np.eye(n)


def kl(S1, S0):
    """KL(N(0,S1) || N(0,S0)) in nats -- the expected LLR under model 1.

    Used only through `kl_cond`: the joint KL diverges with the prior, because
    a diffuse prior on the level means the models disagree infinitely about
    y_1 before any dynamics are involved.
    """
    n = S1.shape[0]
    L0 = np.linalg.cholesky(S0)
    L1 = np.linalg.cholesky(S1)
    M = np.linalg.solve(L0, L1)
    tr = float(np.sum(M * M))
    ld0 = 2.0 * float(np.sum(np.log(np.diag(L0))))
    ld1 = 2.0 * float(np.sum(np.log(np.diag(L1))))
    return 0.5 * (tr - n + ld0 - ld1)


# ----------------------------------------------------------------- part C
def achieved(a, Q, S2, nmax=200):
    """Posterior SD of each derivative from a cold start, vs its steady state.

    Comparing against the truth's own SD is useless: the level has a unit root
    and wanders to an SD of ~550, so the filter beats it on the first point.
    The learning question is how fast the posterior reaches the precision it
    will eventually have, so everything is normalised by the steady-state SD
    (taken at n = nmax).  The filter is given the TRUE parameters, so this is
    the recursion at its most favourable.
    """
    pr = Params(alpha=tuple(a), Q=Q, s2=S2)
    f = OdeFilter(pr, order=3)
    rng = np.random.default_rng(11)
    z = np.zeros(3)
    f.reset()
    post = []
    for t in range(nmax):
        z = np.concatenate([[a @ z + math.sqrt(Q) * rng.standard_normal()], z[:-1]])
        f.update(z[0] + math.sqrt(S2) * rng.standard_normal())
        _, P = f.derivatives()
        post.append(np.sqrt(np.maximum(np.diag(P), 0.0)))
    post = np.array(post)
    return post, post[-1].copy()          # steady-state SD


def main():
    Q, S2 = 1.0, 9.0

    # ---------------------------------------------------------------- A
    print("=== A. the information bound, deterministic signal ===")
    print("  n such that the expected log-likelihood ratio reaches a threshold.")
    print("  r = noise SD / one step of that derivative.\n")
    print(f"  {'r':>5} | " + " | ".join(
        f"{'vel ' + str(T) + ' nats':>13}" for T in THRESH)
        + " | " + " | ".join(f"{'acc ' + str(T) + ' nats':>13}" for T in THRESH))
    rowsA = []
    for r in (0.5, 1.0, 2.0, 3.0, 5.0):
        v = [first_n(lambda n, r=r: nats_poly(n, 1, r), T) for T in THRESH]
        ac = [first_n(lambda n, r=r: nats_poly(n, 2, r), T) for T in THRESH]
        rowsA.append(dict(r=r, vel=v, acc=ac))
        print(f"  {r:5.1f} | " + " | ".join(f"{str(z):>13}" for z in v)
              + " | " + " | ".join(f"{str(z):>13}" for z in ac))
    print("\n  Velocity nats grow as n^3, acceleration as n^5 -- the 'a few more")
    print("  points' intuition is a change of power law, not of constant.")

    # ---------------------------------------------------------------- B
    print("\n=== B. the information bound, the real stochastic models ===")
    print("  exact KL between members of one 3-d family, prior stated on the")
    print("  derivatives and parameterised by the same r = sigma / v as A.\n")
    NMAX = 40
    VEL, ACC = 1.0, 1.0 / 2.88     # the ODE's own accel:velocity SD ratio
    pairs = [("VELOCITY", "FLAT"), ("ACCEL", "VELOCITY"), ("ODE", "FLAT")]

    def curve(h1, h0, r, Qs, L):
        P0 = prior_lag(L, VEL, ACC)
        s2 = (r * VEL) ** 2
        return [kl(cov_y(FAMILY[h1], n, Qs, s2, P0),
                   cov_y(FAMILY[h0], n, Qs, s2, P0)) for n in range(2, NMAX + 1)]

    # sanity: with negligible process noise B must reproduce A
    cA = curve("VELOCITY", "FLAT", 2.0, 1e-9, 1e3)
    nA = first_n(lambda n: nats_poly(n, 1, 2.0), 3.0)
    nB = first_n(lambda n: cA[n - 2] if 2 <= n <= NMAX else 0.0, 3.0, NMAX)
    print(f"  check, Q -> 0, r = 2, 3 nats:  part A says n = {nA}, "
          f"part B says n = {nB}")
    for L in (1e2, 1e3, 1e4):
        c = curve("VELOCITY", "FLAT", 2.0, 1.0, L)
        print(f"    level prior L = {L:.0e}: nats at n=20 = {c[18]:.3f}")

    print(f"\n  {'comparison':>18} {'Q':>6} | " + " | ".join(
        f"{'r=' + str(r):>16}" for r in (1.0, 2.0, 5.0)))
    print(f"  {'':>18} {'':>6} | " + " | ".join(
        f"{'n@1 / n@3 / n@5':>16}" for _ in range(3)))
    rowsB = []
    for h1, h0 in pairs:
        for Qs in (1e-9, 1.0):
            cells, keep = [], {}
            for r in (1.0, 2.0, 5.0):
                c = curve(h1, h0, r, Qs, 1e3)
                keep[r] = c
                ns = [first_n(lambda n, c=c: c[n - 2] if 2 <= n <= NMAX else 0.0,
                              T, NMAX) for T in THRESH]
                cells.append(" / ".join(str(z) for z in ns))
            rowsB.append(dict(pair=f"{h1} vs {h0}", Q=Qs,
                              curves={str(k): v for k, v in keep.items()}))
            print(f"  {h1 + ' vs ' + h0:>18} {Qs:6.0e} | "
                  + " | ".join(f"{c:>16}" for c in cells))
    print("\n  Q -> 0 is the deterministic bound; Q = 1 is the real process,")
    print("  where the dynamics themselves are noisy and detection is slower.")

    # ---------------------------------------------------------------- C
    print("\n=== C. what the filter achieves from a cold start ===")
    post, ss = achieved(FAMILY["ODE"], Q, S2)
    print(f"  steady-state posterior SD: level {ss[0]:.3f}, "
          f"velocity {ss[1]:.3f}, acceleration {ss[2]:.3f}")
    print(f"  {'coordinate':>13} | {'n to 2x':>8} {'n to 1.5x':>10} "
          f"{'n to 1.1x':>10}   (of steady state)")
    rowsC = []
    for i, nm in enumerate(("level", "velocity", "acceleration")):
        ns = []
        for mult in (2.0, 1.5, 1.1):
            below = np.where(post[:, i] <= mult * ss[i])[0]
            ns.append(int(below[0]) + 1 if below.size else None)
        rowsC.append(dict(coord=nm, n_2x=ns[0], n_15x=ns[1], n_11x=ns[2],
                          steady_sd=float(ss[i]), sd=post[:60, i].tolist()))
        print(f"  {nm:>13} | {str(ns[0]):>8} {str(ns[1]):>10} {str(ns[2]):>10}")
    print("\n  This is the recursion at its most favourable -- true parameters,")
    print("  nothing left to estimate.  It is a floor on the filter, not on the")
    print("  problem; B is the floor on the problem.")

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.1))

    ax = axes[0]
    ns = np.arange(2, 61)
    for i, r in enumerate((0.5, 1.0, 2.0, 5.0)):
        ax.plot(ns, [nats_poly(n, 1, r) for n in ns], color=ts.SERIES[i],
                lw=1.8, label=f"velocity, r={r}")
        ax.plot(ns, [nats_poly(n, 2, r) for n in ns], color=ts.SERIES[i],
                lw=1.3, ls="--")
    for T in THRESH:
        ax.axhline(T, color=ts.INK, lw=0.9, ls=":", zorder=0)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim(1e-2, 1e4)
    ax.set_xlim(3, 60)
    ax.set_xticks([3, 5, 8, 12, 20, 40, 60])
    ax.set_xticklabels(["3", "5", "8", "12", "20", "40", "60"])
    ax.set_xlabel("measurements n")
    ax.set_ylabel("expected log-likelihood ratio (nats)")
    ax.set_title("A · The bound: solid velocity ($n^3$),\ndashed acceleration ($n^5$)")
    ax.legend(fontsize=7.5)
    ts.tidy(ax)

    ax = axes[1]
    for i, rec in enumerate([r for r in rowsB if r["Q"] == 1.0]):
        ax.plot(np.arange(2, NMAX + 1), rec["curves"]["2.0"],
                color=ts.SERIES[i], lw=1.8, label=rec["pair"])
    for T in THRESH:
        ax.axhline(T, color=ts.INK, lw=0.9, ls=":", zorder=0)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1e3)
    ax.set_xlabel("measurements n")
    ax.set_ylabel("expected LLR (nats)")
    ax.set_title("B · The real models, exact KL\n($Q=1$, $r=2$)")
    ax.legend(fontsize=7.5)
    ts.tidy(ax)

    ax = axes[2]
    for i, nm in enumerate(("level", "velocity", "acceleration")):
        ax.plot(np.arange(1, 61), post[:60, i] / ss[i],
                color=ts.SERIES[i], lw=1.8, label=nm)
    ax.axhline(1.0, color=ts.INK, lw=1.2, ls="--", zorder=0)
    ax.set_yscale("log")
    ax.set_xlim(1, 60)
    ax.set_xlabel("measurements since a cold start")
    ax.set_ylabel("posterior SD / its steady state")
    ax.set_title("C · What the filter achieves\n(1 = fully converged)")
    ax.legend(fontsize=8)
    ts.tidy(ax)
    ts.save(fig, os.path.join(HERE, "figures", "fig23-detection-latency.png"))

    with open(os.path.join(HERE, "figures", "ode035.json"), "w") as f:
        json.dump(dict(thresholds=list(THRESH), A=rowsA, B=rowsB, C=rowsC,
                       Q=Q, S2=S2, VEL=VEL, ACC=ACC, NMAX=NMAX,
                       check=dict(A=nA, B=nB)), f, indent=1)


if __name__ == "__main__":
    main()
