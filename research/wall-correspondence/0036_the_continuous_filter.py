"""wall-correspondence 0036 -- the continuous filter: what a
continuum limit IS, stated prequentially, and one gap in 0030.

The sibling's last open conjunct (their 0127) is CONTINUITY. Their
version is heavy: does a lattice gauge measure have a continuum
limit. The filter's version is lighter and, per the isomorphism
programme, is where to solve it first: replace a discrete transition
MATRIX with a continuous transition INTEGRAL TRANSFORM, keep the
inputs discrete, and ask what has to be true for the refinement to
converge.

Half of this is already done and was not labelled as continuity.
0030 proved a record's dynamics embeds in continuous time iff its
transfer operator is positive -- T = exp(-H) -- which is exactly
"the TIME direction is continuous underneath". This module adds the
gap in that result, does the STATE direction, and then states the
criterion the physics actually needs.

  s1  A REAL GENERATOR IS NOT A PROBABILISTIC ONE (0030 refined).
      0030 stopped at "a real H exists". For a record to have a
      genuine history in between, H must also be a RATE matrix --
      off-diagonals of -H nonnegative. Measured on random
      3-state records whose transfer operator already passes
      0030's test: a large fraction have a real logarithm that is
      NOT a rate matrix. Those records have continuous time in the
      linear-algebra sense and no probabilistic history in between.
      So "counting buys time" buys the generator, not yet the
      history; the extra condition is named and measured here.
  s2  THE STATE DIRECTION: MATRIX -> INTEGRAL TRANSFORM. Discretise
      an Ornstein-Uhlenbeck record's state at spacing h. The
      transition matrix's generator converges to the DIFFERENTIAL
      operator (sigma^2/2) d2 - theta x d at O(h^2), measured, and
      its kernel is local (off-diagonal weight falls off as a
      Gaussian in |i-j|/h). The integral transform has a local
      generator, which is what makes the limit a field theory
      rather than something nonlocal.
  s3  THE CRITERION, PREQUENTIALLY. Fix the physical process and
      the observation times; refine the model's resolution. THE
      CONTINUUM LIMIT EXISTS IFF THE PREQUENTIAL CODE LENGTH PER
      UNIT PHYSICAL TIME CONVERGES TO A NONTRIVIAL LIMIT UNDER
      REFINEMENT. Measured -- and the measurement corrected the
      expectation: a bad refinement does NOT diverge. Holding the
      dynamics fixed in GRID units converges perfectly well, to
      WHITE NOISE, because the physical correlation length goes to
      zero. Both regimes converge; they converge to different
      limits, and the gap is 0.888 nats/observation. So the
      criterion is not convergence but convergence to a nontrivial
      limit, and the diagnostic is the code-length gap. What
      separates them is exactly xi/a -> infinity -- a critical
      point on the physics side, and triviality is the real
      failure mode of a lattice theory off criticality.
"""

import numpy as np
from scipy.linalg import expm, logm

rng = np.random.default_rng(36)


# ----------------------------------------------------------------
def s1_generator_vs_history():
    print("== s1: a real generator is not a probabilistic one ==")
    print("  0030: a record embeds in continuous time iff T = "
          "exp(-H) for real H.")
    print("  For a genuine HISTORY in between, -H must also be a "
          "rate matrix")
    print("  (off-diagonals >= 0). These are not the same condition.")
    tried = passed_0030 = rate = 0
    bad = None
    while tried < 6000:
        tried += 1
        T = rng.dirichlet(np.ones(3) * 1.2, size=3)
        ev = np.linalg.eigvals(T)
        if np.max(np.abs(ev.imag)) > 1e-9 or np.min(ev.real) <= 1e-9:
            continue                       # no real principal log
        passed_0030 += 1
        Q = np.real(logm(T))               # the would-be generator
        off = Q - np.diag(np.diag(Q))
        if off.min() > -1e-9:
            rate += 1
        elif bad is None and off.min() < -0.02:
            bad = (T, Q)
    frac = rate / passed_0030
    print(f"  random 3-state records: {passed_0030} pass 0030's "
          f"test, of which {rate} ({100 * frac:.1f}%)")
    print("  have a legitimate rate matrix")
    assert 0.0 < frac < 1.0
    T, Q = bad
    off = (Q - np.diag(np.diag(Q))).min()
    print(f"  a counterexample: most negative off-diagonal rate = "
          f"{off:+.4f}")
    print(f"    exp(Q) reproduces T to "
          f"{np.abs(expm(Q) - T).max():.1e}, so the generator is "
          f"real and exact --")
    print("    but the half-step exp(Q/2) has a negative entry "
          f"({expm(Q / 2).min():+.4f}):")
    print("    there is no probability distribution for 'what "
          "happened halfway'")
    assert expm(Q / 2).min() < 0
    print("  SO: counting buys the generator (0030), not yet the "
          "history. The physics")
    print("  side's 'counting buys time' should carry the same "
          "asterisk\n")


# ----------------------------------------------------------------
def ou_matrix(h, theta, sigma, dt, xmax):
    """Transition matrix of an OU record on a grid of spacing h."""
    x = np.arange(-xmax, xmax + 1e-12, h)
    m = x * np.exp(-theta * dt)
    v = sigma ** 2 * (1 - np.exp(-2 * theta * dt)) / (2 * theta)
    D = x[None, :] - m[:, None]
    K = np.exp(-D ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)
    K /= K.sum(1, keepdims=True)
    return x, K


def s2_state_direction():
    print("== s2: the state direction -- matrix to integral "
          "transform ==")
    theta, sigma, xmax = 1.0, 1.0, 6.0
    print("  OU record; generator should converge to "
          "(sigma^2/2) d2 - theta x d.")
    print("  Diffusive refinement dt = h^2/2, so the step error "
          "shrinks with the grid:")
    print("    h        dt        max error vs the differential "
          "operator   ratio")
    prev = None
    for h in (0.4, 0.2, 0.1, 0.05):
        dt = h ** 2 / 2
        x, K = ou_matrix(h, theta, sigma, dt, xmax)
        Q = (K - np.eye(len(x))) / dt          # generator estimate
        f = np.exp(-x ** 2 / 4.0)              # a smooth test state
        d1 = np.gradient(f, h)
        d2 = np.gradient(d1, h)
        exact = sigma ** 2 / 2 * d2 - theta * x * d1
        sel = np.abs(x) < 3.0
        err = float(np.abs((Q @ f - exact)[sel]).max()
                    / np.abs(exact[sel]).max())
        r = "" if prev is None else f"{prev / err:.2f}"
        print(f"   {h:.2f}    {dt:.5f}    {err:.5f}              "
              f"                 {r}")
        prev = err
    assert err < 0.02
    print("  the error falls with h: the transition INTEGRAL "
          "TRANSFORM's generator is the")
    print("  differential operator. And it is LOCAL --")
    x, K = ou_matrix(0.05, theta, sigma, 0.05 ** 2 / 2, xmax)
    row = K[len(x) // 2]
    c = len(x) // 2
    w = [float(row[c - k:c + k + 1].sum()) for k in (2, 5, 10, 20)]
    print("    fraction of a row's weight within k sites: "
          + ", ".join(f"k={k}: {v:.4f}"
                      for k, v in zip((2, 5, 10, 20), w)))
    assert w[-1] > 0.999
    print("  -- so the limit is a field theory, not a nonlocal "
          "one\n")


# ----------------------------------------------------------------
def s3_the_criterion():
    print("== s3: the criterion, prequentially ==")
    print("  Fix the physical process and the observation times; "
          "refine the model's")
    print("  resolution. Two regimes, same refinement.")
    T_PHYS, NOBS = 4.0, 64
    theta_phys, sigma_phys = 1.0, 1.0
    obs_noise = 0.3
    # one ground-truth path, densely sampled, observed at fixed
    # physical times
    fine = 8192
    dtf = T_PHYS / fine
    xs = np.zeros(fine + 1)
    for t in range(fine):
        xs[t + 1] = (xs[t] * np.exp(-theta_phys * dtf)
                     + rng.normal() * sigma_phys
                     * np.sqrt((1 - np.exp(-2 * theta_phys * dtf))
                               / (2 * theta_phys)))
    idx = np.linspace(0, fine, NOBS + 1).astype(int)[1:]
    y = xs[idx] + obs_noise * rng.standard_normal(NOBS)

    def code_length(nsteps, rescale):
        """Kalman filter at nsteps per unit physical time."""
        dt = T_PHYS / nsteps
        if rescale:                    # hold physical xi fixed
            a, q = np.exp(-theta_phys * dt), sigma_phys ** 2 * (
                1 - np.exp(-2 * theta_phys * dt)) / (2 * theta_phys)
        else:                          # hold the dynamics fixed in
            a, q = np.exp(-1.0), 0.5   # GRID units
        m, P = 0.0, sigma_phys ** 2 / (2 * theta_phys)
        per = nsteps // NOBS                    # exact, by design
        step_of = (np.arange(NOBS) + 1) * per
        tot, j = 0.0, 0
        for s in range(1, nsteps + 1):
            m, P = a * m, a * a * P + q
            while j < NOBS and step_of[j] == s:
                S = P + obs_noise ** 2
                r = y[j] - m
                tot += 0.5 * (np.log(2 * np.pi * S) + r * r / S)
                K = P / S
                m, P = m + K * r, (1 - K) * P
                j += 1
        return tot / NOBS
    print("    steps          fixed in GRID units    rescaled to "
          "fixed physical xi")
    grid, resc = [], []
    for n in (64, 128, 256, 512, 1024, 2048):
        g = code_length(n, False)
        r = code_length(n, True)
        grid.append(g)
        resc.append(r)
        print(f"     {n:4d}          {g:8.4f}              "
              f"{r:8.4f}")
    dg = abs(grid[-1] - grid[-2])
    dr = abs(resc[-1] - resc[-2])
    gap = grid[-1] - resc[-1]
    print(f"  last-step change:  grid-fixed {dg:.5f}   rescaled "
          f"{dr:.5f}    -- BOTH converge")
    print(f"  but they converge to DIFFERENT limits: gap = "
          f"{gap:+.4f} nats/observation")
    assert dr < 0.01 and dg < 0.05 and gap > 0.05
    print("  This is the honest shape of the failure, and it is "
          "the physics' shape too:")
    print("  a bad refinement does not diverge, it converges to a "
          "TRIVIAL limit. Holding")
    print("  the dynamics fixed per step sends the physical "
          "correlation length to zero,")
    print("  so the limit is white noise -- a perfectly convergent, "
          "perfectly useless")
    print("  theory. The criterion is convergence to a NONTRIVIAL "
          "limit, and the")
    print("  diagnostic is the code-length gap. What separates "
          "them is xi/a -> infinity.\n")


def s4_the_port():
    print("== s4: the port ==")
    print("  THE CONTINUUM LIMIT EXISTS IFF THE PREQUENTIAL CODE "
          "LENGTH PER UNIT PHYSICAL")
    print("  TIME CONVERGES UNDER REFINEMENT. That is renormalisa"
          "bility as an operational")
    print("  statement, in the program's own currency, and it needs "
          "no continuum manifold")
    print("  to state -- only a sequence of discrete models and a "
          "score.")
    print()
    print("  What it demands of the physics: a nontrivial limit "
          "needs xi/a -> infinity.")
    print("  A lattice theory with a FIXED coupling and no dial has "
          "exactly two ways to")
    print("  get it -- sit at a critical point by accident, or have "
          "the coupling run so")
    print("  that xi/a grows on its own. The second is asymptotic "
          "freedom, and it is what")
    print("  their 0115 measures.")
    print()
    print("  And s1's asterisk ports too: their 'counting buys "
          "time' gives a generator,")
    print("  not automatically a probabilistic history. The "
          "transfer operator being")
    print("  positive is necessary; whether the in-between is a "
          "STATE is a further")
    print("  question, and on their side it is the question of "
          "whether the Osterwalder-")
    print("  Schrader reconstruction's Hilbert space carries a "
          "positive measure at")
    print("  intermediate times\n")


if __name__ == "__main__":
    s1_generator_vs_history()
    s2_state_direction()
    s3_the_criterion()
    s4_the_port()
    print("all assertions passed")
