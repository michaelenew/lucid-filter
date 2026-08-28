"""wall-correspondence 0048 -- the item-2 blocker is an estimation
problem, and it has a name.

Their 0146 delivered 30x the throughput and the spin-2 correlator
did not move off the shuffle floor. So the blocker is not statistics.
Stated as physics it sounds hard -- "a connected correlator of a
composite operator at weak coupling". Stated as ENGINEERING it is a
standard estimation problem with a standard fix, and the filter is
where that fix is obvious.

THE PROBLEM. We estimate E[XY] - E[X]E[Y] where X and Y are products
of many noisy factors. The naive estimator samples every factor, so
its variance carries the fluctuation of every factor -- including the
ones that contribute NOTHING to the long-distance signal.

THE FIX. Rao-Blackwell: replace any sampled quantity by its
conditional expectation given the rest. Var(E[X|Z]) = Var(X) -
E[Var(X|Z)], so it never increases variance and never biases the
estimate. In lattice language this is 'multihit' or 'link
integration'; in filter language it is just refusing to sample what
you can integrate.

  s1  THE PROBLEM, REPRODUCED. A product of k noisy factors with a
      weak correlation buried in it. The naive estimator's variance,
      and how fast it grows with k.
  s2  THE FIX, AND WHY IT BEATS sqrt(n). Replacing each factor by
      its conditional mean removes that factor's variance
      ENTIRELY, so the gain COMPOUNDS: (1 + v/m^2)^k. Exponential
      in the number of integrated factors, against sqrt(n) for
      brute force. Measured.
  s3  AND IT PAYS EVEN FOR STIFF LINKS. I expected the gain to
      vanish for nearly deterministic factors and it does not: the
      baseline variance compounds over all 2k factors while the
      reduced estimator keeps 2(k-j), so the ratio stays large --
      ~8x even at v/m^2 = 0.005. The gain comes from the LENGTH of
      the operator, not the noisiness of each link, which is why
      it suits their case.
  s4  THE PORT SPEC.
"""

import numpy as np

rng = np.random.default_rng(48)


def make(k, n, m=1.0, v=0.35, sig=0.02):
    """Two composite observables X, Y, each a product of k noisy
    factors, sharing a weak common signal `sig`."""
    s = sig * rng.standard_normal(n)                   # the signal
    fx = m + np.sqrt(v) * rng.standard_normal((n, k))
    fy = m + np.sqrt(v) * rng.standard_normal((n, k))
    fx[:, 0] += s
    fy[:, 0] += s
    return fx, fy, s


def naive(fx, fy):
    X, Y = fx.prod(1), fy.prod(1)
    return X * Y - X.mean() * Y.mean()


def rao_blackwell(fx, fy, nint, m=1.0):
    """integrate out the last `nint` factors: replace each by its
    conditional mean, which here is m."""
    X = fx[:, :fx.shape[1] - nint].prod(1) * m ** nint
    Y = fy[:, :fy.shape[1] - nint].prod(1) * m ** nint
    return X * Y - X.mean() * Y.mean()


def s1_problem():
    print("== s1: the problem, reproduced ==")
    n = 200000
    print("  X and Y are products of k noisy factors sharing a weak "
          "signal in factor 0.")
    print("     k    estimate      s.e.        |est|/s.e.")
    for k in (1, 2, 4, 6, 8):
        e = naive(*make(k, n)[:2])
        se = e.std() / np.sqrt(n)
        print(f"    {k:2d}   {e.mean():+.6f}   {se:.6f}    "
              f"{abs(e.mean()) / se:7.2f}")
    print("  the signal is the same at every k; the ERROR grows "
          "with k, because every")
    print("  extra factor adds its own fluctuation to the "
          "estimator and none of it")
    print("  carries signal. That is exactly their situation: a "
          "site operator built from")
    print("  six plaquettes of four links each\n")


def s2_the_fix():
    print("== s2: the fix, and why it beats sqrt(n) ==")
    n, k, m, v = 200000, 8, 1.0, 0.35
    fx, fy, _ = make(k, n, m, v)
    base = naive(fx, fy)
    b_se = base.std() / np.sqrt(n)
    print(f"  k = {k}, naive s.e. = {b_se:.6f}")
    def predicted(j):
        """Var(XY) = (mu^2+sig^2)^2 - mu^4 with sig^2 = (1+v)^k - 1,
        so the reduction is a ratio of those, not (1+v)^j."""
        q = 1 + v / m ** 2
        return (q ** (2 * k) - 1) / (q ** (2 * (k - j)) - 1)
    print("     factors integrated out   s.e.       variance "
          "reduction   predicted")
    for j in (0, 2, 4, 6, 7):
        e = rao_blackwell(fx, fy, j, m)
        se = e.std() / np.sqrt(n)
        red = (b_se / se) ** 2
        print(f"        {j:2d}                  {se:.6f}   "
              f"{red:10.2f}x    {predicted(j):10.2f}x")
    e7 = rao_blackwell(fx, fy, 7, m)
    red7 = (b_se / (e7.std() / np.sqrt(n))) ** 2
    assert red7 > 3
    print()
    print(f"  the reduction COMPOUNDS. The right law is a RATIO "
          f"of product variances,")
    print("  [(1+v)^2k - 1] / [(1+v)^2(k-j) - 1], not the naive "
          "(1+v)^j -- a first pass here")
    print("  used the naive form and under-predicted by 15x.")
    print(f"  At j = 7 that is {red7:.0f}x in variance, which brute "
          f"force would need {red7:.0f}x")
    print("  the samples to match. AND THE ESTIMATE IS UNCHANGED: "
          "Rao-Blackwell is exact,")
    print("  not an approximation.")
    xs = [rao_blackwell(fx, fy, j, m).mean() for j in (0, 4, 7)]
    print(f"    estimates at j = 0, 4, 7: " + ", ".join(
        f"{x:+.6f}" for x in xs) + "  -- same signal")
    assert max(xs) - min(xs) < 4 * b_se
    print()
    print("  THAT is why multilevel beats more sweeps: sqrt(n) "
          "against a product over")
    print("  integrated factors\n")


def s3_it_pays_even_for_stiff_links():
    print("== s3: and it pays even when the links are stiff ==")
    n, k = 200000, 8
    print("  I expected the gain to vanish for nearly "
          "deterministic factors. It does not,")
    print("  and the reason is the ratio law: the BASELINE "
          "compounds over all 2k factors")
    print("  while the reduced estimator keeps only 2(k-j), so the "
          "ratio stays large even")
    print("  for tiny v.")
    print("     v/m^2     reduction at j = 7   predicted   verdict")
    out = []
    for v in (0.35, 0.05, 0.005):
        fx, fy, _ = make(k, n, 1.0, v)
        b = naive(fx, fy).std()
        r = rao_blackwell(fx, fy, 7, 1.0).std()
        red = (b / r) ** 2
        q = 1 + v
        pred = (q ** (2 * k) - 1) / (q ** 2 - 1)
        out.append(red)
        print(f"     {v:.3f}          {red:8.2f}x        "
              f"{pred:8.2f}x    "
              f"{'worth it' if red > 3 else 'marginal'}")
    assert min(out) > 3
    print()
    print("  THAT MATTERS FOR THE PORT. Their links are stiff at "
          "kappa ~ 17, and I was")
    print("  about to tell them to measure the conditional variance "
          "first in case the fix")
    print("  did not pay. It pays anyway -- roughly 8x in variance "
          "even at v/m^2 = 0.005,")
    print("  because the gain comes from the LENGTH of the "
          "operator, not the noisiness of")
    print("  each link. Their site operator is six plaquettes of "
          "four links: long\n")


def s4_port():
    print("== s4: the port spec ==")
    print("  For each link U in a measured plaquette, replace U by "
          "its CONDITIONAL MEAN")
    print("  given everything else -- which for a class-function "
          "weight and staple S is")
    print("      Ubar = c(|S|) * Shat^dagger ,")
    print("  a one-dimensional integral, computed once per link per "
          "measurement. For their")
    print("  Spin(4) weight the two factors couple through the "
          "2-D table, so integrate one")
    print("  factor at a time holding the other: still 1-D "
          "quadrature.")
    print()
    print("  AND s3 says do NOT gate it on the link stiffness. "
          "The gain comes from the")
    print("  LENGTH of the operator, not the noisiness of each "
          "link, so it pays at their")
    print("  coupling: ~8x in variance even for near-deterministic "
          "links, and their site")
    print("  operator is six plaquettes of four links.")
    print()
    print("  This is the whole move: STOP SAMPLING WHAT CAN BE "
          "INTEGRATED. It is not a")
    print("  physics idea, it is an estimator idea, and the filter "
          "has been making it since")
    print("  its first stone -- a posterior mean is always a better "
          "estimator than a draw\n")


if __name__ == "__main__":
    s1_problem()
    s2_the_fix()
    s3_it_pays_even_for_stiff_links()
    s4_port()
    print("all assertions passed")
