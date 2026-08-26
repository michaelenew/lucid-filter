"""wall-correspondence 0045 -- the multiplicity obstruction, in
filter terms, reproduced and then resolved.

Their 0139 is the program's criticality-1 obstruction: the
amplitude's multiplicities come from binning a frame-pair quantity
at a scale s0 nobody fixed, and across s0 the hierarchy swings
10^12. 0043 reduced the coupling to one scalar, kappa =
(8/3)<j(j+1)>, so s0's violence is that it MOVES THE MEAN CASIMIR.

Five stages, in order.

  s1  THE INCOMPATIBILITY. The filter has no object for "a record
      about a PAIR, whose content is the pair's joint magnitude".
      That is what their bivector |B| = |a ^ b| is. Built here: the
      wedge magnitude IS the square root of the determinant of the
      pair's joint precision -- the INFORMATION VOLUME of a record
      pair, as against the information either record carries alone.
      Verified numerically, which is what makes the port legitimate
      rather than a metaphor.
  s2  THE VOLATILITY, REPRODUCED. Bin that information volume at
      width s0, call each bin a sector, give sector j the
      forgetting rate j(j+1) that 0043 measured, and read off the
      mean. The filter reproduces their swing -- same mechanism,
      same orders of magnitude, no geometry anywhere.
  s3  THE RESOLUTION. Equal-width binning is not what a filter
      does. A record spends its capacity to carry the most
      information it can, and at a fixed number of levels that
      means EQUIPROBABLE bins. Equiprobable bins have EQUAL
      multiplicities -- so the profile is FLAT, uniquely, and s0 is
      not a free parameter at all. It was the wrong quantiser.
  s4  THE CLOSED FORM. With flat multiplicities over M sectors the
      mean Casimir collapses to
          kappa = (M + 2)(M - 1) / 3
      exactly. The coupling is then a function of the SECTOR COUNT
      alone -- which is the level. The s0 freedom is gone.
  s5  WHAT REMAINS. One record or two: their unrecorded
      Spin(4) -> SU(2) step. Computed here in closed form too, so
      the whole obstruction reduces to a single structural question
      with two candidate answers rather than a continuum of them.
"""

import numpy as np

rng = np.random.default_rng(45)


# ----------------------------------------------------------------
def s1_the_wedge_is_information_volume():
    print("== s1: the incompatibility, and the object that fixes "
          "it ==")
    print("  Their multiplicities count frame PAIRS by |a ^ b|. The "
          "filter has no object for")
    print("  'a record about a pair, whose content is the pair's "
          "joint magnitude'. It does")
    print("  have one once you look: two records with directions a "
          "and b contribute a joint")
    print("  precision J = a a' + b b', and")
    print("      sqrt(det J) = |a| |b| |sin(angle)| = |a ^ b| .")
    print("  So THE WEDGE IS THE INFORMATION VOLUME OF A RECORD "
          "PAIR. Verified:")
    print("     trial   |a ^ b|      sqrt(det J)     diff")
    for k in range(4):
        a = rng.standard_normal(4)
        b = rng.standard_normal(4)
        J = np.outer(a, a) + np.outer(b, b)
        # the pair spans a plane; det on that plane
        Q, _ = np.linalg.qr(np.stack([a, b], 1))
        J2 = Q.T @ J @ Q
        wedge = np.sqrt(max(np.dot(a, a) * np.dot(b, b)
                            - np.dot(a, b) ** 2, 0))
        print(f"       {k}     {wedge:.6f}     "
              f"{np.sqrt(max(np.linalg.det(J2), 0)):.6f}     "
              f"{abs(wedge - np.sqrt(max(np.linalg.det(J2), 0))):.1e}")
        assert abs(wedge - np.sqrt(max(np.linalg.det(J2), 0))) < 1e-9
    print("  exact. A wedge is not a geometric flourish -- it is "
          "how much a PAIR of records")
    print("  jointly pins down, over and above what either pins "
          "alone. The filter can carry")
    print("  their construction after all\n")


def info_volumes(n=2_000_000):
    """|a ^ b| for record pairs drawn from the Gaussian measure."""
    a = rng.standard_normal((n, 4))
    b = rng.standard_normal((n, 4))
    aa = (a * a).sum(1)
    bb = (b * b).sum(1)
    ab = (a * b).sum(1)
    return np.sqrt(np.maximum(aa * bb - ab * ab, 0))


def kappa_from(mult):
    m = np.asarray(mult, float)
    n = np.arange(1, len(m) + 1)
    if m.sum() <= 0:
        return float("nan")
    return float((2 / 3) * np.sum(m * n * (n * n - 1))
                 / np.sum(m * n))


B0, B1 = 11 / (24 * np.pi ** 2), 17 / (96 * np.pi ** 4)


def xi(beta):
    if not np.isfinite(beta) or beta <= 0:
        return float("nan")
    g2 = 4.0 / beta
    return 1.0 / ((B0 * g2) ** (-B1 / (2 * B0 ** 2))
                  * np.exp(-1 / (2 * B0 * g2)))


def s2_reproduce(s):
    print("== s2: the volatility, reproduced in filter world ==")
    print("  bin the information volume at width s0; each bin is a "
          "sector; sector j forgets")
    print("  at rate j(j+1) (0043 s2, measured). Read the mean "
          "forgetting rate:")
    print("     s0      multiplicities (first 6)          kappa    "
          " induced xi")
    M = 6
    ks = []
    for s0 in (0.5, 0.75, 1.0, 1.5, 2.0):
        idx = np.rint(s / s0).astype(np.int64)
        m = np.array([float((idx == j).sum()) for j in range(M)])
        m = m / max(m.max(), 1)
        k = kappa_from(m)
        ks.append(k)
        print(f"    {s0:.2f}   " + " ".join(f"{v:.2f}" for v in m)
              + f"    {k:7.3f}   {xi(k):.2e}")
    print(f"  kappa {min(ks):.2f} to {max(ks):.2f}; xi "
          f"{xi(min(ks)):.1e} to {xi(max(ks)):.1e}"
          f"  -- a factor {xi(max(ks)) / xi(min(ks)):.0e}")
    assert xi(max(ks)) / xi(min(ks)) > 1e6
    print("  THE FILTER REPRODUCES THE OBSTRUCTION EXACTLY, with no "
          "geometry in it: it is a")
    print("  quantiser-width problem about a pair-information "
          "record\n")
    return M


def s3_resolution(s, M):
    print("== s3: the resolution -- equal width is the wrong "
          "quantiser ==")
    print("  A filter does not bin at an arbitrary width. It spends "
          "its capacity to carry")
    print("  the most information it can, and at a fixed number of "
          "levels the")
    print("  entropy-maximising quantiser is EQUIPROBABLE. "
          "Equiprobable bins carry equal")
    print("  multiplicities -- so the profile is FLAT, uniquely.")
    print("     scheme            multiplicities            "
          "entropy (nats)   kappa")
    ent = lambda m: float(-np.sum((m / m.sum())
                                  * np.log(np.maximum(m / m.sum(),
                                                      1e-300))))
    for lbl, s0 in (("equal width s0=0.5", 0.5),
                    ("equal width s0=1.0", 1.0),
                    ("equal width s0=2.0", 2.0)):
        idx = np.rint(s / s0).astype(np.int64)
        m = np.array([float((idx == j).sum()) for j in range(M)])
        print(f"    {lbl:19s} " + " ".join(f"{v / m.max():.2f}"
                                           for v in m)
              + f"     {ent(m):.4f}      {kappa_from(m):7.3f}")
    edges = np.quantile(s, np.linspace(0, 1, M + 1)[1:-1])
    idx = np.digitize(s, edges)
    m = np.array([float((idx == j).sum()) for j in range(M)])
    print(f"    {'EQUIPROBABLE':19s} " + " ".join(f"{v / m.max():.2f}"
                                                  for v in m)
          + f"     {ent(m):.4f}      {kappa_from(m):7.3f}")
    print(f"  the equiprobable scheme attains ln M = {np.log(M):.4f} "
          f"nats, the maximum available")
    print("  at M levels; every equal-width scheme falls short. And "
          "its multiplicities are")
    print("  flat to sampling error.")
    assert abs(ent(m) - np.log(M)) < 1e-3
    assert abs(kappa_from(m) - kappa_from([1] * M)) < 0.05
    print()
    print("  SO s0 WAS NEVER A FREE PARAMETER -- it was a wrong "
          "quantiser. Fix the number of")
    print("  levels and the capacity-achieving profile is FLAT, "
          "which is the profile their")
    print("  0091 used. Flat counting was right, for a reason "
          "nobody had given\n")


def s4_closed_form():
    print("== s4: the closed form ==")
    print("  with flat multiplicities over M sectors the mean "
          "Casimir collapses:")
    print("      kappa = (2/3) sum n(n^2-1) / sum n = (M+2)(M-1)/3")
    print("     M     numeric      (M+2)(M-1)/3    xi/a")
    for M in (3, 4, 5, 6, 7, 8, 10):
        a = kappa_from([1] * M)
        b = (M + 2) * (M - 1) / 3
        print(f"    {M:2d}   {a:9.4f}     {b:9.4f}    {xi(b):.2e}")
        assert abs(a - b) < 1e-9
    print()
    print("  exact. THE COUPLING IS A FUNCTION OF THE SECTOR COUNT "
          "ALONE. Their level N")
    print("  gives M = N + 1 sectors, so")
    print("      kappa = N(N+3)/3     and     band = 2N+1")
    for N in (3, 5, 7, 13):
        print(f"    N = {N:2d}:  kappa = {N * (N + 3) / 3:8.4f},  "
              f"band = {2 * N + 1:3d},  xi/a = "
              f"{xi(N * (N + 3) / 3):.2e}")
    assert abs(5 * 8 / 3 - kappa_from([1] * 6)) < 1e-9
    print()
    print("  THE 10^12 FREEDOM IS GONE. What was a family indexed "
          "by a bin width is now a")
    print("  single curve indexed by the level -- and the level is "
          "the program's one")
    print("  remaining scale, which 0044 showed is fixed by one "
          "measurement\n")


def s5_what_remains():
    print("== s5: what remains ==")
    print("  One question, not a continuum: ONE RECORD OR TWO. "
          "Fusing a self-dual and an")
    print("  anti-self-dual stream gives chi_n^2 = chi_1 + chi_3 + "
          "... + chi_{2n-1}, so the")
    print("  multiplicities become c'_k = #{n <= M : 2n-1 >= k} for "
          "odd k:")
    print("     M     one record      two records fused    ratio")
    for M in (4, 6, 8):
        one = kappa_from([1] * M)
        c = np.zeros(2 * M)
        for n in range(1, M + 1):
            for k in range(1, 2 * n, 2):
                c[k - 1] += 1
        two = kappa_from(c)
        print(f"    {M:2d}   {one:9.4f}      {two:9.4f}       "
              f"{two / one:.3f}x")
    print()
    print("  So the obstruction has gone from 'a free function' to "
          "'a binary structural")
    print("  question with two computable answers'. That is what "
          "the port has to settle,")
    print("  and it is a question about how many independent frame "
          "records the world writes")
    print("  per event -- answerable, unlike a bin width\n")


if __name__ == "__main__":
    s1_the_wedge_is_information_volume()
    s = info_volumes()
    M = s2_reproduce(s)
    s3_resolution(s, M)
    s4_closed_form()
    s5_what_remains()
    print("all assertions passed")
