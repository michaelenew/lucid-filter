"""wall-correspondence 0047 -- the factor 20: is a cut the same
measurement as a deformation?

Their standing criticality-5 item. Two routes to G disagree by
pi p / alpha = 20.11, field-count independent:

  - the ENTANGLEMENT route reads alpha, the information a boundary
    record reveals about the other side, per unit boundary;
  - the INDUCED route reads p, how stiffly the code length resists
    a smooth deformation.

The program has treated their disagreement as a defect. This asks
the prior question: ARE THEY THE SAME QUANTITY AT ALL? If they are,
their ratio must be a pure number, the same for every field. If it
moves when the field moves, they are two different measurements and
expecting them to agree was the error.

The test is cheap and decisive: give the field a MASS and watch the
ratio.

  s1  BOTH QUANTITIES, ON ONE FIELD WITH ONE REGULATOR. A Gaussian
      field on a lattice: the mutual information across a straight
      cut per unit boundary, and the stiffness of the log-partition
      function against a smooth scale deformation.
  s2  THE MASS SCAN. If cut and deformation are the same
      measurement the ratio is constant.
  s3  WHAT THAT SETTLES.
"""

import numpy as np

L = 26
rng = np.random.default_rng(47)


def precision(m2, lam=None):
    """K = -Laplacian + m^2 on an LxL periodic lattice, with
    optional per-link weights exp(lam_link)."""
    n = L * L
    idx = np.arange(n).reshape(L, L)
    K = np.zeros((n, n))
    for ax in (0, 1):
        nb = np.roll(idx, -1, ax)
        w = np.ones((L, L)) if lam is None else np.exp(
            0.5 * (lam + np.roll(lam, -1, ax)))
        for i in range(L):
            for j in range(L):
                a, b, ww = idx[i, j], nb[i, j], w[i, j]
                K[a, a] += ww
                K[b, b] += ww
                K[a, b] -= ww
                K[b, a] -= ww
    K[np.arange(n), np.arange(n)] += m2
    return K


def mutual_info_across_cut(m2):
    """I(A;B) for the two halves of the lattice, per unit boundary."""
    K = precision(m2)
    C = np.linalg.inv(K)
    idx = np.arange(L * L).reshape(L, L)
    A = idx[:, : L // 2].ravel()
    B = idx[:, L // 2:].ravel()
    sa = np.linalg.slogdet(C[np.ix_(A, A)])[1]
    sb = np.linalg.slogdet(C[np.ix_(B, B)])[1]
    st = np.linalg.slogdet(C)[1]
    I = 0.5 * (sa + sb - st)
    return float(I / (2 * L))        # two cuts on a torus


def stiffness(m2, k=2):
    """second derivative of (1/2) ln det K under a smooth scale
    deformation lam = eps cos(k x), per unit volume per khat^2."""
    x = np.arange(L)
    lam = np.cos(2 * np.pi * k * x / L)[None, :] * np.ones((L, 1))

    def G(eps):
        K = precision(m2, lam=eps * lam)
        return 0.5 * np.linalg.slogdet(K)[1]
    e = 0.02
    d2 = (G(e) + G(-e) - 2 * G(0.0)) / e ** 2
    khat2 = 2 * (1 - np.cos(2 * np.pi * k / L))
    return float(2 * d2 / (L * L) / khat2)


def s1_both():
    print("== s1: both quantities, one field, one regulator ==")
    m2 = 0.05
    a = mutual_info_across_cut(m2)
    p = stiffness(m2)
    print(f"  m^2 = {m2}")
    print(f"    alpha  (mutual information per unit boundary) = "
          f"{a:.6f}")
    print(f"    p      (stiffness per unit volume per khat^2)  = "
          f"{p:.6f}")
    print(f"    pi p / alpha = {np.pi * p / a:.4f}")
    print("  both computed on the same lattice, same field, same "
          "cutoff -- no regulator")
    print("  mismatch anywhere\n")


def s2_mass_scan():
    print("== s2: the mass scan ==")
    print("  if a cut and a deformation are the same measurement, "
          "this ratio is constant.")
    print("     m^2       alpha       p          pi p / alpha")
    rows = []
    for m2 in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        a = mutual_info_across_cut(m2)
        p = stiffness(m2)
        r = np.pi * p / a
        rows.append((m2, a, p, r))
        print(f"    {m2:5.2f}   {a:9.6f}   {p:9.6f}    {r:9.4f}")
    rs = [r for _, _, _, r in rows]
    spread = max(rs) / min(rs)
    print()
    print(f"  the ratio runs {min(rs):.2f} to {max(rs):.2f} -- a "
          f"factor {spread:.1f} across the scan.")
    assert spread > 1.5
    print("  IT IS NOT CONSTANT. A cut and a deformation are NOT "
          "the same measurement:")
    print("  alpha and p respond differently to the very same "
          "change in the field\n")
    return rows


def s3_settles(rows):
    print("== s3: what that settles ==")
    rs = [r for _, _, _, r in rows]
    print("  Their factor 20 was treated as a defect -- two routes "
          "to one constant that")
    print("  ought to agree and do not. The premise is wrong. The "
          "two routes are not two")
    print("  windows on one quantity; they are two DIFFERENT "
          "functionals of the field, and")
    print("  here their ratio moves by a factor "
          f"{max(rs) / min(rs):.1f} under nothing more than a mass.")
    print()
    print("  A cut asks: HOW MUCH DOES A BOUNDARY RECORD REVEAL "
          "ABOUT THE OTHER SIDE.")
    print("  A deformation asks: HOW STIFFLY DOES THE CODE LENGTH "
          "RESIST BEING BENT.")
    print("  Those coincide only when a fluctuation-dissipation "
          "relation ties them, and")
    print("  0019 already measured that this program HAS NO SUCH "
          "RELATION -- the vacuum")
    print("  spectrum is white while the response is Coulomb. "
          "Response and correlation")
    print("  decouple here, and that was recorded three dozen "
          "stones ago.")
    print()
    print("  SO CRITICALITY 5 IS NOT A DISCREPANCY TO RECONCILE. It "
          "is a category error to")
    print("  retire: G should be read off the RESPONSE route, "
          "because G is defined by how")
    print("  matter bends geometry -- a response -- and the "
          "entanglement number is a")
    print("  different observable that happens to carry the same "
          "units.")
    print()
    print("  What that costs them: 0105's l_P = 2.27a came from the "
          "entanglement route, so")
    print("  it must be recomputed from the induced stiffness "
          "instead. That moves l_P and")
    print("  everything hanging off it -- including 0143's "
          "inversion of N from gravity's")
    print("  weakness. Named, not hidden\n")


if __name__ == "__main__":
    s1_both()
    rows = s2_mass_scan()
    s3_settles(rows)
    print("all assertions passed")
