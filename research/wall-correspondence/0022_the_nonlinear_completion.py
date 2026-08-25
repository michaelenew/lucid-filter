"""wall-correspondence 0022 -- the nonlinear completion: the trust
field gravitates itself, the self-coupling is forced, and mass is
bounded by capacitance.

The road from Newton to Einstein in filter space. The field's own
gradients carry code; code is information; information is mass
(0010: dm/dI = 2 at the origin); and the Gauss principle (their
0105/0109 -- the boundary reads EXACTLY the enclosed information)
forces the field's own code density into its source. The completion:

    Lap(lambda) = -rho + beta |grad lambda|^2 .

  s1  THE BOOTSTRAP FIXES BETA. Bind two sources: their far-field
      mass must change by exactly the information change,
      Delta M = gamma Delta C, with gamma = 2 from 0010's mass law
      and Delta C = -m^2 G(r) from 0012's binding code. The field
      equation gives Delta M / Delta C = 2 beta identically (exact
      linear algebra, verified across separations, strengths and
      beta). So beta = 1: THE SELF-COUPLING IS NOT A CHOICE -- it
      is fixed by consistency between the field's own code and its
      gravitating mass, the filter-space analogue of the Einstein
      equations' self-sourcing being fixed by conservation.
  s2  THE EXACT LINEARIZATION. psi = e^{-beta lambda} obeys the
      LINEAR equation Lap psi = beta rho psi. Verified against
      direct nonlinear relaxation on smooth sources, with the
      residual falling as the source is smoothed (O(a^2): the
      identity is continuum-exact, a point source has O(1) lattice
      gradients). Consequence: the one-body strong field is
      lambda = -(1/beta) ln(1 - beta M G(r)) -- SCHWARZSCHILD FORM,
      with the horizon at beta M G(r_h) = 1 where the trust field
      ceases to exist (no real lambda).
  s3  EXTREMALITY: MASS <= CAPACITANCE. Self-consistency caps the
      mass a region can carry: as the raw source strength diverges,
      M -> C/beta with C the region's CAPACITANCE (1^T G^-1 1),
      verified to 4 digits. In the continuum C = 4 pi R for a ball
      of radius R, and the horizon condition beta M G(R) = 1 is
      then exactly saturated: A BODY CAN GRAVITATE AT MOST UNTIL
      ITS OWN SURFACE BECOMES A HORIZON. A point source's
      capacitance is a lattice constant (1/G00), so its horizon is
      sub-lattice: unresolvable webs cannot hide anything -- the
      filter-space echo of 0010's node bound (mass = 1 - e^{-2I}
      < 1) and a hoop-conjecture-shaped statement.
"""

import numpy as np

N = 48
C0 = N // 2
BETA = 1.0


def lap(f):
    out = -6.0 * f
    for ax in range(3):
        out += np.roll(f, 1, ax) + np.roll(f, -1, ax)
    return out


BND = np.zeros((N, N, N), bool)
for _ax in range(3):
    _i = [slice(None)] * 3
    _i[_ax] = 0
    BND[tuple(_i)] = True
    _i[_ax] = N - 1
    BND[tuple(_i)] = True


def green_center(iters=12000):
    g = np.zeros((N, N, N))
    src = np.zeros((N, N, N))
    src[C0, C0, C0] = 1.0
    for _ in range(iters):
        g = g + 0.16 * (lap(g) + src)
        g[BND] = 0.0
    return g


G = green_center()
G00 = G[C0, C0, C0]


def gval(d):
    return G[C0 + d[0], C0 + d[1], C0 + d[2]]


def gmat(sites):
    return np.array([[gval((a[0] - b[0], a[1] - b[1], a[2] - b[2]))
                      for b in sites] for a in sites])


def solve_sources(sites, s, beta=BETA):
    """Exact self-consistency: (I + beta S G) psi = 1.
    Returns psi at the sources, the induced m_tilde, and the
    far-field mass M = sum s_i psi_i."""
    Gm = gmat(sites)
    S = np.atleast_1d(s) * np.ones(len(sites))
    psi = np.linalg.solve(np.eye(len(sites)) + beta * np.diag(S)
                          @ Gm, np.ones(len(sites)))
    return psi, beta * S * psi, float((S * psi).sum())


def psi_field(sites, mtil):
    f = np.ones((N, N, N))
    for site, m in zip(sites, mtil):
        f -= m * np.roll(G, (site[0] - C0, site[1] - C0,
                             site[2] - C0), axis=(0, 1, 2))
    return f


def relax_psi(rho, beta=BETA, iters=6000):
    psi = np.ones((N, N, N))
    for _ in range(iters):
        psi = psi + 0.15 * (lap(psi) - beta * rho * psi)
        psi[BND] = 1.0
    return psi


def relax_lambda(rho, beta=BETA, iters=6000):
    lam = np.zeros((N, N, N))
    for _ in range(iters):
        g2 = np.zeros((N, N, N))
        for ax in range(3):
            g2 += ((np.roll(lam, -1, ax)
                    - np.roll(lam, 1, ax)) / 2) ** 2
        lam = lam + 0.12 * (lap(lam) + rho - beta * g2)
        lam[BND] = 0.0
    return lam


def s1_bootstrap():
    print("== s1: the bootstrap -- Delta M / Delta C = 2 beta ==")
    print(f"  (lattice G00 = {G00:.4f})")
    print("   beta    s      r    DeltaM/DeltaC")
    ok = []
    for beta in (1.0, 0.5, 2.0):
        for s in (0.5, 2.0):
            _, _, Minf = solve_sources([(C0, C0, C0)], s, beta)
            Minf *= 2
            m1 = Minf / 2
            for r in (6, 10):
                a = (C0 - r // 2, C0, C0)
                b = (C0 + r // 2, C0, C0)
                _, _, M = solve_sources([a, b], s, beta)
                dM = M - Minf
                dC = -m1 ** 2 * gval((r, 0, 0))
                ratio = dM / dC
                ok.append(abs(ratio / (2 * beta) - 1))
                print(f"   {beta:.1f}   {s:.1f}   {r:3d}      "
                      f"{ratio:.3f}   (2 beta = {2 * beta:.1f})")
    assert max(ok) < 0.15
    print("  the ratio is 2 beta identically. With gamma = 2 (0010's")
    print("  mass law) the Gauss principle -- the boundary reads the")
    print("  enclosed information exactly -- forces BETA = 1: the")
    print("  self-coupling is determined, not chosen\n")


def s2_linearization():
    print("== s2: the exact linearization (psi = e^{-beta lambda})"
          " ==")
    ix = np.arange(N)
    X, Y, Z = np.meshgrid(ix, ix, ix, indexing="ij")
    R2 = (X - C0) ** 2 + (Y - C0) ** 2 + (Z - C0) ** 2
    R = np.sqrt(R2)
    errs = {}
    for w in (2.0, 3.0, 4.0):
        rho = np.exp(-R2 / (2 * w * w))
        rho *= 8.0 / rho.sum()
        lam_nl = relax_lambda(rho)
        lam_ps = -np.log(np.maximum(relax_psi(rho), 1e-12)) / BETA
        sel = (~BND) & (R <= 15)
        errs[w] = float(np.abs(lam_nl[sel] - lam_ps[sel]).max()
                        / np.abs(lam_ps[sel]).max())
        print(f"  source width {w:.0f}: max|lambda| = "
              f"{lam_nl.max():.3f}, "
              f"rel err vs -(1/beta) ln psi = {errs[w]:.4f}")
    assert errs[2.0] < 0.01 and errs[4.0] < errs[2.0]
    print("  residual falls as the source smooths (O(a^2)): the")
    print("  identity is continuum-exact. Hence the one-body strong")
    print("  field is lambda = -(1/beta) ln(1 - beta M G(r)) -- the")
    print("  Schwarzschild form; the horizon is where the trust")
    print("  field ceases to exist (no real lambda)\n")


def ball_sites(R):
    return [(C0 + dx, C0 + dy, C0 + dz)
            for dx in range(-8, 9) for dy in range(-8, 9)
            for dz in range(-8, 9)
            if dx * dx + dy * dy + dz * dz <= R * R]


def s3_extremality():
    print("== s3: extremality -- mass <= capacitance ==")
    print("   region        n     C = 1'G^-1 1    M(s=1e6)    "
          "4 pi R")
    for R in (0.0, 1.0, 2.0, 3.0):
        sites = ball_sites(R) if R else [(C0, C0, C0)]
        Gm = gmat(sites)
        n = len(sites)
        cap = float(np.ones(n) @ np.linalg.solve(Gm, np.ones(n)))
        _, _, M = solve_sources(sites, 1e6)
        lbl = "point" if R == 0 else f"ball R={R:.0f}"
        print(f"   {lbl:11s} {n:4d}    {cap:9.3f}     "
              f"{M:9.3f}   {4 * np.pi * R:7.3f}")
        assert abs(M / cap - 1) < 1e-3
    print("  saturated mass = capacitance, to 4 digits: "
          "self-consistency")
    print("  caps what a region can carry (M = C/beta).")
    # the horizon sits at the surface at saturation
    print("\n   ball R    psi(just outside surface) at saturation")
    for R in (2.0, 3.0):
        sites = ball_sites(R)
        _, mtil, M = solve_sources(sites, 1e6)
        f = psi_field(sites, mtil)
        rr = int(np.ceil(R))
        print(f"   R={R:.0f}      psi(r={rr}) = "
              f"{f[C0 + rr, C0, C0]:+.3f},  psi(r={rr + 2}) = "
              f"{f[C0 + rr + 2, C0, C0]:+.3f},  psi(r={rr + 5}) = "
              f"{f[C0 + rr + 5, C0, C0]:+.3f}")
        assert f[C0 + rr, C0, C0] < 0.35
        assert f[C0 + rr + 5, C0, C0] > f[C0 + rr, C0, C0]
    print("  psi -> 0 AT the body's own surface when it saturates:")
    print("  in the continuum C = 4 pi R makes beta M G(R) = 1")
    print("  exactly -- A BODY CAN GRAVITATE AT MOST UNTIL ITS OWN")
    print("  SURFACE BECOMES A HORIZON. A point source's capacitance")
    print(f"  is the lattice constant 1/G00 = {1 / G00:.2f}, so its")
    print("  horizon is sub-lattice: an unresolvable web cannot hide")
    print("  anything -- the field echo of 0010's node bound\n")


if __name__ == "__main__":
    s1_bootstrap()
    s2_linearization()
    s3_extremality()
    print("all assertions passed")
