"""wall-correspondence 0010 -- what gravity is in the filter.

The sibling's gravity is built from trust: mass is accumulated
information (their m = (1 - e^-I)/4G), and the metric weight is the
precision channel. Translated to this repository's objects, gravity
has three LOCAL laws, each measurable:

  s1  INERTIA IS PRECISION; MASS IS ABSORPTION. A node that has
      accumulated information I (advantage over its incoming
      channel) responds to an incident influence with transmitted
      fraction f = e^{-2I}: the absorbed fraction 1 - e^{-2I} is
      its mass -- the sibling's saturating mass law, derived from
      Bayes alone. A node with process noise (finite memory) has
      bounded I: bounded mass. A rigid node (no process noise)
      absorbs without bound: f -> 0 -- an asymptotic horizon that
      transmits nothing.
  s2  TRUST WELLS DELAY AND BEND INFLUENCE. On a 2D relay grid
      (each node's estimate relaxes toward its neighbors at its own
      gain), a high-trust region transmits slowly: influence
      arriving from one edge reaches points behind the well LATER
      (Shapiro delay), and the wavefront routes around it
      (lensing): the fastest path avoids the well.
  s3  UNIVERSALITY (the equivalence principle). The response of a
      node to a small incident influence is q x Var(posterior),
      IDENTICALLY, whatever the node's history or posterior shape:
      only the information state at the encounter gravitates.
      Shape enters solely through how that state evolves -- and
      reappears, as it should, in the tides (large influences).
"""

import numpy as np

# ----------------------------------------------------------------------
# s1 -- mass is absorption
# ----------------------------------------------------------------------

def s1_mass():
    print("== s1: inertia = precision; mass = absorption ==")
    q = 1.0
    print("  static node (rigid prior), n messages absorbed:")
    print("   n    f measured   e^{-2I}    mass 1-e^{-2I}")
    for n in (0, 2, 8, 30, 120):
        p = q * (1 + n)                     # prior + n messages
        f = q / (p + q)                     # response to unit shift
        I = 0.5 * np.log((p + q) / q)
        assert abs(f - np.exp(-2 * I)) < 1e-12
        print(f"  {n:4d}   {f:.5f}      {np.exp(-2 * I):.5f}    "
              f"{1 - np.exp(-2 * I):.5f}")
    print("  -> unbounded absorption: f -> 0, mass -> 1: the "
          "asymptotic horizon")
    print("  leaky node (process noise w), steady state:")
    for w in (0.5, 0.05, 0.005):
        V = 1.0
        for _ in range(3000):
            Vp = V + w
            V = Vp / (Vp + 1)
        f = (V + w) / (V + w + 1)
        I = -0.5 * np.log(f)
        print(f"   w = {w:5.3f}: f_ss = {f:.4f}, mass_max = "
              f"{1 - f:.4f}  (I_max = {I:.3f} nats)")
    print("  -> finite memory bounds the mass: only a rigid "
          "(conserved) node can grow a horizon\n")


# ----------------------------------------------------------------------
# s2 -- Shapiro delay and lensing on the relay grid
# ----------------------------------------------------------------------

def s2_wells():
    print("== s2: trust wells delay and bend influence "
          "(41 x 41 relay grid) ==")
    L = 41
    K = np.full((L, L), 0.5)
    cy, cx, rad = L // 2, L // 2, 6
    yy, xx = np.mgrid[0:L, 0:L]
    disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= rad ** 2
    K[disk] = 0.06                          # the trust well
    m = np.zeros((L, L))
    arrive = np.full((L, L), -1)
    for t in range(60000):
        p = np.pad(m, 1, mode='edge')
        nb = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2]
              + p[1:-1, 2:]) / 4
        m = m + K * (nb - m)
        m[:, 0] = 1.0                       # influence source: left
        new = (arrive < 0) & (m > 0.5)
        arrive[new] = t
        if (arrive >= 0).all():
            break
    behind = arrive[cy, cx + rad + 3]       # just behind the well
    offset = arrive[cy + 15, cx + rad + 3]  # same column, clear row
    edge_in = arrive[cy, cx - rad - 1]      # well's near edge
    chord = edge_in + (2 * rad + 3) ** 2 / (0.06 / 4)
    print(f"  arrival just behind the well : t = {behind}")
    print(f"  same column, clear row       : t = {offset}   "
          f"(Shapiro delay: +{behind - offset} steps)")
    print(f"  through-the-well estimate    : t ~ {chord:.0f}")
    assert behind > offset + 100            # the well delays
    assert behind < 0.8 * chord             # ...but influence bent
    print("  arrival behind the well is far earlier than the "
          "through-well estimate: the")
    print("  influence ROUTED AROUND the well -- geodesics avoid "
          "trust concentrations\n")


# ----------------------------------------------------------------------
# s3 -- universality
# ----------------------------------------------------------------------

def s3_universality():
    print("== s3: universality -- only the nats gravitate ==")
    th = np.linspace(-12, 12, 240001)
    q = 1.0

    def response(logprior, delta):
        post0 = logprior - 0.5 * q * (th - 0.0) ** 2
        post1 = logprior - 0.5 * q * (th - delta) ** 2
        w0 = np.exp(post0 - post0.max())
        m0 = np.average(th, weights=w0)
        v0 = np.average(th ** 2, weights=w0) - m0 ** 2
        m1 = np.average(th, weights=np.exp(post1 - post1.max()))
        return (m1 - m0) / delta, v0

    v = 0.25                                 # matched posterior var
    lp_g = -0.5 * th ** 2 / v
    d, w = 0.45, v - 0.45 ** 2
    lp_b = np.logaddexp(-0.5 * (th - d) ** 2 / w,
                        -0.5 * (th + d) ** 2 / w)
    # verify matched variance
    for lp in (lp_g, lp_b):
        pz = np.exp(lp - lp.max())
        var = (np.average(th ** 2, weights=pz)
               - np.average(th, weights=pz) ** 2)
        assert abs(var - v) < 1e-3
    (fg, vg), (fb, vb) = response(lp_g, 0.001), response(lp_b, 0.001)
    print(f"  Gaussian node : f = {fg:.5f} = q Var(post) = "
          f"{q * vg:.5f}  (rel {abs(fg / (q * vg) - 1):.0e})")
    print(f"  bimodal node  : f = {fb:.5f} = q Var(post) = "
          f"{q * vb:.5f}  (rel {abs(fb / (q * vb) - 1):.0e})")
    assert abs(fg / (q * vg) - 1) < 1e-3
    assert abs(fb / (q * vb) - 1) < 1e-3
    (Fg, _), (Fb, _) = response(lp_g, 3.0), response(lp_b, 3.0)
    print(f"  large influence: Gaussian {Fg:.4f} vs bimodal "
          f"{Fb:.4f} -- composition shows in the tides")
    print("  the equivalence principle, exact form: response = "
          "q Var(posterior) IDENTICALLY,")
    print("  whatever the node's history or shape -- only the "
          "information state at the")
    print("  encounter gravitates; shape enters solely through how "
          "that state evolves, and")
    print("  reappears (as it should) in the tides\n")


if __name__ == "__main__":
    s1_mass()
    s2_wells()
    s3_universality()
    print("all assertions passed")
