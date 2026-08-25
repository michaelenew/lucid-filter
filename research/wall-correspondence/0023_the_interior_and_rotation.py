"""wall-correspondence 0023 -- the interior, the rotation sector,
and the evaporation shape.

Three opens left by the nonlinear completion (0022), run together.

  s1  THE INTERIOR. Saturation drives psi -> 0 at every source site,
      so the extremal body's interior has psi = 0 identically:
      infinite trust potential, zero transmission -- the singular
      interior, reached only in the limit. Masses beyond the
      capacitance are not 'complex-field' states: they are
      UNREACHABLE, no source strength produces them. And at a
      FINITE resolution (a deepest representable potential
      lambda_max) the horizon never quite forms: transmission
      floors at e^{-lambda_max} and the mass cap softens to
      (1 - e^{-lambda_max}) C. With the sibling's budget capacity
      lambda_max = ln N (their 0100), a level-N web has minimum
      transmission 1/N and maximum mass (1 - 1/N) C: DISCRETENESS
      REPLACES HORIZONS WITH GREYBODY FLOORS. (Conditional on that
      identification, labelled as such.)
  s2  THE ROTATION SECTOR IS THE ORDER CHANNEL. The completion is
      scalar: its field depends on the source DENSITY alone, so two
      configurations differing only in circulation are bitwise
      identical -- no frame-dragging, exactly zero. What circulation
      IS, in this program's terms, is the ORDER of records around a
      loop, and that is precisely the channel abelian/scalar
      structure cannot carry (0009). Measured: information about
      circulation direction is exactly 0 in every scalar observable
      and positive in the ordered composite. Frame-dragging, if it
      exists here, is order information -- a named next structure,
      not a missing term.
  s3  THE EVAPORATION SHAPE. With T = 2/M (their 0114) and horizon
      area A ~ M^2, a Stefan-type law dM/dt ~ -A T^4 gives
      dM/dt ~ -1/M^2, hence M^3 linear in time and lifetime ~ M0^3:
      Hawking's evaporation shape. Dimensional analysis, not a
      derived emission rate -- the carrier is the source tier
      (0014), whose rate is not computed here. With s1's floor, the
      last stage stalls: a level-N web leaves a remnant.
"""

import numpy as np

TH = np.linspace(1e-7, np.pi - 1e-7, 40001)
HAAR = (2 / np.pi) * np.sin(TH) ** 2


def k_heat(tau, jmax=40):
    out = np.zeros_like(TH)
    for j in np.arange(0, jmax + 0.1, 0.5):
        out += (2 * j + 1) * np.exp(-tau * j * (j + 1)) \
            * np.sin((2 * j + 1) * TH) / np.sin(TH)
    return out


_CDF = None


def sample_su2(tau, n, rng):
    global _CDF
    if _CDF is None:
        p = np.maximum(k_heat(tau), 0) * HAAR
        _CDF = np.cumsum(p) / p.sum()
    th = TH[np.searchsorted(_CDF, rng.random(n))]
    ax = rng.normal(size=(n, 3))
    ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    return np.concatenate([np.cos(th)[:, None],
                           np.sin(th)[:, None] * ax], axis=1)


def qmul(a, b):
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


def cls(q):
    return np.arccos(np.clip(q[..., 0], -1, 1))


def s1_interior():
    print("== s1: the interior ==")
    print("   lambda_max   min transmission   mass cap / C")
    for N in (5, 13, 100):
        lam_max = np.log(N)
        floor = np.exp(-lam_max)
        print(f"   ln {N:<3d} = {lam_max:.2f}      {floor:.4f}"
              f"             {1 - floor:.4f}")
        assert abs(floor - 1.0 / N) < 1e-12
    print("  beyond the capacitance nothing is complex -- those "
          "masses are UNREACHABLE:")
    print("  no source strength produces them (M(s) rises to C and "
          "stops). At the extremal")
    print("  limit psi = 0 through the whole body: infinite "
          "potential, zero transmission.")
    print("  At finite resolution the horizon never quite closes: "
          "with the budget capacity")
    print("  lambda_max = ln N, a level-N web floors transmission "
          "at 1/N and caps mass at")
    print("  (1 - 1/N) C -- DISCRETENESS REPLACES HORIZONS WITH "
          "GREYBODY FLOORS (conditional")
    print("  on that identification)\n")


def s2_rotation():
    print("== s2: the rotation sector is the order channel ==")
    rng = np.random.default_rng(17)
    n = 200000
    K = 4
    els = [sample_su2(0.2, n, rng) for _ in range(K)]
    # a scalar observable is a function of the MULTISET of sources
    # (the density); evaluate symmetrically so float summation
    # order cannot masquerade as physics
    fwd = np.sort(np.stack([cls(e) for e in els]), axis=0)
    rev = np.sort(np.stack([cls(e) for e in reversed(els)]), axis=0)
    same = np.array_equal(fwd, rev)
    print(f"  the source multiset under circulation reversal is "
          f"bitwise identical: {same}")
    print(f"  -> every scalar observable (density, total, moments) "
          f"is unchanged")
    assert same
    print("  -> information about circulation in ANY scalar "
          "observable = 0 exactly")
    # ordered composite: the loop holonomy's class
    def compose(seq):
        acc = seq[0]
        for e in seq[1:]:
            acc = qmul(acc, e)
        return cls(acc)
    t_f = compose(els)
    t_r = compose(list(reversed(els)))
    sig = float(np.std(cls(els[0])))
    sent = rng.random(n) < 0.5
    read = np.where(sent, t_f, t_r) + sig * rng.normal(size=n)
    dec = np.abs(read - t_f) < np.abs(read - t_r)
    perr = float(np.mean(dec != sent))
    hb = (-perr * np.log(max(perr, 1e-12))
          - (1 - perr) * np.log(max(1 - perr, 1e-12)))
    cap = np.log(2) - hb
    print(f"  ordered composite: mean |dtheta| = "
          f"{np.abs(t_f - t_r).mean():.4f}, decode error {perr:.3f},"
          f" capacity {cap:.4f} nats/loop")
    assert cap > 0.002
    print("  circulation IS order information: invisible to the "
          "scalar field, carried by")
    print("  the commutator. Frame-dragging, if this program has "
          "it, is the order channel")
    print("  sourcing a vector sector -- a named structure to "
          "build, not a missing term\n")


def s3_evaporation():
    print("== s3: the evaporation shape ==")
    # dM/dt = -c A T^4 with A = 4 pi r_h^2, r_h = M/(4 pi), T = 2/M
    c = 1.0
    def rate(M):
        A = 4 * np.pi * (M / (4 * np.pi)) ** 2
        T = 2.0 / M
        return -c * A * T ** 4
    for M in (10.0, 20.0, 40.0):
        pred = -c * (4 / np.pi) / M ** 2
        assert abs(rate(M) / pred - 1) < 1e-12
    print("  dM/dt = -c A T^4 = -(4c/pi) / M^2  exactly, so "
          "M^3 is linear in time")
    # integrate to confirm the cube law and the lifetime scaling
    for M0 in (10.0, 20.0):
        M, t, dt = M0, 0.0, 1e-3
        while M > 1e-3 * M0:
            M += rate(M) * dt
            t += dt
        pred_life = M0 ** 3 * np.pi / (12 * c)
        print(f"  M0 = {M0:.0f}: integrated lifetime {t:.1f} vs "
              f"M0^3 pi/(12 c) = {pred_life:.1f}")
        assert abs(t / pred_life - 1) < 0.02
    print("  lifetime ~ M0^3: Hawking's evaporation shape. This is "
          "DIMENSIONAL ANALYSIS --")
    print("  the emission rate is not derived; its carrier is the "
          "source tier (0014).")
    print("  With s1's floor the last stage stalls: a level-N web "
          "leaves a remnant\n")


if __name__ == "__main__":
    s1_interior()
    s2_rotation()
    s3_evaporation()
    print("all assertions passed")
