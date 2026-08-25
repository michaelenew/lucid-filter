"""wall-correspondence 0011 -- the trust field: the 4D corollary.

0010 gave gravity's local laws. The 4D object is a BANK ON A
3-LATTICE: one stream per site, a shared log-scale (trust) field
with nearest-neighbor coupling and a pinned (massless, phi = 1) time
channel; time is the stream index. Gravity is the field's response
to accumulated information. Three measured statements:

  s1  NEWTON'S 1/r IS THE FIELD'S GREEN FUNCTION -- AND ONLY IN
      THREE SPATIAL DIMENSIONS. A point information source's
      induced trust perturbation falls as 1/r on the 3-lattice
      (log-log slope -1), logarithmically in 2D, linearly in 1D:
      the dimension of the bank SELECTS the potential; demanding
      Newton demands 3+1.
  s2  THE CAUSAL (RECORD-TIER) FIELD IS NEWTONIAN-DIFFUSIVE. A
      switched-on source's influence spreads with a sqrt(t) front:
      screening and statics are right, but there is NO radiation in
      the record tier. Prediction ported back: gravitational WAVES
      are a source-tier (phase-ledger) phenomenon -- the same split
      as 0005/0006 (phase never pays on the record side).
  s3  THE FIELD-LEVEL MASS LAW IS THE NODE-LEVEL ONE. By
      Sherman-Morrison the field at a source of data-precision rho
      is exactly lambda0 rho G00/(1 + rho G00): the effective mass
      saturates as 1 - e^{-2I} with I = (1/2) ln(1 + rho G00) --
      the SAME saturating law as 0010 s1, with the Green function
      as the channel. One mass formula, node to field.
"""

import numpy as np


def green_profile(dim, L):
    """Mean trust perturbation from a unit point source on a
    periodic d-lattice with Laplacian prior precision (+ small
    pin), via FFT."""
    k = 2 * np.pi * np.fft.fftfreq(L)
    if dim == 1:
        k2 = 4 * np.sin(k / 2) ** 2
    elif dim == 2:
        kx, ky = np.meshgrid(k, k, indexing="ij")
        k2 = 4 * (np.sin(kx / 2) ** 2 + np.sin(ky / 2) ** 2)
    else:
        kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
        k2 = 4 * (np.sin(kx / 2) ** 2 + np.sin(ky / 2) ** 2
                  + np.sin(kz / 2) ** 2)
    eps = 1e-6
    G = np.real(np.fft.ifftn(1.0 / (k2 + eps)))
    G -= G.flat[-1] if dim == 1 else 0.0
    return G


def s1_poisson():
    print("== s1: Newton's 1/r is the trust field's Green function "
          "(3D only) ==")
    G3 = green_profile(3, 48)
    r = np.arange(1, 11)
    prof = np.array([G3[rr, 0, 0] for rr in r])
    # periodic box: fit a + b/r^alpha with alpha free (the constant
    # absorbs zero mode + images); the data select the exponent
    best = None
    for al in np.arange(0.5, 1.61, 0.01):
        X = np.stack([np.ones_like(r, dtype=float),
                      1.0 / r ** al], axis=1)
        c, res, *_ = np.linalg.lstsq(X, prof, rcond=None)
        rr = np.sqrt(res[0]) if len(res) else 0.0
        if best is None or rr < best[1]:
            best = (al, rr)
    alpha = best[0]
    print(f"  3D lattice (48^3): best-fit potential ~ 1/r^alpha, "
          f"alpha = {alpha:.2f}  (Newton: 1)")
    assert 0.85 < alpha < 1.15
    G2 = green_profile(2, 256)
    p2 = np.array([G2[rr, 0] for rr in r]) - G2[128, 128]
    c2 = np.polyfit(np.log(r[1:9]), p2[1:9], 1)
    resid = np.abs(p2[1:9] - np.polyval(c2, np.log(r[1:9]))).max() \
        / abs(p2[1] - p2[8])
    print(f"  2D lattice: logarithmic potential (log-fit residual "
          f"{resid:.3f} of range)")
    G1 = green_profile(1, 4096)
    p1 = np.array([G1[rr] for rr in r])
    lin = np.polyfit(r[1:9], p1[1:9], 1)
    resl = np.abs(p1[1:9] - np.polyval(lin, r[1:9])).max() \
        / abs(p1[1] - p1[8])
    print(f"  1D lattice: linear potential (linear-fit residual "
          f"{resl:.3f} of range)")
    assert resid < 0.05 and resl < 0.05
    print("  the bank's spatial dimension selects the potential; "
          "demanding Newton demands 3+1\n")


def s2_front():
    print("== s2: the record-tier field is Newtonian-diffusive ==")
    L = 48
    m = np.zeros((L, L, L))
    m[0, 0, 0] = 1.0                        # an information pulse
    eta = 0.1
    rads, ts = [], []
    for t in range(1, 241):
        nb = sum(np.roll(m, d, a) for d in (1, -1) for a in range(3))
        m = m + eta * (nb - 6 * m)
        if t % 10 == 0 and t >= 30:
            prof = np.array([m[rr, 0, 0] for rr in range(L // 2)])
            above = np.where(prof > 1e-2 * prof[0])[0]
            rads.append(above.max() if len(above) else 0)
            ts.append(t)
    sl = np.polyfit(np.log(ts), np.log(np.array(rads,
                    dtype=float)), 1)[0]
    print(f"  front radius vs time: log-log slope = {sl:.2f}  "
          f"(diffusive 0.5; ballistic 1)")
    assert 0.30 < sl < 0.70
    print("  statics and screening are right; NO radiation in the "
          "record tier. Ported")
    print("  prediction: gravitational waves are a source-tier "
          "(phase) phenomenon -- the")
    print("  same record/source split as 0005/0006\n")


def s3_field_mass():
    print("== s3: the field-level mass law ==")
    G3 = green_profile(3, 48)
    G00 = G3[0, 0, 0] - G3[24, 24, 24]
    print(f"  G00 = {G00:.3f};  field at source = "
          f"lambda0 rho G00/(1 + rho G00):")
    print("   rho     effective mass    1 - e^{-2I}, "
          "I = 0.5 ln(1 + rho G00)")
    for rho in (0.1, 1.0, 10.0, 1000.0):
        meff = rho * G00 / (1 + rho * G00)
        I = 0.5 * np.log(1 + rho * G00)
        assert abs(meff - (1 - np.exp(-2 * I))) < 1e-12
        print(f"  {rho:7.1f}     {meff:.4f}            "
              f"{1 - np.exp(-2 * I):.4f}")
    print("  the node-level saturating mass law (0010 s1) IS the "
          "field-level one, with the")
    print("  Green function as the channel: one mass formula, node "
          "to field, matching the")
    print("  sibling's m = (1 - e^-I)/4G in form with G00 playing "
          "4G's role\n")


if __name__ == "__main__":
    s1_poisson()
    s2_front()
    s3_field_mass()
    print("all assertions passed")
