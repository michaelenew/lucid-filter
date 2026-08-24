"""wall-correspondence 0012 -- the force law: attraction is a code
gradient.

0011 gave statics (the 1/r potential). This stone derives the FORCE:
no dynamics rule is posited -- the force is the gradient of the
joint description length.

  THEOREM (exact, Gaussian field algebra). For sources s1, s2
  coupled to the trust field at separation r, the minimized field
  code is
      C(r) = const - s1 s2 G(r),
  G the field's Green function. On the 3-lattice G ~ 1/r, so the
  code gradient is an ATTRACTIVE 1/r^2 force between like sources;
  and if a source's location is a learnable parameter, its MAP
  estimate drifts down the code gradient: bodies fall because
  falling compresses.

  s1  C(r) measured on the 3-lattice: fit C(r) = a - b/r^alpha,
      alpha = 1 -> force ~ 1/r^2, attractive.
  s2  WHY GRAVITY ONLY ATTRACTS. Opposite-sign sources REPEL under
      the same algebra (their fields cancel; separating them
      compresses). But a source's strength is accumulated
      information, and information is NONNEGATIVE -- mass
      1 - e^{-2I} >= 0 always (0010) -- so all physical sources are
      like-signed: universal attraction is the positivity of
      information.
  s3  The fall, demonstrated: the code landscape over source-2
      positions has its gradient pointing at source 1 from every
      site; the MAP location walks to contact.
"""

import numpy as np

L = 32


def green3():
    k = 2 * np.pi * np.fft.fftfreq(L)
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    k2 = 4 * (np.sin(kx / 2) ** 2 + np.sin(ky / 2) ** 2
              + np.sin(kz / 2) ** 2)
    k2.flat[0] = np.inf                     # zero mode projected
    return np.real(np.fft.ifftn(1.0 / k2))


G = green3()


def code(pos1, pos2, s1=1.0, s2=1.0):
    """Minimized field code -1/2 s^T G s (interaction part kept)."""
    d = tuple((np.array(pos2) - np.array(pos1)) % L)
    return -0.5 * (s1 ** 2 * G[0, 0, 0] + s2 ** 2 * G[0, 0, 0]
                   + 2 * s1 * s2 * G[d])


def s1_force():
    print("== s1: the interaction code and the 1/r^2 force ==")
    rs = np.arange(1, 11)
    C = np.array([code((0, 0, 0), (r, 0, 0)) for r in rs])
    best = None
    for al in np.arange(0.5, 1.61, 0.01):
        X = np.stack([np.ones_like(rs, float), 1 / rs ** al], 1)
        c, res, *_ = np.linalg.lstsq(X, C, rcond=None)
        if best is None or res[0] < best[1]:
            best = (al, res[0], c)
    al, _, c = best
    print(f"  C(r) = a - b/r^alpha: alpha = {al:.2f} (Newton: 1), "
          f"b = {-c[1]:.3f} > 0 (attractive)")
    assert 0.9 < al < 1.1 and c[1] < 0
    F = -(C[1:] - C[:-1])
    print(f"  force -dC/dr at r = 2..5: "
          + ", ".join(f"{f:+.4f}" for f in F[1:5])
          + "  (ratios ~ (r+1)^2/r^2)")
    print("  closer is cheaper: the code gradient is an attractive "
          "inverse-square force\n")


def s2_signs():
    print("== s2: why gravity only attracts ==")
    like = code((0, 0, 0), (3, 0, 0), 1, 1) \
        - code((0, 0, 0), (10, 0, 0), 1, 1)
    unlike = code((0, 0, 0), (3, 0, 0), 1, -1) \
        - code((0, 0, 0), (10, 0, 0), 1, -1)
    print(f"  like sources   : C(3) - C(10) = {like:+.4f}  "
          f"(negative: attract)")
    print(f"  unlike sources : C(3) - C(10) = {unlike:+.4f}  "
          f"(positive: repel)")
    assert like < 0 < unlike
    print("  the algebra allows both signs -- but source strength "
          "is accumulated")
    print("  information, and information is nonnegative "
          "(mass = 1 - e^{-2I} >= 0, 0010):")
    print("  every physical source is like-signed. Universal "
          "attraction = positivity of")
    print("  information\n")


def s3_fall():
    print("== s3: the fall ==")
    pos = np.array([9, 6, 4])
    path = [tuple(int(v) for v in pos)]
    for _ in range(40):
        best = None
        for d in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                  [0, 0, 1], [0, 0, -1], [0, 0, 0]):
            p = (pos + d) % L
            cc = code((0, 0, 0), tuple(p))
            if best is None or cc < best[0]:
                best = (cc, p)
        if (best[1] == pos).all():
            break
        pos = best[1]
        path.append(tuple(int(v) for v in pos))
    print(f"  MAP drift of source 2 from (9,6,4): "
          + " -> ".join(str(p) for p in path[:4]) + " -> ... -> "
          + str(path[-1]) + f"  ({len(path) - 1} steps)")
    assert path[-1] == (0, 0, 0)
    print("  the learnable source location walks all the way to "
          "merger (point sources have")
    print("  no hard core): bodies fall because falling compresses "
          "the record\n")


if __name__ == "__main__":
    s1_force()
    s2_signs()
    s3_fall()
    print("all assertions passed")
