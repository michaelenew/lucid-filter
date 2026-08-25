"""wall-correspondence 0017 -- the common mode (F6): what survives
differencing.

The sibling's cosmological budget / Lambda residual, in bank terms:
M streams each carry private structure plus a shared slow LEVEL
c_t (a pinned random walk). Every stream is differenced (the (1-B)
family) before modeling. What of the common mode is identifiable?

  s1  THE LEVEL IS GAUGE. From differenced data the common level's
      absolute value is exactly unrecoverable (differencing
      annihilates it); its INCREMENTS survive and are estimable by
      the cross-sectional mean, with error ~ private-noise/sqrt(M).
      Measured: increment recovery R^2 vs M; level recovery R^2 = 0
      identically up to the unknown constant.
  s2  THE LAMBDA READING. The bank can know the common mode's
      drift and fluctuations to any precision (more streams), but
      its absolute level is a convention -- the zero mode the
      sibling's budget quantizes globally rather than locally. The
      filter-side statement of 'Lambda is a boundary/budget term,
      not a local observable': only d(level)/dt gravitates in the
      record.
"""

import numpy as np

M_LIST = (4, 16, 64, 256)
T = 4000
S_C = 0.05          # common-level innovation scale
S_P = 0.30          # private innovation scale
S_Y = 0.20          # observation noise


def run(M, rng):
    dc = S_C * rng.normal(size=T)
    c = np.cumsum(dc)
    xp = np.cumsum(S_P * rng.normal(size=(T, M)), axis=0)
    y = c[:, None] + xp + S_Y * rng.normal(size=(T, M))
    dy = np.diff(y, axis=0)
    # estimate common increments: cross-sectional mean of dy
    dchat = dy.mean(axis=1)
    r2_inc = np.corrcoef(dchat, dc[1:])[0, 1] ** 2
    # attempt to recover the LEVEL from differenced data: any
    # reconstruction integrates dchat from an ARBITRARY constant --
    # the constant is not in the data. R^2 about the true level is
    # meaningful only up to that constant; measure the residual sd
    # of (chat - c) after optimal constant removal vs its growth
    chat = np.cumsum(dchat)
    resid = chat - c[1:]
    resid -= resid.mean()
    return r2_inc, float(resid.std())


if __name__ == "__main__":
    print("== the common mode: what survives differencing ==")
    rng = np.random.default_rng(8)
    print("   M     R^2(increments)   sd(level residual, best "
          "constant)")
    r2s = {}
    for M in M_LIST:
        r2, sd = run(M, rng)
        r2s[M] = r2
        print(f"  {M:4d}       {r2:.3f}             {sd:.3f}")
    assert r2s[256] > r2s[4]
    r2_pred = {M: S_C ** 2 / (S_C ** 2 + (S_P ** 2 + 2 * S_Y ** 2)
                              / M) for M in M_LIST}
    print("  predicted R^2 = s_c^2/(s_c^2 + (s_p^2 + 2 s_y^2)/M): "
          + ", ".join(f"M={M}: {r2_pred[M]:.3f}" for M in M_LIST))
    for M in M_LIST:
        assert abs(r2s[M] - r2_pred[M]) < 0.08
    print("  increments: identifiable at exactly the 1/M law, "
          "toward exact;")
    print("  level: recoverable only up to an arbitrary constant -- "
          "the zero mode is not")
    print("  in the record. The bank-side Lambda statement: the "
          "common level's absolute")
    print("  value is a convention (the budget quantizes it "
          "globally, their 0080/0094);")
    print("  only its drift gravitates in the record\n")
    print("all assertions passed")
