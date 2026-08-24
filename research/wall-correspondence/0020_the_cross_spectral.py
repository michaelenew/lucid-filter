"""wall-correspondence 0020 -- the tangle (F5): a parameter no
marginal can see, and the monogamy budget.

The sibling's entanglement tier, in bank terms. Two claims, both
measured:

  s1  INVISIBILITY AND TRACKING. Construct two streams whose
      marginals are exactly white whatever theta, with theta carried
      only in the cross-spectral PHASE. A marginal-based tracker is
      blind by construction (its sufficient statistics are
      theta-independent -- verified numerically); the cross-spectral
      tracker recovers a slowly wandering theta_t far below the
      blind RMSE.
  s2  MONOGAMY IS POSITIVE-DEFINITENESS. For jointly Gaussian
      streams with rho_23 = 0, the correlation matrix is PSD iff
      rho_12^2 + rho_13^2 <= 1 -- in mutual-information form
      e^{-2 I(1;2)} + e^{-2 I(1;3)} >= 1: sharing information with
      one partner CAPS what can be shared with another. Verified:
      the Cholesky boundary sits exactly on the budget curve, and
      measured MI rates on generated banks trace it.
"""

import numpy as np

T = 60000
RHO = 0.8


def gen_pair(theta_series, rng):
    """y1 white; y2 = RHO * (phase-shifted y1) + noise, phase theta
    applied in the frequency domain per block."""
    B = 512
    n = len(theta_series)
    y1 = np.empty((n, B))
    y2 = np.empty((n, B))
    for i, th in enumerate(theta_series):
        e = rng.standard_normal(B)
        E = np.fft.rfft(e)
        ph = np.exp(1j * th * np.ones_like(E.real))
        ph[0] = 1.0
        shifted = np.fft.irfft(E * ph, B)
        y1[i] = e
        y2[i] = RHO * shifted + np.sqrt(1 - RHO ** 2) \
            * rng.standard_normal(B)
    return y1, y2


def s1_tracking():
    print("== s1: the marginal-invisible parameter, tracked ==")
    rng = np.random.default_rng(3)
    n = 400
    th = np.cumsum(0.05 * rng.standard_normal(n))
    y1, y2 = gen_pair(th, rng)
    # marginal blindness: per-block marginal statistics vs theta
    v1, v2 = y1.var(axis=1), y2.var(axis=1)
    a1 = (y1[:, 1:] * y1[:, :-1]).mean(axis=1)
    for stat in (v1, v2, a1):
        c = np.corrcoef(stat, th)[0, 1]
        assert abs(c) < 0.15
    print("  marginal statistics (variances, autocorrs) carry no "
          "theta (|corr| < 0.15)")
    # cross-spectral tracker: phase of cross-spectrum per block,
    # lightly smoothed
    est = []
    z = 0
    for i in range(n):
        X1 = np.fft.rfft(y1[i])
        X2 = np.fft.rfft(y2[i])
        cross = (X2[1:] * np.conj(X1[1:])).sum()
        ph = np.angle(cross)
        z = 0.7 * z + 0.3 * ph
        est.append(z)
    est = np.array(est)
    err = np.sqrt(np.mean((est[50:] - th[50:]) ** 2))
    blind = np.sqrt(np.mean(th[50:] ** 2))
    print(f"  cross-spectral tracker RMSE = {err:.3f} vs "
          f"blind-guess scale {blind:.3f}")
    assert err < 0.4 * blind
    print("  the parameter lives only in the pair: the tangle, "
          "tracked\n")


def s2_monogamy():
    print("== s2: monogamy = positive-definiteness ==")
    # boundary: rho12^2 + rho13^2 = 1 (rho23 = 0)
    for r12 in (0.3, 0.6, 0.9):
        r13_max = np.sqrt(1 - r12 ** 2)
        ok = np.array([[1, r12, r13_max - 1e-9],
                       [r12, 1, 0],
                       [r13_max - 1e-9, 0, 1]])
        bad = np.array([[1, r12, r13_max + 1e-3],
                        [r12, 1, 0],
                        [r13_max + 1e-3, 0, 1]])
        np.linalg.cholesky(ok)
        try:
            np.linalg.cholesky(bad)
            raise AssertionError("budget violated")
        except np.linalg.LinAlgError:
            pass
        I12 = -0.5 * np.log(1 - r12 ** 2)
        I13 = -0.5 * np.log(1 - r13_max ** 2)
        bud = np.exp(-2 * I12) + np.exp(-2 * I13)
        print(f"  rho12 = {r12:.1f}: max rho13 = {r13_max:.3f}; "
              f"e^-2I12 + e^-2I13 = {bud:.6f} (= 1 at the "
              f"boundary)")
        assert abs(bud - 1) < 1e-6
    # measured MI on generated banks at the boundary
    rng = np.random.default_rng(9)
    r12 = 0.6
    r13 = np.sqrt(1 - r12 ** 2) - 1e-6
    C = np.array([[1, r12, r13], [r12, 1, 0], [r13, 0, 1]])
    Lc = np.linalg.cholesky(C)
    X = rng.standard_normal((3, T))
    Y = Lc @ X
    m12 = -0.5 * np.log(1 - np.corrcoef(Y[0], Y[1])[0, 1] ** 2)
    m13 = -0.5 * np.log(1 - np.corrcoef(Y[0], Y[2])[0, 1] ** 2)
    bud = np.exp(-2 * m12) + np.exp(-2 * m13)
    print(f"  measured on a generated bank at the boundary: "
          f"e^-2I12 + e^-2I13 = {bud:.4f}")
    assert abs(bud - 1) < 0.02
    print("  sharing with one partner caps sharing with another: "
          "the filter-side CKW,")
    print("  exact in the Gaussian tier -- the budget is the "
          "positive-definiteness of the")
    print("  world's correlation matrix\n")


if __name__ == "__main__":
    s1_tracking()
    s2_monogamy()
    print("all assertions passed")
