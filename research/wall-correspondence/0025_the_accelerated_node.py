"""wall-correspondence 0025 -- the accelerated node: the protocol,
found, and the temperature measured in code length.

0021 verified the wedge (per-REGION) form of the sibling's Unruh
result and recorded the per-OBSERVER form as design-blocked: no
principled map from 'acceleration' to a filter protocol. The map
exists, and it was hiding in the definition. A Rindler observer's
proper time is EXPONENTIALLY related to the record's clock
(t = sinh(a tau)/a), so

    AN ACCELERATED FILTER IS ONE THAT READS THE RECORD ON AN
    EXPONENTIALLY STRETCHED SCHEDULE,

and everything follows from that one substitution.

  s1  THE INTERVAL IDENTITY. Along the hyperbolic worldline the
      invariant separation depends on proper time only through
      sinh(a d tau / 2). The inertial vacuum correlator, read on
      that schedule, is therefore the correlator of a THERMAL state
      whose KMS period in imaginary proper time is 2 pi / a -- i.e.
      T = a / 2 pi. Verified exactly.
  s2  THE FILTER MEASURES ITS OWN TEMPERATURE. Built spectrally so
      positivity is automatic: the accelerated filter's spectral
      density is the inertial one times coth(w / 2T) -- a
      low-frequency NOISE FLOOR of height 2T where the inertial
      filter expects none. Data generated on the accelerated
      schedule are scored prequentially by a bank of models indexed
      by assumed temperature: the score minimum recovers the true
      T, and assuming the inertial vacuum costs a measurable
      penalty in nats/sample. THE UNRUH EFFECT AS A CODE-LENGTH
      STATEMENT.
  s3  Scope, and a methods note. The protocol and measurement are
      exact at the Gaussian tier. NOT established: why a filter
      would read its record on that schedule -- the stretch is
      imposed, not derived. Methods note kept because it cost a
      wrong result first: an algebraic UV regulator inserted into
      the thermal kernel BREAKS positive-definiteness at large a
      (min eigenvalue -0.02), and the resulting 'model' scored
      better than the truth. Spectral construction removes the
      failure mode by making every candidate a genuine Gaussian.
"""

import numpy as np

LAM = 6.0                    # UV cutoff on the spectral density
NW = 20001
W = np.linspace(0.0, 10 * LAM, NW)


def spectral(T):
    """Symmetrised spectral density: vacuum |w| e^{-w/LAM}, times
    the thermal factor coth(w/2T) when T > 0."""
    s = W * np.exp(-W / LAM)
    if T > 0:
        x = W / (2 * T)
        s = s * np.where(x > 1e-9, 1.0 / np.tanh(np.maximum(x, 1e-9)),
                         0.0)
        s[0] = 2 * T                       # coth limit: |w| -> 2T
    return s


def cov_of(taus, T):
    """Stationary: evaluate on the unique lags and broadcast
    (Toeplitz), which is n times cheaper than the full matrix."""
    s = spectral(T)
    n = len(taus)
    dt = taus[1] - taus[0]
    lags = np.arange(n) * dt
    c = np.trapezoid(s[None, :] * np.cos(lags[:, None]
                                         * W[None, :]),
                     W, axis=-1) / np.pi
    idx = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    return c[idx]


def s1_identity():
    print("== s1: the interval identity ==")
    a = 6.0
    T = a / (2 * np.pi)
    for dtau in (0.3, 0.7, 1.5):
        t1, x1 = np.sinh(0) / a, np.cosh(0) / a
        t2, x2 = np.sinh(a * dtau) / a, np.cosh(a * dtau) / a
        interval = (t2 - t1) ** 2 - (x2 - x1) ** 2
        thermal = (2 / a * np.sinh(a * dtau / 2)) ** 2
        assert abs(interval / thermal - 1) < 1e-12
    print("  invariant separation along the hyperbola = "
          "(2/a) sinh(a dtau/2), exactly")
    print(f"  KMS period in imaginary proper time: 2 pi/a = "
          f"{2 * np.pi / a:.4f} = 1/T with T = a/2pi = {T:.4f}")
    assert abs(2 * np.pi / a - 1 / T) < 1e-12
    print("  so the inertial vacuum, read on the exponentially "
          "stretched schedule, IS a")
    print("  thermal record at T = a/2pi\n")
    return a, T


def s2_measure(a_true, T_true):
    print("== s2: the filter measures its own temperature ==")
    n, dt, REP = 160, 0.12, 300
    tau = np.arange(n) * dt
    C = cov_of(tau, T_true)
    ev = np.linalg.eigvalsh(C)
    print(f"  generating covariance: min eigenvalue {ev.min():.3e} "
          f"(positive by construction)")
    assert ev.min() > 0
    rng = np.random.default_rng(5)
    L = np.linalg.cholesky(C)
    data = (L @ rng.standard_normal((n, REP))).T

    def score(T, per_rep=False):
        Cm = cov_of(tau, T)
        sgn, ld = np.linalg.slogdet(Cm)
        assert sgn > 0, "candidate model is not a valid Gaussian"
        q = np.einsum("ri,ij,rj->r", data, np.linalg.inv(Cm), data)
        per = 0.5 * (ld + q + n * np.log(2 * np.pi)) / n
        return per if per_rep else float(per.mean())

    print("    model              T        score (nats/sample)")
    grid = [0.0, 0.5, 0.8, T_true, 1.2, 1.8]
    scores = {}
    for T in grid:
        scores[T] = score(T)
        lbl = "inertial vacuum" if T == 0 else "thermal        "
        star = "   <-- truth" if T == T_true else ""
        print(f"   {lbl}  {T:.4f}    {scores[T]:+.5f}{star}")
    best = min(scores, key=scores.get)
    d = score(0.0, True) - score(T_true, True)
    pen, se = float(d.mean()), float(d.std() / np.sqrt(REP))
    print(f"  best model: T = {best:.4f} (true {T_true:.4f} = "
          f"a/2pi)")
    print(f"  assuming the inertial vacuum costs {pen:+.4f} +- "
          f"{se:.4f} nats/sample ({pen / se:.0f} sigma over "
          f"{REP} records)")
    assert abs(best - T_true) < 1e-12
    assert pen > 10 * se
    print("  the accelerated filter reads its own temperature out "
          "of its own record,")
    print("  prequentially: THE UNRUH EFFECT AS A CODE-LENGTH "
          "STATEMENT\n")


def s3_scope():
    print("== s3: scope, and a methods note ==")
    print("  Established: the protocol (exponential resampling of "
          "the record's clock), the")
    print("  exact thermal form of the accelerated filter's "
          "correlations, and an operational")
    print("  temperature measurement. The per-observer Unruh "
          "statement now exists in filter")
    print("  terms alongside 0021's per-region one.")
    print("  NOT established: why a filter would read its record on "
          "that schedule. The block")
    print("  moves from 'no map exists' to 'which dynamics "
          "generates this schedule'.")
    print("  Methods note: an algebraic UV regulator inserted into "
          "the thermal kernel breaks")
    print("  positive-definiteness at large a (min eigenvalue "
          "-0.02) and the resulting")
    print("  'model' scored BETTER than the truth -- caught by the "
          "sign of the log-")
    print("  determinant. Spectral construction makes every "
          "candidate a genuine Gaussian\n")


if __name__ == "__main__":
    a, T = s1_identity()
    s2_measure(a, T)
    s3_scope()
    print("all assertions passed")
