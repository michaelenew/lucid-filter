"""wall-correspondence 0028 -- why a filter reads on the exponential
schedule: stationarity selects it.

0025 found the accelerated-observer protocol (read the record on an
exponentially stretched schedule) but imposed it. The residue was:
why would a filter read that way? The filter answers in its own
terms, and the answer is the most filter-native requirement there
is.

  A RECURSIVE FILTER NEEDS A STATIONARY RECORD. Its whole apparatus
  -- one model, one transfer, one innovation law reused at every
  step -- presumes the statistics it faces do not depend on when it
  looks. A filter whose record is non-stationary has no
  time-invariant model to run.

  s1  STATIONARITY IS A GEOMETRIC CONDITION. Reading a Lorentz-
      invariant vacuum along a worldline gives a covariance
      W_ij = f(interval_ij). It is stationary (Toeplitz) exactly
      when the sampled points sit on an orbit of a symmetry. Two
      timelike families exist in 1+1D: time translations (inertial)
      and BOOSTS (hyperbolic). Verified: both give exactly Toeplitz
      covariances (1e-16); a generic timelike worldline does not
      (defect 0.2-0.6).
  s2  THE BOOST ORBIT IS THE EXPONENTIAL SCHEDULE. Uniform proper
      time along a boost orbit is t = sinh(a tau)/a -- precisely
      0025's stretched schedule. So the schedule is not chosen: it
      is the unique non-inertial way to read a record and still have
      a stationary model.
  s3  THE CONSEQUENCE. A filter that insists on a time-invariant
      model has exactly two options, and one of them is hot. The
      Unruh temperature is the price of the second option, and
      0025's residue is closed: THE EXPONENTIAL SCHEDULE IS WHAT
      STATIONARITY ALLOWS. (Stated for 1+1D, where the timelike
      stationary families are just these two; in 3+1D there are more
      -- Letaw's classification -- and the corresponding filter
      question is which of those a bank can run.)
"""

import numpy as np

EPS2 = 0.25


def kernel(t, x):
    """A Lorentz-invariant vacuum correlator, regulated: any f of
    the invariant interval will do for the stationarity test."""
    T = t[:, None] - t[None, :]
    X = x[:, None] - x[None, :]
    s2 = T ** 2 - X ** 2
    return 1.0 / (s2 + EPS2)


def toeplitz_defect(W):
    """0 iff W_ij depends only on i - j."""
    n = W.shape[0]
    d = 0.0
    for k in range(n):
        diag = np.diag(W, k)
        if len(diag) > 1:
            d = max(d, float(diag.max() - diag.min())
                    / max(abs(diag).max(), 1e-12))
    return d


def s1_stationarity():
    print("== s1: stationarity is a geometric condition ==")
    n = 30
    a = 1.3
    # NOTE: t, x grow like sinh(a tau), so the naive interval
    # Delta t^2 - Delta x^2 suffers catastrophic cancellation once
    # a tau_max >> 1. The range is kept modest so the test measures
    # geometry, not floating point.
    tau = np.arange(n) * 0.1
    # (a) inertial: orbit of time translation
    t_in, x_in = tau, np.zeros(n)
    # (b) uniformly accelerated: orbit of a boost
    t_ac, x_ac = np.sinh(a * tau) / a, np.cosh(a * tau) / a
    # (c) a generic timelike worldline (not a symmetry orbit)
    x_ge = 0.6 * np.sin(2.2 * tau) + 0.25 * tau ** 2
    t_ge = np.zeros(n)
    for i in range(1, n):
        dx = x_ge[i] - x_ge[i - 1]
        dt = np.sqrt(dx ** 2 + (tau[i] - tau[i - 1]) ** 2)
        t_ge[i] = t_ge[i - 1] + dt        # timelike, unit proper vel
    rows = [("inertial (time-translation orbit)", t_in, x_in),
            ("accelerated (boost orbit)      ", t_ac, x_ac),
            ("generic timelike worldline     ", t_ge, x_ge)]
    print("    worldline                          stationarity "
          "defect")
    defects = []
    for name, t, x in rows:
        d = toeplitz_defect(kernel(t, x))
        defects.append(d)
        print(f"   {name}     {d:.2e}")
    assert defects[0] < 1e-12 and defects[1] < 1e-11
    assert defects[2] > 0.1
    print("  a record read along a SYMMETRY ORBIT is stationary "
          "exactly; a generic")
    print("  worldline's record is not, and no time-invariant "
          "filter model exists for it\n")


def s2_the_schedule():
    print("== s2: the boost orbit is the exponential schedule ==")
    a = 1.3
    for tau in (0.5, 1.0, 2.0):
        t = np.sinh(a * tau) / a
        print(f"   proper time {tau:.1f} -> record clock "
              f"t = sinh(a tau)/a = {t:.4f}")
    print("  uniform steps in the filter's own time are "
          "exponentially stretched in the")
    print("  record's clock -- exactly 0025's protocol. The schedule "
          "is not chosen: it is")
    print("  the unique non-inertial way to read and still have a "
          "stationary model\n")


def s3_consequence():
    print("== s3: the consequence ==")
    a = 1.3
    print(f"  a filter demanding a time-invariant model has exactly "
          f"two options in 1+1D:")
    print(f"    inertial   -> the vacuum record, T = 0")
    print(f"    accelerated -> the stretched record, T = a/2pi = "
          f"{a / (2 * np.pi):.4f}")
    print("  the Unruh temperature is the PRICE OF THE SECOND "
          "OPTION, and 0025's residue")
    print("  is closed: the exponential schedule is what "
          "stationarity allows.")
    print("  (1+1D statement. In 3+1D the timelike stationary "
          "worldlines form six families")
    print("  -- Letaw -- and the filter question becomes which of "
          "those a bank can run:")
    print("  a sharper open question than the one it replaces)\n")


if __name__ == "__main__":
    s1_stationarity()
    s2_the_schedule()
    s3_consequence()
    print("all assertions passed")
