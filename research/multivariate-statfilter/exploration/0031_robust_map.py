"""Probe 0031 -- the robust measurement update, DERIVED (replacing the empirical 4-sigma gate).

The 0030 robust gate inflated a sensor's R by (1 + max(nis - 16, 0)) on a clearly-white outlier
-- the 16 (4-sigma) was measured "pretty good", not derived, and the max()/if-else is a hard
transition.  Both are foreign to the rest of the filter.

Theory.  The measurement variance R_i = rho_i e^{eta_i} is UNCERTAIN -- its log-scale breathes
with a posterior.  Marginalising the Gaussian measurement likelihood over that scale uncertainty
gives a HEAVY-TAILED predictive: a large innovation is partly explained by a larger scale, so it
inflates R and is down-weighted -- SMOOTHLY, no threshold.  Concretely, with innovation variance
S(eta) = c + rho e^{eta} (c = the state's share (H Pp H^T)_ii) and a Gaussian scale prior
N(mu, sig2), the MAP condition L'(eta) = 0 is

        eta - mu = 1/2 sig2 (1 - c/S)(e^2/S - 1).                      (*)

It is exactly the shape the theory should give:
  * e^2 = S (consistent)  -> RHS 0 -> eta = mu (no change);
  * e^2 >> S (outlier)    -> RHS > 0 -> eta up -> S up -> down-weight, smoothly;
  * signed (raises on e^2>S, lowers on e^2<S) -> NO branch;
  * the factor (1 - c/S) = the sensor's OWN share of the innovation variance -> a *derived*
    attribution of the excess to the sensor, not a gate;
  * the only constant is sig2 = the class scale-swing s^2 -- already in the model, no new knob.

This probe checks (a) the Newton solver hits (*), (b) the resulting R-inflation is a smooth
heavy-tail (compare to the old hinge), and (c) the derived weight vs the 4-sigma weight.
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
from statfilter import AdaptiveKalmanFilter  # noqa: E402

np.set_printoptions(precision=4, suppress=True)


def check_map():
    rho, c, mu = 1.0, 0.3, 0.0
    print("(a) Newton solve hits the MAP condition (*)  [residual should be ~0]:")
    for ei2 in (1.0, 4.0, 25.0, 100.0, 400.0):
        eta = AdaptiveKalmanFilter._robust_eta(ei2, c, rho, mu, 0.25)
        S = c + rho * math.exp(eta)
        resid = (eta - mu) - 0.5 * 0.25 * (1 - c / S) * (ei2 / S - 1)
        print(f"    e^2/S0={ei2 / (c + rho):6.1f}  eta_MAP={eta:+.3f}  R_inflation={math.exp(eta):7.2f}x"
              f"  (*)-residual={resid:+.1e}")


def compare_weights():
    """Derived heavy-tail vs the old 4-sigma hinge, as a function of the outlier size."""
    rho, c, mu, sig2 = 1.0, 0.05, 0.0, 0.25
    print("\n(b) R-inflation vs outlier size:  DERIVED heavy-tail (smooth) vs 4-sigma hinge (kink):")
    print(f"    {'nis':>6} {'derived':>9} {'4-sigma gate':>13}")
    for nis in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 60.0, 225.0):
        eta = AdaptiveKalmanFilter._robust_eta(nis * (c + rho * math.exp(mu)), c, rho, mu, sig2)
        derived = math.exp(eta)
        hinge = 1.0 + max(nis - 16.0, 0.0)
        print(f"    {nis:6.1f} {derived:9.2f} {hinge:13.2f}")
    print("    -> derived responds smoothly from nis~1 (no flat-then-hinge at 16); no threshold.")


if __name__ == "__main__":
    check_map()
    compare_weights()
