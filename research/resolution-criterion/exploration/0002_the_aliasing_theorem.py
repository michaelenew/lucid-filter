"""0002 -- one theorem replaces the Sparrow factor at all three sites.

Every resolution site in the filter has the same form:  a continuous parameter
with a local blur width `b` (one over the root Fisher information over the
relevant horizon), gridded uniformly at spacing `gap = c * b`, and mixed over.
The shipped `c = 1.5` is the Sparrow optical analogy, at all three:

    walk grid      gap = 1.5 * s                 b = s (class prior SD)
    hazard ladder  gap = 1.5 * 1 e-fold          b = 1/sqrt(n), n = 1 event
    split ladder   gap = 1.5 * sqrt(2/mem)       b = sqrt(2/mem) (arclength, I = 1/step)

THEOREM (uniform quadrature of a Gaussian).  A uniform grid of spacing `Delta`
applied to a Gaussian integrand of SD `b` has, by Poisson summation, relative
error in its normalisation

    err(c) = 2 * sum_{m>=1} exp(-2 pi^2 m^2 / c^2)  ~  2 exp(-2 pi^2 / c^2),   c = Delta / b,

a universal function of `c` alone, and the same rate governs the first and second
moments.  So `c = 1.5` is not an analogy: it is the spacing at which the grid
reproduces a Gaussian of width `b` to relative aliasing

    eps0 = err(1.5) = 2 exp(-2 pi^2 / 2.25) ~ 3.1e-4,

and it is that SAME tolerance at all three sites.  That is the sharp statement
AUD-1 said was missing.

THE WALK'S EXTRA, ABSOLUTE BOUND.  The walk grid also has to let the window
CENTRE move across it (the walk), which needs the between-node SCORE to keep its
sign.  The score is the derivative, so its aliasing is the likelihood's spectral
alias times omega = 2 pi/Delta.  With 0001's exact spectrum
|g^(omega)| = 1/sqrt(u cosh(pi omega)):

    score alias(Delta) ~ (2 pi/Delta) * 2 * sqrt2 * exp(-pi^2 / Delta).

Setting it to the same eps0 gives an ABSOLUTE spacing in nats -- and the
prediction to check is whether it lands on the dead-zone onset adaptive-grid
MEASURED at ~0.7-0.8 nats (finding 11), which has never had a derivation.

Run: python3 0002_the_aliasing_theorem.py
"""
import math

import numpy as np


def poisson_err(c, terms=6):
    return 2.0 * sum(math.exp(-2 * math.pi ** 2 * m * m / c / c) for m in range(1, terms + 1))


def measured_err(c, nphase=64):
    """Worst-case-over-phase relative error of a uniform grid, spacing c*b, on N(0,b^2):
    normalisation, mean, and variance."""
    b = 1.0
    D = c * b
    worst = np.zeros(3)
    for ph in np.linspace(0, D, nphase, endpoint=False):
        x = ph + D * np.arange(-400, 401)
        w = D * np.exp(-0.5 * (x / b) ** 2) / (b * math.sqrt(2 * math.pi))
        Z = w.sum()
        mean = (w * x).sum() / Z
        var = (w * (x - mean) ** 2).sum() / Z
        worst = np.maximum(worst, [abs(Z - 1.0), abs(mean), abs(var - b * b) / (b * b)])
    return worst


def theorem_check():
    print("1. THEOREM: uniform grid at c*b on a Gaussian of SD b -- worst-case over grid phase")
    print(f"   {'c':>5s}{'err theory':>13s}{'|Z-1| meas':>13s}{'|mean| meas':>13s}"
          f"{'|dvar|/var':>13s}")
    for c in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0):
        th = poisson_err(c)
        m = measured_err(c)
        print(f"   {c:5.2f}{th:13.3e}{m[0]:13.3e}{m[1]:13.3e}{m[2]:13.3e}")
    print("   (normalisation matches the Poisson formula; mean and variance share its rate.)")


def three_sites():
    eps0 = poisson_err(1.5)
    print(f"\n2. THE THREE SITES at the shipped c = 1.5  ->  eps0 = {eps0:.3e}")
    print(f"   {'site':>14s}{'blur b':>22s}{'gap':>18s}{'c':>6s}{'aliasing':>12s}")
    print(f"   {'walk grid':>14s}{'s  (class prior SD)':>22s}{'1.5 s':>18s}{1.5:6.2f}{eps0:12.2e}")
    print(f"   {'hazard ladder':>14s}{'1 e-fold (n=1 event)':>22s}{'1.5 nats':>18s}{1.5:6.2f}{eps0:12.2e}")
    print(f"   {'split ladder':>14s}{'sqrt(2/mem)':>22s}{'1.5 sqrt(2/mem)':>18s}{1.5:6.2f}{eps0:12.2e}")
    print("   One tolerance, three sites.  The constant was never three separate analogies.")
    return eps0


def walk_absolute_bound(eps0):
    print("\n3. THE WALK'S ABSOLUTE BOUND from the exact log-scale spectrum (0001)")
    # normalisation alias of the likelihood itself:  2*sqrt2*exp(-pi^2/Delta)
    # score alias (the walk stalls):  (2pi/Delta) * 2*sqrt2*exp(-pi^2/Delta)
    def norm_alias(D):
        return 2 * math.sqrt(2) * math.exp(-math.pi ** 2 / D)

    def score_alias(D):
        return (2 * math.pi / D) * norm_alias(D)

    def solve(f, target):
        lo, hi = 0.2, 6.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(mid) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    D_norm = solve(norm_alias, eps0)
    D_score = solve(score_alias, eps0)
    print(f"   likelihood normalisation alias = eps0  ->  Delta = {D_norm:.3f} nats")
    print(f"   likelihood SCORE alias (walk stalls) = eps0  ->  Delta = {D_score:.3f} nats")
    print(f"   adaptive-grid finding 11 MEASURED the dead-zone onset at ~0.7-0.8 nats,")
    print(f"   order-independent, with no derivation.  The score bound at the filter's own")
    print(f"   tolerance lands at {D_score:.2f} nats.")
    print(f"\n   Consequence: the walk needs BOTH  gap <= 1.5 s  (resolve the prior)  AND")
    print(f"   gap <= {D_score:.2f} nats  (the walk can cross the grid).  The second is absolute.")
    print(f"   It binds once 1.5 s > {D_score:.2f}, i.e. s > {D_score / 1.5:.2f}.  In the shipped box")
    print(f"   s in (0.2, 0.4, 0.8, 1.6, 3.2), gap in (0.3, 0.6, 1.2, 2.4, 4.8): the three")
    print(f"   largest-s members exceed it per member.  The BANK covers that -- small-s members")
    print(f"   carry a fine grid where the walk must move, large-s members carry reach -- which")
    print(f"   is why the shipped filter works despite the per-member violation.")
    return D_norm, D_score


if __name__ == "__main__":
    print(__doc__.split("Run:")[0])
    theorem_check()
    eps0 = three_sites()
    walk_absolute_bound(eps0)
