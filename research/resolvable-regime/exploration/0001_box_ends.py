"""0001 — the arithmetic behind the derived box ends, and the Fisher check.

Stdlib only.  Three things:

  1. The per-step log-scale increment SD across the shipped `(phi, s)` box,
     against the derived resolution ceiling sqrt(2) (C5).
  2. The derived admissible band [sqrt(2/N), sqrt(2)] and its rung count,
     against the shipped `_SS` ladder.
  3. A numerical confirmation that the Fisher information a single Gaussian
     observation carries about its own log-variance is exactly 1/2, which is
     what sets the blur width sqrt(2) that both ends are measured in.

Run: python3 0001_box_ends.py
"""
import math
import random

PHIS = (0.70, 0.85, 0.95)
SS = (0.20, 0.40, 0.80, 1.60, 3.20)
CEIL = math.sqrt(2.0)          # one-event blur width on a log-scale, f = 1
GAP = 1.5                      # the filter's own Sparrow spacing (_GAP_FACTOR)


def box_increments():
    """Per-step increment SD of the log-scale, s * sqrt(1 - phi^2), over the box."""
    print("per-step increment SD = s * sqrt(1 - phi^2)")
    print("        " + "  ".join(f"s={s:4.2f}" for s in SS))
    over = 0
    for p in PHIS:
        row = [s * math.sqrt(1.0 - p * p) for s in SS]
        over += sum(1 for v in row if v > CEIL)
        print(f"phi={p:.2f} " + "  ".join(f"{v:6.3f}" for v in row))
    allv = [s * math.sqrt(1.0 - p * p) for p in PHIS for s in SS]
    print(f"\nceiling sqrt(2) = {CEIL:.4f}; {over} of {len(allv)} members sit above it")
    print(f"box increment range: {min(allv):.4f} -> {max(allv):.4f}")


def derived_band(n_eff=1000, f=1.0):
    """Ceiling from one-event resolution, floor from N-event visibility."""
    lo, hi = math.sqrt(2.0 / (f * f * n_eff)), math.sqrt(2.0) / f
    print(f"\nderived band at N={n_eff}, f={f}: [{lo:.4f}, {hi:.4f}]"
          f"  dynamic range {hi / lo:.1f}")
    print(f"rungs = log(N)/gap = {math.log(n_eff) / GAP:.2f}"
          f"   ladder ratio in sigma^2 = e^{GAP} = {math.exp(GAP):.2f}")
    print(f"shipped _SS: {len(SS)} rungs, ratio in s^2 = {(SS[1] / SS[0]) ** 2:.1f}"
          f"  (log-ratio {math.log((SS[1] / SS[0]) ** 2):.3f} vs derived {GAP})")


def fisher_logvariance(n=2_000_000, seed=0):
    """I(lambda) for y ~ N(0, e^lambda): score is -1/2 + y^2 e^-lambda / 2."""
    random.seed(seed)
    tot = 0.0
    for _ in range(n):
        z = random.gauss(0.0, 1.0)
        tot += (-0.5 + 0.5 * z * z) ** 2
    print(f"\nFisher I(lambda) numeric: {tot / n:.5f}   analytic: 0.5"
          f"   blur width sqrt(1/I) = {math.sqrt(2.0):.4f}")


if __name__ == "__main__":
    box_increments()
    derived_band()
    fisher_logvariance()
