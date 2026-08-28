"""wall-correspondence 0038 -- the acceptance criterion is part of
the model, and choosing it late has a measurable price.

Twice running, the sibling's stated criterion was the weak link
rather than the physics. Their 0121 tested whether four fitted
exponents supported a power law with |mean - 2| < 0.8, and it PASSED
on estimates of -0.36, 1.02, 4.15 and 4.98 -- a mean that is
arithmetic, not evidence. Their 0120's first verdict compared the
wrong quantity and concluded the opposite of the truth.

This is the program's own principle turned on its own practice. A
prequential score requires the model to be committed BEFORE the
data; an acceptance criterion IS part of the model, so choosing it
after seeing the spread is the same error as fitting on the test
set. That is an argument. What follows is the measurement, because
an engineer should pick a criterion by its operating characteristic
and not by taste.

  s1  THE FALSE-POSITIVE RATE. Under a null where the items have NO
      common exponent -- each drawn from a broad range, which is
      exactly what "no power law describes this" looks like -- how
      often does each test say yes? The mean-based test says yes
      about half the time. The per-item consistency test says yes
      about one time in a hundred.
  s2  THE POWER. A strict test is worthless if it also rejects the
      truth. When the truth IS p = 2 with realistic per-item noise,
      how often does each test say yes? Measured across a range of
      noise, so the trade is visible rather than asserted.
  s3  THE OPERATING CHARACTERISTIC. The two rates together, and the
      noise level at which the consistency test stops being usable
      -- which is the real design constraint, and which converts
      "write the criterion down first" from a moral into a
      specification: it tells you how many items and how much
      per-item precision the measurement must buy.
"""

import numpy as np

rng = np.random.default_rng(38)

TARGET = 2.0
N_ITEMS = 4
TRIALS = 40000

# the two tests, exactly as they were written
def test_mean(exps, tol=0.8):
    return abs(np.mean(exps) - TARGET) < tol


def test_consistency(exps, lo=1.0, hi=3.0, spread=1.5):
    return (all(lo <= e <= hi for e in exps)
            and (max(exps) - min(exps)) <= spread)


def s1_false_positive():
    print("== s1: the false-positive rate ==")
    print("  null: the items have NO common exponent -- each drawn "
          "uniformly from a broad")
    print("  range, which is what 'no power law describes this' "
          "actually looks like.")
    print("     null range        mean-based test    consistency "
          "test   ratio")
    out = {}
    for lo, hi in ((-1.0, 6.0), (0.0, 5.0), (-2.0, 8.0)):
        e = rng.uniform(lo, hi, (TRIALS, N_ITEMS))
        fm = float(np.mean([test_mean(x) for x in e]))
        fc = float(np.mean([test_consistency(x) for x in e]))
        out[(lo, hi)] = (fm, fc)
        print(f"    [{lo:+.0f}, {hi:+.0f}]          {fm:.4f}"
              f"             {fc:.4f}          "
              f"{fm / max(fc, 1e-9):6.0f}x")
    fm, fc = out[(-1.0, 6.0)]
    print(f"  on the range that matches what their 0121 actually "
          f"saw (-0.36 to 4.98), the")
    print(f"  mean-based test is wrong {100 * fm:.0f}% of the time "
          f"and the consistency test")
    print(f"  {100 * fc:.1f}% -- a factor {fm / fc:.0f}.")
    assert fm > 10 * fc
    print("  THAT IS THE PRICE OF THE CRITERION, and it was paid "
          "twice before being seen\n")
    return out


def s2_power():
    print("== s2: the power ==")
    print("  truth: every item really has p = 2, with per-item "
          "noise sigma (fitting four")
    print("  or five points of a noisy log-log slope is not "
          "precise).")
    print("     sigma    mean-based test    consistency test")
    rows = []
    for sig in (0.2, 0.4, 0.6, 0.8, 1.2):
        e = TARGET + sig * rng.standard_normal((TRIALS, N_ITEMS))
        pm = float(np.mean([test_mean(x) for x in e]))
        pc = float(np.mean([test_consistency(x) for x in e]))
        rows.append((sig, pm, pc))
        print(f"    {sig:.1f}       {pm:.4f}             {pc:.4f}")
    assert rows[0][2] > 0.8
    print("  the consistency test keeps most of its power while the "
          "per-item scatter stays")
    print("  below about half a unit, and collapses beyond it. THAT "
          "IS THE DESIGN")
    print("  CONSTRAINT, not a matter of taste\n")
    return rows


def s3_operating_characteristic(fp, power):
    print("== s3: the operating characteristic ==")
    fm, fc = fp[(-1.0, 6.0)]
    print("     sigma   consistency: power   false-positive   "
          "usable?")
    best = None
    for sig, _pm, pc in power:
        use = pc > 0.5 and fc < 0.05
        if use and best is None:
            best = sig
        print(f"    {sig:.1f}          {pc:.3f}              "
              f"{fc:.3f}          {'yes' if use else 'no'}")
    print()
    print("  So the specification, which is what 'write the "
          "criterion down first' really")
    print("  means once it is quantified:")
    print(f"    - fix the test before the run (worth a factor "
          f"{fm / fc:.0f} in false positives);")
    print("    - require PER-ITEM agreement, never an aggregate the "
          "spread contradicts;")
    print("    - and size the measurement so the per-item scatter "
          "lands below ~0.5,")
    print("      because past that the honest test has no power and "
          "the run is wasted")
    print("      BEFORE it starts -- which is a thing you can know "
          "in advance.")
    print()
    print("  Their 0121 failed the last clause: its usable window "
          "in the probe scale spanned")
    print("  a factor 1.6, which cannot pin a slope to +-0.5. The "
          "run was unable to answer")
    print("  its question before it was launched, and the criterion "
          "hid that instead of")
    print("  surfacing it. A pre-registered power requirement would "
          "have caught it.\n")


if __name__ == "__main__":
    fp = s1_false_positive()
    power = s2_power()
    s3_operating_characteristic(fp, power)
    print("all assertions passed")
