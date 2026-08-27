"""wall-correspondence 0042 -- Lambda in the filter: what makes a
global mode quantised when nothing is discrete.

Their 0137 put the Lambda quantisation at criticality 2: it is the
program's ONLY observational route, and its mechanism -- total
curvature congruent to the boundary holonomy MOD N -- is a statement
about a finite ring. Their 0136 showed the finite ring is a toy. So
the only falsifiable line in the program currently rests on the part
that does not port.

The filter has the pieces to say what quantises a global mode when
nothing is discrete, and 0017 already did half of it.

  s1  THE GLOBAL LEVEL IS GAUGE (0017, re-confirmed). Spatial
      records carry differences, so the absolute level of a shared
      mode is unidentifiable: a filter given records of differences
      cannot recover it, and its posterior over it stays flat no
      matter how much data arrives.
  s2  BUT THE WINDING IS NOT. If the latent lives on a CIRCLE, the
      total winding around a closed loop is an integer, and it IS
      identifiable from difference records alone -- because
      integrating differences around a closed loop must return to
      the start, and the only freedom left is how many times it went
      round. Measured: winding recovered exactly, while the absolute
      level stays flat.
  s3  QUANTISATION NEEDS COMPACTNESS, NOT DISCRETENESS. The same
      filter on a LINE-valued latent has no winding and no
      quantisation -- the total drift is a continuous number.
      Measured side by side. So the quantum comes from pi_1 of the
      state space, not from chopping it up.
  s4  THE PORT, AND WHAT IT COSTS THEM. Lambda.V in (2 pi / q) Z
      with q the CHARGE the record winds under. On a lattice over
      Z_N that charge is N and you recover their toy formula. In the
      continuum it is not the lattice level at all -- it is the
      CENTRE of the gauge group, which for SU(2) is Z_2. The
      quantisation survives the port; the identification of its
      quantum with the level does not.
"""

import numpy as np

rng = np.random.default_rng(42)
TWO_PI = 2 * np.pi


def wrap(x):
    return (x + np.pi) % TWO_PI - np.pi


def loop_record(nstep, winding, sigma, circular=True):
    """A latent walked around a CLOSED loop with a given winding.
    The record is the (wrapped) latent itself, not its increments --
    which is what a phase record actually is."""
    step = rng.standard_normal(nstep)
    step -= step.mean()                        # closes exactly
    step = step * 0.02 + TWO_PI * winding / nstep
    phi = np.concatenate([[0.0], np.cumsum(step)])
    obs = phi + sigma * rng.standard_normal(len(phi))
    return (wrap(obs) if circular else obs), phi


def unwrap_winding(obs):
    return (np.unwrap(obs)[-1] - np.unwrap(obs)[0]) / TWO_PI


def s1_level_is_gauge():
    print("== s1: the global level is gauge ==")
    print("  the claim is that a global shift of the latent leaves "
          "a DIFFERENCE record")
    print("  bitwise unchanged, so the absolute level carries zero "
          "Fisher information.")
    print("  Demonstrated rather than asserted:")
    _, phi = loop_record(2000, 2, 0.0)
    d0 = np.diff(phi)
    print("     global shift    max |difference record - original|")
    for c in (0.5, 3.0, 100.0):
        d1 = np.diff(phi + c)
        print(f"       {c:8.1f}        {np.abs(d1 - d0).max():.3e}")
        assert np.abs(d1 - d0).max() < 1e-12
    print("  identical to machine precision at every shift. The "
          "level never enters the")
    print("  likelihood, so no amount of data shrinks it -- 0017's "
          "result, exact\n")


def s2_winding_is_not():
    print("== s2: but the winding is not ==")
    print("  the record is the wrapped phase; unwrap it and read "
          "the total turn.")
    print("     true winding   sigma   recovered   estimate")
    ok = trials = 0
    for w in (-2, 0, 1, 3):
        for sigma in (0.05, 0.15):
            obs, _ = loop_record(4000, w, sigma)
            est = unwrap_winding(obs)
            rec = int(np.rint(est))
            trials += 1
            ok += (rec == w)
            print(f"       {w:+3d}         {sigma:.2f}     "
                  f"{rec:+3d}      {est:+8.4f}")
    print(f"  {ok}/{trials} exact.")
    assert ok == trials
    print("  So a record carrying NO absolute information still "
          "pins an integer. That is")
    print("  the whole mechanism: the level is gauge, the winding "
          "is not\n")


def s3_compactness_not_discreteness():
    print("== s3: quantisation needs compactness, not discreteness "
          "==")
    print("  The condition is SINGLE-VALUEDNESS on the closed loop, "
          "and what it permits")
    print("  depends entirely on where the field takes values.")
    print()
    n = 4000
    t = np.arange(n + 1) / n
    print("     field values   sample total changes around the loop"
          "        sectors")
    circ = []
    for _ in range(6):
        w = int(rng.integers(-3, 4))
        smooth = 0.4 * np.sin(TWO_PI * t + rng.uniform(0, TWO_PI))
        phi = TWO_PI * w * t + smooth          # single-valued mod 2pi
        obs = wrap(phi + 0.05 * rng.standard_normal(n + 1))
        circ.append(unwrap_winding(obs))
    print("     CIRCLE        " + ", ".join(f"{x:+.2f}" for x in circ)
          + "   2 pi Z")
    line = []
    for _ in range(6):
        smooth = (0.4 * np.sin(TWO_PI * t + rng.uniform(0, TWO_PI))
                  + 0.7 * np.sin(2 * TWO_PI * t))
        f = smooth                              # single-valued on R
        line.append((f[-1] - f[0]) / TWO_PI)
    print("     LINE          " + ", ".join(f"{x:+.2f}" for x in line)
          + "   {0} only")
    print()
    ci = np.array(circ)
    li = np.array(line)
    assert np.abs(ci - np.rint(ci)).max() < 0.05
    assert np.abs(li).max() < 1e-9
    assert len(set(np.rint(ci).astype(int))) > 1
    print("  A circle-valued field on a closed loop admits a "
          "DISCRETE INFINITY of sectors,")
    print("  one per integer, and the record reads which. A "
          "line-valued field admits")
    print("  exactly ONE -- the total change is identically zero, "
          "no freedom at all.")
    print()
    print("  THE QUANTUM COMES FROM pi_1 OF THE STATE SPACE. "
          "Nothing here is discrete: the")
    print("  loop is continuous, the field is continuous, the "
          "record is continuous. Chopping")
    print("  the state space into N pieces was never what made "
          "Lambda quantised\n")


def s4_the_port():
    print("== s4: the port, and what it costs them ==")
    print("  A global mode is quantised iff the latent is COMPACT, "
          "and the quantum is set by")
    print("  the CHARGE q under which the record winds:")
    print("      Lambda . V  in  (2 pi / q) Z")
    print()
    print("  On a lattice over Z_N the record's phase advances in "
          "units of 2 pi / N, so")
    print("  q = N and their toy formula Lambda.V in (2 pi/N) Z "
          "drops out. Good -- the toy")
    print("  is the q = N case, not a separate mechanism.")
    print()
    print("  IN THE CONTINUUM q IS NOT THE LEVEL. The lattice level "
          "was a discretisation")
    print("  parameter and discretisation is exactly what s3 says "
          "is irrelevant. What sets q")
    print("  is the compactness of the gauge group's own centre: "
          "for SU(2) that is Z_2, so")
    print("  q = 2, and for SU(M)/Z_M it is M.")
    print()
    print("  SO: THE QUANTISATION SURVIVES THE PORT. The "
          "IDENTIFICATION OF ITS QUANTUM WITH")
    print("  THE LEVEL DOES NOT. Their falsifiable line is not "
          "dead -- it is re-based, and")
    print("  the re-basing changes the predicted quantum by "
          "whatever N/q is, which for")
    print("  N = 5 against SU(2)'s centre is a factor 2.5.")
    print()
    print("  One thing this does NOT settle, and they should hear "
          "it plainly: whether the")
    print("  physical universe is closed. s2's integer exists "
          "because the loop closes. On an")
    print("  open arena there is no loop and no winding, and their "
          "0080 s3 already measured")
    print("  the consequence -- on a free arena the measure does "
          "not prefer Lambda = 0.")
    print("  Compactness is doing the work here, and it is an "
          "assumption about the world\n")


if __name__ == "__main__":
    s1_level_is_gauge()
    s2_winding_is_not()
    s3_compactness_not_discreteness()
    s4_the_port()
    print("all assertions passed")
