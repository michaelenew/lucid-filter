"""wall-correspondence 0044 -- is the record's precision forced?

0043 reduced three open problems to one: N, s0 and their requirement
(D) are all "what sets the precision of the world's record". Until
that is answered the program cannot emit a falsifiable number --
every candidate (G = 5.165 a^2, xi/a ~ 10^13, the Lambda quantum)
carries a free parameter inside it.

This asks whether the precision is free at all. In a bank, no node's
record precision is externally given: a node learns about its
neighbour through a record whose quality is limited by the
NEIGHBOUR'S OWN state, which is limited in turn. The precision is
therefore a FIXED POINT of its own propagation, not an input -- the
same move that forced beta = 1 in 0022.

  s1  THE CAVITY EQUATION. Each node's posterior precision is built
      from its neighbours' precisions through link records of
      precision q. Solved numerically and in closed form: for
      degree d the fixed point is p* = (d-1) q, attractive.
      So GIVEN the link quality, the state precision is DERIVED --
      no freedom left in p.
  s2  CLOSING THE LOOP. But q is not external either: a link record
      is a comparison of two states, so its quality is limited by
      those states. Setting q = c p makes the equation scale-free
      and the fixed point collapses -- there is no p* unless
      c(d-1) = 1 exactly. Measured: the flow runs to zero below
      that and diverges above it.
  s3  WHAT THAT MEANS, STATED WITHOUT SPIN. Self-consistency does
      NOT pick a precision. It picks a CRITICAL RELATION between
      the link and state channels, and leaves the overall scale
      free -- which is the scale invariance the program has been
      calling "no dial". So the precision is not forced, and the
      number does not fall out this way.
  s4  WHERE THAT LEAVES A NUMBER. What survives is sharper than
      what went in: the missing input is a single SCALE, and any
      one measured quantity fixes it. That is what "N has the
      epistemic status of a measured coupling" has meant all along
      -- now with the reason attached.
"""

import numpy as np

rng = np.random.default_rng(44)


def cavity_step(p, q, d):
    """One round of belief propagation on a d-regular tree: a node's
    precision is the sum over neighbours of the series combination
    of the link record and the neighbour's own cavity precision."""
    return (d - 1) * (q * p) / (q + p) if p > 0 else 0.0


def s1_cavity():
    print("== s1: the cavity equation ==")
    print("  a node learns its neighbour only as well as the "
          "neighbour knows itself:")
    print("      p' = (d-1) * [ 1/q + 1/p ]^-1")
    print("     d    q      fixed point (iterated)   closed form "
          "(d-2)q ... check")
    for d, q in ((3, 1.0), (4, 1.0), (4, 2.5), (6, 0.4)):
        p = 10.0
        for _ in range(4000):
            p = cavity_step(p, q, d)
        closed = (d - 2) * q
        print(f"    {d}   {q:.1f}        {p:10.6f}            "
              f"{closed:10.6f}")
        assert abs(p - closed) < 1e-6
    print("  the fixed point is p* = (d-2) q, and it is attractive: "
          "start anywhere positive")
    print("  and the iteration lands on it.")
    print()
    print("  SO GIVEN THE LINK QUALITY, THE STATE PRECISION IS "
          "DERIVED. There is no freedom")
    print("  left in p at all -- it is whatever the network's own "
          "propagation makes it\n")
    return


def s2_close_the_loop():
    print("== s2: closing the loop ==")
    print("  but q is not external either. A link record COMPARES "
          "TWO STATES, so its quality")
    print("  is limited by them: q = c p. Substitute and the map "
          "becomes LINEAR --")
    print("      p' = [ (d-2) c / (c+1) ] p  =  lambda p")
    print("  so p cancels out of the fixed-point condition "
          "entirely, and lambda alone decides.")
    print("     d    c       lambda    flow of p over 200 rounds"
          "                    fate")
    for d, c in ((4, 0.4), (4, 1.0), (4, 2.0), (6, 0.25), (6, 1 / 3)):
        p = 1.0
        traj = []
        lam = (d - 2) * c / (c + 1)
        for k in range(200):
            q = c * p
            p = (d - 2) * (q * p) / (q + p)
            if k in (0, 49, 199):
                traj.append(p)
        fate = ("collapses to 0" if traj[-1] < 1e-6
                else "diverges" if traj[-1] > 1e6 else "MARGINAL")
        print(f"    {d}   {c:.3f}   {lam:.4f}   "
              + " -> ".join(f"{x:.3e}" for x in traj)
              + f"   {fate}")
    print()
    print("  lambda = 1 exactly when c (d - 3) = 1. Below it the "
          "network forgets faster than")
    print("  it learns and every precision runs to zero; above it "
          "precision runs away.")
    ok = 0
    for d in (4, 5, 6, 8):
        c = 1.0 / (d - 3)
        for start in (0.05, 1.0, 40.0):
            p = start
            for _ in range(2000):
                q = c * p
                p = (d - 2) * (q * p) / (q + p)
            ok += abs(p - start) < 1e-9
        print(f"    d = {d}, c_crit = 1/(d-3) = {c:.4f}: p holds at "
              f"its starting value from 0.05, 1.0 and 40.0")
    assert ok == 12
    print()
    print("  AT CRITICALITY EVERY p IS A FIXED POINT -- a LINE of "
          "them through the origin,")
    print("  not a point on it. The scale never gets picked\n")


def s3_no_spin():
    print("== s3: what that means, stated without spin ==")
    print("  SELF-CONSISTENCY DOES NOT PICK A PRECISION. It picks a "
          "CRITICAL RELATION")
    print("  between the link channel and the state channel, and "
          "leaves the overall scale")
    print("  free. The fixed point is a ray through the origin, not "
          "a point on it.")
    print()
    print("  That is not a failure of the calculation; it is the "
          "same scale invariance the")
    print("  program has been calling 'no dial'. A theory with no "
          "dial cannot manufacture a")
    print("  scale out of its own consistency -- if it could, the "
          "dial would be back.")
    print()
    print("  So: the precision is NOT forced, and the number does "
          "NOT fall out this way.")
    print("  I expected it might. It does not\n")


def s4_where_that_leaves_it():
    print("== s4: where that leaves a number ==")
    print("  What survives is sharper than what went in. Before, "
          "three separate quantities")
    print("  were undetermined (N, s0, the coupling). Now they are "
          "ONE quantity -- a single")
    print("  overall scale -- and the criticality relation fixes "
          "everything else about the")
    print("  network given it.")
    print()
    print("  A one-parameter theory is not a prediction machine, "
          "but it is not nothing: any")
    print("  ONE measured quantity fixes the scale, and then "
          "EVERYTHING ELSE IS A PREDICTION.")
    print("  That is exactly what 'N has the epistemic status of a "
          "measured coupling' has")
    print("  meant since their 0106 -- now with a reason attached "
          "rather than a shrug.")
    print()
    print("  The honest programme for a number is therefore:")
    print("    1. fix the scale with ONE measurement (their n* = 58 "
          "prices this);")
    print("    2. compute a SECOND, independent quantity from it;")
    print("    3. compare that one with the world.")
    print("  Step 2 is the one nobody has done. Every number the "
          "program quotes today is")
    print("  step 1 wearing different clothes -- G, xi/a and the "
          "Lambda quantum are all the")
    print("  same scale seen through different windows, which is "
          "why none of them predicts\n")


if __name__ == "__main__":
    s1_cavity()
    s2_close_the_loop()
    s3_no_spin()
    s4_where_that_leaves_it()
    print("all assertions passed")
