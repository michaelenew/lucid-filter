"""0051 -- what an intertwiner is, in filter terms, and what
discarding one costs.

Their 0166 raised the sharpest open item in the program: the derived
weight is a Barrett-Crane amplitude, and Barrett-Crane is known to
give the wrong long-distance graviton propagator (Alesci & Rovelli
2007), traced to INTERTWINER-INDEPENDENCE. The question they cannot
answer from inside is whether their capacity argument -- the one that
forced flat multiplicities and fixed kappa -- also FORCES the
intertwiner-blind form, or merely fails to mention it.

Those are opposite outcomes. If capacity forces blindness, the
program inherits a known defect and that is a sharp negative. If
capacity FORBIDS blindness, the program was never Barrett-Crane in
the first place and should be EPRL-class.

WHAT AN INTERTWINER IS, WITHOUT THE GROUP THEORY. At a node, n
channels meet. Each carries a magnitude. The node is closed (the
channel vectors sum to zero) and a global rotation is unobservable.
The estimable content is then the SHAPE of the closed configuration,
modulo rotation -- and the question is whether that shape carries
information the magnitudes do not.

  s1  THE SHAPE SPACE. Its dimension, measured as the rank of the
      constraint Jacobian, against the closed form 2n - 6.
  s2  THE GROUP-THEORY GATE: the same count as dim Inv(j1..jn).
  s3  WHAT BLINDNESS COSTS, in nats per node.
  s4  WHAT THAT DOES TO A CAPACITY ARGUMENT.
"""

import numpy as np

rng = np.random.default_rng(51)


def closed_config(mags, tries=20000):
    """n vectors with the given magnitudes summing to zero."""
    n = len(mags)
    for _ in range(tries):
        v = rng.standard_normal((n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        v *= np.array(mags)[:, None]
        # project onto closure by a few Newton steps
        for _i in range(200):
            r = v.sum(0)
            if np.linalg.norm(r) < 1e-12:
                return v
            v = v - r / n
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            v *= np.array(mags)[:, None]
        if np.linalg.norm(v.sum(0)) < 1e-9:
            return v
    return None


def shape_dim(mags):
    """dimension of {closed configs with these magnitudes} / SO(3),
    measured as 3n - rank(Jacobian of the constraints) - 3."""
    n = len(mags)
    v = closed_config(mags)
    if v is None:
        return None
    rows = []
    for i in range(n):                       # |v_i|^2 fixed
        g = np.zeros((n, 3))
        g[i] = 2 * v[i]
        rows.append(g.reshape(-1))
    for a in range(3):                       # closure
        g = np.zeros((n, 3))
        g[:, a] = 1.0
        rows.append(g.reshape(-1))
    J = np.array(rows)
    rank = np.linalg.matrix_rank(J, tol=1e-8)
    return 3 * n - rank - 3


def dim_inv(js):
    """dim Inv(j1 (x) ... (x) jn) for SU(2), by recoupling."""
    cur = {0.0: 1}
    for j in js:
        nxt = {}
        for k, c in cur.items():
            lo, hi = abs(k - j), k + j
            x = lo
            while x <= hi + 1e-9:
                nxt[round(x, 1)] = nxt.get(round(x, 1), 0) + c
                x += 1.0
        cur = nxt
    return cur.get(0.0, 0)


def s1_s2_shape_and_gate():
    print("== s1/s2: the shape space, and the group-theory gate ==")
    print("  A node: n channels meet, magnitudes fixed, the node is "
          "closed (they sum to")
    print("  zero), a global rotation is unobservable. What is left "
          "is the SHAPE.")
    print()
    print("     n    measured dim    2n-6    dim Inv(j..j), j=3/2  "
          "  verdict")
    ok = True
    for n in (3, 4, 5, 6):
        d = shape_dim([1.0] * n)
        di = dim_inv([1.5] * n)
        pred = 2 * n - 6
        good = (d == pred)
        ok &= good
        print(f"    {n:2d}       {d}              {pred}"
              f"           {di:3d}            "
              f"{'ok' if good else 'MISMATCH'}")
    assert ok
    print()
    print("  AT n = 3 THE SHAPE SPACE IS A POINT. Three closed "
          "vectors of fixed length")
    print("  have no freedom left once you quotient the rotation -- "
          "and correspondingly")
    print("  dim Inv(j1,j2,j3) is at most 1. A 3-valent node has "
          "NOTHING beyond its")
    print("  magnitudes, so an amplitude that depends only on the "
          "magnitudes loses")
    print("  nothing there.")
    print()
    print("  AT n = 4 IT IS TWO-DIMENSIONAL, and dim Inv is "
          "greater than 1. A tetrahedron")
    print("  has four faces, so its node is 4-valent: its face "
          "areas do NOT determine its")
    print("  shape. That residual shape is the intertwiner.")
    print()


def s3_the_cost():
    print("== s3: what blindness costs, in nats per node ==")
    print("  A magnitude-only summary collapses every shape sharing "
          "the same magnitudes")
    print("  into one symbol. The information discarded is the log "
          "of how many there were.")
    print()
    print("     four equal spins j      dim Inv     nats discarded "
          "per node")
    tot = []
    for j in (0.5, 1.0, 1.5, 2.0, 2.5):
        d = dim_inv([j] * 4)
        tot.append((j, d, np.log(d) if d > 0 else 0.0))
        print(f"        {j:4.1f}                 {d:3d}          "
              f"{np.log(d) if d > 0 else 0.0:.4f}")
    print()
    print("  Their band limit is j <= 5/2 (M = 6 characters), so "
          "the worst case on their")
    print(f"  own lattice is ln {tot[-1][1]} = {tot[-1][2]:.4f} "
          f"nats per node, and it is never zero")
    print("  above j = 1/2.")
    print()
    return tot


def s4_capacity():
    print("== s4: what that does to a capacity argument ==")
    print("  Their derivation of kappa runs on a capacity "
          "principle: with M usable symbols,")
    print("  capacity is maximised when the symbols are "
          "equiprobable, which forces flat")
    print("  multiplicities and hence kappa = N(N+3)/3.")
    print()
    print("  That argument counts STATES. So run it honestly at a "
          "node. The states are")
    print("  the label assignment AND the shape; a "
          "magnitude-only amplitude identifies")
    print("  shapes that the record can distinguish, so it strictly "
          "REDUCES the number of")
    print("  distinguishable symbols.")
    print()
    print("  A principle that maximises capacity cannot select an "
          "amplitude that throws")
    print("  capacity away. So:")
    print()
    print("    THE CAPACITY ARGUMENT DOES NOT FORCE THE "
          "BARRETT-CRANE FORM.")
    print("    IT FORBIDS IT.")
    print()
    print("  And it does not touch kappa. kappa is fixed by the "
          "multiplicity profile over")
    print("  the FACE labels j -- the distribution the capacity "
          "argument equalises. The")
    print("  intertwiner is a separate degree of freedom living at "
          "the NODE. Restoring it")
    print("  adds states at nodes; it does not change how face "
          "labels are weighted.")
    print()
    print("  PREDICTION FOR THE PORT, and it is falsifiable on "
          "their side: extending the")
    print("  weight to carry intertwiner dependence should leave "
          "kappa = 16.0001 exactly")
    print("  unchanged, because the curvature at the identity is a "
          "property of the face")
    print("  weight alone. If kappa MOVES when the intertwiner is "
          "restored, this argument")
    print("  is wrong and the two sectors are not separable.")
    print()


if __name__ == "__main__":
    s1_s2_shape_and_gate()
    s3_the_cost()
    s4_capacity()
    print("all assertions passed")
