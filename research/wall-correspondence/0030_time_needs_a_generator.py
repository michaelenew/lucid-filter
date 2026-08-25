"""wall-correspondence 0030 -- reflection positivity, in filter
terms: a record has continuous time underneath iff its dynamics has
a generator.

The sibling found (their 0111) that reflection positivity -- the
condition for an Osterwalder-Schrader reconstruction, hence for a
Lorentzian theory -- holds for their measure because their amplitude
is a COUNT. This is the filter-side statement of the same condition,
manufactured deliberately: the correspondence's fringe is built, not
found, and building it has paid before.

  s1  RP IS THE EMBEDDING PROBLEM. A record's one-step transfer
      operator T is positive semi-definite iff T = exp(-H) for a
      real generator H -- iff the discrete dynamics EMBEDS in a
      continuous-time flow. A transfer operator with a negative
      eigenvalue has no real logarithm: its dynamics exists at
      integer steps and at no time in between. Demonstrated on the
      two-state chain: persistent chains embed (generator computed),
      anti-persistent ones provably do not.
  s2  THE SAME DISTINCTION THAT PICKED GRAVITY. 0019 found the
      record structure selects the massless momentum: ratio records
      -> k = 0 (smooth), sum records -> k = pi (staggered). The
      embedding criterion is the same split in the time direction:
      persistent (positive) transfer embeds, alternating (negative)
      does not. Stated as a structural parallel -- 0019's
      measurement was spatial and this one is temporal -- not as a
      proven identity.
  s3  THE PORT. counting => nonnegative character coefficients =>
      positive transfer => a generator exists => the record has
      continuous time underneath. In filter terms: A COUNT-GENERATED
      RECORD IS ONE YOU CAN ALWAYS ASK 'WHAT HAPPENED IN BETWEEN'.
      That is what the sibling's 'counting buys time' means on this
      side, and it is the founding premise read backwards: the
      world is filterable because it is counted.
"""

import numpy as np


def two_state(p):
    return np.array([[1 - p, p], [p, 1 - p]])


def real_generator(T):
    """Return H with exp(-H) = T if a real one exists, else None."""
    w, V = np.linalg.eigh(T)
    if w.min() <= 0:
        return None
    return V @ np.diag(-np.log(w)) @ V.T


def s1_embedding():
    print("== s1: RP is the embedding problem ==")
    print("   p      eigenvalues        embeds?   generator")
    for p in (0.1, 0.3, 0.5, 0.7, 0.9):
        T = two_state(p)
        w = np.linalg.eigvalsh(T)
        H = real_generator(T)
        if H is not None:
            back = np.linalg.eigh(H)
            E = back[1] @ np.diag(np.exp(-back[0])) @ back[1].T
            ok = np.abs(E - T).max() < 1e-12
            off = H[0, 1]
            print(f"  {p:.1f}   [{w[0]:+.2f}, {w[1]:+.2f}]      YES"
                  f"      off-diagonal rate {off:+.3f}"
                  f"{'' if ok else '  (RECONSTRUCTION FAILED)'}")
            assert ok
        else:
            print(f"  {p:.1f}   [{w[0]:+.2f}, {w[1]:+.2f}]      NO "
                  f"      (negative eigenvalue: no real logarithm)")
    assert real_generator(two_state(0.7)) is None
    assert real_generator(two_state(0.3)) is not None
    print("  a transfer operator with a negative eigenvalue has no "
          "real generator: its")
    print("  dynamics exists at integer steps and AT NO TIME IN "
          "BETWEEN. Positivity of the")
    print("  transfer operator -- reflection positivity -- is "
          "exactly the condition that")
    print("  continuous time exists underneath the record\n")


def s2_same_split():
    print("== s2: the same distinction that picked gravity ==")
    # a persistent and an anti-persistent record, same |correlation|
    for name, phi in (("persistent  (smooth, k = 0)", +0.6),
                      ("alternating (staggered, k = pi)", -0.6)):
        T = np.array([[(1 + phi) / 2, (1 - phi) / 2],
                      [(1 - phi) / 2, (1 + phi) / 2]])
        w = np.linalg.eigvalsh(T)
        emb = real_generator(T) is not None
        print(f"   {name:32s} lag-1 corr {phi:+.1f}  "
              f"min eigenvalue {w.min():+.2f}  "
              f"{'embeds' if emb else 'NO generator'}")
    assert real_generator(np.array([[0.2, 0.8], [0.8, 0.2]])) is None
    print("  0019 found the record structure selects the massless "
          "momentum -- ratio records")
    print("  k = 0, sum records k = pi. The embedding criterion is "
          "that same split in the")
    print("  time direction. (Structural parallel: 0019's "
          "measurement was spatial, this")
    print("  one temporal -- flagged, not claimed as an identity)\n")


def s3_the_port():
    print("== s3: the port ==")
    print("  counting  =>  nonnegative character coefficients  "
          "(fusion multiplicities >= 0)")
    print("            =>  positive transfer operator")
    print("            =>  a real generator exists")
    print("            =>  continuous time underneath the record")
    print("  In filter terms: A COUNT-GENERATED RECORD IS ONE YOU "
          "CAN ALWAYS ASK 'WHAT")
    print("  HAPPENED IN BETWEEN'. The sibling's 'counting buys "
          "time' says here that the")
    print("  world is filterable in continuous time BECAUSE it is "
          "counted -- the founding")
    print("  premise read backwards\n")


if __name__ == "__main__":
    s1_embedding()
    s2_same_split()
    s3_the_port()
    print("all assertions passed")
