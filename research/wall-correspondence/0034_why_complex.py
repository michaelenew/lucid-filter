"""wall-correspondence 0034 -- why complex: the record squeezes the
amplitude algebra from both sides.

0033 narrowed the source ledger's observable content to INTERFERENCE
specifically. What was left open is the last structural question in
the program: why do amplitudes compose by COMPLEX MULTIPLICATION?

The premise is not new here -- it is the two-ledger theorem plus the
prequential tier. Concatenating two segments ADDS two things: the
code length (their 0095: the action IS a prequential code length,
additive by the chain rule) and the phase (the source ledger's own
additivity). An amplitude is the single object carrying both, so
composition must be an associative product on a real vector space
with a unit, in which |z w| = |z||w| -- moduli multiply because
code lengths add.

That is exactly the hypothesis of Frobenius's theorem, and its
answer is a trichotomy: R, C, H. The theorem is a century old and
this module does not reprove it; what is new is that the FILTER CAN
MEASURE WHICH ONE, because the two survivors that are not C each
predict a signal in a channel the record has already been read.

  s1  THE TRICHOTOMY, AND THE ONE STRUCTURAL DIFFERENCE. R, C and H
      each satisfy unital + norm-multiplicative + associative
      exactly (residual ~1e-30); a 40-restart search at n = 3 finds
      no solution at all (floor ~1e+1). The one property that
      separates the survivors: R and C are commutative, H is not.
  s2  R IS TOO SMALL -- 0033's FRINGE EXCLUDES IT. R's
      unit-modulus group is {+1,-1}, discrete. The only CONTINUOUS
      additive phase ledger into a discrete group is the constant
      one, so a real amplitude ledger cannot vary its phase with a
      continuously-turned knob at all. Its best predictor is
      therefore a constant, and the best constant is EXACTLY the
      incoherent model: E_delta[p_coh] = (a1^2+a2^2)/(a1+a2)^2
      identically. So the real ledger's penalty is 0033's own
      measured 0.302 nats/trial -- that number was already the
      exclusion of R, unrecognised.
  s3  H IS TOO BIG -- 0033's EXACT ZERO EXCLUDES IT. H's
      unit-modulus group is S^3 = SU(2), nonabelian, so two
      influences composed in the two orders give different
      interference against a reference: measured 0.019 nats/trial
      of ORDER information leaking into the INTERFERENCE channel.
      The complex ledger leaks zero (machine precision), and 0033
      measured exactly zero in the record. A sub-result worth
      naming: the leak vanishes identically when the reference is
      REAL, because Re(ab) = Re(ba) -- the leak is visible only
      through a reference that carries a phase of its own.
      Consequence for this program: the S^3 the filter adopted for
      noncommutativity (0089) belongs in the RECORD ledger, where
      0009 and 0033 put it, and must not be the amplitude algebra.
"""

import numpy as np
from scipy.optimize import least_squares

rng = np.random.default_rng(34)


# ----------------------------------------------------------------
# s1: the trichotomy
# ----------------------------------------------------------------
def struct_R():
    T = np.zeros((1, 1, 1))
    T[0, 0, 0] = 1.0
    return T


def struct_C():
    T = np.zeros((2, 2, 2))
    T[0, 0, 0] = 1.0
    T[0, 1, 1] = -1.0
    T[1, 0, 1] = 1.0
    T[1, 1, 0] = 1.0
    return T


def struct_H():
    T = np.zeros((4, 4, 4))
    e = np.eye(4)

    def qm(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                         w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
    for j in range(4):
        for k in range(4):
            T[:, j, k] = qm(e[j], e[k])
    return T


def prod(T, a, b):
    return np.einsum("ijk,mj,mk->mi", T, a, b)


def algebra_residual(x, n, Z, W, A, B, C):
    T = x.reshape(n, n, n)
    e = np.zeros(n)
    e[0] = 1.0
    r = [(np.einsum("ijk,k->ij", T, e) - np.eye(n)).ravel(),
         (np.einsum("ijk,j->ik", T, e) - np.eye(n)).ravel(),
         np.sum(prod(T, Z, W) ** 2, 1)
         - np.sum(Z ** 2, 1) * np.sum(W ** 2, 1),
         (prod(T, prod(T, A, B), C)
          - prod(T, A, prod(T, B, C))).ravel()]
    return np.concatenate(r)


def s1_trichotomy():
    print("== s1: the trichotomy, and the one structural difference"
          " ==")
    print("  requirement: unital, |z w| = |z||w| (code lengths add),"
          " associative")
    for name, T in (("R", struct_R()), ("C", struct_C()),
                    ("H", struct_H())):
        n = T.shape[0]
        Z = rng.standard_normal((40, n))
        W = rng.standard_normal((40, n))
        A, B, Cc = (rng.standard_normal((16, n)) for _ in range(3))
        res = float((algebra_residual(T.ravel(), n, Z, W, A, B, Cc)
                     ** 2).sum())
        a2 = rng.standard_normal((400, n))
        b2 = rng.standard_normal((400, n))
        comm = float(np.abs(prod(T, a2, b2) - prod(T, b2, a2)).max())
        print(f"  {name} (n = {n}): residual^2 = {res:.2e}   "
              f"commutator sup = {comm:.3e}")
        assert res < 1e-20
    # the excluded dimension
    n = 3
    Z = rng.standard_normal((40, n))
    W = rng.standard_normal((40, n))
    A, B, Cc = (rng.standard_normal((16, n)) for _ in range(3))
    best = np.inf
    for _ in range(40):
        s = least_squares(algebra_residual,
                          rng.standard_normal(n ** 3) * 0.7,
                          args=(n, Z, W, A, B, Cc),
                          xtol=1e-14, ftol=1e-14, gtol=1e-14,
                          max_nfev=1500)
        best = min(best, 2 * s.cost)
    print(f"  n = 3: 40-restart search floor = {best:.2e}  -- NO "
          f"solution (Frobenius)")
    assert best > 1.0
    print("  so the ledger structure alone leaves exactly three "
          "candidates, and the only")
    print("  property separating them is COMMUTATIVITY: R and C "
          "commute, H does not\n")


# ----------------------------------------------------------------
# s2: R is too small
# ----------------------------------------------------------------
A1, A2 = 0.75, 0.66        # 0033's record B, unchanged


def s2_real_too_small():
    print("== s2: R is too small -- 0033's fringe excludes it ==")
    print("  R's unit-modulus group is {+1,-1}: discrete. A "
          "continuous additive phase")
    print("  ledger delta -> s(delta) in {+1,-1} with "
          "s(d1+d2) = s(d1)s(d2) must be constant.")
    n = 200000
    delta = rng.uniform(0, 2 * np.pi, n)
    p_coh = np.abs(A1 + A2 * np.exp(1j * delta)) ** 2 / (A1 + A2) ** 2
    bits = rng.random(n) < p_coh

    def code(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-np.mean(np.where(bits, np.log(p),
                                       np.log(1 - p))))
    # the two constants a real ledger can offer, and the best
    # constant of any kind
    q_plus = 1.0
    q_minus = (A1 - A2) ** 2 / (A1 + A2) ** 2
    p_inc = (A1 ** 2 + A2 ** 2) / (A1 + A2) ** 2
    print(f"  its two available constants: s = +1 -> q = "
          f"{q_plus:.4f},  s = -1 -> q = {q_minus:.6f}")
    print(f"  best constant predictor of any kind: q = E[p] = "
          f"{np.mean(p_coh):.6f}")
    print(f"  incoherent model (0033):              q = "
          f"{p_inc:.6f}")
    assert abs(np.mean(p_coh) - p_inc) < 2e-3
    # the identity, exactly
    exact = (A1 ** 2 + A2 ** 2) / (A1 + A2) ** 2
    an = (1 / (2 * np.pi)) * np.trapezoid(
        np.abs(A1 + A2 * np.exp(1j * np.linspace(0, 2 * np.pi, 200001)))
        ** 2 / (A1 + A2) ** 2, np.linspace(0, 2 * np.pi, 200001))
    print(f"  identity E_delta[p_coh] = (a1^2+a2^2)/(a1+a2)^2 : "
          f"{an:.9f} vs {exact:.9f}")
    assert abs(an - exact) < 1e-9
    c_coh = code(p_coh)
    c_real = min(code(np.full(n, q_minus)), code(np.full(n, p_inc)))
    print(f"  coherent (C) {c_coh:.5f} nats/trial;  best real "
          f"ledger {c_real:.5f};  gap {c_real - c_coh:+.5f}")
    assert c_real - c_coh > 0.25
    print("  THE BEST REAL LEDGER IS THE INCOHERENT MODEL, exactly. "
          "0033's measured 0.302")
    print("  nats/trial was the exclusion of R all along\n")
    return c_real - c_coh


# ----------------------------------------------------------------
# s3: H is too big
# ----------------------------------------------------------------
def qmul(a, b):
    w1, x1, y1, z1 = a.T
    w2, x2, y2, z2 = b.T
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)


def qunit(axis, th):
    ax = np.asarray(axis, float)
    ax = ax / np.linalg.norm(ax)
    return np.stack([np.cos(th / 2)] + [np.sin(th / 2) * ax[i]
                                        for i in range(3)], 1)


def s3_quaternion_too_big():
    print("== s3: H is too big -- 0033's exact zero excludes it ==")
    n = 200000
    t1 = rng.uniform(0, 2 * np.pi, n)
    t2 = rng.uniform(0, 2 * np.pi, n)
    q1 = qunit([0, 0, 1], t1)          # two influences, different
    q2 = qunit([1, 0, 0], t2)          # generators
    REF = 0.8
    refs = {
        "real reference":
            np.tile(np.array([REF, 0.0, 0.0, 0.0]), (n, 1)),
        "reference with its own phase":
            np.tile(REF * qunit([1, 1, 0], np.array([1.1]))[0],
                    (n, 1)),
    }
    for name, R in refs.items():
        d1 = qmul(q1, q2) + R
        d2 = qmul(q2, q1) + R
        p1 = (d1 ** 2).sum(1) / (1 + REF) ** 2
        p2 = (d2 ** 2).sum(1) / (1 + REF) ** 2
        order = rng.random(n) < 0.5
        ptrue = np.where(order, p1, p2)
        bits = rng.random(n) < ptrue

        def code(p):
            p = np.clip(p, 1e-12, 1 - 1e-12)
            return float(-np.mean(np.where(bits, np.log(p),
                                           np.log(1 - p))))
        aware, blind = code(ptrue), code(0.5 * (p1 + p2))
        print(f"  H, {name}: order-aware {aware:.5f}, order-blind "
              f"{blind:.5f}")
        print(f"      leak {blind - aware:+.5f} nats/trial   "
              f"(max|p1-p2| = {np.abs(p1 - p2).max():.4f})")
        if "real" == name.split()[0]:
            assert np.abs(p1 - p2).max() < 1e-12
        else:
            assert blind - aware > 0.005
    print("  (the real-reference row is an identity, not an "
          "accident: Re(ab) = Re(ba) in H,")
    print("   so the leak needs a reference carrying a phase of its"
          " own)")
    c1 = np.exp(1j * t1)
    c2 = np.exp(1j * t2)
    r = REF * np.exp(1j * 1.1)
    b1 = np.abs(c1 * c2 + r) ** 2
    b2 = np.abs(c2 * c1 + r) ** 2
    print(f"  C, same construction: max|p1 - p2| = "
          f"{np.abs(b1 - b2).max():.1e}  (machine zero)")
    assert np.abs(b1 - b2).max() < 1e-12
    print("  the record (0033 s2) measured EXACTLY zero order "
          "information in the")
    print("  interference channel. H predicts a leak; C predicts "
          "none; the record says none.\n")


def s4_the_answer(gapR):
    print("== s4: the answer ==")
    print("  two additive ledgers          -> unital, "
          "norm-multiplicative, associative")
    print("  Frobenius                     -> R, C or H")
    print(f"  a continuously-turned phase   -> not R   "
          f"(costs {gapR:.3f} nats/trial, 0033 s2)")
    print("  an order-blind interference   -> not H   "
          "(H leaks 0.019 nats/trial, record has 0)")
    print("  ---------------------------------------------------")
    print("  AMPLITUDES COMPOSE BY COMPLEX MULTIPLICATION because "
          "the record carries two")
    print("  additive ledgers, turns its phase continuously, and "
          "shows no order in the")
    print("  channel where the phases meet. Each clause is a "
          "measurement this program")
    print("  has already made; none of them is a postulate about "
          "Hilbert space.")
    print("  Corollary for the filter: S^3 (0089) is a RECORD-ledger"
          " structure. Putting it")
    print("  in the amplitude would forge an order signal the "
          "record does not have.\n")


if __name__ == "__main__":
    s1_trichotomy()
    g = s2_real_too_small()
    s3_quaternion_too_big()
    s4_the_answer(g)
    print("all assertions passed")
