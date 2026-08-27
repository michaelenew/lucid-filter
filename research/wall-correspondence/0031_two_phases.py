"""wall-correspondence 0031 -- the two phases: one is gauge, the
other is compositional.

The sibling's 0124 exposed a conflation this correspondence has been
carrying: 'the phase' has meant two different objects.

  (i)  the FACTORISATION phase -- a static weight W has 2^n
       amplitudes A with |A|^2 = W (their 0119), all agreeing on
       every record-side observable;
  (ii) the DYNAMICAL phase -- what the 0006 detector measures at
       0.087 nats/bit on a monitored two-state stream.

If these are the same object, the detector is measuring something
provably unobservable, which cannot be. This module separates them
inside one generator, so the difference is exhibited rather than
argued.

  s1  THE FACTORISATION PHASE IS GAUGE -- EVEN DYNAMICALLY.
      Re-phasing each Kraus operator, M_b -> e^{i phi_b} M_b,
      changes the amplitude at every step while leaving |.|^2
      alone. Result: the generated record is BITWISE IDENTICAL and
      the detector's advantage is unchanged to machine precision.
      A factorisation choice is invisible even when the amplitude
      is propagated.
  s2  THE RELATIVE PHASE IS PHYSICAL. Changing a phase BETWEEN the
      components that later interfere -- U -> U diag(1, e^{i
      delta}) -- changes the record measurably, and moves the
      detector's advantage. Same amount of 'phase', entirely
      different status.
  s3  WHAT THIS MEANS FOR THE SOURCE LEDGER. Its observable content
      is not a hidden field attached to the weight; it is the
      RELATIVE phase between alternatives that later compose. The
      static Euclidean weight carries none of it -- all its
      factorisations agree. So on the physics side the source
      ledger cannot live in the weight; it lives in how amplitudes
      COMPOSE, which is the same structure the order channel
      (0009 / their 0108) already measures. The ledger is a
      statement about composition, not about a field.
"""

import numpy as np


def kraus(k, phases=(0.0, 0.0)):
    a, b = np.sqrt((1 + k) / 2), np.sqrt((1 - k) / 2)
    Mp = np.exp(1j * phases[0]) * np.array([[a, 0], [0, b]],
                                           dtype=complex)
    Mm = np.exp(1j * phases[1]) * np.array([[b, 0], [0, a]],
                                           dtype=complex)
    return Mp, Mm


def unitary(theta, rel=0.0):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    U = np.array([[c, -1j * s], [-1j * s, c]])
    return U @ np.diag([1.0, np.exp(1j * rel)])


def gen(theta, k, T, seed, phases=(0.0, 0.0), rel=0.0):
    rng = np.random.default_rng(seed)
    U = unitary(theta, rel)
    Mp, Mm = kraus(k, phases)
    psi = np.array([1.0, 0.0], dtype=complex)
    bits = np.empty(T, dtype=int)
    for t in range(T):
        psi = U @ psi
        pp = float(np.linalg.norm(Mp @ psi) ** 2)
        if rng.random() < pp:
            bits[t], psi = 1, Mp @ psi / np.sqrt(pp)
        else:
            bits[t], psi = 0, Mm @ psi / np.sqrt(1 - pp)
    return bits


def code_quantum(bits, theta, k, phases=(0.0, 0.0), rel=0.0):
    U = unitary(theta, rel)
    Mp, Mm = kraus(k, phases)
    psi = np.array([1.0, 0.0], dtype=complex)
    code = 0.0
    for t, b in enumerate(bits):
        psi = U @ psi
        pp = min(max(float(np.linalg.norm(Mp @ psi) ** 2), 1e-12),
                 1 - 1e-12)
        p = pp if b else 1 - pp
        M = Mp if b else Mm
        psi = M @ psi / np.linalg.norm(M @ psi)
        if t >= 200:
            code -= np.log(p)
    return code / (len(bits) - 200)


def code_classical(bits, theta, k, rel=0.0):
    P = np.abs(unitary(theta, rel)) ** 2
    lp = np.array([(1 + k) / 2, (1 - k) / 2])
    w = np.array([1.0, 0.0])
    code = 0.0
    for t, b in enumerate(bits):
        w = P @ w
        like = lp if b else 1 - lp
        p = min(max(float(w @ like), 1e-12), 1 - 1e-12)
        if t >= 200:
            code -= np.log(p)
        w = w * like / p
    return code / (len(bits) - 200)


TH, K, T = 1.0, 0.5, 20000


def s1_gauge():
    print("== s1: the factorisation phase is gauge, even "
          "dynamically ==")
    base = gen(TH, K, T, 11)
    for ph in ((0.7, -1.3), (2.0, 2.0), (-0.4, 3.1)):
        alt = gen(TH, K, T, 11, phases=ph)
        same = bool(np.array_equal(base, alt))
        dq = abs(code_quantum(base, TH, K)
                 - code_quantum(base, TH, K, phases=ph))
        print(f"  Kraus phases {str(ph):>14s}: record identical "
              f"{same}, |d code| = {dq:.1e}")
        assert same and dq < 1e-12
    adv0 = code_classical(base, TH, K) - code_quantum(base, TH, K)
    adv1 = code_classical(base, TH, K) - code_quantum(
        base, TH, K, phases=(0.7, -1.3))
    print(f"  detector advantage: {adv0:.5f} vs {adv1:.5f} nats/bit "
          f"(identical)")
    assert abs(adv0 - adv1) < 1e-12
    print("  a factorisation choice is INVISIBLE even when the "
          "amplitude is propagated\n")
    return adv0


def s2_relative(adv0):
    print("== s2: the relative phase is physical ==")
    print("   rel phase   record differs?   detector advantage")
    for rel in (0.0, 0.3, 0.9, 1.8):
        b = gen(TH, K, T, 11, rel=rel)
        base = gen(TH, K, T, 11)
        diff = float(np.mean(b != base))
        adv = code_classical(b, TH, K, rel=rel) \
            - code_quantum(b, TH, K, rel=rel)
        print(f"     {rel:.1f}         {diff:.3f}            "
              f"{adv:.5f}")
        if rel > 0:
            assert diff > 0.01
    print("  the SAME amount of phase, entirely different status: "
          "between-component phase")
    print("  moves the record and the measurable advantage\n")


def s3_meaning():
    print("== s3: what this means for the source ledger ==")
    print("  Its observable content is NOT a hidden field attached "
          "to the weight -- all")
    print("  factorisations of a weight agree on everything. It is "
          "the RELATIVE phase")
    print("  between alternatives that later COMPOSE.")
    print("  Consequence for the physics side: the static Euclidean "
          "weight cannot carry the")
    print("  source ledger, because its factorisations are "
          "indistinguishable. The ledger")
    print("  lives in how amplitudes compose -- the same structure "
          "the order channel")
    print("  (0009, their 0108) already measures. THE SOURCE LEDGER "
          "IS A STATEMENT ABOUT")
    print("  COMPOSITION, NOT ABOUT A FIELD\n")


if __name__ == "__main__":
    a = s1_gauge()
    s2_relative(a)
    s3_meaning()
    print("all assertions passed")
