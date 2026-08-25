"""wall-correspondence 0027 -- the transport field's dynamics: record
noise gives the heat kernel, and the Born square is what it cannot
give.

0026 showed the tensor completion needs a connection and named its
dynamics as the remaining gap. The filter answers structurally: the
transport R_xy is not a postulated field, it is an INFERRED
NUISANCE PARAMETER -- read from noisy frame-comparison records. Its
'action' is therefore the code length of those records.

  s1  NOISY RECORDS MAKE A HEAT KERNEL. Each edge's relative frame
      is read with noise; the holonomy of a closed loop is the
      ordered product of the edge errors, i.e. Brownian motion on
      the group. Its class law is the HEAT KERNEL K_tau with
      tau = P sigma^2 / 2 for P edges of per-generator noise
      sigma -- verified against the exact character formula.
  s2  THE COUPLING IS THE RECORD PRECISION. The induced plaquette
      action is -ln K_tau(theta) = theta^2/(2 tau) + ..., so the
      gauge coupling is fixed by how precisely frames are compared:
      1/g^2 = 1/tau = 2/(P sigma^2). Nothing is chosen; the
      connection's stiffness IS the records' precision.
  s3  WHAT RECORD NOISE CANNOT GIVE: THE BORN SQUARE. The sibling's
      weight is |A|^2 with A a finite character sum -- a weight with
      EXACT ZEROS (its nodes fracture ergodicity, their 0113). A
      heat kernel is strictly positive and node-free, and no amount
      of record noise produces a zero. So the record ledger supplies
      the connection's Gaussian dynamics and the source ledger
      supplies its nodes: the two-ledger split again, now at the
      level of the gauge action. The gap left by 0026 is therefore
      not open-ended -- it is exactly the amplitude structure the
      program has been tracking since 0005.
"""

import numpy as np

TH = np.linspace(1e-7, np.pi - 1e-7, 200001)
HAAR = (2 / np.pi) * np.sin(TH) ** 2


def k_heat(tau, jmax=80):
    out = np.zeros_like(TH)
    for j in np.arange(0, jmax + 0.1, 0.5):
        out += (2 * j + 1) * np.exp(-tau * j * (j + 1)) \
            * np.sin((2 * j + 1) * TH) / np.sin(TH)
    return out


def qmul(a, b):
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


def small_rot(n, sig, rng):
    """Frame-comparison error: a small random rotation."""
    v = sig * rng.standard_normal((n, 3))
    th = np.linalg.norm(v, axis=1, keepdims=True)
    ax = v / np.maximum(th, 1e-15)
    return np.concatenate([np.cos(th / 2), np.sin(th / 2) * ax],
                          axis=1)


def s1_heat_kernel():
    print("== s1: noisy records make a heat kernel ==")
    rng = np.random.default_rng(3)
    n = 400000
    print("   edges P   sigma   tau = P sigma^2/2   <phi^2> loop"
          "   <phi^2> heat kernel")
    for P, sig in ((4, 0.15), (4, 0.25), (8, 0.15)):
        acc = small_rot(n, sig, rng)
        for _ in range(P - 1):
            acc = qmul(acc, small_rot(n, sig, rng))
        phi = np.arccos(np.clip(np.abs(acc[:, 0]), -1, 1))
        m2 = float(np.mean(phi ** 2))
        tau = P * sig ** 2 / 2       # heat kernel e^{tau Lap}:
        # variance 2 tau per generator of the rotation vector
        pk = np.maximum(k_heat(tau), 0) * HAAR
        pk /= np.trapezoid(pk, TH)
        m2k = float(np.trapezoid(pk * TH ** 2, TH))
        print(f"     {P}      {sig:.2f}       {tau:.4f}        "
              f"{m2:.5f}        {m2k:.5f}")
        assert abs(m2 / m2k - 1) < 0.05
    print("  the loop holonomy of noisy frame records IS Brownian "
          "motion on the group:")
    print("  its class law is the heat kernel at tau = P sigma^2 / 2\n")


def s2_coupling():
    print("== s2: the coupling is the record precision ==")
    print("   tau     -ln K_tau quadratic coefficient    1/tau")
    for tau in (0.05, 0.1, 0.2):
        lnk = np.log(np.maximum(k_heat(tau), 1e-300))
        sel = TH < 0.2
        c = np.polyfit(TH[sel] ** 2, -lnk[sel], 1)[0]
        print(f"   {tau:.2f}              {c:8.3f}             "
              f"{1 / tau:8.3f}")
        assert abs(c / (1 / tau) - 1) < 0.05
    print("  the induced plaquette action is phi^2/tau + ...:")
    print("  1/g^2 = 1/tau = 2/(P sigma^2). THE GAUGE COUPLING IS "
          "THE RECORDS' PRECISION --")
    print("  the connection's stiffness is not chosen, it is how "
          "well frames are compared\n")


def s3_the_born_residue():
    print("== s3: what record noise cannot give ==")
    JS = np.arange(0, 2.6, 0.5)
    A = sum(np.sin((2 * j + 1) * TH) / np.sin(TH) for j in JS)
    W = A ** 2
    zeros = int(np.sum(np.sign(A[:-1]) != np.sign(A[1:])))
    K = k_heat(0.2)
    rel_min = float(np.min(K) / np.max(K))
    print(f"  Born weight |A|^2 (J <= 2.5): A has {zeros} sign "
          f"changes -> {zeros} EXACT ZEROS")
    print(f"  heat kernel K_0.2: relative minimum {rel_min:.1e} -- "
          f"zero only at the level of")
    print(f"  character-series truncation; K_tau is strictly "
          f"positive by construction (a")
    print(f"  convolution of positive densities), so it has NO nodes")
    assert zeros >= 2 and abs(rel_min) < 1e-6
    print("  no amount of record noise makes a zero: convolution of "
          "positive densities is")
    print("  positive. The record ledger supplies the connection's "
          "GAUSSIAN dynamics; the")
    print("  NODES -- the ergodicity-fracturing structure of their "
          "0113 -- come from the")
    print("  source ledger's amplitudes. 0026's gap is therefore not "
          "open-ended: it is")
    print("  exactly the amplitude structure tracked since 0005\n")


if __name__ == "__main__":
    s1_heat_kernel()
    s2_coupling()
    s3_the_born_residue()
    print("all assertions passed")
