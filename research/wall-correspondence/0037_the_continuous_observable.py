"""wall-correspondence 0037 -- the fully continuous filter, and the
observation as an arbitrary kernel.

0036 made the state and time directions continuous and gave the
criterion (a continuum limit exists iff the prequential code length
per unit physical time converges to a NONTRIVIAL limit). What was
still discrete there is the OBSERVATION: a fixed measurement matrix
reading the state at points. This module makes that continuous too --
an observation is an arbitrary kernel integrated against the field --
and the payoff is immediate and specific, because the sibling's 0129
failed for exactly the want of one.

Their problem: the anisotropy of a local operator's connected
correlator came out +5.6 +- 4.8, because at weak coupling that
correlator is O(g^4). Their named way in was "an operator whose
connected correlator is not O(g^4)". THAT IS AN OBSERVATION KERNEL,
and choosing it is a filter question with a filter answer.

  s1  THE FULLY CONTINUOUS FILTER. Continuous state, continuous
      time, continuous observation: the discrete Bayes recursion
      converges to the Kalman-Bucy equations, and the steady-state
      posterior converges to the algebraic Riccati solution. Rate
      measured, so "continuous observation" is a limit and not a
      redefinition.
  s2  WHICH KERNEL, AND HOW MUCH IT BUYS. A field with a
      long-ranged signal and white ultraviolet noise, observed
      through a kernel of width w. Signal-to-noise on the
      correlator at separation r rises steeply with w: measured
      62x at w = r/2 over the local operator, in d = 4 with 300
      samples. The honest statement of what that is: EACH w
      DEFINES A DIFFERENT OBSERVABLE, not a better estimate of the
      same one -- the measured value moves by more than its error
      across the range, and at large w it is heavily distorted
      (down 25x by w = 5). So the kernel buys statistics, and every
      comparison must be made at MATCHED w against a baseline
      computed for that same w. Prescription: w ~ r/3 to r/2.
  s3  A KERNEL CAN FORGE THE ANSWER -- AND THE GOOD ONE STILL
      FORGES SOME. On a field that is isotropic BY CONSTRUCTION, a
      cubic kernel manufactures an anisotropy of +0.1556 +- 0.0036.
      A radially symmetric one manufactures +0.0204 +- 0.0006 --
      7.6x smaller, but still 34 sigma from zero, so "use a radial
      kernel and the probe is clean" IS FALSE and I had written it
      before measuring. What is true: the radial kernel makes the
      artefact small instead of dominant, and A KINEMATIC BASELINE
      MUST BE SUBTRACTED EITHER WAY. The wrong kernel does not lose
      signal, it manufactures it.
"""

import numpy as np

rng = np.random.default_rng(37)


# ----------------------------------------------------------------
def s1_kalman_bucy():
    print("== s1: the fully continuous filter ==")
    th, sig2, H, R = 0.8, 1.0, 1.0, 0.4
    # algebraic Riccati: -2 th P + sig2 - P^2 H^2 / R = 0
    a, b, c = H ** 2 / R, 2 * th, -sig2
    P_inf = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    print(f"  continuous-time steady state (algebraic Riccati): "
          f"P = {P_inf:.6f}")
    print("     dt          discrete-filter P        error      "
          "ratio")
    prev = None
    for dt in (0.2, 0.1, 0.05, 0.025, 0.0125):
        A = np.exp(-th * dt)
        Q = sig2 * (1 - A * A) / (2 * th)
        Rk = R / dt                       # observation noise per step
        P = 1.0
        for _ in range(int(200 / dt)):
            P = A * A * P + Q
            P = P - P * P / (P + Rk)
        err = abs(P - P_inf)
        r = "" if prev is None else f"{prev / err:.2f}"
        print(f"   {dt:.4f}      {P:.6f}            {err:.2e}   {r}")
        prev = err
    assert err < 0.01
    print("  the discrete recursion converges to Kalman-Bucy at "
          "first order in dt:")
    print("  continuous observation is a LIMIT of discrete "
          "observation, not a redefinition\n")


# ----------------------------------------------------------------
def make_field(L, d, xi, noise, n):
    """n samples of: long-ranged Gaussian signal + white UV noise."""
    ax = 2 * np.pi * np.fft.fftfreq(L)
    g = np.meshgrid(*([ax] * d), indexing="ij")
    k2 = sum(gi ** 2 for gi in g)
    S = 1.0 / (k2 + 1.0 / xi ** 2)          # signal power
    amp = np.sqrt(S)
    out = []
    for _ in range(n):
        w = rng.standard_normal((L,) * d)
        f = np.real(np.fft.ifftn(np.fft.fftn(w) * amp))
        f = f / f.std()
        out.append(f + noise * rng.standard_normal((L,) * d))
    return out


def corr_at(fields, L, d, w, pts):
    ax = 2 * np.pi * np.fft.fftfreq(L)
    g = np.meshgrid(*([ax] * d), indexing="ij")
    k2 = sum(gi ** 2 for gi in g)
    K = np.exp(-(w ** 2) * k2)
    vals = {p: [] for p in pts}
    for f in fields:
        P = np.abs(np.fft.fftn(f - f.mean())) ** 2
        C = np.real(np.fft.ifftn(P * K)) / f.size
        for p in pts:
            vals[p].append(float(C[p]))
    return {p: (float(np.mean(v)),
                float(np.std(v) / np.sqrt(len(v))))
            for p, v in vals.items()}


def s2_which_kernel():
    print("== s2: which kernel, and how much it buys ==")
    L, d, xi, noise, n = 16, 4, 4.0, 3.0, 300
    fields = make_field(L, d, xi, noise, n)
    r = 6
    pt = (r, 0, 0, 0)
    print(f"  d = {d}, signal correlation length xi = {xi}, white "
          f"noise amplitude {noise}, {n} samples")
    print(f"  estimating C(r = {r}) through a kernel of width w:")
    print("     w        C(r)          error        |C|/err     "
          "gain vs w=0")
    base = None
    rows = []
    for w in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        m, e = corr_at(fields, L, d, w, [pt])[pt]
        snr = abs(m) / e if e else float("inf")
        if base is None:
            base = snr
        rows.append((w, m, e, snr))
        print(f"   {w:.1f}    {m:+.6f}    {e:.6f}    {snr:7.2f}    "
              f"{snr / base:7.2f}")
    snrs = [s for _, _, _, s in rows]
    best = rows[int(np.argmax(snrs))]
    print(f"  best signal-to-noise at w = {best[0]}: "
          f"|C|/err = {best[3]:.1f}, a {best[3] / base:.0f}x gain "
          f"over the local operator")
    assert best[3] > 3 * base
    vals = [m for _, m, _, _ in rows]
    print(f"  BUT the VALUE moves across the range "
          f"({min(vals):+.6f} to {max(vals):+.6f}), by far more "
          f"than its error:")
    print("  each w is a DIFFERENT OBSERVABLE, not a better "
          "estimate of the same one. At large")
    print("  w the correlator is heavily distorted -- the kernel "
          "averages over the very")
    print("  separation being measured. So the kernel buys "
          "STATISTICS, and every comparison")
    print(f"  must be at MATCHED w against a baseline computed for "
          f"that w. Take w ~ r/3 to r/2\n")
    return best[0]


# ----------------------------------------------------------------
def s3_kernel_can_forge(wbest):
    print("== s3: a kernel can forge the answer ==")
    L, d, xi, noise, n = 16, 4, 4.0, 3.0, 300
    fields = make_field(L, d, xi, noise, n)   # ISOTROPIC by
    pairs = [((4, 0, 0, 0), (2, 2, 2, 2))]    # construction
    ax = 2 * np.pi * np.fft.fftfreq(L)
    g = np.meshgrid(*([ax] * d), indexing="ij")
    k2 = sum(gi ** 2 for gi in g)

    def box_kernel(b):
        """separable cubic (box) smoothing of half-width b"""
        K = np.ones((L,) * d)
        for gi in g:
            K = K * (np.sinc(gi * b / (2 * np.pi)) ** 2)
        return K
    kerns = {"radial Gaussian": np.exp(-(wbest ** 2) * k2),
             "cubic box": box_kernel(2 * wbest)}
    print("  the field is isotropic BY CONSTRUCTION, so any measured "
          "anisotropy is the")
    print("  probe's own:")
    print("     kernel              pair                    "
          "anisotropy")
    res = {}
    for name, K in kerns.items():
        vals = {}
        for p in {v for pr in pairs for v in pr}:
            acc = []
            for f in fields:
                P = np.abs(np.fft.fftn(f - f.mean())) ** 2
                C = np.real(np.fft.ifftn(P * K)) / f.size
                acc.append(float(C[p]))
            vals[p] = (float(np.mean(acc)),
                       float(np.std(acc) / np.sqrt(len(acc))))
        for a, b in pairs:
            ma, ea = vals[a]
            mb, eb = vals[b]
            A = (ma - mb) / (0.5 * (ma + mb))
            eA = abs(A) * np.sqrt((ea / ma) ** 2 + (eb / mb) ** 2)
            res[name] = (A, eA)
            print(f"   {name:18s}  {str(a):13s} vs {str(b):11s}  "
                  f"{A:+.4f} +- {eA:.4f}")
    gA, ge = res["radial Gaussian"]
    bA, be = res["cubic box"]
    print(f"  the box kernel's spurious anisotropy is "
          f"{abs(bA) / max(abs(gA), 1e-9):.1f}x the radial one's, "
          f"at {abs(bA) / be:.0f} sigma from zero.")
    print(f"  BUT THE RADIAL KERNEL IS NOT CLEAN EITHER: "
          f"{gA:+.4f} +- {ge:.4f} is {abs(gA) / ge:.0f} sigma from "
          f"zero on a field that is")
    print("  isotropic by construction. I wrote 'a radial kernel "
          "injects none' before")
    print("  measuring; that is false. What is true is that it "
          "makes the artefact small")
    print("  instead of dominant -- AND A KINEMATIC BASELINE MUST "
          "BE SUBTRACTED EITHER WAY.")
    assert abs(bA) > 3 * abs(gA) and abs(gA) > 3 * ge
    print("  THE WRONG KERNEL DOES NOT LOSE SIGNAL, IT MANUFACTURES "
          "IT; the right kernel")
    print("  manufactures less of it, and you still have to "
          "subtract what it makes\n")


def s4_the_port():
    print("== s4: the port ==")
    print("  For their 0118: smear the local operator with a "
          "RADIALLY SYMMETRIC kernel of")
    print("  width w ~ r/3 -- exp(-w^2 k^2) in the CONTINUUM "
          "momentum, which is exactly")
    print("  isotropic -- then SUBTRACT the free-field baseline "
          "computed at the SAME w,")
    print("  because s3 shows even the radial kernel manufactures "
          "some. The statistical")
    print("  gain over the local operator is the 62x measured in "
          "s2.")
    print()
    print("  And the general statement, which is what makes this a "
          "tier and not a trick:")
    print("  AN OBSERVABLE IS A CHOICE OF OBSERVATION KERNEL, and "
          "in a filter that choice")
    print("  is made by maximising information about the mode of "
          "interest subject to not")
    print("  contaminating it. Physics calls the first half "
          "'improving the overlap' and")
    print("  usually leaves the second half implicit; s3 shows the "
          "second half is where")
    print("  the errors that survive peer review live\n")


if __name__ == "__main__":
    s1_kalman_bucy()
    w = s2_which_kernel()
    s3_kernel_can_forge(w)
    s4_the_port()
    print("all assertions passed")
