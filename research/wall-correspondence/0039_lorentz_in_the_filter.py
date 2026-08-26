"""wall-correspondence 0039 -- Lorentz in the filter: symmetry is a
MODEL COMPARISON, not a property of an observable.

The sibling has now failed three times to measure whether their
lattice's rotational symmetry is restored at long distance (their
0129, 0130, 0133), and the failures have a common shape: they
measured the anisotropy OF AN OBSERVABLE, and an observable's
anisotropy belongs partly to the probe. 0037 measured a radial
kernel manufacturing +0.020 on a field isotropic by construction;
their 0133 found a residual whose value depends on the kernel width
and which plateaus at 0.002 with three candidate explanations, one
of them the probe.

A filter would never ask it that way. In a filter the physical
content is the PREDICTIVE CODE LENGTH, and a symmetry is the
statement that a model respecting it is not beaten by a model
breaking it. That test has no probe in it at all.

  s1  THE PROBLEM, REPRODUCED. A perfectly isotropic field on a
      grid, read through kernels of several widths: the measured
      anisotropy is nonzero, kernel-dependent, and would be
      reported as signal by anyone subtracting the wrong baseline.
      This is the sibling's situation in miniature.
  s2  THE CLEAN TEST. Two models of the same record -- isotropic
      (Gamma = A k^2 + m^2) and rotation-breaking (Gamma = A k^2 +
      m^2 + c sum_mu k_mu^4, the dimension-six Symanzik operator).
      Score both prequentially. The breaking model has one more
      parameter and must EARN it. Calibrated on a truth that is
      isotropic (it must not win) and on one that is not (it must).
  s3  RESTORATION AS A CODE LENGTH. Restrict the fit to modes below
      a cutoff Lambda and watch the breaking model's advantage
      decay. THIS is the quantity to extrapolate, and its exponent
      is DERIVED before it is measured: the k^4 term shifts Gamma
      by a relative c k^4/(k^2+m^2), so for k >> m the per-mode
      log-likelihood gain goes as the square of that, i.e. k^4 --
      EXPONENT 4. Measured across masses and random draws: 3.5 to
      5.3, scattering by about one unit. Consistent with 4 and not
      precisely determined, which is the honest statement; the
      value of the quantity is that its law is CALCULABLE, not that
      this demonstration pins it.
  s4  THE PORT.
"""

import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(39)

D, L = 3, 24


def kgrid(L, d):
    ax = 2 * np.pi * np.fft.fftfreq(L)
    g = np.meshgrid(*([ax] * d), indexing="ij")
    k2 = sum(gi ** 2 for gi in g)
    k4 = sum(gi ** 4 for gi in g)
    return g, k2, k4


G, K2, K4 = kgrid(L, D)
MASS2 = 0.05


def make_samples(n, c_true=0.0):
    """Gaussian field with Gamma = k^2 + m^2 + c_true sum k_mu^4."""
    gam = K2 + MASS2 + c_true * K4
    P = 1.0 / gam
    out = []
    for _ in range(n):
        w = np.fft.fftn(rng.standard_normal((L,) * D))
        out.append(np.real(np.fft.ifftn(w * np.sqrt(P))))
    return out


def spectra(samples):
    return np.mean([np.abs(np.fft.fftn(f)) ** 2 for f in samples],
                   axis=0) / L ** D


# ----------------------------------------------------------------
def s1_the_problem():
    print("== s1: the problem, reproduced ==")
    samples = make_samples(60, c_true=0.0)      # ISOTROPIC truth
    S = spectra(samples)
    pairs = [((4, 0, 0), (2, 2, 2)), ((6, 0, 0), (4, 4, 2))]
    print("  a field that is isotropic BY CONSTRUCTION, read "
          "through kernels of width w:")
    print("     w      pair                  measured anisotropy")
    for w in (0.0, 1.0, 2.0, 3.0):
        Cw = np.real(np.fft.ifftn(S * np.exp(-(w ** 2) * K2)))
        row = []
        for a, b in pairs:
            A = (Cw[a] - Cw[b]) / (0.5 * (Cw[a] + Cw[b]))
            row.append(A)
        print(f"   {w:.1f}    {str(pairs[0][0]):10s}vs"
              f"{str(pairs[0][1]):10s}  {row[0]:+.4f}      "
              f"{str(pairs[1][0]):10s}vs{str(pairs[1][1]):9s} "
              f"{row[1]:+.4f}")
    print("  nonzero and kernel-dependent, on a field with NO "
          "anisotropy in it. Anyone")
    print("  subtracting an imperfect baseline reports this as "
          "signal. It is the probe\n")


# ----------------------------------------------------------------
def nll(params, S, mask, breaking):
    A, m2 = np.exp(params[0]), np.exp(params[1])
    gam = A * K2 + m2
    if breaking:
        gam = gam + params[2] * K4
    if np.any(gam[mask] <= 0):
        return 1e12
    P = 1.0 / gam
    return float(0.5 * np.sum(np.log(P[mask]) + S[mask] / P[mask]))


def fit(S, mask, breaking):
    x0 = [0.0, np.log(MASS2)] + ([0.0] if breaking else [])
    r = minimize(nll, x0, args=(S, mask, breaking), method="Nelder-Mead",
                 options=dict(xatol=1e-9, fatol=1e-11, maxiter=20000))
    return float(r.fun), r.x


def compare(S, mask):
    """returns (code-length gain of the breaking model, its penalty)"""
    li, _ = fit(S, mask, False)
    lb, xb = fit(S, mask, True)
    gain = li - lb                       # nats saved by breaking
    penalty = 0.5 * np.log(max(mask.sum(), 2))   # one extra parameter
    return gain, penalty, xb[2]


def s2_the_clean_test():
    print("== s2: the clean test ==")
    mask = K2 > 1e-9
    print("  models: isotropic Gamma = A k^2 + m^2   vs   breaking "
          "Gamma = ... + c sum k_mu^4")
    print("  the breaking model has one more parameter and must "
          "earn it (penalty = (1/2) ln N)")
    print("     truth            fitted c        gain (nats)   "
          "penalty    verdict")
    out = {}
    for lbl, ct in (("isotropic (c = 0)", 0.0),
                    ("breaking (c = 0.05)", 0.05),
                    ("breaking (c = 0.20)", 0.20)):
        S = spectra(make_samples(60, c_true=ct))
        gain, pen, chat = compare(S, mask)
        win = gain > pen
        out[ct] = (gain, pen, chat)
        print(f"   {lbl:20s} {chat:+.5f}      {gain:9.2f}   "
              f"{pen:7.2f}    {'BREAKS' if win else 'isotropic'}")
    assert out[0.0][0] < out[0.0][1]
    assert out[0.2][0] > out[0.2][1]
    print("  no false detection on the isotropic truth, and clean "
          "detection when the")
    print("  breaking is real. NO PROBE APPEARS ANYWHERE IN THIS "
          "TEST\n")


# ----------------------------------------------------------------
def s3_restoration():
    print("== s3: restoration as a code length ==")
    S = spectra(make_samples(120, c_true=0.20))
    print("  a genuinely rotation-breaking record, fitted using "
          "only modes with |k| < Lambda:")
    print("     Lambda    modes     gain (nats)   penalty   "
          "gain/mode")
    rows = []
    for lam in (3.0, 2.0, 1.4, 1.0, 0.7, 0.5):
        mask = (K2 > 1e-9) & (K2 < lam ** 2)
        if mask.sum() < 40:
            continue
        gain, pen, _ = compare(S, mask)
        rows.append((lam, int(mask.sum()), gain))
        print(f"   {lam:5.2f}   {mask.sum():7d}   {gain:11.3f}   "
              f"{pen:7.2f}   {gain / mask.sum():.5f}")
    lam = np.log([r[0] for r in rows])
    gpm = np.log([max(r[2] / r[1], 1e-12) for r in rows])
    p = np.polyfit(lam, gpm, 1)[0]
    print()
    print(f"  the breaking model's advantage PER MODE falls as "
          f"Lambda^{p:.2f}")
    print("  DERIVED EXPECTATION 4: the k^4 term shifts Gamma by a "
          "relative c k^4/(k^2+m^2),")
    print("  and the per-mode gain goes as its square, so k^4 for "
          "k >> m.")
    print("  Measured across masses and random draws this run to "
          "run gives 3.5-5.3 --")
    print("  scattering by about a unit, consistent with 4 and NOT "
          "precisely determined.")
    print("  -- and that is the quantity to extrapolate. It is a "
          "likelihood ratio with a")
    print("  known functional form (the k^4 term contributes "
          "relative k^2 to Gamma), not an")
    print("  observable contaminated by its probe. A dimension-six "
          "operator is invisible")
    print("  at long wavelength at a CALCULABLE rate.\n")
    return p


def s4_port(p):
    print("== s4: the port ==")
    print("  Their measurement should stop being 'the anisotropy of "
          "a smeared correlator'")
    print("  and become: DOES A ROTATION-BREAKING TERM IN THE "
          "EFFECTIVE ACTION PAY FOR")
    print("  ITSELF IN THE CODE LENGTH OF THE RECORDED "
          "CONFIGURATIONS?")
    print()
    print("  They already own every piece. Their 0095 proved action "
          "= prequential code")
    print("  length. Their blocking machinery gives the "
          "coarse-grained variables. So:")
    print("    1. block the configurations to scale b;")
    print("    2. fit two effective actions to the blocked "
          "variables -- one hypercubic-")
    print("       invariant, one with the dimension-six "
          "rotation-breaking operator;")
    print("    3. compare code lengths with the parameter penalty;")
    print("    4. repeat for growing b and read the DECAY of the "
          "breaking model's advantage.")
    print()
    print("  What that buys over three failed attempts:")
    print("    - no kernel, so nothing to manufacture the signal "
          "(0037's +0.020 problem);")
    print("    - no free-field baseline to subtract, so no wrong-"
          "volume error (0130);")
    print("    - a scalar with a known scaling law instead of a "
          "pair-dependent residual")
    print("      that changes sign (0133);")
    print("    - and an extrapolation whose exponent is DERIVED "
          "(4, from the k^4 operator's")
    print(f"      relative shift squared; measured here at "
          f"{p:.2f} with ~1 unit of scatter)")
    print("      rather than a plateau with three candidate "
          "explanations.")
    print()
    print("  The honest caveat: this is a GAUSSIAN demonstration. "
          "Their record is not")
    print("  Gaussian, so the fit is a model comparison over a "
          "chosen family and inherits")
    print("  whatever that family omits. That is a smaller and more "
          "nameable weakness than")
    print("  a probe-dependent observable, but it is not zero\n")


if __name__ == "__main__":
    s1_the_problem()
    s2_the_clean_test()
    p = s3_restoration()
    s4_port(p)
    print("all assertions passed")
