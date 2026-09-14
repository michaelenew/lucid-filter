"""0001 -- the log-scale likelihood has an exact, universal Fourier spectrum.

The Sparrow proxy (`_GAP_FACTOR = 1.5`, and the same rule at `_HAZARD_GAP` and the
split ladder) sets the grid spacing by ANALOGY to the optical two-point resolution
limit (Sparrow 1916).  The parent audit concedes that transfer is unproven
(adaptive-grid finding 11, "SPECULATIVE").  adaptive-grid also measured that the
between-node ripple is ALIASING (Nyquist), not an ODE -- which says the honest
criterion is a sampling/quadrature statement about the log-scale likelihood, and
that statement needs the likelihood's spectrum.

This probe computes that spectrum in closed form and verifies it.

THE OBJECT.  One observation `y ~ N(0, e^lambda)` gives, as a function of the
log-variance `lambda`, the likelihood

    g(lambda) = (2 pi e^lambda)^{-1/2} exp(-(u/2) e^{-lambda}),     u = y^2.

The grid represents the reweighting of nodes by this factor, so its resolvability
is the resolvability of `g` -- and a uniform grid samples `g`, so aliasing is set
by `g`'s Fourier transform.

THE RESULT (derived, then verified below).  Substituting `t = (u/2) e^{-lambda}`
turns the transform into a Gamma integral:

    hat g(omega) = (2 pi)^{-1/2} (u/2)^{-i omega} (2/u)^{1/2} Gamma(1/2 + i omega),

so, using |Gamma(1/2 + i y)|^2 = pi / cosh(pi y),

    |hat g(omega)| = 1 / sqrt( u * cosh(pi omega) ).

Two facts fall out, and they are the whole point:

 1. The spectrum decays EXACTLY as e^{-pi|omega|/2} (since 1/sqrt(cosh) ~ sqrt2
    e^{-pi|omega|/2}).  The log-scale likelihood is not band-limited, but its
    band has an exact, closed-form exponential edge.  This is the rigorous object
    the Sparrow analogy was standing in for.

 2. The SHAPE is universal: `u` (the data) sets only the amplitude `1/sqrt(u)`
    and the peak location (`lambda* = log u`), never the bandwidth.  So the
    likelihood-driven resolution is an ABSOLUTE spacing in nats, independent of
    the class scale `s`.  The shipped rule `gap = 1.5 * s` scales with `s`; the
    two agree only in the prior-dominated (small-s) regime and diverge where the
    posterior is likelihood-dominated -- the large-s regime where the span-6
    experiment overflowed the arm (resolvable-regime 0017).

Run: python3 0001_the_likelihood_spectrum.py
"""
import cmath
import math

import numpy as np


def cgamma(z):
    """Lanczos complex Gamma (no scipy in this environment)."""
    g = 7
    c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    if z.real < 0.5:
        return math.pi / (cmath.sin(math.pi * z) * cgamma(1 - z))
    z -= 1
    x = c[0]
    for i in range(1, g + 2):
        x += c[i] / (z + i)
    t = z + g + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * cmath.exp(-t) * x


_trapz = getattr(np, "trapezoid", None) or (
    lambda y, x: np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)))


def g_of_lambda(lam, u):
    return (2 * np.pi) ** -0.5 * np.exp(-lam / 2) * np.exp(-0.5 * u * np.exp(-lam))


def check_dc_and_gamma():
    print("1. DC term  hat g(0) = integral g dlambda  should equal 1/sqrt(u):")
    for u in (1.0, 4.0, 0.25):
        lam = np.linspace(-80, 80, 1 << 18)
        val = _trapz(g_of_lambda(lam, u), lam)
        print(f"   u={u:5.2f}: numeric {val:.8f}   1/sqrt(u) {1 / math.sqrt(u):.8f}")
    print("\n2. |Gamma(1/2 + i y)|^2 = pi / cosh(pi y):")
    for y in (0.7, 1.5, 3.0):
        lhs = abs(cgamma(complex(0.5, y))) ** 2
        rhs = math.pi / math.cosh(math.pi * y)
        print(f"   y={y}: |Gamma|^2 {lhs:.8e}   pi/cosh {rhs:.8e}   ratio {lhs / rhs:.8f}")


def check_spectrum():
    print("\n3. Full spectrum |hat g(omega)| = 1/sqrt(u cosh(pi omega)), by FFT:")
    L, N = 60.0, 1 << 16
    lam = np.linspace(-L / 2, L / 2, N, endpoint=False)
    dl = lam[1] - lam[0]
    om = np.fft.fftshift(np.fft.fftfreq(N, d=dl)) * 2 * math.pi
    for u in (1.0, 4.0, 0.25):
        G = g_of_lambda(lam, u)
        Gh = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(G))) * dl
        mag = np.abs(Gh)
        theo = 1.0 / np.sqrt(u * np.cosh(np.pi * om))
        band = (om > 0.5) & (om < 5.0)
        slope = np.polyfit(om[band], np.log(mag[band] + 1e-300), 1)[0]
        rel = np.max(np.abs(mag[band] - theo[band]) / theo[band])
        print(f"   u={u:5.2f}: decay slope {slope:.5f} (exact -pi/2 = {-math.pi / 2:.5f}); "
              f"max rel err vs closed form over omega in [0.5,5]: {rel:.2e}")


def aliasing_spacing():
    """Poisson-summation aliasing of the uniform-grid quadrature of g.

    The grid { lambda0 + k*Delta } sampling g has quadrature error dominated by the
    first spectral alias at omega = 2 pi / Delta:

        relative alias  ~  |hat g(2 pi/Delta)| / |hat g(0)|
                        =  1 / sqrt(cosh(2 pi^2 / Delta))
                        ~  sqrt(2) * exp(-pi^2 / Delta).

    Inverting for a tolerated alias eps gives an ABSOLUTE spacing in nats:

        Delta(eps)  =  pi^2 / ln(sqrt(2)/eps).
    """
    print("\n4. Derived spacing from the aliasing bound  Delta = pi^2 / ln(sqrt2/eps):")
    print(f"   {'tolerated alias eps':>22s}{'Delta (nats)':>14s}"
          f"{'rel. alias at Delta':>22s}")
    for eps in (1e-1, 1e-2, 1e-3, 1e-4):
        D = math.pi ** 2 / math.log(math.sqrt(2) / eps)
        actual = 1.0 / math.sqrt(math.cosh(2 * math.pi ** 2 / D))
        print(f"   {eps:22.0e}{D:14.3f}{actual:22.2e}")
    print("   (context: the shipped rule gives gap = 1.5*s nats; the measured "
          "dead-zone\n    onset is ~0.7-0.8 nats, adaptive-grid finding 11.  Which "
          "tolerance / which\n    criterion the shipped 1.5 encodes is the open this "
          "workstream must close.)")


if __name__ == "__main__":
    print(__doc__.split("Run:")[0])
    check_dc_and_gamma()
    check_spectrum()
    aliasing_spacing()
