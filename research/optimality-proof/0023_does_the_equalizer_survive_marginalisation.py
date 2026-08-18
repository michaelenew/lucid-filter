"""Gap 2 of layer 2: does Theorem C's equalizer survive integrating lambda out?

Theorem C (output/02) proves E_p[-log p*(lambda)] is the SAME for every member of
C(gamma_0, gamma_1), because -log p* is affine in exactly the statistics the
class fixes.  That is an exact equalizer, and it gives minimaxity in three lines.

But fit() does not code lambda.  It codes x, with lambda integrated out.  The
observable quantity is E_p[-log m*(x)] with m*(x) = int p(x|lambda) p*(lambda),
which equals E_p[g(lambda)] for g(lambda) = E_{x|lambda}[-log m*(x)].  g is the
log of an integral, so there is no reason for it to be affine in lambda's second
moments, and the equalizer has no reason to survive.

A concrete mechanism for failure: H(x|lambda) involves log(a + sigma^2 e^lambda),
whose second derivative in lambda is a c e^l / (a + c e^l)^2 > 0 -- CONVEX.  So
H(x|lambda) depends on lambda's marginal SHAPE, not only its variance, and among
fixed-variance laws the Gaussian does not maximise a convex functional.  Heavier
lambda tails should therefore make x harder to code.

Two families, both matched EXACTLY in (gamma_0, gamma_1) so Theorem C's equalizer
covers all of them:

  A  AR(1) with non-Gaussian innovations.  lambda_t = phi lambda_{t-1} + eps_t
     with eps standardised to nu = gamma_0 (1 - phi^2) fixes every autocovariance
     regardless of eps's shape, so only the MARGINAL differs.  Run in the
     moderate regime (phi = 0.5) where the MA weights decay fast enough that the
     marginal still reflects eps rather than being Gaussianised by the CLT.
  B  AR(2) with gamma_2 swept.  Gaussian marginals throughout, so this isolates
     the DEPENDENCE direction rather than the marginal-shape direction.

Measured: mean per-point code length -loglik/n of the observable under the
filter's AR(1) model at the class parameters.  Flat across a family => the
equalizer survives it.  Varying => it does not, and gap 2 closes negatively.

Run: python3 0023_does_the_equalizer_survive_marginalisation.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as tdist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "lucid"))
from statfilter import AdaptiveFilter, Params           # noqa: E402

_il = __import__("importlib.util", fromlist=["util"])
_spec = _il.spec_from_file_location(
    "leak1", Path(__file__).resolve().parent / "0003_leak1_shape_adversary.py")
leak1 = _il.module_from_spec(_spec)
_spec.loader.exec_module(leak1)
_s2 = _il.spec_from_file_location(
    "mx", Path(__file__).resolve().parent
    / "0014_is_the_AR1_member_least_favourable.py")
mx = _il.module_from_spec(_s2)
_s2.loader.exec_module(mx)

N = leak1.N
SEEDS = tuple(range(401, 461))         # 60 seeds
BURN = 400
S, PHI = 0.80, 0.50                    # moderate regime, as in 0016
G0 = S * S


def standardised(rng, shape, n):
    """Mean-zero, unit-variance draws of the requested shape."""
    if shape == "gaussian":
        return rng.normal(0.0, 1.0, n)
    if shape == "two-point":                       # lightest tail, kurt 1
        return np.where(rng.uniform(size=n) < 0.5, -1.0, 1.0)
    if shape == "uniform":                         # kurt 1.8
        return (rng.uniform(size=n) - 0.5) * 2.0 * np.sqrt(3.0)
    if shape == "t5":                              # kurt 9
        return tdist.rvs(5.0, size=n, random_state=rng) / np.sqrt(5.0 / 3.0)
    if shape == "t3":                              # kurt infinite
        return tdist.rvs(3.0, size=n, random_state=rng) / np.sqrt(3.0)
    raise ValueError(shape)


def make_ar1(seed, shape):
    """AR(1) log-scale, innovations of the given shape.  gamma_0, gamma_1 exact."""
    rng = np.random.default_rng(seed)
    nu = G0 * (1.0 - PHI * PHI)
    e = standardised(rng, shape, N + BURN) * np.sqrt(nu)
    lam = np.zeros(N + BURN)
    for t in range(1, N + BURN):
        lam[t] = PHI * lam[t - 1] + e[t]
    lam = lam[BURN:]
    R = leak1.S2_TRUE * np.exp(lam)
    theta = np.cumsum(rng.normal(0.0, np.sqrt(leak1.Q_TRUE), N))
    return theta + rng.normal(0.0, 1.0, N) * np.sqrt(R), lam


def code_length(series):
    f = AdaptiveFilter(Params(Q=leak1.Q_TRUE, s2=leak1.S2_TRUE, phi_P=0.0,
                              s_P=0.0, phi_M=PHI, s_M=S))
    return np.array([-f.filter(x).loglik / len(x) for x in series])


def report(title, rows, ref_key):
    ref = rows[ref_key]
    print(f"  {title}")
    print(f"    {'member':22s} {'code len':>10s} {'vs p*':>9s} {'se':>6s} {'t':>6s}"
          f" {'kurt(lam)':>10s}")
    for label, (cl, kl) in rows.items():
        d = cl - ref[0]
        pct = 100.0 * d.mean() / ref[0].mean()
        se = 100.0 * d.std(ddof=1) / np.sqrt(len(d)) / ref[0].mean()
        t = pct / se if se > 0 else 0.0
        mark = "  <- p* (Gaussian AR(1))" if label == ref_key else ""
        print(f"    {label:22s} {cl.mean():10.5f} {pct:+8.3f}% {se:6.3f} "
              f"{t:6.1f} {kl:10.2f}{mark}")
    print()


def main():
    print("=" * 78)
    print("Does Theorem C's equalizer survive marginalising lambda out?")
    print(f"  class fixed at gamma_0={G0:.3f}, gamma_1={PHI*G0:.3f} for EVERY row")
    print(f"  {len(SEEDS)} seeds, n={N}; code length = -loglik/n of the OBSERVABLE")
    print("  Theorem C: the LATENT code length is identical across all of these.")
    print()

    rows = {}
    for shape in ("two-point", "uniform", "gaussian", "t5", "t3"):
        got = [make_ar1(sd, shape) for sd in SEEDS]
        lam = np.concatenate([g[1] for g in got])
        kurt = float(np.mean((lam - lam.mean()) ** 4) / lam.var() ** 2)
        rows[f"AR(1), {shape} innov"] = (code_length([g[0] for g in got]), kurt)
    report("FAMILY A -- same autocovariances, different lambda MARGINAL",
           rows, "AR(1), gaussian innov")

    rows2, peak = {}, PHI * PHI
    for rho2 in (0.05, 0.15, peak, 0.40, 0.65, 0.90):
        if not np.isfinite(mx.ar2_coeffs(G0, PHI, rho2)[2]):
            continue
        got = [mx.make(sd, G0, PHI, rho2) for sd in SEEDS]
        lam = np.concatenate([np.log(g[3] / leak1.S2_TRUE) for g in got])
        kurt = float(np.mean((lam - lam.mean()) ** 4) / lam.var() ** 2)
        key = f"AR(2), rho_2={rho2:.4f}"
        rows2[key] = (code_length([g[0] for g in got]), kurt)
    report("FAMILY B -- Gaussian marginals, different DEPENDENCE (gamma_2)",
           rows2, f"AR(2), rho_2={peak:.4f}")

    print("  READ: flat within a family => the equalizer survives that direction.")
    print("  Any row with code length ABOVE p* means p* is not least favourable")
    print("  at the observable, so Theorem C does not transfer and gap 2 closes")
    print("  negatively.  Family A tests marginal shape, family B dependence.")


if __name__ == "__main__":
    main()
