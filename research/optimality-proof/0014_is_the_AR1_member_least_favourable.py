"""Layer 2's missing step: is the max-entropy member of the class the worst one?

The class fixes only gamma_0 and gamma_1 of the log-scale.  Burg says the
maximum-entropy-rate member is the Gaussian AR(1), and that is exactly the
filter's model.  Theorem A (output/01) is the same shape of statement one level
down: at fixed variance the max-entropy law is the Gaussian, and the Gaussian is
exactly least favourable.  If max-entropy => least favourable holds at layer 2
as well, then the filter models the WORST member of its own class, the saddle
argument of Theorem A goes through verbatim, and layer 2 closes.

So: hold (gamma_0, gamma_1) fixed, vary gamma_2, and see where the risk peaks.

An AR(2) log-scale realises any admissible (rho_1, rho_2).  Yule-Walker gives
    a1 = rho_1 (1 - rho_2) / (1 - rho_1^2),  a2 = (rho_2 - rho_1^2) / (1 - rho_1^2)
and innovation variance gamma_0 (1 - a1 rho_1 - a2 rho_2).  The AR(1) member is
rho_2 = rho_1^2 exactly, so the prediction is an INTERIOR maximum there, with
risk falling off on BOTH sides.  That is a distinctive signature: no monotone
confound (more persistence is easier, say) can produce it.

Note which side the shape adversary lives on.  Adding i.i.d. eta to an AR(1)
gives rho_2 / rho_1^2 = gamma_0_total / gamma_0_AR > 1, so a heavy tail always
lands on the rho_2 > rho_1^2 side -- never at the peak.

The filter is held FIXED at the AR(1) model matching (gamma_0, gamma_1), i.e. it
is the Bayes rule for the max-entropy member.  Raw MSE is the quantity the
minimax statement is about; the oracle ratio is reported alongside because the
adversary is also changing how hard the path intrinsically is.

Run: python3 0014_is_the_AR1_member_least_favourable.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "lucid"))
from statfilter import AdaptiveFilter, Params           # noqa: E402

_il = __import__("importlib.util", fromlist=["util"])
_spec = _il.spec_from_file_location(
    "leak1", Path(__file__).resolve().parent / "0003_leak1_shape_adversary.py")
leak1 = _il.module_from_spec(_spec)
_spec.loader.exec_module(leak1)

N = leak1.N
SEEDS = tuple(range(101, 141))        # 40 seeds
BURN = 500

REGIMES = [("persistent", 0.55, 0.93), ("moderate", 0.80, 0.50)]


def ar2_coeffs(g0, rho1, rho2):
    """(a1, a2, innovation sd) for an AR(2) with the given autocorrelations."""
    d = 1.0 - rho1 * rho1
    a1 = rho1 * (1.0 - rho2) / d
    a2 = (rho2 - rho1 * rho1) / d
    var_e = g0 * (1.0 - a1 * rho1 - a2 * rho2)
    ok = (abs(a2) < 1.0) and (a1 + a2 < 1.0) and (a2 - a1 < 1.0) and var_e > 0
    return a1, a2, (np.sqrt(var_e) if ok else np.nan)


def make(seed, g0, rho1, rho2):
    """Series whose log-scale is AR(2) with the prescribed autocorrelations."""
    a1, a2, sd = ar2_coeffs(g0, rho1, rho2)
    if not np.isfinite(sd):
        return None
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, 1.0, N + BURN) * sd
    lam = np.zeros(N + BURN)
    for t in range(2, N + BURN):
        lam[t] = a1 * lam[t - 1] + a2 * lam[t - 2] + e[t]
    lam = lam[BURN:]
    Rpath = leak1.S2_TRUE * np.exp(lam)
    Qpath = np.full(N, leak1.Q_TRUE)
    w = np.sqrt(Qpath) * rng.normal(0.0, 1.0, N)
    v = np.sqrt(Rpath) * rng.normal(0.0, 1.0, N)
    theta = np.cumsum(w)
    return theta + v, theta, Qpath, Rpath


def main():
    print("=" * 78)
    print("Is the AR(1) (max-entropy) member the worst member of the class?")
    print(f"  gamma_0, gamma_1 held fixed; gamma_2 varied; {len(SEEDS)} seeds, n={N}")
    print("  filter FIXED at the AR(1) model -- the Bayes rule for max entropy")
    print("  prediction: raw MSE peaks at rho_2 = rho_1^2 and falls off both sides")

    for name, s, phi in REGIMES:
        g0 = s * s
        peak = phi * phi
        grid = sorted({round(peak * f, 4) for f in
                       (0.70, 0.80, 0.90, 1.00, 1.06, 1.12)} | {round(peak, 4)})
        print()
        print(f"  {name.upper()}  s={s}, phi={phi}  ->  max-entropy rho_2 = {peak:.4f}")
        print(f"    {'rho_2':>8s} {'rho_2/rho_1^2':>13s} {'MSE':>9s} "
              f"{'vs peak':>9s} {'se':>6s} {'oracle':>8s} {'ratio':>7s}")
        ref = None
        rows = []
        for rho2 in grid:
            if rho2 >= 1.0:
                continue
            got = [make(sd, g0, phi, rho2) for sd in SEEDS]
            if any(g is None for g in got):
                continue
            f = AdaptiveFilter(Params(Q=leak1.Q_TRUE, s2=leak1.S2_TRUE,
                                      phi_P=0.0, s_P=0.0, phi_M=phi, s_M=s))
            mse = np.array([np.mean((th - f.filter(x).mean) ** 2)
                            for x, th, _, _ in got])
            orc = np.array([leak1.oracle_mse(*g) for g in got])
            rows.append((rho2, mse, orc))
            if abs(rho2 - peak) < 1e-9:
                ref = mse
        for rho2, mse, orc in rows:
            d = mse - ref
            pct = 100.0 * d.mean() / ref.mean()
            se = 100.0 * d.std(ddof=1) / np.sqrt(len(d)) / ref.mean()
            mark = "  <- max-entropy (AR(1))" if abs(rho2 - peak) < 1e-9 else ""
            print(f"    {rho2:8.4f} {rho2/peak:13.2f} {mse.mean():9.5f} "
                  f"{pct:+8.2f}% {se:6.2f} {orc.mean():8.5f} "
                  f"{mse.mean()/orc.mean():7.4f}{mark}")
    print()
    print("  READ: negative 'vs peak' on BOTH sides => max-entropy is least")
    print("  favourable at layer 2, and Theorem A's saddle argument transfers.")
    print("  Monotone in rho_2 => it does not, and the Burg step is decorative.")


if __name__ == "__main__":
    main()
