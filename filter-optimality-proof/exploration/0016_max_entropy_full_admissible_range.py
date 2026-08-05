"""0014 again, over the FULL admissible range of gamma_2 rather than a narrow band.

0014 swept rho_2 only over +-12% of the max-entropy value rho_1^2, which for
rho_1 = 0.5 is [0.175, 0.28] out of an admissible (-0.5, 1).  That is far too
narrow to locate a maximum, so its readings do not decide anything.

Admissible range.  For (1, rho_1, rho_2) to extend to a PSD Toeplitz sequence
with a stationary AR(2) realisation, rho_2 must satisfy

    2 rho_1^2 - 1  <  rho_2  <  1

so at rho_1 = 0.5 that is (-0.5, 1) and at rho_1 = 0.93 it is (0.730, 1).  This
sweeps both essentially to their endpoints.

Reporting both quantities, because they answer different questions:
  raw MSE   -- what the minimax statement over the class is actually about
  ratio     -- filter MSE / path-oracle MSE, which removes the fact that the
               adversary is also changing how hard the path intrinsically is

The max-entropy (AR(1)) member sits at rho_2 = rho_1^2.  Least-favourability
predicts an interior maximum of RAW MSE there.

Run: python3 0016_max_entropy_full_admissible_range.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "adaptive-random-walk-filter" / "output"))
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

SEEDS = tuple(range(201, 281))        # 80 seeds
REGIMES = [("moderate", 0.80, 0.50), ("persistent", 0.55, 0.93)]


def grid_for(rho1):
    lo, hi = 2.0 * rho1 * rho1 - 1.0, 1.0
    inner = np.linspace(lo + 0.04 * (hi - lo), hi - 0.04 * (hi - lo), 9)
    return sorted(set(np.round(np.append(inner, rho1 * rho1), 4)))


def main():
    print("=" * 78)
    print("Max-entropy least-favourable, full admissible range of gamma_2")
    print(f"  gamma_0, gamma_1 fixed; {len(SEEDS)} seeds, n={leak1.N}")
    print("  filter FIXED at the AR(1) model (Bayes rule for the max-ent member)")

    for name, s, phi in REGIMES:
        g0, peak = s * s, phi * phi
        print()
        print(f"  {name.upper()}  s={s}, phi={phi}  ->  max-entropy rho_2 = {peak:.4f}"
              f"   admissible ({2*peak-1:.3f}, 1)")
        print(f"    {'rho_2':>8s} {'MSE':>9s} {'vs max-ent':>11s} {'se':>6s} "
              f"{'oracle':>8s} {'ratio':>7s}")
        rows, ref = [], None
        for rho2 in grid_for(phi):
            if not np.isfinite(mx.ar2_coeffs(g0, phi, rho2)[2]):
                continue
            got = [mx.make(sd, g0, phi, rho2) for sd in SEEDS]
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
            mark = "  <- max-entropy" if abs(rho2 - peak) < 1e-9 else ""
            print(f"    {rho2:8.4f} {mse.mean():9.5f} {pct:+10.2f}% {se:6.2f} "
                  f"{orc.mean():8.5f} {mse.mean()/orc.mean():7.4f}{mark}")
    print()
    print("  READ: raw MSE peaking at the marked row, falling both sides =>")
    print("  max-entropy is least favourable and Theorem A's saddle transfers.")
    print("  Monotone across the full range => it is not, and the class defined")
    print("  by (gamma_0, gamma_1) alone has its worst member at an endpoint.")


if __name__ == "__main__":
    main()
