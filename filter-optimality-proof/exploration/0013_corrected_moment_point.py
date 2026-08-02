"""The moment-matched prediction of 0005 was miscalculated.  Re-measure.

0005 section 3 predicted that an i.i.d. shape with kurtosis kappa adds
log(kappa/3) to the log-scale variance gamma_0.  That is right only for
LOGNORMAL mixing.  In general a Gaussian scale mixture eps = sqrt(u) z with
E u = 1 shifts the log-scale by eta = log u, so the correct increment to gamma_0
is Var(log u), and

    log(kappa/3) = log E[u^2]   is not   Var(log u).

For Student-t5 the mixing law is u = 3 / chi2_5, giving Var(log u) =
trigamma(5/2) = 0.4904 -- against the 1.0986 the wrong formula supplied.  So the
predicted relocation is (s, phi) = (0.890, 0.355), not (1.184, 0.201), and 0007
measured (0.907 +- 0.065, 0.488 +- 0.089).  The "25-30% shortfall" that 0009
attributed to a KL-projection was arithmetic.

This re-runs 0011's paired contrast with the corrected point included, to see
whether it is competitive with where fit() actually lands.  Everything is paired
across seeds, so differences are far sharper than the individual MSEs.

Run: python3 0013_corrected_moment_point.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.special import polygamma

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "adaptive-random-walk-filter" / "output"))
from statfilter import AdaptiveFilter, Params           # noqa: E402

_il = __import__("importlib.util", fromlist=["util"])
_spec = _il.spec_from_file_location(
    "leak1", Path(__file__).resolve().parent / "0003_leak1_shape_adversary.py")
leak1 = _il.module_from_spec(_spec)
_spec.loader.exec_module(leak1)

SEEDS = tuple(range(11, 51))          # 40 seeds

S2, PHI = leak1.S_M ** 2, leak1.PHI_M
V_RIGHT = float(polygamma(1, 2.5))                     # Var(log u), t5
V_WRONG = float(np.log(3.0))                           # log(kappa/3)


def relocate(v):
    st2 = S2 + v
    return np.sqrt(st2), PHI * S2 / st2


def main():
    s_r, p_r = relocate(V_RIGHT)
    s_w, p_w = relocate(V_WRONG)
    print("=" * 78)
    print("Corrected moment-matched relocation for Student-t5")
    print(f"  Var(log u) = trigamma(5/2) = {V_RIGHT:.4f}   "
          f"log(kappa/3) = {V_WRONG:.4f}")
    print(f"  corrected prediction : s={s_r:.3f}  phi={p_r:.3f}")
    print(f"  0005's prediction    : s={s_w:.3f}  phi={p_w:.3f}")
    print(f"  0007 measured fit()  : s=0.907+-0.065  phi=0.488+-0.089")
    print()

    points = {
        "corrected moment point": (round(s_r, 2), round(p_r, 2)),
        "fit() / 0007 estimate ": (0.90, 0.45),
        "0005 moment point     ": (1.20, 0.15),
        "the truth             ": (0.55, 0.93),
    }
    series = [leak1.make_series(seed, "student-t5") for seed in SEEDS]
    res = {}
    for label, (s, ph) in points.items():
        f = AdaptiveFilter(Params(Q=leak1.Q_TRUE, s2=leak1.S2_TRUE,
                                  phi_P=0.0, s_P=0.0,
                                  phi_M=float(ph), s_M=float(s)))
        res[label] = np.array([np.mean((th - f.filter(x).mean) ** 2)
                               for x, th, _, _ in series])

    best = min(res, key=lambda k: res[k].mean())
    print(f"  STUDENT-t5, {len(SEEDS)} seeds, paired.  Best: {best.strip()}")
    print(f"    {'parameters':24s} {'(s, phi)':>14s} {'MSE':>9s} "
          f"{'vs best':>9s} {'se':>6s} {'t':>6s}")
    for label, v in sorted(res.items(), key=lambda kv: kv[1].mean()):
        d = v - res[best]
        pct = 100.0 * d.mean() / res[best].mean()
        se = 100.0 * d.std(ddof=1) / np.sqrt(len(d)) / res[best].mean()
        t = pct / se if se > 0 else 0.0
        s, ph = points[label]
        print(f"    {label:24s} {f'({s:.2f}, {ph:.2f})':>14s} {v.mean():9.5f} "
              f"{pct:+8.2f}% {se:6.2f} {t:6.1f}")
    print()
    print("  READ: if the corrected moment point is statistically tied with")
    print("  where fit() lands, then the shape adversary IS an exact relocation")
    print("  along the gamma_1 level set, fit() finds it, and 0009's")
    print("  'fit targets the KL-projection, not the class' was an artifact of")
    print("  the arithmetic error rather than a real effect.")


if __name__ == "__main__":
    main()
