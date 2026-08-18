"""How big is the log-loss / squared-error seam, with paired standard errors?

0010 scanned (s_M, phi_M) and found the loglik argmax and the MSE argmin land
far apart in parameter space but within ~0.6% in MSE.  The MSE surface is flat
in phi_M, so those argmins are noisy and the 0.6% needs an error bar before it
means anything.

All contrasts here are PAIRED -- the same series filtered with different
parameters -- so the standard error of the difference is far smaller than the
standard error of either MSE, and the comparison is much sharper than 0010's.

Reference points (t5): truth (0.55, 0.93); ML / KL-projection (0.90, 0.45);
moment-matched (1.20, 0.15); 0010's MSE argmin (0.75, 0.15) and loglik argmax
(0.90, 0.75).

Run: python3 0011_seam_size_paired.py
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

SEEDS = tuple(range(11, 41))          # 30 seeds

POINTS = {
    "gaussian": {
        "truth                (0.55, 0.93)": (0.55, 0.93),
        "0010 loglik argmax   (0.60, 0.88)": (0.60, 0.88),
        "0010 MSE argmin      (0.45, 0.88)": (0.45, 0.88),
        "homoscedastic        (0.00, ----)": (0.0, 0.0),
    },
    "student-t5": {
        "truth                (0.55, 0.93)": (0.55, 0.93),
        "ML / KL-projection   (0.90, 0.45)": (0.90, 0.45),
        "0010 loglik argmax   (0.90, 0.75)": (0.90, 0.75),
        "0010 MSE argmin      (0.75, 0.15)": (0.75, 0.15),
        "moment-matched       (1.20, 0.15)": (1.20, 0.15),
        "homoscedastic        (0.00, ----)": (0.0, 0.0),
    },
}


def run(shape):
    series = [leak1.make_series(seed, shape) for seed in SEEDS]
    out = {}
    for label, (s, ph) in POINTS[shape].items():
        f = AdaptiveFilter(Params(Q=leak1.Q_TRUE, s2=leak1.S2_TRUE,
                                  phi_P=0.0, s_P=0.0,
                                  phi_M=float(ph), s_M=float(s)))
        out[label] = np.array([np.mean((theta - f.filter(x).mean) ** 2)
                               for x, theta, _, _ in series])
    return out


def main():
    print("=" * 78)
    print("Size of the log-loss / squared-error seam, paired across seeds")
    print(f"  Q and s2 at truth; {len(SEEDS)} seeds; n={leak1.N}")
    print("  'vs best' is the PAIRED mean difference in MSE, as a percentage,")
    print("  with the paired standard error.  |t| < 2 means indistinguishable.")
    for shape in ("gaussian", "student-t5"):
        res = run(shape)
        best = min(res, key=lambda k: res[k].mean())
        print()
        print(f"  {shape.upper()}   (best grid point: {best.strip()})")
        print(f"    {'parameters':36s} {'MSE':>8s} {'vs best':>9s} {'se':>7s} {'t':>6s}")
        for label, v in sorted(res.items(), key=lambda kv: kv[1].mean()):
            d = v - res[best]
            pct = 100.0 * d.mean() / res[best].mean()
            se = 100.0 * d.std(ddof=1) / np.sqrt(len(d)) / res[best].mean()
            t = pct / se if se > 0 else 0.0
            print(f"    {label:36s} {v.mean():8.5f} {pct:+8.2f}% "
                  f"{se:7.2f} {t:6.1f}")
    print()
    print("  READ: on t5, if 'ML / KL-projection' is within a few tenths of a")
    print("  percent of the best point while 'truth' is materially worse, then")
    print("  fit() targeting log-loss is not costing the filter anything, and")
    print("  the KL-projection beats the true parameters for tracking.")


if __name__ == "__main__":
    main()
