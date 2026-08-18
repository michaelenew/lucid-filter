"""Is the t5 fitting shortfall in 0007 numerical, or is it the KL-projection?

0007 found fit() relocating in the predicted direction under a heavy-tailed
shape but landing ~25-30% short of the moment-matched (s_M, phi_M).  Three
candidate causes were listed; the cheapest to test is quadrature truncation.
The default 5-node Gauss-Hermite grid reaches only about +-2.86 SD of the
log-scale, so a large s_M is represented worse than a small one.

If raising the order moves the t5 fit toward the predicted s_M = 1.184, the
shortfall is numerical and the identification argument of 0005 is exactly right.
If it does not move, the shortfall is the KL-projection onto the Gaussian-AR(1)
log-scale family, and the moment formula has to be replaced by a projection
calculation.

Run: python3 0008_quadrature_order_and_the_shortfall.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "lucid"))
from statfilter import AdaptiveFilter                   # noqa: E402

_il = __import__("importlib.util", fromlist=["util"])
_spec = _il.spec_from_file_location(
    "leak1", Path(__file__).resolve().parent / "0003_leak1_shape_adversary.py")
leak1 = _il.module_from_spec(_spec)
_spec.loader.exec_module(leak1)

SEEDS = (11, 12, 13, 14)
PRED = {"gaussian": (0.550, 0.930), "student-t5": (1.184, 0.201)}


def main():
    print("=" * 78)
    print("Does quadrature order explain the 0007 shortfall?")
    print(f"  truth s_M={leak1.S_M}, phi_M={leak1.PHI_M}; {len(SEEDS)} seeds")
    print("  grid reach is roughly +-2.86 SD at order 5, +-3.19 at 7, +-3.75 at 9")
    print()
    for shape, (sp, pp) in PRED.items():
        print(f"  {shape}  (predicted s_M={sp:.3f}, phi_M={pp:.3f})")
        print(f"    {'order':>5s} {'s_M fit':>16s} {'phi_M fit':>16s}")
        for order in (5, 7, 9, 13):
            sf, pf = [], []
            for seed in SEEDS:
                x, _, _, _ = leak1.make_series(seed, shape)
                p = AdaptiveFilter.fit(x, order=order).params
                sf.append(p.s_M)
                pf.append(p.phi_M)
            sf, pf = np.array(sf), np.array(pf)
            print(f"    {order:5d} {sf.mean():8.3f} +- "
                  f"{sf.std(ddof=1)/np.sqrt(len(sf)):.3f}"
                  f" {pf.mean():10.3f} +- {pf.std(ddof=1)/np.sqrt(len(pf)):.3f}")
        print()
    print("  READ: t5 s_M trending toward 1.184 as order rises -> numerical.")
    print("  Flat in order -> the shortfall is the KL-projection, and the")
    print("  moment-matching formula in 0005 needs replacing.")


if __name__ == "__main__":
    main()
