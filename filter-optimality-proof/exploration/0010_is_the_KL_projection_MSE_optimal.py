"""Does maximising likelihood also minimise tracking MSE under misspecification?

0009 established that fit() returns the KL-projection of the truth onto the
Gaussian-AR(1)-log-scale family, not the truth's own (s, phi).  The KL-projection
is by construction the LOG-LOSS-optimal representative.  The filter is used for
SQUARED ERROR.  Whether those pick the same point is the layer-1/layer-2 seam of
0001 section 6, and on a single series it is directly checkable.

Scan (s_M, phi_M) on a grid with Q and s2 held at truth, and report both
surfaces: the log-likelihood argmax (what fit() targets) and the MSE argmin
(what the filter is for).  Reference points for t5 data:

    ML / KL-projection, measured in 0007-0008 : (0.90, 0.49)
    moment-matched prediction from 0005       : (1.18, 0.20)
    the truth                                 : (0.55, 0.93)

Gaussian data is the control: the truth IS in the family there, so both optima
should sit near (0.55, 0.93) and near each other.

Run: python3 0010_is_the_KL_projection_MSE_optimal.py
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

SEEDS = tuple(range(11, 21))          # 10 seeds
S_GRID = np.array([0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20, 1.40, 1.60])
P_GRID = np.array([0.00, 0.15, 0.30, 0.45, 0.60, 0.75, 0.88, 0.95])


def surfaces(shape):
    """Mean MSE and mean per-point loglik over the (s_M, phi_M) grid."""
    series = [leak1.make_series(seed, shape) for seed in SEEDS]
    mse = np.zeros((len(S_GRID), len(P_GRID)))
    llk = np.zeros((len(S_GRID), len(P_GRID)))
    for i, s in enumerate(S_GRID):
        for j, ph in enumerate(P_GRID):
            f = AdaptiveFilter(Params(Q=leak1.Q_TRUE, s2=leak1.S2_TRUE,
                                      phi_P=0.0, s_P=0.0,
                                      phi_M=float(ph), s_M=float(s)))
            m, l = [], []
            for x, theta, _, _ in series:
                r = f.filter(x)
                m.append(np.mean((theta - r.mean) ** 2))
                l.append(r.loglik / len(x))
            mse[i, j] = np.mean(m)
            llk[i, j] = np.mean(l)
    return mse, llk


def argbest(surf, best):
    idx = np.unravel_index(best(surf), surf.shape)
    return S_GRID[idx[0]], P_GRID[idx[1]], surf[idx]


def show(name, shape, refs):
    mse, llk = surfaces(shape)
    s_ml, p_ml, v_ml = argbest(llk, np.argmax)
    s_ms, p_ms, v_ms = argbest(mse, np.argmin)
    print(f"  {name}")
    print(f"    loglik argmax (what fit targets) : s_M={s_ml:.2f}  "
          f"phi_M={p_ml:.2f}   ({v_ml:.5f} nats/pt)")
    print(f"    MSE argmin    (what filter is for): s_M={s_ms:.2f}  "
          f"phi_M={p_ms:.2f}   (MSE {v_ms:.5f})")
    print(f"    MSE at the loglik argmax          : {mse[list(S_GRID).index(s_ml)][list(P_GRID).index(p_ml)]:.5f}"
          f"   -> penalty {mse[list(S_GRID).index(s_ml)][list(P_GRID).index(p_ml)]/v_ms - 1:+.2%}")
    for label, (s, p) in refs.items():
        i = int(np.argmin(abs(S_GRID - s)))
        j = int(np.argmin(abs(P_GRID - p)))
        print(f"    at {label:26s} (s={S_GRID[i]:.2f}, phi={P_GRID[j]:.2f}): "
              f"MSE {mse[i, j]:.5f}  ({mse[i, j]/v_ms - 1:+.2%})  "
              f"loglik {llk[i, j]:.5f}")
    print()
    return mse, llk


def main():
    print("=" * 78)
    print("Does the log-loss optimum coincide with the squared-error optimum?")
    print(f"  Q and s2 held at truth; {len(SEEDS)} seeds; n={leak1.N}")
    print(f"  truth (s_M, phi_M) = ({leak1.S_M}, {leak1.PHI_M})")
    print()
    show("GAUSSIAN data (control -- truth is in the family)", "gaussian",
         {"the truth": (0.55, 0.93)})
    show("STUDENT-t5 data (misspecified)", "student-t5",
         {"the truth": (0.55, 0.93),
          "ML / KL-projection": (0.90, 0.49),
          "moment-matched prediction": (1.18, 0.20)})
    print("  READ: if the two optima coincide on the t5 panel, the seam is benign")
    print("  in practice and maximising likelihood is a sound proxy for the")
    print("  filter's actual purpose.  If the MSE optimum sits well away from the")
    print("  loglik optimum, fit() is optimising the wrong criterion.")


if __name__ == "__main__":
    main()
