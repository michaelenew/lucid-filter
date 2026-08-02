"""Completes 0018's last row, which crashed fit(), and documents why.

BUG in the parent workstream (adaptive-random-walk-filter/output/statfilter/core.py):

    def _expit(z: float) -> float:
        return 1.0 / (1.0 + math.exp(-z))

overflows with OverflowError once z < -709, and fit()'s inner ll() catches only
ValueError, so the exception escapes and kills the fit.  _logit clamps p to
[1e-9, 1-1e-9], i.e. |logit| <= 20.7, so the STARTS are always safe -- but
stage 1 is an unconstrained Nelder-Mead search and on extreme data it walks out
there.  0018's two-point p=0.30 mixing law (Var(log u) = 2.96, noise variance
0.074x baseline 70% of the time) is enough to trigger it.

Two-line fix, not applied here because that file is the parent workstream's
deliverable:

    def _expit(z: float) -> float:
        if z >= 0.0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

and widening ll()'s except to (ValueError, OverflowError) as a belt-and-braces
guard.  The patched form is algebraically identical and overflow-free on both
branches.

This script monkeypatches the module global -- Params._from_vec resolves _expit
at call time, so patching core._expit is enough -- and runs the missing row.

Run: python3 0021_complete_row_and_expit_overflow.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "adaptive-random-walk-filter" / "output"))
import statfilter.core as core                          # noqa: E402
from statfilter import AdaptiveFilter                   # noqa: E402


def _expit_safe(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


core._expit = _expit_safe                               # the patch

_il = __import__("importlib.util", fromlist=["util"])
_spec = _il.spec_from_file_location(
    "kvu", Path(__file__).resolve().parent / "0018_kurtosis_vs_var_log_u.py")
kvu = _il.module_from_spec(_spec)
_spec.loader.exec_module(kvu)


def main():
    print("=" * 78)
    print("0018's missing row, with _expit patched to be overflow-safe")
    print(f"  kappa = {kvu.KURT} as in every other row; "
          f"log(kappa/3) = {np.log(kvu.KURT/3):.4f}")
    print()
    sf, pf, vlu = [], [], None
    for seed in kvu.SEEDS:
        series, vlu = kvu.make(seed, 0.30)
        p = AdaptiveFilter.fit(series[0], order=kvu.ORDER).params
        sf.append(p.s_M)
        pf.append(p.phi_M)
    sf, pf = np.array(sf), np.array(pf)
    s_b = np.sqrt(kvu.S2T + vlu)
    p_b = kvu.PHIT * kvu.S2T / (kvu.S2T + vlu)
    s_k = np.sqrt(kvu.S2T + np.log(kvu.KURT / 3.0))
    print(f"  two-pt p=.30   Var(log u) = {vlu:.4f}")
    print(f"    s_M   Theorem B {s_b:.3f}   kurtosis {s_k:.3f}   "
          f"fitted {sf.mean():.3f} +- {sf.std(ddof=1)/np.sqrt(len(sf)):.3f}")
    print(f"    phi_M Theorem B {p_b:.3f}                  "
          f"fitted {pf.mean():.3f} +- {pf.std(ddof=1)/np.sqrt(len(pf)):.3f}")
    print()
    print("  With the patch the fit completes, so the crash was the overflow")
    print("  and not a property of the data.")


if __name__ == "__main__":
    main()
