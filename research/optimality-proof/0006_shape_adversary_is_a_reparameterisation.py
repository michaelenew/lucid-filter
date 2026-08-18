"""Is the shape adversary just a move *within* the model, at different (s, phi)?

0005 argued that an increment law with kurtosis >= 3 IS a Gaussian scale
mixture, so it is already a member of C(s, phi) at some other (s, phi) -- the
filter is not being attacked from outside its class, it is being relocated
inside it.  If that is right it is not a hand-wave, it is arithmetic, and it
makes a sharp falsifiable prediction about what fit() must return.

The arithmetic.  Generate v_t = sqrt(s2 e^{lam_t}) u_t with lam a stationary
AR(1) of variance s_M^2 and persistence phi_M, and u_t i.i.d. of unit variance
and kurtosis k.  Marginal kurtosis of v is k * e^{s_M^2}.  A Gaussian scale
mixture of total log-scale variance s_tot^2 has kurtosis 3 e^{s_tot^2}.  Equate:

    s_tot^2 = s_M^2 + log(k / 3)

and since u is i.i.d. it adds nothing to the lag-1 autocovariance, so

    phi_eff = phi_M * s_M^2 / s_tot^2 .

So the prediction is not "the fit degrades" but "the fit moves, to here":

    kurtosis > 3  ->  larger s_M, smaller phi_M, by the amounts above
    kurtosis < 3  ->  s_tot^2 < 0, i.e. NO representation exists, so s_M -> 0

That last line is the whole limitation of the filter in one equation.

Run: python3 0006_shape_adversary_is_a_reparameterisation.py
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

S_M, PHI_M = leak1.S_M, leak1.PHI_M          # 0.55, 0.93
SEEDS = (11, 12, 13, 14)
KURT = {"two-point": 1.0, "uniform": 1.8, "gaussian": 3.0, "student-t5": 9.0}


def predict(kurt):
    """(s_tot, phi_eff) implied by the scale-mixture identification."""
    s2tot = S_M ** 2 + np.log(kurt / 3.0)
    if s2tot <= 0:
        return 0.0, float("nan")             # no representation exists
    s_tot = np.sqrt(s2tot)
    return s_tot, PHI_M * S_M ** 2 / s2tot


def main():
    print("=" * 78)
    print("Does fit() relocate to the predicted (s_M, phi_M) under a shape change?")
    print(f"  truth: s_M={S_M}, phi_M={PHI_M}, Q={leak1.Q_TRUE}, "
          f"s2={leak1.S2_TRUE}, n={leak1.N}, {len(SEEDS)} seeds")
    print()
    print(f"  {'shape':12s} {'kurt':>5s} | {'s_M pred':>9s} {'s_M fit':>18s}"
          f" | {'phi_M pred':>10s} {'phi_M fit':>18s}")
    print("  " + "-" * 82)
    for shape, kurt in KURT.items():
        s_pred, phi_pred = predict(kurt)
        sf, pf = [], []
        for seed in SEEDS:
            x, _, _, _ = leak1.make_series(seed, shape)
            p = AdaptiveFilter.fit(x).params
            sf.append(p.s_M)
            pf.append(p.phi_M)
        sf, pf = np.array(sf), np.array(pf)
        pp = "   n/a" if np.isnan(phi_pred) else f"{phi_pred:10.3f}"
        print(f"  {shape:12s} {kurt:5.1f} | {s_pred:9.3f} "
              f"{np.mean(sf):7.3f} +- {sf.std(ddof=1)/np.sqrt(len(sf)):.3f}"
              f"       | {pp} "
              f"{np.mean(pf):7.3f} +- {pf.std(ddof=1)/np.sqrt(len(pf)):.3f}")
    print()
    print("  READ.  If the fitted values track the predicted ones, the shape")
    print("  adversary is a reparameterisation and Leak 1 collapses into Leak 3")
    print("  (parameter estimation), which is already an open item rather than a")
    print("  new one.  If they do not, the identification argument is wrong.")
    print("  For kurtosis < 3 no representation exists and s_M should collapse")
    print("  toward 0 -- that is the light-tail limitation, stated as arithmetic.")


if __name__ == "__main__":
    main()
