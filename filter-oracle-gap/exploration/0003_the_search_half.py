"""0003 -- The search half: basin indifference, and the screen that handicaps
every live-channel start.

`0002` showed the model half: GPB1 flattens the likelihood in the scale
coordinates and IMM restores its curvature (argmin back on the generating
s_P, 5x the depth).  This probe measures the two symptoms that a flat surface
plus a biased start screen would produce, and tests the one-line repair
FILTER-NOTES section 1 proposed.

  A  BASIN INDIFFERENCE, synthetically.  Crypto section 4 reported two optima
     -- persistent-moderate (phi_P ~ 0.8, s_P ~ 1.2) and impulsive-large
     (phi_P ~ 0.25, s_P ~ 1.9) -- separated by 0.003 nats/pt in-sample and
     0.042 out-of-sample: the optimiser picks the answer, not the data.
     Generate data from EACH basin's parameters and score BOTH hypotheses
     through both filters.  If GPB1's collapse is what erases the in-sample
     separation, the IMM in-sample gap should be several times GPB1's and
     correctly signed on both datasets.

  B  THE SCREEN, on 0029-style data (ODE with an AR(1) log-scale, true
     s_P = 0.8, the dataset whose fitted zero `0039` showed was WRONG -- the
     likelihood's argmin is at the truth and the fit still lands on 0).
     Run the shipped fit; then re-rank the same start screen with
     FILTER-NOTES section 1's correction (log Q -= s_P^2/2 on every start
     that proposes s_P > 0, plus splits that reach past 0.6) and polish from
     the corrected winner with the SAME optimiser and the SAME (GPB1)
     likelihood.  If the search half is real, the corrected screen alone --
     no model change -- should move the endpoint off the boundary.

Run:  python3 0003_the_search_half.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ode-adaptive-filter", "output"))

from odefilter import OdeFilter, Params  # noqa: E402
from odefilter.core import (_bounds, _loglik_batch, _logit,  # noqa: E402
                            _PHI_GRID)

ALPHA3 = np.array([2.785218519281637, -2.6855430450862655, 0.9003245225862656])
Q0, S20 = 1.0, 9.0
FIG = os.path.join(HERE, "figures")

sys.path.insert(0, os.path.join(ROOT, "filter-oracle-gap", "exploration"))
from importlib import import_module  # noqa: E402

_m2 = import_module("0002_per_node_covariances")
ode, logscale, imm_run, gpb1_run = _m2.ode, _m2.logscale, _m2.imm_run, _m2.gpb1_run


# ----------------------------------------------------------------------- A
def part_a():
    """Two basins, two datasets, both hypotheses, both filters."""
    H = {"persistent-moderate": dict(phi_P=0.80, s_P=1.2),
         "impulsive-large": dict(phi_P=0.25, s_P=1.9)}
    n = 2000
    print("A.  BASIN INDIFFERENCE -- in-sample nll/pt, alpha = (1,), s2 = 0.04")
    print(f"    {'data from':22s} {'filter':6s} {'nll(pm)':>9s} {'nll(il)':>9s}"
          f" {'gap':>8s} {'right?':>7s}")
    out = {}
    for src, hp in H.items():
        rng = np.random.default_rng(5 if "pers" in src else 6)
        lam = logscale(n, rng, hp["phi_P"], hp["s_P"])
        x = np.cumsum(np.sqrt(Q0 * np.exp(lam)) * rng.standard_normal(n))
        y = x + 0.2 * rng.standard_normal(n)
        row = {}
        for fname, run in (("gpb1", gpb1_run), ("imm", imm_run)):
            nlls = {}
            for hname, hh in H.items():
                pr = Params((1.0,), Q0, 0.04, phi_P=hh["phi_P"], s_P=hh["s_P"])
                nlls[hname] = run(y, pr, order=7, burn=50)[0]
            gap = nlls["impulsive-large"] - nlls["persistent-moderate"]
            want = "pm" if "pers" in src else "il"
            got = "pm" if gap > 0 else "il"
            row[fname] = dict(nlls=nlls, gap=float(gap))
            print(f"    {src:22s} {fname:6s} "
                  f"{nlls['persistent-moderate']:9.4f} "
                  f"{nlls['impulsive-large']:9.4f} {gap:+8.4f} "
                  f"{'YES' if got == want else 'no':>7s}")
        out[src] = row
    print("    gap = nll(il) - nll(pm): positive means the data prefers the")
    print("    persistent-moderate hypothesis.  The question is magnitude:")
    print("    crypto measured 0.003 in-sample under gpb1 against 0.042 of")
    print("    out-of-sample consequence.")
    return out


# ----------------------------------------------------------------------- B
def part_b():
    """The fit, then the corrected screen, same likelihood and optimiser."""
    rng = np.random.default_rng(19)                    # 0038 D / 0002 C data
    n = 900
    y = (ode(n, ALPHA3, Q0 * np.exp(logscale(n, rng, 0.9, 0.8)), rng)
         + math.sqrt(S20) * rng.standard_normal(n))

    print("\nB.  THE SCREEN, on data generated with s_P = 0.8")
    f = OdeFilter.fit(y, p=3, dynamics=False, max_iter=200)
    pr = f.params
    nll_ship = -f.loglik(y) / n
    print(f"    shipped fit:    s_P {pr.s_P:.4f}  phi_P {pr.phi_P:.3f}  "
          f"Q {pr.Q:.3f}  s_M {pr.s_M:.4f}  nll/pt {nll_ship:.4f}")

    # ---- rebuild the pass-3 screen from the same fitted base, twice
    p = 3
    v0 = pr._vec()
    d = np.diff(y[np.isfinite(y)])
    g0 = float(np.mean(d * d))
    bounds = _bounds(p, g0)
    lo, hi = np.array(bounds).T
    off = math.log(1e-6)
    base = v0.copy()
    base[p + 4] = base[p + 5] = off                    # channels off at the base
    base[p + 6], base[p + 7] = _logit(0.9), off

    def screen(splits, corrected):
        starts = []
        for pp in _PHI_GRID:
            for pm in _PHI_GRID:
                for sp, sm in splits:
                    v = base.copy()
                    v[p + 2], v[p + 3] = _logit(pp), _logit(pm)
                    v[p + 4], v[p + 5] = math.log(sp), math.log(sm)
                    if corrected:                      # FILTER-NOTES section 1
                        v[p] -= max(sp, sm) ** 2 / 2.0
                    starts.append(v)
        V = np.clip(np.array(starts), lo, hi)
        val = _loglik_batch(y, V, p, 5, 3, with_A=False)
        return V, val

    SPLITS_OLD = ((0.03, 0.03), (0.5, 0.5), (0.03, 0.5), (0.5, 0.03))
    SPLITS_NEW = SPLITS_OLD + ((1.0, 1.0), (1.0, 0.03), (0.03, 1.0),
                               (1.6, 0.03))

    rows = {}
    for name, splits, corr in (("screen as shipped", SPLITS_OLD, False),
                               ("wider splits only", SPLITS_NEW, False),
                               ("corrected Q only", SPLITS_OLD, True),
                               ("corrected + wider", SPLITS_NEW, True)):
        V, val = screen(splits, corr)
        best = V[int(np.argmax(val))]
        g = OdeFilter(order=5)
        vec, nll = g._polish(y, [best], list(range(p, p + 6)), bounds, n, p,
                             200, with_A=False)
        vec, nll = g._polish(y, [vec], list(range(p + 6)), bounds, n, p,
                             200, with_A=False)
        prf = Params._from_vec(vec, p)
        rows[name] = dict(s_P=float(prf.s_P), phi_P=float(prf.phi_P),
                          Q=float(prf.Q), s_M=float(prf.s_M),
                          nll=float(nll))
        print(f"    {name:18s} ->  s_P {prf.s_P:.4f}  phi_P {prf.phi_P:.3f}  "
              f"Q {prf.Q:.3f}  s_M {prf.s_M:.4f}  nll/pt {nll:.4f}")
    print("    (truth: s_P = 0.8, phi_P = 0.9, Q = 1, s_M = 0)")
    return dict(shipped=dict(s_P=float(pr.s_P), phi_P=float(pr.phi_P),
                             Q=float(pr.Q), nll=float(nll_ship)), screens=rows)


def main():
    os.makedirs(FIG, exist_ok=True)
    a = part_a()
    b = part_b()
    with open(os.path.join(FIG, "gap0003.json"), "w") as fh:
        json.dump(dict(A=a, B=b), fh, indent=1)
    print("\nwrote", os.path.join(FIG, "gap0003.json"))


if __name__ == "__main__":
    main()
