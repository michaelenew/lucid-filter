"""0010 -- The oracle gap, rechecked on consolidated main, in two currencies.

Prompted by: "the oracle gap got to less than 1%".  Both that number and the
repository's recorded 89.5% are correct; they are the same measurement in
different denominators, and this probe pins both on the consolidated tree.

  span currency   fraction of the static-to-oracle span closed
  nll currency    residual gap as a fraction of the oracle's own
                  negative log-likelihood -- the "less than 1%" reading

THE PARENT'S LEDGER, so the tiers are not confused (oracle-gap/0007,
each superseding the last as the account of record):

  80.0%   SUPERSEDED -- forced channel under the removed GPB1 collapse
  89.5%   SUPERSEDED as the account -- forced AR(1)-log-scale channel under
          the shipped IMM recursion (80.0 + the 9.5 collapse repair).  It
          remains the correct value FOR THAT CONFIGURATION, which is what
          the shipped default runs and what the gate below constructs.
  96.3%   CURRENT -- the causal ceiling, decomposed and owned point by
          point: +6.8% is the AR(1)-vs-regime channel model (a KEPT model
          commitment, roadmap item 4), +3.7% irreducible detection lag.
          Nothing in the span is unaccounted.

Part 1 reruns the gate construction
(test_forced_channel_extracts_most_of_the_oracle_gap: x8 process-noise
regime, forced live channel, scored against a Kalman filter told Q_t
exactly) over four seeds.  By construction this measures the FORCED-CHANNEL
TIER, so reproducing ~89.5% on the canonical seed is a no-regression check
of the consolidated tree against the catalogue -- not the current account
of the gap, which is the 96.3% ceiling above.  The residual gap is 0.3-0.5%
of the oracle nll on every seed even at this tier.

Part 2 does the same for THIS workstream's filter: the fractional face fit
(split kernel, wide-q profile, K=25) against a true-parameter oracle
(nu, Q, sigma^2 known, K=200 kernel), prequential on the 0005-C protocol.
The FRAC-oracle gap is 0.001-0.016 nats/pt -- 0.1-1.0% of the oracle nll --
with the span fraction against the fitted parent (AR(1)) rising from ~16%
at nu=0.7 (where the span is nearly nil: an AR(1)+noise fit is close to
oracle-grade for ONE-STEP prediction of pure long memory) to 99.6% at
nu=1.7 (where the parent's span is 3 nats/pt and the momentum kernel closes
essentially all of it).

The two oracles differ in kind and the numbers must not be cross-quoted:
the parent's knows the NOISE SCHEDULE on heteroscedastic data (adaptivity);
the fractional one knows the TRUE PARAMETERS on homoscedastic data (the
face).  The adaptivity question for the fractional kernel is SUMMARY item 2
and remains unmeasured.

Run:  python 0010_the_oracle_gap_in_two_currencies.py        (~10 min)
"""
import sys
import math
import pathlib
import importlib
import re

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "lucid"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from odefilter import OdeFilter, Params  # noqa: E402
from odefilter.core import _face_optimum, _iv_alpha  # noqa: E402

m2 = importlib.import_module("0002_is_nu_learnable")
m3 = importlib.import_module("0003_one_coordinate_vs_p_free_ones")
m4 = importlib.import_module("0004_the_integer_part_must_be_exact")
m5 = importlib.import_module("0005_the_q_ridge")


# ---------------------------------------------------- part 1: the parent
def ar_qseq(n, alpha, Qseq, S2, rng):
    p = len(alpha)
    z = np.zeros(p)
    x = np.zeros(n)
    for t in range(n):
        xn = (float(np.dot(alpha, z))
              + math.sqrt(Qseq[t]) * rng.standard_normal())
        z = np.concatenate([[xn], z[:-1]])
        x[t] = xn
    return x, x + math.sqrt(S2) * rng.standard_normal(n)


def oracle_nll(y, alpha, Qseq, S2, burn):
    """Exact Kalman told Q_t -- the test battery's construction, verbatim."""
    a = np.asarray(alpha)
    p = a.size
    F = np.zeros((p, p))
    F[0] = a
    F[1:, :-1] = np.eye(p - 1)
    m = np.zeros(p)
    P = np.eye(p) * (S2 + float(np.max(Qseq))) * p
    e1 = np.zeros(p)
    e1[0] = 1.0
    nll, k = 0.0, 0
    for t, yt in enumerate(y):
        m = F @ m
        A = F @ P @ F.T
        S = A[0, 0] + Qseq[t] + S2
        e = float(yt) - m[0]
        if t >= burn:
            nll += 0.5 * (e * e / S + math.log(S) + math.log(2 * math.pi))
            k += 1
        row = A[:, 0] + Qseq[t] * e1
        m = m + row / S * e
        P = A
        P[0, 0] += Qseq[t]
        P -= np.outer(row / S, row)
    return nll / k


def alpha3():
    src = (pathlib.Path(__file__).resolve().parents[2] / "ode-filter"
           / "output" / "tests" / "test_odefilter.py").read_text()
    return tuple(float(v)
                 for v in re.search(r"ALPHA3 = \(([^)]*)\)", src).group(1).split(","))


def main():
    ALPHA3 = alpha3()

    print("== 1: the parent against the noise-schedule oracle"
          " (the gate's construction, 4 seeds) ==")
    print(f"{'seed':>5} {'span (nats/pt)':>14} {'% span closed':>13}"
          f" {'gap (nats/pt)':>13} {'% of oracle nll':>15}")
    closed, rel = [], []
    for seed in (4, 5, 6, 7):
        rng = np.random.default_rng(seed)
        n, lo, hi = 900, 400, 560
        Qseq = np.full(n, 1.0)
        Qseq[lo:hi] = 8.0
        x, y = ar_qseq(n, ALPHA3, Qseq, 9.0, rng)
        burn = 60
        nll_o = oracle_nll(y, ALPHA3, Qseq, 9.0, burn)
        nll_s = oracle_nll(y, ALPHA3, np.full(n, 1.0), 9.0, burn)
        span = nll_s - nll_o
        pr = Params(alpha=ALPHA3, Q=1.0, s2=9.0, phi_P=0.9, s_P=0.8)
        f = OdeFilter(pr, order=5).reset()
        ll = [f.update(float(v)).loglik for v in y]
        nll = -float(np.mean(ll[burn:]))
        closed.append(100 * (nll_s - nll) / span)
        rel.append(100 * (nll - nll_o) / nll_o)
        print(f"{seed:>5} {span:>14.4f} {closed[-1]:>12.1f}%"
              f" {nll - nll_o:>13.4f} {rel[-1]:>14.2f}%")
    print(f"mean: {np.mean(closed):.1f}% of span closed;"
          f" residual gap {np.mean(rel):.2f}% of the oracle nll"
          " (every seed < 1%)")

    print("\n== 2: the fractional face fit against a true-parameter oracle"
          " (0005-C protocol, 3 seeds) ==")
    n, half = 1600, 800
    print(f"{'truth':>6} {'oracle':>9} {'FRAC':>9} {'AR(1)':>9}"
          f" {'gap':>8} {'% of nll':>9} {'% span closed':>13}")
    for nu0 in (0.7, 1.0, 1.3, 1.7):
        orc, fr, a1 = [], [], []
        for sd in (0, 1, 2):
            y, s2t = m2.simulate(nu0, n, 0.5, sd)
            ao = m4.gl_split(nu0, 200)
            orc.append(m3.kalman_ll(y, ao, 1.0, s2t, half) / half)
            a, Q, s2, _ = m5.fit_wide(y[:half])
            fr.append(m3.kalman_ll(y, a, Q, s2, half) / half)
            al, Qp, s2p, _ = _face_optimum(y[:half], _iv_alpha(y[:half], 1))
            a1.append(m3.kalman_ll(y, al, Qp, s2p, half) / half)
        orc, fr, a1 = map(np.mean, (orc, fr, a1))
        span = orc - a1
        pct = 100 * (fr - a1) / span if abs(span) > 1e-9 else math.nan
        print(f"{nu0:>6} {orc:>9.4f} {fr:>9.4f} {a1:>9.4f}"
              f" {fr - orc:>8.4f} {100 * (orc - fr) / abs(orc):>8.2f}%"
              f" {pct:>12.1f}%")


if __name__ == "__main__":
    main()
