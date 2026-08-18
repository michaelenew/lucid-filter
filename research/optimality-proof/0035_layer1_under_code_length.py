"""0035 -- Is layer 1 a log-loss theorem too?  The cross-entropy equalizer.

The seam: layer 1 (output/01, the Kalman filter is exactly minimax over noise
shapes given the variance path) is a SQUARED-ERROR argument; layer 2
(output/02, the log-scale AR(1) is least favourable over the two-moment
class) is a LOG-LOSS argument.  Two losses, no single theorem -- recorded in
SUMMARY as the live leak 2 since 0012/0024.

The candidate removal, checked here before it is written up as a theorem:
Theorem A's equalizer property should transfer to code length VERBATIM,
because the Kalman filter's predictive density is Gaussian with a mean that
is LINEAR in the data and a variance from the Riccati recursion that depends
on nothing but the (given) variance path.  Its cross-entropy under any
mean-zero shape p with those variances is

    E_p[-log m(x_t | past)] = 1/2 ( log 2 pi S_t + E_p[e_t^2] / S_t )

and E_p[e_t^2] is a second-moment functional of a fixed linear map of the
noises, so it equals the Riccati S_t for EVERY shape in the class:

    E_p[-log m] = 1/2 ( log 2 pi S_t + 1 ),   constant across shapes.

Equalizer + exact-Bayes-at-the-Gaussian + weak duality is then the same
three lines as Theorem A, under code length.  If this holds, both layers are
log-loss theorems and the seam is removed where it exists (the practice was
already all-log-loss: fit() maximises it and every oracle-gap number is nll).

  A  THE EQUALIZER.  A fixed heteroscedastic variance path, five mean-zero
     unit-variance shapes (gaussian, t5, uniform, two-point, skewed
     two-point).  Per-shape mean code length and mean squared innovation of
     the FIXED Kalman filter.  Both must agree across shapes within Monte
     Carlo error, and the code length must sit on the closed form
     1/2(log 2 pi S_t + 1) averaged over the path.

  B  THE DELIMITER.  The same sweep through the ADAPTIVE filter (scale
     channels live).  Its predictive variance is data-dependent, the
     cross-entropy argument does not apply, and the per-shape code lengths
     must spread -- this is leak 1's leverage, and it delimits the theorem
     exactly as in the MSE case (0003).

Run:  python3 0035_layer1_under_code_length.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "lucid"))

from statfilter import AdaptiveFilter, Params  # noqa: E402

LOG2PI = math.log(2.0 * math.pi)
FIG = os.path.join(HERE, "figures")


def shapes(rng, n):
    """Five mean-zero, unit-variance shapes."""
    g = rng.standard_normal(n)
    t5 = rng.standard_t(5, size=n) / math.sqrt(5.0 / 3.0)
    u = rng.uniform(-math.sqrt(3.0), math.sqrt(3.0), size=n)
    tp = rng.choice([-1.0, 1.0], size=n)
    # skewed two-point: values a, b with p, 1-p; mean 0, var 1
    p = 0.8
    a = math.sqrt((1 - p) / p)
    b = -math.sqrt(p / (1 - p))
    sk = np.where(rng.uniform(size=n) < p, a, b)
    return {"gaussian": g, "t5": t5, "uniform": u,
            "two-point": tp, "skewed": sk}


def kalman_path(n, Q, Rseq):
    """Riccati gains and predictive variances for the KNOWN variance path.

    Local-level model: theta_t = theta_{t-1} + w_t, x_t = theta_t + v_t.
    Returns (K_t, S_t) with S_t the one-step predictive variance of x_t.
    """
    P = 1e6                       # diffuse start; discard the burn-in later
    Ks, Ss = np.empty(n), np.empty(n)
    for t in range(n):
        Pp = P + Q
        S = Pp + Rseq[t]
        K = Pp / S
        Ks[t], Ss[t] = K, S
        P = Pp * (1.0 - K)
    return Ks, Ss


def run_fixed_kf(x, Q, Rseq, burn):
    """Code length and squared innovation of the fixed-gain-path KF."""
    n = x.size
    Ks, Ss = kalman_path(n, Q, Rseq)
    m = 0.0
    cl, se, k = 0.0, 0.0, 0
    for t in range(n):
        e = x[t] - m
        if t >= burn:
            cl += 0.5 * (LOG2PI + math.log(Ss[t]) + e * e / Ss[t])
            se += e * e
            k += 1
        m = m + Ks[t] * e
    return cl / k, se / k


def main():
    os.makedirs(FIG, exist_ok=True)
    n, burn, seeds = 4000, 50, 40
    Q = 0.3
    out = {}

    # a genuinely heteroscedastic, KNOWN measurement-variance path
    base = np.ones(n)
    base[1000:1600] = 6.0
    base[2500:2800] = 0.25
    Rseq = base

    # the closed form the equalizer predicts, path-averaged
    _, Ss = kalman_path(n, Q, Rseq)
    closed = 0.5 * float(np.mean(LOG2PI + np.log(Ss[burn:]) + 1.0))

    print("A.  THE EQUALIZER -- fixed KF, known variance path")
    print(f"    closed form 1/2 E[log 2 pi S + 1] = {closed:.5f}")
    print(f"    {'shape':10s} {'code len':>10s} {'se':>7s} {'mse':>9s} {'se':>7s}")
    resA = {}
    for name in ("gaussian", "t5", "uniform", "two-point", "skewed"):
        cls_, mses = [], []
        for s in range(seeds):
            # BOTH noises take the shape; only their variances are fixed
            w = math.sqrt(Q) * shapes(np.random.default_rng(1000 + s), n)[name]
            v = np.sqrt(Rseq) * shapes(np.random.default_rng(5000 + s), n)[name]
            theta = np.cumsum(w)
            x = theta + v
            cl, se = run_fixed_kf(x, Q, Rseq, burn)
            cls_.append(cl)
            mses.append(se)
        resA[name] = dict(cl=float(np.mean(cls_)),
                          cl_se=float(np.std(cls_) / math.sqrt(seeds)),
                          mse=float(np.mean(mses)),
                          mse_se=float(np.std(mses) / math.sqrt(seeds)))
        r = resA[name]
        print(f"    {name:10s} {r['cl']:10.5f} {r['cl_se']:7.5f}"
              f" {r['mse']:9.5f} {r['mse_se']:7.5f}")
    spread_cl = (max(r["cl"] for r in resA.values())
                 - min(r["cl"] for r in resA.values()))
    typ_se = float(np.mean([r["cl_se"] for r in resA.values()]))
    print(f"    code-length spread across shapes {spread_cl:.5f}"
          f"  (~{spread_cl/typ_se:.1f} se)   closed-form gap "
          f"{max(abs(r['cl']-closed) for r in resA.values()):.5f}")
    out["A"] = dict(closed=closed, shapes=resA,
                    spread=spread_cl, typ_se=typ_se)

    # ------------------------------------------------------------------ B
    print("\nB.  THE DELIMITER -- adaptive filter, scale channels live")
    pr = Params(Q=Q, s2=1.0, phi_M=0.7, s_M=0.9)
    resB = {}
    for name in ("gaussian", "t5", "uniform", "two-point", "skewed"):
        cls_ = []
        for s in range(12):
            w = math.sqrt(Q) * shapes(np.random.default_rng(1000 + s), n)[name]
            v = np.sqrt(Rseq) * shapes(np.random.default_rng(5000 + s), n)[name]
            x = np.cumsum(w) + v
            f = AdaptiveFilter(pr, order=5)
            cls_.append(-f.loglik(x) / n)
        resB[name] = dict(cl=float(np.mean(cls_)),
                          cl_se=float(np.std(cls_) / math.sqrt(12)))
        print(f"    {name:10s} {resB[name]['cl']:10.5f}"
              f" {resB[name]['cl_se']:7.5f}")
    spread_b = (max(r["cl"] for r in resB.values())
                - min(r["cl"] for r in resB.values()))
    print(f"    spread {spread_b:.5f} -- the adaptive filter is NOT an"
          f" equalizer; the theorem is about the fixed-path KF, as in MSE")
    out["B"] = dict(shapes=resB, spread=spread_b)

    with open(os.path.join(FIG, "fop0035.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote", os.path.join(FIG, "fop0035.json"))


if __name__ == "__main__":
    main()
