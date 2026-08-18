"""Can PEM replace ML in fit(), or does it fail to identify the noise scale?

0019 found PEM (minimising mean squared one-step innovation) never worse than ML
and matching the MSE optimum in 8 of 11 regimes -- but it scanned only
(s_M, phi_M) with Q and sigma^2 pinned at truth, which is exactly the slice where
the following problem cannot appear.

The worry, analytically.  Scale Q -> cQ and sigma^2 -> c sigma^2.  In a plain
Kalman filter the gain K = P^- / (P^- + R) is invariant, so the predicted mean is
invariant, so the innovations are invariant: PEM is EXACTLY FLAT along that ray
and identifies the signal-to-noise ratio only.  ML is not flat, because the
predictive DENSITY depends on absolute scale even when the predictive mean does
not.

The adaptive machinery should break the exact invariance, because the log-scale
inference compares e_t^2 against an absolute predictive variance rather than a
relative one.  Whether it breaks it ENOUGH to identify the scale is the question.

Part A sweeps the pure scale ray (c Q, c sigma^2) with s_M, phi_M at truth and
reports all three criteria.  A flat or near-flat PEM curve against a sharply
peaked ML curve means PEM cannot replace ML outright.
Part B sweeps the SNR direction (c Q, sigma^2 / c) as a control, where PEM should
be well behaved.

Run: python3 0025_does_PEM_identify_scale.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "lucid"))
from statfilter import AdaptiveFilter, Params           # noqa: E402

N = 1200
Q_T, S2_T, S_T, PHI_T = 0.05, 1.0, 0.55, 0.93
SEEDS = tuple(range(501, 531))                          # 30 seeds
C_GRID = np.array([0.25, 0.5, 0.71, 1.0, 1.41, 2.0, 4.0])


def make(seed):
    rng = np.random.default_rng(seed)
    nu = S_T * S_T * (1.0 - PHI_T * PHI_T)
    lam = np.empty(N)
    lam[0] = rng.normal(0.0, S_T)
    zz = rng.normal(0.0, np.sqrt(nu), N)
    for t in range(1, N):
        lam[t] = PHI_T * lam[t - 1] + zz[t]
    theta = np.cumsum(rng.normal(0.0, np.sqrt(Q_T), N))
    return theta + rng.normal(0.0, 1.0, N) * np.sqrt(S2_T * np.exp(lam)), theta


def criteria(series, q, s2):
    f = AdaptiveFilter(Params(Q=q, s2=s2, phi_P=0.0, s_P=0.0,
                              phi_M=PHI_T, s_M=S_T))
    mse, llk, pem = [], [], []
    for x, theta in series:
        r = f.filter(x)
        mse.append(np.mean((theta - r.mean) ** 2))
        llk.append(r.loglik / len(x))
        pem.append(np.mean(r.innovation ** 2))
    return np.array(mse), float(np.mean(llk)), float(np.mean(pem))


def sweep(series, title, qfun, s2fun):
    print(f"  {title}")
    print(f"    {'c':>6s} {'Q':>8s} {'sigma^2':>8s} {'theta-MSE':>10s} "
          f"{'loglik/pt':>10s} {'PEM':>10s} {'PEM rel':>9s}")
    rows = []
    for c in C_GRID:
        m, l, p = criteria(series, qfun(c), s2fun(c))
        rows.append((c, qfun(c), s2fun(c), m.mean(), l, p))
    p1 = [r[5] for r in rows if abs(r[0] - 1.0) < 1e-9][0]
    for c, q, s2, m, l, p in rows:
        print(f"    {c:6.2f} {q:8.4f} {s2:8.4f} {m:10.5f} {l:10.5f} "
              f"{p:10.5f} {100*(p/p1-1):+8.3f}%")
    ml = max(rows, key=lambda r: r[4])
    pm = min(rows, key=lambda r: r[5])
    ms = min(rows, key=lambda r: r[3])
    spread = 100.0 * (max(r[5] for r in rows) / min(r[5] for r in rows) - 1.0)
    print(f"    argmax loglik c={ml[0]:.2f}   argmin PEM c={pm[0]:.2f}   "
          f"argmin theta-MSE c={ms[0]:.2f}")
    print(f"    PEM total spread across the sweep: {spread:.3f}%")
    print()


def main():
    print("=" * 78)
    print("Does PEM identify the noise scale, or only the SNR?")
    print(f"  truth Q={Q_T}, sigma^2={S2_T}, s_M={S_T}, phi_M={PHI_T}; "
          f"{len(SEEDS)} seeds, n={N}")
    print("  s_M and phi_M held at truth throughout; only Q and sigma^2 move")
    print()
    series = [make(sd) for sd in SEEDS]
    sweep(series, "PART A -- pure scale ray (c Q, c sigma^2), SNR constant",
          lambda c: Q_T * c, lambda c: S2_T * c)
    sweep(series, "PART B -- SNR direction (c Q, sigma^2 / c), control",
          lambda c: Q_T * c, lambda c: S2_T / c)
    print("  READ: in part A a flat PEM against a peaked loglik means PEM cannot")
    print("  identify absolute scale and cannot replace ML outright -- it would")
    print("  need pairing with a scale condition such as the variogram identity")
    print("  gamma_0 = Q + 2 sigma^2 that stage 0 already uses.  A PEM curve with")
    print("  a clear interior minimum in part A means the adaptive machinery")
    print("  breaks the invariance and PEM is a viable drop-in.")


if __name__ == "__main__":
    main()
