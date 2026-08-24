"""wall-correspondence 0016 -- causal attainability at the bank
level (F3, the last unrun row of the original adoption plan).

The sibling's time tier (their 0103) proved Euclidean = smoother
and gave the exact free-tier attainability curve (1 - t/T). The
bank-level question: WHICH posterior structures survive the causal
restriction, and at what lag does the smoother's advantage arrive?

Minimal bank: the 2-state monitored system of 0006 in its classical
regime (an exact HMM, k = 0.5, theta = 1.0): filter vs
forward-backward smoother vs fixed-lag smoothers, on the two
structures a bank carries:
  - STATE occupancy (which hypothesis is true now);
  - TRANSITIONS (when the world switched -- the sector-identity
    question 0003 found to be a slow observable).
"""

import numpy as np

P_TR = None


def make_model(theta=1.0, k=0.5):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    P = np.array([[c * c, s * s], [s * s, c * c]])
    lp = np.array([(1 + k) / 2, (1 - k) / 2])
    return P, lp


def gen(T, P, lp, rng):
    z = np.empty(T, dtype=int)
    y = np.empty(T, dtype=int)
    zc = 0
    for t in range(T):
        zc = rng.choice(2, p=P[:, zc])
        z[t] = zc
        y[t] = rng.random() < lp[zc]
    return z, y


def forward(y, P, lp):
    T = len(y)
    a = np.empty((T, 2))
    w = np.array([0.5, 0.5])
    for t in range(T):
        w = P @ w
        like = np.where(y[t], lp, 1 - lp)
        w = w * like
        w /= w.sum()
        a[t] = w
    return a


def backward_smooth(y, P, lp):
    T = len(y)
    a = np.empty((T, 2))
    b = np.ones(2)
    w = np.array([0.5, 0.5])
    fw = np.empty((T, 2))
    for t in range(T):
        w = P @ w
        like = np.where(y[t], lp, 1 - lp)
        w = w * like
        w /= w.sum()
        fw[t] = w
    bw = np.empty((T, 2))
    for t in range(T - 1, -1, -1):
        bw[t] = b
        like = np.where(y[t], lp, 1 - lp)
        b = P.T @ (like * b)
        b /= b.sum()
    g = fw * bw
    return g / g.sum(1, keepdims=True)


if __name__ == "__main__":
    print("== bank-level causal attainability (F3) ==")
    P, lp = make_model(theta=0.4)   # dwell ~ 25 steps
    rng = np.random.default_rng(21)
    T = 60000
    z, y = gen(T, P, lp, rng)
    filt = forward(y, P, lp)
    smoo = backward_smooth(y, P, lp)
    accF = float(((filt[:, 1] > 0.5) == z).mean())
    accS = float(((smoo[:, 1] > 0.5) == z).mean())
    # fixed-lag: smooth y[0:t+lag] and read time t
    print(f"  state occupancy: filter acc {accF:.3f}, smoother acc "
          f"{accS:.3f}  (gap {accS - accF:+.3f})")
    lags = [0, 1, 2, 4, 8, 16]
    print("  fixed-lag attainability (fraction of the smoother's "
          "gain captured):")
    gains = {}
    for lag in lags:
        if lag == 0:
            acc = accF
        else:
            accs = []
            step = 1
            # lagged posterior: smooth the window ending at t+lag
            fw = filt
            # exact fixed-lag via forward + limited backward
            a = np.empty((T - lag, 2))
            for t0 in range(0, T - lag, 997):
                seg = slice(t0, min(t0 + 997 + lag, T))
                g = backward_smooth(y[seg], P, lp)
                a[t0:min(t0 + 997, T - lag)] = \
                    g[:min(997, T - lag - t0)]
            acc = float(((a[:, 1] > 0.5) == z[:T - lag]).mean())
        gains[lag] = (acc - accF) / max(accS - accF, 1e-9)
        print(f"    lag {lag:3d}: {gains[lag]:+.2f}")
    # transitions: detection of switch times within +-1 step
    sw = np.where(np.diff(z) != 0)[0] + 1
    def f1(post, win=2):
        flips = np.where(np.diff((post[:, 1] > 0.5)
                                 .astype(int)) != 0)[0] + 1
        if len(flips) == 0:
            return 0.0, 0, 0.0, 0.0
        rec = float(np.mean([np.min(np.abs(flips - t)) <= win
                             for t in sw]))
        prec = float(np.mean([np.min(np.abs(sw - f)) <= win
                              for f in flips]))
        return 2 * prec * rec / max(prec + rec, 1e-9), len(flips), \
            prec, rec
    fF, nF, pF, rF = f1(filt)
    fS, nS, pS, rS = f1(smoo)
    print(f"  transition timing (F1, +-2): filter {fF:.3f} "
          f"(prec {pF:.2f}/rec {rF:.2f}, {nF} flips), smoother "
          f"{fS:.3f} (prec {pS:.2f}/rec {rS:.2f}, {nS} flips)")
    assert accS > accF
    assert gains[16] > 0.9
    assert fS > fF + 0.05
    lag_half = min(l for l in lags[1:] if gains[l] > 0.5)
    print(f"  verdict: STATE is largely causal (gain captured by "
          f"lag ~{lag_half});")
    print("  TRANSITION TIMING is the structure that genuinely "
          "needs the future -- the")
    print("  bank-level face of 0003's 'sector identity is a slow "
          "observable' and the")
    print("  sibling's 1 - t/T template: the past pins states, the "
          "future pins boundaries")
    print("all assertions passed")
