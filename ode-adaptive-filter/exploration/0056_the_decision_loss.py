"""0056 -- The decision loss: h-step cross-forecasts score the trusted
distribution, and the lead time is the forecastable horizon.

0047 section 4 item 4, under the decision frame chosen this session: tau's
calibration is scored by the loss of the decisions it feeds, and every
consumer of a lead/lag acts through CROSS-FORECASTS -- the leader's history
predicting the follower at horizon h.  The score is the prequential log score
of h-step forecasts of y2, swept over h.  No free parameters (it is log-loss
again), no proxy truth (the forecasts verify against realised observations),
and the horizon axis is the decision class; a trading rule picks its h, the
score is defined for all of them.

Two predictions, both measured here:

  (a) THE KNEE.  Forecasting y2 at t+h needs x(t+h-tau): for h < tau that is
      the leader's OBSERVED past (tracking-grade accuracy, h-independent);
      for h > tau it is extrapolation (variance grows on the memory scale).
      So the oracle loss-vs-h curve is flat until h = tau and rises after:
      the lead time IS the horizon out to which the leader forecasts the
      follower at tracking grade.  Swept at tau in {0.5, 1.5, 2.5, 3.5}.

  (b) THE INSTRUMENT.  0047 found that a tau-overconfident member (restart-
      only staircase) is prequentially near-tied at one step.  The prediction
      was that the decision loss would expose it at depth -- the gap opening
      as h grows through the ramp.  MEASURED RESULT: THE PREDICTION INVERTS.
      The gap is ~5 millinats/pt at EVERY horizon and both members calibrate
      fine in y-space at all h -- the forecast's tau-sensitivity does not
      grow with h while the irreducible variance does.  The decision loss
      does not fail to see the tau-band miscalibration; it PRICES it, and
      for every cross-forecast consumer the price is negligible.  The ramp
      "problem" of 0046/0050 is real only for consumers that act on tau
      DIRECTLY (alignment-type decisions), whose decision loss is the
      tau-RMS/coverage 0050 already reports.  Calibration-in-consequences,
      not calibration-in-tau, is the right currency -- and it is cheap here.
      (The kinetic member's forecasts freeze the tau nodes over the horizon
      -- conservative, biases against it; both members' mild overdispersion
      at large h, E[e^2/S] ~ 0.75, is that conservatism showing.)

Outputs: figures/fig39-decision-loss.png, figures/ode056.json
"""
import json
import numpy as np

import theory_style as ts
import matplotlib.pyplot as plt

from importlib import import_module
d43 = import_module("0043_the_delay_row")
d46 = import_module("0046_online_learned_offset")
d50 = import_module("0050_the_persistence_axis")
d54 = import_module("0054_which_series_leads")

FINE, K = 32, 4
N = d46.N
SIG2 = 0.3
C = 0.7
HS = [1, 2, 3, 4, 6, 8, 12]
STRIDE = 2                     # score every STRIDE-th origin
BURN = 60


def powers(F, Q, hmax):
    Fh, Qh = {0: np.eye(len(F))}, {0: np.zeros_like(Q)}
    for h in range(1, hmax + 1):
        Fh[h] = F @ Fh[h - 1]
        Qh[h] = F @ Qh[h - 1] @ F.T + Q
    return Fh, Qh


def run_and_score(mx, F, Q, y1, y2, t0, t1):
    Fh, Qh = powers(F, Q, max(HS))
    H2, R2 = mx.H2, mx.R2
    ll = {h: 0.0 for h in HS}; cal = {h: [] for h in HS}; n = {h: 0 for h in HS}
    for t in range(N):
        mx.step(y1[t], y2[t])
        if not (t0 <= t < t1 and (t - t0) % STRIDE == 0):
            continue
        for h in HS:
            if t + h >= N:
                continue
            mh = mx.m @ Fh[h].T
            Ph = np.matmul(np.matmul(Fh[h], mx.P), Fh[h].T) + Qh[h]
            mean = np.einsum('bi,bi->b', H2, mh)
            S = np.einsum('bi,bij,bj->b', H2, Ph, H2) + R2
            y = y2[t + h]
            lp = -0.5 * (np.log(2 * np.pi * S) + (y - mean) ** 2 / S)
            mxv = lp.max()
            ll[h] += np.log(np.sum(mx.w * np.exp(lp - mxv))) + mxv
            pm = float(np.sum(mx.w * mean))
            pv = float(np.sum(mx.w * (S + mean ** 2)) - pm ** 2)
            cal[h].append((y - pm) ** 2 / pv)
            n[h] += 1
    return {h: ll[h] / max(n[h], 1) for h in HS}, \
           {h: float(np.mean(cal[h])) for h in HS}


def oracle_knee(tau, seed):
    """Single-node filter with tau and c known, static truth."""
    path = np.full(N, int(round(tau * FINE)))
    sys_, y1, y2 = d54.simulate(path, seed)
    G, LLt, read = sys_
    d = G.shape[0]
    F, Q, C0 = d43.aug_model(G, LLt, K, diffuse=False)
    _, _, C0d = d43.aug_model(G, LLt, K, diffuse=True)
    lvl = np.array([i * d for i in range(K + 1)])
    C0[np.ix_(lvl, lvl)] = C0d[np.ix_(lvl, lvl)]
    D = d * (K + 1)
    h1 = np.zeros(D); h1[:d] = read
    row, vb = d43.delay_row(G, LLt, read, tau, K, d)
    H2 = (C * row)[None, :]
    R2 = np.array([C * C * vb + SIG2 ** 2])
    mx = d46.Mixture(F, Q, C0, h1, H2, R2, D, 0.0, 0.0)
    ll, cal = run_and_score(mx, F, Q, y1, y2, BURN, N - max(HS))
    return ll


if __name__ == "__main__":
    res = {}

    # (a) the knee: oracle loss vs horizon at four lead times
    knee = {}
    for tau in (0.5, 1.5, 2.5, 3.5):
        knee[tau] = oracle_knee(tau, 5601 + int(tau * 2))
    res["a_knee"] = {str(t): {str(h): float(v) for h, v in d.items()}
                     for t, d in knee.items()}

    # (b) the instrument: staircase vs kinetic on the ramp
    sys_, y1, y2 = d46.simulate(4601, "moving", True)
    G, LLt, read, F, Q, C0, h1, H2, R2, D = d46.coupled_setup()
    nt, nc, nv = len(d46.TAUS), len(d46.CS), len(d50.VDOTS)
    H2k = np.repeat(H2.reshape(nt, nc, D), nv, axis=0).reshape(-1, D)
    R2k = np.repeat(R2.reshape(nt, nc), nv, axis=0).reshape(-1)
    Tk = np.kron(d50.kinetic_kernel(), np.eye(nc))
    members = {
        "staircase": d46.Mixture(F, Q, C0, h1, H2, R2, D, 0.0, 1e-2),
        "kinetic": d46.Mixture(F, Q, C0, h1, H2k, R2k, D, T=Tk),
    }
    res["b_ramp"] = {}
    for nm, mx in members.items():
        ll, cal = run_and_score(mx, F, Q, y1, y2, 630, N - max(HS))
        res["b_ramp"][nm] = {"ll": {str(h): float(v) for h, v in ll.items()},
                             "cal": {str(h): float(v) for h, v in cal.items()}}
    res["b_gap_vs_h"] = {str(h): float(res["b_ramp"]["kinetic"]["ll"][str(h)]
                                       - res["b_ramp"]["staircase"]["ll"][str(h)])
                         for h in HS}
    print(json.dumps(res, indent=1))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))
    ax = axes[0]
    for i, (tau, d) in enumerate(knee.items()):
        hs = sorted(d); vals = [d[h] for h in hs]
        ax.plot(hs, vals, "o-", color=ts.SERIES[i], label=rf"$\tau={tau}$")
        ax.axvline(tau, color=ts.SERIES[i], lw=0.7, ls=":")
    ax.set_title("the knee: the lead time is the forecastable horizon")
    ax.set_xlabel("forecast horizon $h$"); ax.set_ylabel("log score / point")
    ax.legend(); ts.tidy(ax)

    ax = axes[1]
    for nm, col in (("staircase", ts.SERIES[3]), ("kinetic", ts.SERIES[2])):
        hs = HS
        ax.plot(hs, [res["b_ramp"][nm]["ll"][str(h)] for h in hs], "o-",
                color=col, label=nm)
    ax.set_title("ramp: near-tied at every depth (gap $\\approx$ 5 mnats/pt)")
    ax.set_xlabel("$h$"); ax.set_ylabel("log score / point")
    ax.legend(); ts.tidy(ax)

    ax = axes[2]
    for nm, col in (("staircase", ts.SERIES[3]), ("kinetic", ts.SERIES[2])):
        ax.plot(HS, [res["b_ramp"][nm]["cal"][str(h)] for h in HS], "o-",
                color=col, label=nm)
    ax.axhline(1.0, color=ts.INK2, lw=0.8, ls="--")
    ax.set_title(r"calibration $E[e^2/S]$ vs horizon (1 = honest)")
    ax.set_xlabel("$h$"); ax.set_ylabel("calibration")
    ax.legend(); ts.tidy(ax)
    ts.save(fig, "figures/fig39-decision-loss.png")

    with open("figures/ode056.json", "w") as f:
        json.dump(res, f, indent=1)
    print("wrote figures/ode056.json")
