"""The bank in action: no (phi, s) set -- the data finds the ridge and averages it.

WalkingBank is the ridge theory (finding 14) made into a filter.  The caller
supplies only the model class (a stationary AR(1) log-scale) and a broad grid box;
the bank runs a WalkingFilter at each (phi, s), and online Bayesian model
averaging pours weight onto the ridge the data supports while integrating over the
sloppy direction.  No number is set.

Measured on a non-stationary process scale (quiet, then a loud regime):
(a) the model-averaged process log-scale vs truth, next to a single WalkingFilter
    given the true (phi, s) -- the bank matches it without being told;
(b) what the data learned: phi_hat, s_hat over time settle onto the ridge, and the
    effective model count sheds from the full grid onto a handful;
(c) parity: level- and scale-tracking RMSE, bank vs the oracle single filter --
    the bank pays ~nothing for not being told (phi, s).

Run: python 0028_result_bank.py   (~10 s)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lucid"))
sys.path.insert(0, os.path.join(HERE, "..", "random-walk-filter", "scripts"))

from statfilter import WalkingBank, WalkingFilter  # noqa: E402
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

PHI, S, Q, S2, NT, SHIFT = 0.9, 0.45, 1.0, 1.0, 2200, 1100


def make(seed):
    rng = np.random.default_rng(seed)
    z = 0.0; lam = np.zeros(NT)
    for t in range(NT):
        z = PHI * z + np.sqrt(S * S * (1 - PHI ** 2)) * rng.standard_normal()
        lam[t] = z
    lam[:SHIFT] += -0.75
    lam[SHIFT:] += 2.25
    theta = np.cumsum(rng.standard_normal(NT) * np.sqrt(Q * np.exp(lam)))
    x = theta + rng.standard_normal(NT) * np.sqrt(S2)
    return x, theta, lam


def main():
    x, theta, lam = make(7)
    warm = 100

    t0 = time.time()
    rb = WalkingBank(Q=Q, s2=S2).filter(x)           # nothing but Q, s2, the class
    t_bank = time.time() - t0
    ro = WalkingFilter(Q=Q, s2=S2, phi=PHI, s=S).filter(x)   # told the true (phi, s)

    def align(v):
        return v - np.median(v[warm:warm + 300]) + np.median(lam[warm:warm + 300])
    def s_rmse(v):
        return float(np.sqrt(np.mean((align(v)[warm:] - lam[warm:]) ** 2)))
    def l_rmse(m):
        return float(np.sqrt(np.mean((m[warm:] - theta[warm:]) ** 2)))

    print(f"[bank] {len(WalkingBank(Q,S2).filters)} models, no (phi,s) set, {t_bank*1000:.0f} ms")
    print(f"[bank]  scale-RMSE {s_rmse(rb.process_scale):.3f} | level-RMSE {l_rmse(rb.mean):.3f}")
    print(f"[oracle single]  scale-RMSE {s_rmse(ro.process_scale):.3f} | level-RMSE {l_rmse(ro.mean):.3f}")
    print(f"[learned] phi_hat {rb.phi_hat[-1]:.3f}, s_hat {rb.s_hat[-1]:.3f} "
          f"(truth {PHI}, {S}; ridge, not the point)")
    print(f"[weights] n_eff end {rb.n_eff[-1]:.1f} of {len(WalkingBank(Q,S2).filters)}")

    tt = np.arange(NT)
    fig, ax = plt.subplots(1, 3, figsize=(16.2, 4.6))

    a = ts.tidy(ax[0])
    a.plot(tt, align(lam), color=ts.INK, lw=1.7, label="true process log-scale")
    a.plot(tt, align(rb.process_scale), color=ts.SERIES[3], lw=1.2,
           label=f"bank (no φ,s set)  RMSE {s_rmse(rb.process_scale):.2f}")
    a.plot(tt, align(ro.process_scale), color=ts.SERIES[2], lw=1.0, ls="--",
           label=f"oracle single  RMSE {s_rmse(ro.process_scale):.2f}")
    a.axvline(SHIFT, color=ts.SERIES[1], lw=1.0, ls=":")
    a.set_xlabel("step"); a.set_ylabel("process log-scale (nats)")
    a.set_title("(a) the bank tracks the scale with no (φ,s) supplied")
    a.legend(loc="upper left", fontsize=7.2)

    a = ts.tidy(ax[1])
    a.plot(tt, rb.phi_hat, color=ts.SERIES[5], lw=1.5, label="phi_hat (learned)")
    a.plot(tt, rb.s_hat, color=ts.SERIES[3], lw=1.5, label="s_hat (learned)")
    a.axhline(PHI, color=ts.SERIES[5], lw=0.8, ls=":")
    a.axhline(S, color=ts.SERIES[3], lw=0.8, ls=":")
    a.set_xlabel("step"); a.set_ylabel("posterior-mean (φ, s)")
    a.set_title("(b) what the data learned — settles onto the ridge")
    a.legend(loc="center right", fontsize=7.4)
    a2 = a.twinx()
    a2.plot(tt, rb.n_eff, color=ts.INK2, lw=1.0, alpha=0.7)
    a2.set_ylabel("n_eff models", fontsize=8, color=ts.INK2)

    a = ts.tidy(ax[2])
    xw = np.arange(2); w = 0.34
    a.bar(xw - w/2, [s_rmse(rb.process_scale), l_rmse(rb.mean)], w,
          color=ts.SERIES[3], label="bank (no φ,s)")
    a.bar(xw + w/2, [s_rmse(ro.process_scale), l_rmse(ro.mean)], w,
          color=ts.SERIES[2], label="oracle single (told φ,s)")
    a.set_xticks(xw); a.set_xticklabels(["scale RMSE", "level RMSE"])
    a.set_ylabel("RMSE")
    a.set_title("(c) parity — the bank pays ~nothing for not being told")
    a.legend(loc="upper right", fontsize=7.6)
    ts.save(fig, os.path.join(HERE, "figures", "0027-walking-bank.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
