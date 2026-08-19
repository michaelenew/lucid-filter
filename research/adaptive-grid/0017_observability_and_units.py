"""The 3-nat feature = observability saturation, and q_mu/settling in one unit.

Three linked questions from the step-size study:
  A. exactly where the recovery U-minimum sits (it looked near +3 nats);
  B. a natural common unit for q_mu and settling time -> a defensible q_mu rule.

The dynamics give both.  The tracker is a scalar Kalman filter on the window
centre mu; near a regime its settling time constant is

    tau ~ 1 / sqrt(q_mu * I),                      I = per-step Fisher info

(the dimensionless tracking index r = q_mu * I = q_mu/R sets it, R=1/I).  So
q_mu [nats^2], I [1/nats^2] and tau [steps] satisfy the invariant

    q_mu * I * tau^2  ~  const,

which is the shared unit: pick a settling budget tau*, and q_mu = 1/(I*tau*^2)
follows from the grid-readable I.  And the recovery of a jump to a destination d,

    t_rec ~ ln(|d|/thr) / sqrt(I(d)),

is minimised where sqrt(I(d)) stops growing fast enough to beat ln|d| -- i.e. at
the knee of the OBSERVABILITY curve I(d), which saturates (~1/2) once the
process dominates the measurement (SNR = e^d >> 1).  Prediction: the U-minimum
sits at that knee, ~2-3 nats.

Measured
--------
(a) I(d): per-step Fisher information at a centred regime, vs d -- the
    observability, showing saturation;
(b) recovery-min vs d, dense, with the location of the minimum, overlaid with
    the prediction ln(|d|/thr)/sqrt(I(d));
(c) the dimensional law: recovery time vs 1/sqrt(q_mu*I) at a fixed regime --
    a straight line through the origin verifies tau ~ 1/sqrt(q_mu*I).

Run: python 0017_observability_and_units.py   (heavy; ~3-4 min)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "random-walk-filter", "scripts"))

from gridlab import simulate  # noqa: E402
from moving_grid import MovingChannel  # noqa: E402
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

JT, NT, THRESH = 100, 1100, 0.5
QGRID = np.logspace(-5.0, -0.3, 10)


def observability(d, nseed=40, nt=700):
    """Mean per-step Fisher info I_t at a centred regime of loudness d."""
    acc = []
    for sd in range(nseed):
        rng = np.random.default_rng(sd)
        x = simulate(rng, d, 1.0, 1.0, nt)          # truth constant at d
        f = MovingChannel(1.0, 1.0, phi=0.9, s=0.30, order=5,
                          step="kalman_auto", q_mu=1e-6, P0=1.0)
        f.reset(mu=d)                                # centred
        fish = [f.update(v)["fisher"] for v in x]
        acc.append(np.mean(fish[nt // 4:]))
    return float(np.mean(acc))


def recovery(d, q_mu, nseed=40):
    lam = np.zeros(NT); lam[JT:] = d
    E = np.zeros((nseed, NT))
    for sd in range(nseed):
        rng = np.random.default_rng(700 + sd)
        st = rng.normal(0.0, np.sqrt(np.exp(lam)))
        x = np.cumsum(st) + rng.normal(0.0, 1.0, NT)
        f = MovingChannel(1.0, 1.0, phi=0.9, s=0.30, order=5,
                          step="kalman_auto", q_mu=q_mu, P0=25.0)
        f.reset(mu=0.0)
        E[sd] = [f.update(v)["logscale"] for v in x]
    r = np.sqrt(((E - lam) ** 2).mean(0))[JT:]
    hit = np.where(r < THRESH)[0]
    if hit.size == 0:
        return float(r.size)
    i = hit[0]
    return 0.0 if i == 0 else float((i - 1) + (r[i - 1] - THRESH) / (r[i - 1] - r[i]))


def main():
    dd = np.round(np.arange(-2.0, 6.01, 0.5), 2)
    Iof = np.array([observability(d) for d in dd])
    recmin = np.zeros(dd.size); qopt = np.zeros(dd.size)
    for k, d in enumerate(dd):
        rc = np.array([recovery(d, q) for q in QGRID])
        j = int(np.argmin(rc)); recmin[k] = rc[j]; qopt[k] = QGRID[j]

    kmin = int(np.argmin(recmin))
    # prediction ln(|d|/thr)/sqrt(I(d)) for d>0, scaled to the measured minimum
    pos = dd > 0
    pred = np.log(np.maximum(dd, 1e-9) / THRESH) / np.sqrt(np.maximum(Iof, 1e-6))
    scale = recmin[pos].min() / pred[pos].min()
    kpred = int(np.where(pos)[0][np.argmin(pred[pos])])
    print(f"[U-min] measured recovery minimum at d={dd[kmin]:+.2f} ({recmin[kmin]:.0f} steps)")
    print(f"[prediction] ln|d|/sqrt(I) minimum at d={dd[kpred]:+.2f}")
    print(f"[observability] I saturates: I(+1)={Iof[dd==1][0]:.3f}, "
          f"I(+3)={Iof[dd==3][0]:.3f}, I(+5)={Iof[dd==5][0]:.3f}")

    # (c) dimensional law at a fixed observable regime d0=+3
    d0 = 3.0
    I0 = float(Iof[dd == d0][0])
    qs = np.logspace(-4.5, -1.5, 9)
    trec = np.array([recovery(d0, q) for q in qs])
    xdim = 1.0 / np.sqrt(qs * I0)

    fig, ax = plt.subplots(1, 3, figsize=(16.2, 4.5))

    a = ts.tidy(ax[0])
    a.plot(dd, Iof, color=ts.SERIES[2], lw=2.0, marker="o", ms=3.5)
    a.axhline(0.5, color=ts.INK2, lw=1.0, ls=":", label="chi-square ceiling ½")
    a.axvline(dd[kmin], color=ts.SERIES[7], lw=1.1, ls="--", label=f"recovery min d={dd[kmin]:+.1f}")
    a.set_xlabel("destination d  (nats)"); a.set_ylabel("observability  I  (per-step Fisher info)")
    a.set_title("(a) observability saturates near the recovery-min")
    a.legend(loc="lower right", fontsize=8)

    a = ts.tidy(ax[1])
    a.plot(dd, recmin, color=ts.SERIES[1], lw=2.0, marker="o", ms=3.5, label="measured min recovery")
    a.plot(dd[pos], scale * pred[pos], color=ts.SERIES[5], lw=1.5, ls="--",
           label="prediction  ln(d/thr)/√I  (scaled)")
    a.scatter([dd[kmin]], [recmin[kmin]], facecolors="none", edgecolors=ts.SERIES[7],
              s=150, lw=2.0, zorder=5)
    a.set_xlabel("destination d  (nats)"); a.set_ylabel("min recovery time (steps)")
    a.set_title(f"(b) U-minimum at d≈{dd[kmin]:+.1f} nats")
    a.legend(loc="upper center", fontsize=8)

    a = ts.tidy(ax[2])
    a.plot(xdim, trec, color=ts.SERIES[3], lw=1.6, marker="o", ms=5)
    cf = np.polyfit(xdim, trec, 1)
    a.plot(xdim, np.polyval(cf, xdim), color=ts.INK2, lw=1.1, ls="--",
           label=f"linear: slope {cf[0]:.2f}")
    a.set_xlabel("1 / √(q_mu · I)   (settling-time scale, steps)")
    a.set_ylabel("recovery time at d=+3 (steps)")
    a.set_title("(c) tau ~ 1/√(q_mu·I): the shared unit")
    a.legend(loc="upper left", fontsize=8)
    ts.save(fig, os.path.join(HERE, "figures", "0016-observability-units.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
