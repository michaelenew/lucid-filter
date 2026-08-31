"""0004 -- what the sensor-bias cell can actually deliver, and what the gauge keeps.

0003 test 2 returned a negative worth taking seriously: on one level read by two sensors, the
channel recovers the relative bias almost exactly (0.966 against a truth of 1.0) and the state
estimate does NOT improve.  That is not a defect of the estimator.  0002 already said why --
the identifiable space is the RELATIVE bias and the COMMON MODE is gauge -- and the
consequence had not been drawn: knowing that sensor 2 reads b high tells you the two
disagree, not which one is right.  Any estimator, told the same data, splits the difference.

So the deliverable has to be stated per consumer, and this probe measures all three:

  STATE      -- the residual error should be exactly the gauge component of the bias, i.e.
                the mean bias under an orthonormal quotient basis (b/m for one biased sensor
                out of m).  Redundancy buys this down; the channel cannot.
  CALIBRATION-- 0001 measured the blind filter at 3.6x to 20x overconfident.  The channel
                should return it to ~1, because it explains the disagreement rather than
                absorbing it as noise.
  READ-OUT   -- "sensor 3 reads 1.96 high relative to the others" is the identifiable content,
                and it is the whole point of a lucid filter: the caller can recalibrate, drop
                the sensor, or route around it.  The blind filter cannot say it at all -- 0001
                measured eta_1 and eta_2 rising TOGETHER below 4 sigma, because a second-moment
                channel sees only e^2 and the sign pattern is what identifies a bias.

The sweep is over m, because redundancy is the lever that moves the gauge.

Run: python3 0004_redundancy_and_the_gauge.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))

_g = {}
exec(open(os.path.join(HERE, "0003_the_two_stage_channel.py")).read().split("def main()")[0], _g)
mean_basis, two_stage_kf = _g["mean_basis"], _g["two_stage_kf"]
augmented_kf, simulate = _g["augmented_kf"], _g["simulate"]

from lucid import LucidFilter                                    # noqa: E402

Q_TRUE, R_TRUE, T, T0, SEEDS = 0.02, 1.0, 900, 400, (11, 12, 13, 14)


def run(m, bias, lo=500):
    """One level, m direct sensors, the LAST one biased by `bias` from T0."""
    F, H = np.eye(1), np.ones((m, 1))
    Q, R = np.eye(1) * Q_TRUE, np.eye(m) * R_TRUE
    B = mean_basis(F, H)
    D, C = B[:1], B[1:]
    c_true = np.zeros(m)
    c_true[-1] = bias

    out = np.zeros(7)
    for seed in SEEDS:
        th, Y = simulate(F, H, Q, R, np.zeros(1), c_true, T=T, seed=seed, t0=T0)
        xb, Pb, _ = augmented_kf(Y, F, H, Q, R, np.zeros((1, 0)), np.zeros((m, 0)))
        xc, Pc, _, bh = two_stage_kf(Y, F, H, Q, R, D, C, qb=1e-4)
        step = c_true * (np.arange(T) >= T0)[:, None]
        xt, Pt, _ = augmented_kf(Y - step, F, H, Q, R, np.zeros((1, 0)), np.zeros((m, 0)))
        eb, ec, et = (xb[lo:, 0] - th[lo:, 0], xc[lo:, 0] - th[lo:, 0], xt[lo:, 0] - th[lo:, 0])
        # what the channel says each sensor reads, relative to the others
        c_hat = C @ bh[-1]
        c_hat = c_hat - c_hat.mean()                 # the quotient's own convention
        out += np.array([
            np.sqrt(np.mean(eb ** 2)), np.mean(eb ** 2 / Pb[lo:, 0, 0]),
            np.sqrt(np.mean(ec ** 2)), np.mean(ec ** 2 / Pc[lo:, 0, 0]),
            np.sqrt(np.mean(et ** 2)),
            c_hat[-1] - c_hat[:-1].mean(),           # the read-out: last sensor vs the rest
            np.mean(ec),                             # residual state bias
        ]) / len(SEEDS)
    return out


def lucid_scales(m, bias):
    """What the SHIPPED filter says about the same data -- the second-moment read-out."""
    F, H = np.eye(1), np.ones((m, 1))
    Q, R = np.eye(1) * Q_TRUE, np.eye(m) * R_TRUE
    c_true = np.zeros(m)
    c_true[-1] = bias
    eta = np.zeros(m)
    for seed in SEEDS:
        _, Y = simulate(F, H, Q, R, np.zeros(1), c_true, T=T, seed=seed, t0=T0)
        r = LucidFilter(H=H, measurement=np.full(m, R_TRUE)).filter(Y)
        eta += r.measurement_scale[T0 + 100:].mean(axis=0) / len(SEEDS)
    return eta


def main():
    print("=" * 90)
    print("One level, m direct sensors, the LAST one biased by 2.0 sigma from t = 400")
    print("=" * 90)
    print(f"{'m':>2} | {'blind':>16} | {'mean channel':>16} | {'told the bias':>8} | "
          f"{'read-out':>16} | gauge")
    print(f"{'':>2} | {'rmse    calib':>16} | {'rmse    calib':>16} | {'rmse':>8} | "
          f"{'last vs rest':>16} |")
    for m in (2, 3, 5, 8):
        o = run(m, 2.0)
        print(f"{m:2d} | {o[0]:7.3f} {o[1]:8.2f} | {o[2]:7.3f} {o[3]:8.2f} | {o[4]:8.3f} | "
              f"{o[5]:8.3f} (2.000) | {2.0/m:.3f}")

    print()
    print("The residual state error IS the gauge component -- one biased sensor out of m")
    print("displaces the consensus by b/m, and no estimator can know it was that sensor:")
    print(f"{'m':>2} | {'residual state bias':>19} | {'b/m predicted':>13}")
    for m in (2, 3, 5, 8):
        o = run(m, 2.0)
        print(f"{m:2d} | {o[6]:19.3f} | {2.0/m:13.3f}")

    print()
    print("=" * 90)
    print("What the SHIPPED second-moment channel says about the same data (eta per sensor)")
    print("=" * 90)
    for m in (2, 3, 5):
        eta = lucid_scales(m, 2.0)
        print(f"  m = {m}: " + "  ".join(f"{e:+.2f}" for e in eta)
              + "   <- the biased sensor is the last one")


if __name__ == "__main__":
    main()
