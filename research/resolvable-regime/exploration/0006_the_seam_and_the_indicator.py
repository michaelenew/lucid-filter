"""0006 -- the one surviving idea, turned into a confidence read-out.

By this point C2 (0002) and C5 (0005) are falsified and 0003 has shown the
stationary class matches a correctly-specified wandering class even on a
wandering truth.  What survives is a single number: the Fisher information one
Gaussian observation carries about its own log-variance is exactly 1/2, so the
one-event blur width on a log-scale is **sqrt(2) nats**.

0005 found that number playing the opposite role to the one 0001 assigned it.
The two box members whose per-step motion exceeds sqrt(2) are not inert: removing
them costs +10.0% on the sensor-degradation window.  They are load-bearing
because a REGIME CHANGE IS A JUMP -- a sensor that goes 3x noisier moves its
log-scale by 2 ln 3 = 2.20 nats in one step, which is sub-sampling-interval
structure at every sampling rate.  So:

    sqrt(2) is not a ceiling on the class.  It is the SEAM inside it.
    Below it, a member models drift the record can resolve and track.
    Above it, a member models a step the record can only notice after the fact.
    The filter needs both, and the seam is where they divide.

That gives a read-out that costs nothing, because the numbers already exist:

    RESOLUTION WEIGHT  =  the bank's posterior weight on members above the seam.

Near zero: the filter is tracking drift it can resolve, and its attribution of
that drift is meant to be believed.  Near one: the filter is explaining the data
with steps it could not have seen coming, and its attribution is a guess -- the
honest reading is "something changed faster than I can resolve", not "sensor 3
degraded by exactly this much".

Two things measured here:

  1. Does the indicator move where it should?  On the README hero series -- calm,
     a level jump, a sensor 3x noisier -- it should sit low in the calm stretch
     and rise at both events.
  2. Does it mean what it claims?  VALIDATION BY DECIMATION (C6): filter the
     record at every sample and at every OTHER sample, and compare the two
     answers on the common grid.  Where the indicator is low the two rates should
     agree (the answer is a property of the world); where it is high they should
     not (the answer is a property of the schedule).  Correlation between the two
     is the test -- the indicator is only worth reporting if it predicts the
     disagreement.

Run: python3 0006_the_seam_and_the_indicator.py
"""
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from lucid import LucidFilter                                      # noqa: E402

PHIS, SS = (0.70, 0.85, 0.95), (0.20, 0.40, 0.80, 1.60, 3.20)
SEAM = math.sqrt(2.0)
N, JUMP_AT, JUMP, NOISE_AT = 900, 380, 9.0, 600
Q_TRUE, S2_A, S2_C = 0.02, 1.0, 9.0
NSEED = 10


def seam_mask(f):
    """Which of the bank's cells sit above the seam.

    Cells are enumerated `for ph in phis for sv in ss for base in bases`, so the
    (phi, s) of cell c follows from the base count.
    """
    nc = f._nc
    nb = nc // (len(PHIS) * len(SS))
    mask = np.zeros(nc, bool)
    for c in range(nc):
        isv = (c // nb) % len(SS)
        iphi = c // (nb * len(SS))
        inc = SS[isv] * math.sqrt(1.0 - PHIS[iphi] ** 2)
        mask[c] = inc > SEAM
    return mask


def run(y, dt=1.0):
    """Filter a series, returning the state track and the per-step resolution weight."""
    f = LucidFilter(phis=PHIS, ss=SS, timestep=1.0)
    mask = None
    est, rw = [], []
    for t in range(y.size):
        st = f.update(np.array([y[t]]), t=float(t) * dt)
        if mask is None:
            mask = seam_mask(f)
        w = np.exp(f._logw - f._logw.max())
        w = w / w.sum()
        est.append(float(np.asarray(st.mean).reshape(-1)[0]))
        rw.append(float(w[mask].sum()))
    return np.array(est), np.array(rw)


def generate(seed, s2c=S2_C, jump=JUMP):
    rng = np.random.default_rng(seed)
    theta = np.cumsum(rng.normal(0.0, math.sqrt(Q_TRUE), N))
    theta[JUMP_AT:] += jump
    sd = np.where(np.arange(N) < NOISE_AT, math.sqrt(S2_A), math.sqrt(s2c))
    return theta, theta + rng.normal(0.0, sd)


# ------------------------------------------------------------------ part 1
def part1():
    print("\n=== 1. does the indicator move where it should? ===")
    wins = [("calm        ", slice(80, JUMP_AT - 10)),
            ("level jump  ", slice(JUMP_AT, JUMP_AT + 40)),
            ("calm again  ", slice(JUMP_AT + 120, NOISE_AT - 10)),
            ("sensor x3   ", slice(NOISE_AT, NOISE_AT + 40)),
            ("settled loud", slice(NOISE_AT + 120, N))]
    rows = []
    for seed in range(11, 11 + NSEED):
        _, y = generate(seed)
        _, rw = run(y)
        rows.append([float(np.mean(rw[sl])) for _, sl in wins])
    a = np.array(rows)
    m, e = a.mean(0), a.std(0, ddof=1) / math.sqrt(NSEED)
    print(f"  resolution weight (bank mass above the seam), {NSEED} seeds:")
    for i, (nm, _) in enumerate(wins):
        bar = "#" * int(round(60 * m[i]))
        print(f"    {nm}  {m[i]:.4f} +- {e[i]:.4f}  {bar}")
    return m


# ------------------------------------------------------------------ part 2
def part2():
    print("\n=== 2. does it predict the rate-dependence? (C6, by decimation) ===")
    print("  filter every sample vs every OTHER sample; compare on the common grid.")
    print("  disagreement = RMS difference of the two state tracks, in units of the")
    print("  measurement SD of the segment.  Sweep the size of the sensor's jump,")
    print("  which is the thing that is sub-interval by construction.")
    print(f"\n  {'sensor jump':>12s}{'nats':>7s}{'res. weight':>13s}"
          f"{'disagreement':>15s}")
    pts = []
    for s2c in (1.0, 2.0, 4.0, 9.0, 25.0, 100.0):
        rws, digs = [], []
        for seed in range(11, 11 + 6):
            theta, y = generate(seed, s2c=s2c)
            est_f, rw = run(y)
            est_h, _ = run(y[::2], dt=2.0)
            sl = slice(NOISE_AT, N)
            # the coarse track lands on the even samples; compare there
            idx = np.arange(NOISE_AT, N, 2)
            hidx = idx // 2
            d = est_f[idx] - est_h[hidx]
            digs.append(float(np.sqrt(np.mean(d ** 2)) / math.sqrt(s2c)))
            rws.append(float(np.mean(rw[sl])))
        pts.append((s2c, math.log(s2c), np.mean(rws), np.mean(digs)))
        print(f"  {s2c:12.0f}{math.log(s2c):7.2f}{np.mean(rws):13.4f}"
              f"{np.mean(digs):15.4f}")
    a = np.array([[p[2], p[3]] for p in pts])
    if a[:, 0].std() > 0:
        r = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
        print(f"\n  correlation(resolution weight, disagreement) = {r:+.4f}")
        print("  the indicator is worth reporting only if this is strongly positive.")
    return pts


if __name__ == "__main__":
    print(__doc__.split("Run:")[0])
    part1()
    part2()
