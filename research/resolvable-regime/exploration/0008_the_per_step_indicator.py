"""0008 -- the indicator, rebuilt: per-step walk motion, not bank weight.

0006 built the confidence read-out out of the BANK's posterior weight above the
seam.  It failed both tests, and the reason is structural rather than a tuning
problem: the bank's weights are a running product of predictive likelihoods on
the `forget` memory (~1000 steps), so they integrate the record instead of
scoring the moment.  0006 part 1 shows the ratchet directly -- the weight goes
0.0001 in the calm stretch, 0.37 at the jump, and then STAYS at 0.41 and 0.52
through the calm that follows.  That is a statement about the record, not about
now, and it cannot predict a per-segment disagreement.

The per-step quantity tied to the same derived number is the WALK's own motion.
The filter reports each log-scale per step; the step it just took is
`|lam_t - lam_{t-1}|`, and the one-event blur width is sqrt(2) nats.  So:

    STEP RATIO  =  |lam_t - lam_{t-1}| / sqrt(2)

Below 1 the walk moved less than one observation can locate, so the record
carries the motion.  Above 1 the walk moved further than any single observation
could have justified -- the filter is extrapolating, and the honest reading is
"something changed faster than I can resolve".

Two tests, and the second one is the one that matters:

  1. Does it alarm rather than ratchet?  It must return to baseline after an
     event, which is what 0006's construction failed to do.

  2. Does it predict rate-dependence?  DECIMATION, with the data-volume effect
     controlled.  0006's decimation metric was contaminated: comparing the state
     tracks at Delta and 2Delta gave 0.242 disagreement even with NO regime
     change at all, because halving the data just makes the estimate worse.  Here
     the comparison is on the INFERRED LOG-SCALE path -- which is what
     resolvability is about -- normalised by that path's own spread, and swept
     over the speed of the log-scale rather than over the size of one jump.  A
     slow log-scale should give the same answer at both rates; a fast one should
     not.

Run: python3 0008_the_per_step_indicator.py
"""
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from lucid import LucidFilter                                      # noqa: E402

SEAM = math.sqrt(2.0)
N, JUMP_AT, JUMP, NOISE_AT = 900, 380, 9.0, 600
Q_TRUE, S2_A, S2_C = 0.02, 1.0, 9.0


def run(y, dt=1.0):
    """Returns the state track, the inferred measurement log-scale, and the step ratio."""
    f = LucidFilter(timestep=1.0)
    est, lam = [], []
    for t in range(y.size):
        st = f.update(np.array([y[t]]), t=float(t) * dt)
        est.append(float(np.asarray(st.mean).reshape(-1)[0]))
        lam.append(float(np.asarray(st.measurement_scale).reshape(-1)[0]))
    est, lam = np.array(est), np.array(lam)
    step = np.abs(np.diff(lam, prepend=lam[0])) / SEAM
    return est, lam, step


# ------------------------------------------------------------- test 1: alarm
def test1(nseed=10):
    print("\n=== 1. does it alarm, or does it ratchet? ===")
    wins = [("calm         ", slice(80, JUMP_AT - 10)),
            ("level jump   ", slice(JUMP_AT, JUMP_AT + 20)),
            ("calm again   ", slice(JUMP_AT + 120, NOISE_AT - 10)),
            ("sensor x3    ", slice(NOISE_AT, NOISE_AT + 20)),
            ("settled loud ", slice(NOISE_AT + 120, N))]
    rows = []
    for seed in range(11, 11 + nseed):
        rng = np.random.default_rng(seed)
        th = np.cumsum(rng.normal(0.0, math.sqrt(Q_TRUE), N))
        th[JUMP_AT:] += JUMP
        sd = np.where(np.arange(N) < NOISE_AT, math.sqrt(S2_A), math.sqrt(S2_C))
        y = th + rng.normal(0.0, sd)
        _, _, step = run(y)
        rows.append([float(np.max(step[sl])) for _, sl in wins])
    a = np.array(rows)
    m, e = a.mean(0), a.std(0, ddof=1) / math.sqrt(nseed)
    print(f"  peak step ratio in each window, {nseed} seeds "
          f"(1.0 = one blur width per step):")
    for i, (nm, _) in enumerate(wins):
        print(f"    {nm} {m[i]:6.3f} +- {e[i]:.3f}  "
              f"{'#' * int(round(30 * min(m[i], 2.0)))}")
    ratchet = m[2] / m[0]
    print(f"  return-to-baseline after the jump: calm-again / calm = {ratchet:.2f}"
          f"   (0006's bank-weight version was {0.4098/0.0001:.0f}x)")
    return m


# ------------------------------- test 2: decimation, data volume controlled
def sim_speed(sigma_lam, seed, n=1200):
    """A log-scale wandering at a chosen speed, sampled at every step."""
    rng = np.random.default_rng(seed)
    lam = np.cumsum(rng.normal(0.0, sigma_lam, n))
    lam -= lam.mean()
    th = np.cumsum(rng.normal(0.0, math.sqrt(Q_TRUE), n))
    return th, th + rng.normal(0.0, np.sqrt(S2_A * np.exp(lam))), lam


def test2(nseed=6):
    print("\n=== 2. does it predict rate-dependence? (decimation, volume controlled) ===")
    print("  compare the INFERRED log-scale path at Delta and at 2Delta on the common")
    print("  grid, normalised by that path's own spread.  Sweep the log-scale's speed.")
    print(f"\n  {'lam step SD':>12s}{'/sqrt(2)':>10s}{'step ratio':>12s}"
          f"{'disagree (nats)':>17s}{'/ true spread':>15s}")
    pts = []
    for sig in (0.01, 0.03, 0.10, 0.30, 0.80, 1.50, 3.00):
        srs, dis, raws = [], [], []
        for seed in range(21, 21 + nseed):
            th, y, lam = sim_speed(sig, seed)
            _, lf, stf = run(y)
            _, lh, _ = run(y[::2], dt=2.0)
            idx = np.arange(0, y.size, 2)
            hidx = idx // 2
            d = lf[idx] - lh[hidx]
            # normalise by the TRUE path's spread, not the inferred one.  The
            # inferred spread grows with the log-scale's speed, so dividing by it
            # cancels the very effect under test -- the mistake the first run of
            # this probe made, and the reason its correlation came out -0.65.
            spread = max(float(np.std(lam)), 1e-6)
            dis.append(float(np.sqrt(np.mean(d ** 2)) / spread))
            srs.append(float(np.mean(stf)))
            raws.append(float(np.sqrt(np.mean(d ** 2))))
        pts.append((sig, sig / SEAM, np.mean(srs), np.mean(dis), np.mean(raws)))
        print(f"  {sig:12.2f}{sig / SEAM:10.3f}{np.mean(srs):12.4f}"
              f"{np.mean(raws):17.4f}{np.mean(dis):15.4f}")
    a = np.array([[p[2], p[3]] for p in pts])
    r = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
    rs = float(np.corrcoef(np.argsort(np.argsort(a[:, 0])),
                           np.argsort(np.argsort(a[:, 1])))[0, 1])
    print(f"\n  correlation(step ratio, lam disagreement) = {r:+.4f}  "
          f"(rank {rs:+.4f})")
    print("  0006's bank-weight version scored -0.0758.")
    return pts


if __name__ == "__main__":
    print(__doc__.split("Run:")[0])
    test1()
    test2()
