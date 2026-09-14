"""The split channel's Fisher information, along the ladder's OWN direction:
split varied at FIXED TOTAL Q+R (the null direction of the per-step Fisher).

Claim under test (sequence-demix 0002 / `_rung_odds`, graded AUDIT[derived+proxy]):
 "the per-step divergence between two gains is 0.5 dt^2 in the arclength
  t = arccos(1-K)" -- i.e. Fisher information exactly 1 per step in t.

Paired: the same record scores every rung, so the KL difference is common-random-number
paired and far less noisy than either log-loss alone.
"""
import math
import numpy as np

TOTAL = 1.0

def gain_from_q(q):
    return (-q + math.sqrt(q * q + 4.0 * q)) / 2.0

def q_from_t(t):
    K = 1.0 - math.cos(t)
    return K * K / (1.0 - K)

def simulate(q, T, seed):
    R = TOTAL / (1.0 + q)
    Q = TOTAL - R
    rng = np.random.default_rng(seed)
    th = np.cumsum(rng.normal(0.0, math.sqrt(Q), T))
    return th + rng.normal(0.0, math.sqrt(R), T)

def logloss(y, q):
    R = TOTAL / (1.0 + q)
    K = gain_from_q(q)
    S = R / (1.0 - K)
    m, tot = y[0], 0.0
    for t in range(1, y.size):
        e = y[t] - m
        tot += 0.5 * (math.log(2.0 * math.pi * S) + e * e / S)
        m = m + K * e
    return tot / (y.size - 1)

T, NSEED = 300_000, 6
print(f"total Q+R fixed at {TOTAL}; T={T}, {NSEED} paired seeds")
print(f"\n{'q0':>7s}{'K0':>8s}{'t0':>8s}{'dt':>7s}"
      f"{'KL(+)+KL(-)':>14s}{'/dt^2 = I_t':>13s}")
for q0 in (0.005, 0.02, 0.10, 0.50, 2.0):
    K0 = gain_from_q(q0)
    t0 = math.acos(1.0 - K0)
    ys = [simulate(q0, T, s) for s in range(NSEED)]
    base = np.mean([logloss(y, q0) for y in ys])
    out = []
    for dt in (0.06, 0.12):
        tp, tm = t0 + dt, t0 - dt
        if not (0.0 < tm and tp < math.pi / 2):
            continue
        klp = np.mean([logloss(y, q_from_t(tp)) for y in ys]) - base
        klm = np.mean([logloss(y, q_from_t(tm)) for y in ys]) - base
        I = (klp + klm) / (dt * dt)
        out.append((dt, klp + klm, I))
    for i, (dt, s, I) in enumerate(out):
        lbl = f"{q0:7.3f}{K0:8.4f}{t0:8.4f}" if i == 0 else " " * 23
        print(f"{lbl}{dt:7.2f}{s:14.6f}{I:13.3f}")
print("\nthe claim is I_t = 1.000 at every operating point.")
