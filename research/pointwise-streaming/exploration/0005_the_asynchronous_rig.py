"""0005 -- the acceptance rig: three sensors, three rates, no common schedule.

The workstream's definition of done.  Nothing about this rig is synchronous and nothing
about it is uniform: a 100 Hz rate gyro, a 5 Hz absolute fix, and a 12 Hz second
absolute with jitter, none of them phase-locked, one of them failing mid-run.  The
filter is driven ENTIRELY through the public point API -- ``observe(sensor, value, t=)``
in timestamp order -- and is told nothing about the rates, the schedule, or the failure.

Four contenders on the identical event stream:

    oracle      a Kalman filter told the true schedule AND the true noise at every
                instant -- the bound, not a contender
    lucid       LucidFilter.observe(), told the nominal model and nothing about noise
    fixed       the same model frozen at the base noise: isolates what the noise
                adaptation is worth, since it gets the asynchrony for free
    gridded     what the old API forced: bin onto the fast grid and drop any row that
                is not complete.  In an asynchronous stream almost none are.

PREDICTION (before the run): lucid sits near the oracle throughout and pulls away from
fixed exactly where the failing sensor is hot; gridded is not close to any of them,
because "drop the incomplete row" discards ~all of the stream when the rates are
coprime.  The failing sensor is identified by its own chip with no leak onto the others.

    python research/pointwise-streaming/exploration/0005_the_asynchronous_rig.py
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
from lucid import LucidFilter                                             # noqa: E402

OUT = os.path.join(HERE, "figures", "pw0005.json")

NOMINAL = 0.01                      # 100 Hz -- the fast sensor's rate, the model's unit
DURATION = 16.0
RATE_HZ = (5.0, 100.0, 12.0)        # sensor 0 absolute, 1 rate, 2 second absolute
SIGMA = np.array([0.25, 0.02, 0.40])
QV = 0.6                            # velocity diffusion, per sqrt(second)
FAIL = (2, 6.0, 11.0, 10.0)         # sensor 2 degrades x10 over [6 s, 11 s)


def schedule(seed):
    """Three sensors on their own clocks, out of phase, sensor 2 jittered."""
    r = np.random.default_rng(seed)
    ev = []
    for i, hz in enumerate(RATE_HZ):
        step = 1.0 / hz
        t = step * (0.37 * i)                            # out of phase on purpose
        while t < DURATION:
            ev.append((i, t))
            t += step * (1.0 + (0.35 * r.standard_normal() if i == 2 else 0.0))
            t = max(t, 1e-9)
    ev.sort(key=lambda p: p[1])
    return ev


def simulate(seed):
    """Truth is a continuous double integrator sampled at the event instants."""
    r = np.random.default_rng(seed + 1)
    ev = schedule(seed)
    x = np.zeros(2)
    prev = 0.0
    out = []
    for i, t in ev:
        d = t - prev
        prev = t
        # exact discretisation of dv = QV dW, dp = v dt
        x = np.array([x[0] + d * x[1], x[1]])
        if d > 0:
            sv = QV * math.sqrt(d)
            wv = sv * r.standard_normal()
            x = np.array([x[0] + 0.5 * d * wv, x[1] + wv])
        sd = SIGMA[i]
        j, t0, t1, fac = FAIL
        if i == j and t0 <= t < t1:
            sd *= fac
        val = (x[0] if i != 1 else x[1]) + sd * r.standard_normal()
        out.append((i, t, val, x.copy(), sd))
    return out


H = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])


def nominal_model():
    F = np.array([[1.0, NOMINAL], [0.0, 1.0]])
    Q = np.array([[QV ** 2 * NOMINAL ** 3 / 3.0, QV ** 2 * NOMINAL ** 2 / 2.0],
                  [QV ** 2 * NOMINAL ** 2 / 2.0, QV ** 2 * NOMINAL]])
    return F, Q


def kalman(stream, adaptive_sd=None):
    """A plain Kalman filter over the same event stream.  ``adaptive_sd`` None means the
    base noise (the ``fixed`` contender); a list means the true per-event sd (``oracle``)."""
    m = np.zeros(2); P = np.eye(2) * 10.0
    prev = 0.0
    est = np.empty((len(stream), 2))
    for k, (i, t, val, _truth, sd_true) in enumerate(stream):
        d = t - prev
        prev = t
        F = np.array([[1.0, d], [0.0, 1.0]])
        Q = np.array([[QV ** 2 * d ** 3 / 3.0, QV ** 2 * d ** 2 / 2.0],
                      [QV ** 2 * d ** 2 / 2.0, QV ** 2 * d]])
        m = F @ m
        P = F @ P @ F.T + Q
        h = H[i]
        sd = sd_true if adaptive_sd == "oracle" else SIGMA[i]
        S = float(h @ P @ h + sd ** 2)
        K = (P @ h) / S
        m = m + K * (val - float(h @ m))
        P = P - np.outer(K, h @ P)
        est[k] = m
    return est


def gridded(stream):
    """What the old API forced: bin onto the fast grid, drop any incomplete row."""
    T = int(DURATION / NOMINAL)
    Y = np.full((T, 3), np.nan)
    for i, t, val, _tr, _sd in stream:
        k = min(int(round(t / NOMINAL)), T - 1)
        Y[k, i] = val
    complete = np.isfinite(Y).all(1)
    Y[~complete] = np.nan
    F, Q = nominal_model()
    r = LucidFilter(dynamics=F, H=H, process=Q, measurement=SIGMA ** 2).filter(Y)
    # read the gridded estimate back at each event instant
    return r, complete.sum(), T


def run(seed):
    stream = simulate(seed)
    truth = np.array([s[3] for s in stream])
    F, Q = nominal_model()
    f = LucidFilter(dynamics=F, H=H, process=Q, measurement=SIGMA ** 2, timestep=NOMINAL)
    est = np.empty((len(stream), 2)); ms = np.empty((len(stream), 3))
    t0 = time.perf_counter()
    for k, (i, t, val, _tr, _sd) in enumerate(stream):
        st = f.observe(i, val, t=t)
        est[k] = st.mean
        ms[k] = st.measurement_scale
    wall = time.perf_counter() - t0
    orc = kalman(stream, adaptive_sd="oracle")
    fx = kalman(stream)
    g, kept, rows = gridded(stream)
    times = np.array([s[1] for s in stream])
    gk = np.clip((times / NOMINAL).round().astype(int), 0, rows - 1)
    gest = g.mean[gk]

    def rmse(e, lo=1.0, hi=DURATION):
        w = (times >= lo) & (times < hi)
        return float(np.sqrt(np.mean((e[w, 0] - truth[w, 0]) ** 2)))

    j, t0f, t1f, _ = FAIL
    calm = (times >= 1.0) & (times < t0f)
    hot = (times >= t0f + 1.0) & (times < t1f)
    return dict(
        events=len(stream), wall_ms_per_event=1e3 * wall / len(stream),
        gridded_rows_kept=int(kept), gridded_rows=int(rows),
        all_lucid=rmse(est), all_oracle=rmse(orc), all_fixed=rmse(fx),
        all_gridded=rmse(gest),
        calm_lucid=rmse(est, 1.0, t0f), calm_oracle=rmse(orc, 1.0, t0f),
        calm_fixed=rmse(fx, 1.0, t0f),
        hot_lucid=rmse(est, t0f + 1.0, t1f), hot_oracle=rmse(orc, t0f + 1.0, t1f),
        hot_fixed=rmse(fx, t0f + 1.0, t1f),
        chip_hot=float(np.mean(ms[hot, j]) - np.mean(ms[calm, j])),
        chip_leak=float(max(abs(np.mean(ms[hot, i2]) - np.mean(ms[calm, i2]))
                            for i2 in range(3) if i2 != j)))


if __name__ == "__main__":
    seeds = 5
    rows = [run(10 * s) for s in range(seeds)]

    def avg(k):
        return float(np.mean([r[k] for r in rows]))

    def se(k):
        return float(np.std([r[k] for r in rows]) / math.sqrt(seeds))

    print("=" * 78)
    print("THE ASYNCHRONOUS RIG -- 5 Hz absolute, 100 Hz rate, 12 Hz jittered absolute")
    print(f"   {avg('events'):.0f} events over {DURATION:.0f} s, no two sensors phase-locked;")
    print(f"   sensor {FAIL[0]} degrades x{FAIL[3]:.0f} over [{FAIL[1]:.0f}s, {FAIL[2]:.0f}s)")
    print(f"   driven entirely through observe(sensor, value, t=)")
    print("=" * 78)
    print(f"   the gridded baseline keeps {avg('gridded_rows_kept'):.0f}"
          f" of {avg('gridded_rows'):.0f} rows"
          f"  ({avg('gridded_rows_kept') / avg('gridded_rows'):.1%} of the fast grid)")
    print()
    print(f"   {'position RMSE (m)':<22} {'whole run':>12} {'calm':>12} {'sensor 2 hot':>14}")
    for tag in ("oracle", "lucid", "fixed"):
        print(f"   {tag:<22} {avg('all_' + tag):>12.4f} {avg('calm_' + tag):>12.4f}"
              f" {avg('hot_' + tag):>14.4f}")
    print(f"   {'gridded (old API)':<22} {avg('all_gridded'):>12.4f}")
    print()
    print(f"   {'ratio to oracle':<22} {'whole run':>12} {'calm':>12} {'sensor 2 hot':>14}")
    for tag in ("lucid", "fixed"):
        print(f"   {tag:<22} {avg('all_' + tag) / avg('all_oracle'):>12.3f}"
              f" {avg('calm_' + tag) / avg('calm_oracle'):>12.3f}"
              f" {avg('hot_' + tag) / avg('hot_oracle'):>14.3f}")
    print(f"   {'gridded (old API)':<22} {avg('all_gridded') / avg('all_oracle'):>12.3f}")
    print()
    print(f"   the failing sensor's own chip rises {avg('chip_hot'):+.2f} nats"
          f"  (truth {math.log(FAIL[3] ** 2):.2f});"
          f" worst leak onto a healthy one {avg('chip_leak'):+.2f}")
    print(f"   cost {avg('wall_ms_per_event'):.2f} ms per event in pure numpy"
          f"  (+/- {se('wall_ms_per_event'):.2f})")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(seeds=rows), open(OUT, "w"), indent=1)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")
