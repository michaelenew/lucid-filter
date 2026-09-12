"""0015 -- the diagonalised geometry (span 6, gap 1.5), resolved at high seed count.

0012-0014 established: gap = 1.5 is pinned by code length (t = 6.4 / 10.5 against
its factor-2 neighbours); span is a code-length plateau over 1.5-12; and within
that plateau span 6 is the jump-window RMSE preference (0.640 vs 0.694).  At 24
seeds span 6 scored t = -1.46 on total code length -- better in sign, unresolved.

This probe resolves the comparison before the constant is changed in the filter:
SHIPPED (span 3.0) against DIAGONALISED (span 6.0), both at gap 1.5, paired on
the same records at 96 seeds.  Everything reported as a paired difference:

  * total prequential code length, and each regime block's own (exact prefix
    differences: full run, run to t=600, run to t=380)
  * RMSE in the steady window, the jump window, the degraded-sensor window
  * calibration E[e^2/S] in the degraded window
  * jump settle steps

Also separates the REGRET question the RMSE/code-length gap raises: total code
length = irreducible entropy of the data + the filter's regret.  The entropy
floor is identical for both configurations and cancels in every paired
difference, but it is ~97% of the raw number, which is why a 19% RMSE change in
a 40-step window reads as a fraction of a percent of total code length.  The
paired t is floor-free; the percentage is not.  Both are reported.

Run: python3 0015_span6_analysis.py
"""
import math
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "lucid", "filter", "lucid.py")

SEED, N_T, JUMP_AT, JUMP, NOISE_AT = 11, 900, 380, 9.0, 600
Q_TRUE, S2_A, S2_C = 0.02, 1.0, 9.0
NSEED = 96
PHIS, SS = (0.70, 0.85, 0.95), (0.20, 0.40, 0.80, 1.60, 3.20)


def load_copy(name):
    mod = types.ModuleType(name)
    mod.__file__ = SRC
    sys.modules[name] = mod
    mod.__dict__["__name__"] = name
    exec(compile(open(SRC).read(), SRC, "exec"), mod.__dict__)
    return mod


def generate(seed):
    rng = np.random.default_rng(seed)
    th = np.cumsum(rng.normal(0.0, math.sqrt(Q_TRUE), N_T))
    th[JUMP_AT:] += JUMP
    sd = np.where(np.arange(N_T) < NOISE_AT, math.sqrt(S2_A), math.sqrt(S2_C))
    return th, th + rng.normal(0.0, sd)


def settle(err):
    seg = np.abs(err[JUMP_AT:JUMP_AT + 120]) < 1.0
    for i in range(seg.size):
        if seg[i:].all():
            return i
    return seg.size


def run_config(mod, span, theta, y):
    mod._SPAN_S = span
    f = mod.LucidFilter(phis=PHIS, ss=SS)
    r = f.filter(y.reshape(-1, 1))
    m = np.asarray(r.mean).reshape(-1)
    v = np.asarray(r.var).reshape(-1)
    L_all = float(np.asarray(r.loglik))
    L_600 = float(np.asarray(mod.LucidFilter(phis=PHIS, ss=SS)
                             .filter(y[:NOISE_AT].reshape(-1, 1)).loglik))
    L_380 = float(np.asarray(mod.LucidFilter(phis=PHIS, ss=SS)
                             .filter(y[:JUMP_AT].reshape(-1, 1)).loglik))
    e = m - theta
    out = {
        "cl_total": -L_all / N_T,
        "cl_A": -L_380 / JUMP_AT,
        "cl_B": -(L_600 - L_380) / (NOISE_AT - JUMP_AT),
        "cl_C": -(L_all - L_600) / (N_T - NOISE_AT),
        "mse_steady": float(np.mean(e[80:JUMP_AT] ** 2)),
        "mse_jump": float(np.mean(e[JUMP_AT:JUMP_AT + 40] ** 2)),
        "mse_C": float(np.mean(e[NOISE_AT + 40:] ** 2)),
        "calib": float(np.mean(e[NOISE_AT + 40:] ** 2 / v[NOISE_AT + 40:])),
        "settle": float(settle(e)),
    }
    return out


KEYS = ["cl_total", "cl_A", "cl_B", "cl_C",
        "mse_steady", "mse_jump", "mse_C", "calib", "settle"]

if __name__ == "__main__":
    print(__doc__.split("Run:")[0])
    mod = load_copy("lucid_span")
    A, B = [], []          # shipped, diagonalised
    for s in range(SEED, SEED + NSEED):
        theta, y = generate(s)
        A.append(run_config(mod, 3.0, theta, y))
        B.append(run_config(mod, 6.0, theta, y))
        if (s - SEED + 1) % 16 == 0:
            print(f"  ...{s - SEED + 1}/{NSEED} seeds")
    mod._SPAN_S = 3.0

    print(f"\npaired over {NSEED} seeds -- DIAGONALISED (span 6) minus SHIPPED (span 3)")
    print(f"{'metric':>12s}{'shipped':>12s}{'span 6':>12s}{'diff':>12s}"
          f"{'sem':>11s}{'t':>8s}{'diff %':>9s}")
    for k in KEYS:
        a = np.array([r[k] for r in A])
        b = np.array([r[k] for r in B])
        d = b - a
        sem = d.std(ddof=1) / math.sqrt(NSEED)
        t = d.mean() / sem if sem > 0 else 0.0
        pct = 100.0 * d.mean() / abs(a.mean()) if a.mean() != 0 else 0.0
        print(f"{k:>12s}{a.mean():12.5f}{b.mean():12.5f}{d.mean():+12.5f}"
              f"{sem:11.5f}{t:8.2f}{pct:+8.2f}%")
    print("\nnegative diff = span 6 better, on every row (all metrics lower-is-better")
    print("except none; calib target is 1.0 -- read its two means, not the diff).")
