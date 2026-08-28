"""Probe 0041 -- is the confound-confirmation bound really a fixed ~1/beta, or an EMA artifact?

Every reach mechanism (0038 sigma-point, 0039 adaptive gain, 0040 KDE) trips the SAME wall: you may
not inflate R until C1 confirms the surprise is WHITE (a sensor failure), not CORRELATED (a process
disturbance) -- Prop 1.  We have been reading that confirmation off a fixed-rate EMA (beta=0.02), so
the misfire window at a process onset is ~1/beta ~ 50 steps, and 0039's reach went net-negative
because the inflated gain opens on a process onset for those ~50 steps before C1 catches up.

But detection theory says distinguishing white from a lag-1 correlation rho should take ~1/rho^2
samples -- FAST for a strong disturbance (large rho), slow only for a subtle one.  A fixed-beta EMA
cannot adapt: it always settles at rate beta regardless of how obvious the correlation is.  So the
~1/beta wall may be an ESTIMATOR artifact, not fundamental.  Test: the onset detection delay of a
fixed-beta EMA vs a CUSUM (Page's test -- optimal sequential change detection) on the two cases the
filter must tell apart:
  * PROCESS onset: innovations become AR(1)-correlated (rho > 0).  Detector should fire "correlated"
    fast -> the misfire window (steps still calling it white -> reach opens) must be small.
  * SENSOR onset: innovations stay white (rho = 0) but the variance jumps.  Detector should keep
    saying "white" -> the reach opens promptly and stays open.
"""
import math

import numpy as np

_BETA = 0.02
THR = 2.0 * math.sqrt(_BETA)          # EMA 2-sigma whiteness threshold (production)
T_PRE = 400                            # white run-in before the onset
T_POST = 300
_RHOS = (0.2, 0.35, 0.5, 0.7, 0.9)    # process persistence at onset (detectability)


def gen_stream(rng, rho, var_post=1.0):
    """White for T_PRE, then AR(1)(rho) with stationary variance var_post for T_POST.
    rho=0 with var_post>1 is a SENSOR burst (white, bigger); rho>0 is a PROCESS burst (correlated)."""
    n = T_PRE + T_POST
    e = rng.standard_normal(n)         # white run-in, unit variance
    if rho == 0.0:
        e[T_PRE:] = math.sqrt(var_post) * rng.standard_normal(T_POST)
    else:
        sig_i = math.sqrt(var_post * (1 - rho * rho))   # innovation std so stationary var = var_post
        x = e[T_PRE - 1]
        for t in range(T_PRE, n):
            x = rho * x + sig_i * rng.standard_normal(); e[t] = x
    return e


def ema_delay(e):
    """Fixed-beta EMA of rho1 = C1/C0; first post-onset step it crosses THR (declares correlated)."""
    C0 = 1.0; C1 = 0.0; prev = e[0]
    for t in range(1, len(e)):
        C0 = (1 - _BETA) * C0 + _BETA * e[t] * e[t]
        C1 = (1 - _BETA) * C1 + _BETA * e[t] * prev
        prev = e[t]
        if t >= T_PRE and C1 / (C0 + 1e-12) > THR:
            return t - T_PRE
    return T_POST


def cusum_delay(e, rho_design=0.4, h=6.0):
    """Page's CUSUM on the per-sample log-LR of AR(1)(rho_design) vs white.
    increment = rho*(e_t e_{t-1})/C0 - rho^2/2 * (e_{t-1}^2/C0)  (positive drift under correlated).
    Reset to 0 (one-sided); declare when S > h.  h sets the false-alarm rate; measured below."""
    C0 = 1.0; S = 0.0; prev = e[0]
    for t in range(1, len(e)):
        C0 = (1 - _BETA) * C0 + _BETA * e[t] * e[t]            # scale normaliser (same info as EMA)
        inc = rho_design * (e[t] * prev) / (C0 + 1e-12) - 0.5 * rho_design * rho_design * (prev * prev) / (C0 + 1e-12)
        prev = e[t]
        S = max(0.0, S + inc)
        if t >= T_PRE and S > h:
            return t - T_PRE
        if t == T_PRE - 1:
            S = 0.0                                            # reset at the (known-onset) boundary? no -- see note
    return T_POST


def cusum_delay_noreset(e, rho_design=0.4, h=6.0):
    """Same CUSUM but WITHOUT the boundary reset -- honest online detector that does not know the
    onset time.  This is the fair comparison to the EMA (which also does not know the onset)."""
    C0 = 1.0; S = 0.0; prev = e[0]
    for t in range(1, len(e)):
        C0 = (1 - _BETA) * C0 + _BETA * e[t] * e[t]
        inc = rho_design * (e[t] * prev) / (C0 + 1e-12) - 0.5 * rho_design * rho_design * (prev * prev) / (C0 + 1e-12)
        prev = e[t]
        S = max(0.0, S + inc)
        if t >= T_PRE and S > h:
            return t - T_PRE
    return T_POST


def measure_false_alarm(detector, nseed=200, **kw):
    """Fraction of pure-white streams that (wrongly) declare correlated within T_POST of the boundary."""
    fa = 0
    for s in range(nseed):
        rng = np.random.default_rng(10000 + s)
        e = gen_stream(rng, 0.0, var_post=1.0)                 # pure white throughout
        if detector(e, **kw) < T_POST:
            fa += 1
    return fa / nseed


def main(nseed=200):
    print(f"confound-confirmation delay (steps after onset), beta={_BETA}, EMA thr={THR:.3f}")
    # calibrate CUSUM h to ~ the EMA's white false-alarm rate for a fair race
    fa_ema = measure_false_alarm(cusum_delay_noreset, nseed=nseed, rho_design=0.4, h=0.0)  # placeholder
    fa_ema = None
    print(f"\n  process onset (correlated) -- want SMALL delay (misfire window):")
    print(f"  {'rho':>5} {'EMA':>8} {'CUSUM h=6':>11} {'CUSUM h=10':>11}")
    for rho in _RHOS:
        de = []; dc6 = []; dc10 = []
        for s in range(nseed):
            rng = np.random.default_rng(s)
            e = gen_stream(rng, rho, var_post=1.0)
            de.append(ema_delay(e)); dc6.append(cusum_delay_noreset(e, 0.4, 6.0)); dc10.append(cusum_delay_noreset(e, 0.4, 10.0))
        print(f"  {rho:5.2f} {np.mean(de):8.1f} {np.mean(dc6):11.1f} {np.mean(dc10):11.1f}")
    # false-alarm rates on pure white (the cost of the speed)
    print(f"\n  false alarm on pure white (declares correlated within {T_POST} steps):")
    fa_e = measure_false_alarm(lambda e: ema_delay(e), nseed=nseed)
    fa_c6 = measure_false_alarm(cusum_delay_noreset, nseed=nseed, rho_design=0.4, h=6.0)
    fa_c10 = measure_false_alarm(cusum_delay_noreset, nseed=nseed, rho_design=0.4, h=10.0)
    print(f"    EMA={fa_e:.3f}   CUSUM h=6={fa_c6:.3f}   CUSUM h=10={fa_c10:.3f}")


if __name__ == "__main__":
    main()


def _delay_at(detector, rho, nseed, **kw):
    d = []
    for s in range(nseed):
        rng = np.random.default_rng(s)
        d.append(detector(gen_stream(rng, rho, var_post=1.0), **kw))
    return float(np.mean(d))


def frontier(nseed=300, rho=0.5):
    """Trace each detector's (false-alarm rate, mean detection delay) frontier at a representative
    process rho.  The lower-left frontier is the better detector.  If they coincide, the fixed-beta
    EMA is already near the optimal (CUSUM) detection frontier -> the confound bound is real."""
    print(f"\nspeed / false-alarm FRONTIER at process rho={rho} ({nseed} seeds)")
    print(f"  {'EMA thr':>9} {'FA':>6} {'delay':>7}     {'CUSUM h':>8} {'FA':>6} {'delay':>7}")
    ema_thrs = [0.20, 0.25, 0.283, 0.35, 0.45, 0.60]
    cus_hs = [3.0, 4.5, 6.0, 8.0, 11.0, 15.0]
    for et, ch in zip(ema_thrs, cus_hs):
        fa_e = measure_false_alarm(lambda e: ema_delay_thr(e, et), nseed=nseed)
        de = _delay_at(lambda e, **k: ema_delay_thr(e, et), rho, nseed)
        fa_c = measure_false_alarm(cusum_delay_noreset, nseed=nseed, rho_design=0.4, h=ch)
        dc = _delay_at(cusum_delay_noreset, rho, nseed, rho_design=0.4, h=ch)
        print(f"  {et:9.3f} {fa_e:6.3f} {de:7.1f}     {ch:8.1f} {fa_c:6.3f} {dc:7.1f}")


def ema_delay_thr(e, thr):
    C0 = 1.0; C1 = 0.0; prev = e[0]
    for t in range(1, len(e)):
        C0 = (1 - _BETA) * C0 + _BETA * e[t] * e[t]
        C1 = (1 - _BETA) * C1 + _BETA * e[t] * prev
        prev = e[t]
        if t >= T_PRE and C1 / (C0 + 1e-12) > thr:
            return t - T_PRE
    return T_POST
