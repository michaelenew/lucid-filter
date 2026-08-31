"""0009: the hazard is not a labeled prior -- ladder it and let the evidence weight it.

THE CHARGE (review, 2026-08-31).  A pinned hazard rho fails the monotonicity test that
separates a compute budget from a tuning constant: calm accuracy degrades monotonically in
rho (0006: 1/7000 -> 1.148, 1/50000 -> 1.066) while detection delay improves monotonically
(log(1/rho)/KL, 0001) -- two opposing monotone effects is a trade-off, and a parameter on a
trade-off is a knob.  Worse, a pinned rho has the user TELLING the filter the regime, where
everything else here reads the regime off the data (the (phi, s) box, the split ladder, the
offset classes).  The fix is the house rule for a nuisance: grid it, weight it by evidence.

THE PROBE.  The 0001 scalar rig (A 0.90 -> 0.55 at t* = 1500), read by TWO sensors of the
same state at 2x the single-sensor variance -- same total information, but no singly-read
mode, so the (orthogonal) split ladder stays out of the bank and the probe measures the
hazard machinery alone at 24x less compute.  20 seeds,
with a no-change arm for calm costs.  Contenders: the derived hazard LADDER
(faults=True: rungs at gap 1.5 nats from 1/2 down, see `_HAZARD_GAP`) against the
retired decade box and PINNED hazards
spanning it, plus a rung-count perturbation (one extra decade below) for the inertness claim.
Measured per arm: detection delay (first fault > 0.5 after t*, the 0001 reporting
convention), false-crossing fraction on [300, t*), state RMSE on calm/recovery/settled
windows, and the hazard READOUT (does the filter report the regime the data supports?).
"""
import sys, time
import numpy as np

sys.path.insert(0, "/home/user/lucid-filter")
from lucid.statfilter.lucid import LucidFilter

T, TSTAR, BURN = 2500, 1500, 300
A0, A1 = 0.90, 0.55
QS, RS = 0.3, 0.5     # sd of process / measurement noise


def series(seed, change=True):
    rng = np.random.default_rng(seed)
    x = 0.0
    xs = np.empty(T); ys = np.empty((T, 2))
    for t in range(T):
        a = A1 if (change and t >= TSTAR) else A0
        x = a * x + QS * rng.standard_normal()
        xs[t] = x
        ys[t] = x + RS * np.sqrt(2.0) * rng.standard_normal(2)
    return xs, ys


def run(faults, seed, change=True):
    xs, ys = series(seed, change)
    f = LucidFilter(dynamics=[[A0]], H=[[1.0], [1.0]], process=[[QS ** 2]],
                    measurement=[2 * RS ** 2, 2 * RS ** 2],
                    faults=faults, phis=(0.70, 0.95), ss=(0.30, 0.80))
    r = f.filter(ys)
    err = (r.mean[:, 0] - xs) ** 2
    out = dict(
        rmse_calm=np.sqrt(err[BURN:TSTAR].mean()),
        false=float((r.fault[BURN:TSTAR] > 0.5).mean()),
        hz_calm=float(r.hazard[TSTAR - 1]),
    )
    if change:
        cross = np.flatnonzero(r.fault[TSTAR:] > 0.5)
        out["delay"] = float(cross[0]) if cross.size else np.inf
        out["rmse_rec"] = np.sqrt(err[TSTAR:TSTAR + 200].mean())
        out["rmse_set"] = np.sqrt(err[TSTAR + 200:].mean())
        out["hz_end"] = float(r.hazard[-1])
    return out


def agg(rows, key):
    v = np.array([r[key] for r in rows if key in r])
    v = v[np.isfinite(v)]
    return (v.mean(), v.std() / max(np.sqrt(len(v)), 1)) if v.size else (np.inf, 0.0)


def run_recurrent(faults, seed, period=150, nev=8):
    """A fault-RICH world: A alternates 0.90 <-> 0.55 every `period` steps, `nev` events.

    The regime-reading claim in one rig: a ladder should climb its rungs here (the quiet
    between events is short), detect later events faster than its own calm operating point,
    and report a hazard near the actual event rate 1/period -- all without being retold.
    """
    rng = np.random.default_rng(seed)
    Tr = 400 + period * nev
    x = 0.0
    xs = np.empty(Tr); ys = np.empty((Tr, 2))
    marks = []
    a = A0
    for t in range(Tr):
        if t >= 400 and (t - 400) % period == 0:
            a = A1 if a == A0 else A0
            marks.append(t)
        x = a * x + QS * rng.standard_normal()
        xs[t] = x
        ys[t] = x + RS * np.sqrt(2.0) * rng.standard_normal(2)
    f = LucidFilter(dynamics=[[A0]], H=[[1.0], [1.0]], process=[[QS ** 2]],
                    measurement=[2 * RS ** 2, 2 * RS ** 2],
                    faults=faults, phis=(0.70, 0.95), ss=(0.30, 0.80))
    r = f.filter(ys)
    err = (r.mean[:, 0] - xs) ** 2
    delays = []
    for tm in marks[::2]:                       # the A0 -> A1 edges (leaving nominal)
        w = r.fault[tm:tm + period]
        c = np.flatnonzero(w > 0.5)
        delays.append(float(c[0]) if c.size else float(period))
    return dict(delay_late=float(np.mean(delays[2:])), delay_first=delays[0],
                rmse=np.sqrt(err[400:].mean()), hz_end=float(r.hazard[-1]))


def main(ns=20):
    import math as _m
    E15 = tuple(0.5 * _m.exp(-1.5 * j) for j in range(6))     # the shipped box (derived gap)
    arms = [
        ("box e^1.5 (faults=True)", True),
        ("box + rung below", E15 + (E15[-1] * _m.exp(-1.5),)),
        ("box @ decades (retired)", (0.5, 0.05, 5e-3, 5e-4)),
        ("pinned 0.5", 0.5),
        ("pinned 0.05", 0.05),
        ("pinned 5e-3", 5e-3),
        ("pinned 5e-4", 5e-4),
    ]
    print(f"scalar rig: A {A0} -> {A1} at t*={TSTAR}, T={T}, {ns} seeds; "
          f"no-change arm same seeds")
    hdr = f"{'arm':>22} | {'delay':>12} | {'false%':>7} | {'calm rmse':>16} | " \
          f"{'recov rmse':>10} | {'settled':>8} | {'hz(calm)':>9} | {'hz(end)':>9}"
    print(hdr); print("-" * len(hdr))
    for name, fa in arms:
        t0 = time.time()
        ch = [run(fa, 100 + s, change=True) for s in range(ns)]
        nc = [run(fa, 900 + s, change=False) for s in range(ns)]
        d, dse = agg(ch, "delay")
        fp = np.mean([r["false"] for r in ch] + [r["false"] for r in nc])
        cr, crse = agg(nc, "rmse_calm")
        rr, _ = agg(ch, "rmse_rec")
        sr, _ = agg(ch, "rmse_set")
        hzc, _ = agg(nc, "hz_calm")
        hze, _ = agg(ch, "hz_end")
        print(f"{name:>22} | {d:6.1f} ± {dse:4.1f} | {100 * fp:6.2f}% | "
              f"{cr:.5f} ± {crse:.5f} | {rr:10.4f} | {sr:8.4f} | {hzc:9.3g} | {hze:9.3g}"
              f"   [{time.time() - t0:.0f}s]")

    print()
    print(f"fault-RICH world (A alternates every 150 steps, 8 events), {ns} seeds:")
    hdr2 = f"{'arm':>22} | {'first delay':>11} | {'late delays':>11} | {'rmse':>8} | {'hz(end)':>9}"
    print(hdr2); print("-" * len(hdr2))
    for name, fa in [("box e^1.5 (faults=True)", True),
                     ("box @ decades (retired)", (0.5, 0.05, 5e-3, 5e-4)),
                     ("pinned 5e-4", 5e-4), ("pinned 5e-3", 5e-3)]:
        rows = [run_recurrent(fa, 500 + s_) for s_ in range(ns)]
        d0, d0se = agg(rows, "delay_first")
        dl, dlse = agg(rows, "delay_late")
        rm, _ = agg(rows, "rmse")
        hz, _ = agg(rows, "hz_end")
        print(f"{name:>22} | {d0:5.1f} ± {d0se:3.1f} | {dl:5.1f} ± {dlse:3.1f} | "
              f"{rm:8.4f} | {hz:9.3g}")


if __name__ == "__main__":
    main(ns=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
