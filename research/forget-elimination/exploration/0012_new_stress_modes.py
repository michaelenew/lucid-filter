"""0012 -- new stress modes for the battery, chosen to be unlike the rigs the work was tuned on.

Scalar local level  x_{t+1} = x_t + w,  y = x + v  with time-varying (q_t, r_t); the filter is
`LucidFilter()` with its defaults (it is told neither).  Arms: A default (forget = 0.999),
S forget=None.  References: the oracle Kalman filter (told q_t, r_t) and a fixed Kalman filter
(told the record's first-regime q, r).  Score: RMSE over the oracle's; the worst 200-step
block; log-likelihood per step.

Every (mode, seed, arm) run is checkpointed to `_cache/`.
Run from the repository root:  python research/forget-elimination/exploration/0012_new_stress_modes.py [mode ...]
"""
import math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from lucid.filter.lucid import LucidFilter
from rig import cached
TAG = "v4"
Q0 = 0.02
SEEDS = list(range(int(os.environ.get("NSEEDS", 8))))        # NSEEDS=24 to resolve a close mode

def make(mode, seed):
    """-> dict(y, x, q, r, dt, mis)  (q, r: the true per-step variances; dt: elapsed nominal steps)."""
    rng = np.random.default_rng(1000 + seed)
    if mode == "ramp" or mode == "gappy":
        T = 6000; t = np.arange(T)
        up = np.clip((t - 1000) / 2000, 0, 1); dn = np.clip((t - 3000) / 2000, 0, 1)
        sd = 8.0 ** (up - dn); q = np.full(T, Q0)
    elif mode.startswith("flicker"):
        D = int(mode.split()[1]); T = 4000
        sd = np.where((np.arange(T) // D) % 2 == 0, 1.0, 5.0); q = np.full(T, Q0)
    elif mode == "outliers":
        T = 4000; sd = np.where(rng.random(T) < 0.03, 15.0, 1.0); q = np.full(T, Q0)
    elif mode == "stale":
        T = 12000; sd = np.where(np.arange(T) < 10000, 1.0, 4.0); q = np.full(T, Q0)
    elif mode == "manoeuvre":
        T = 6000; q = np.where((np.arange(T) % 1500) >= 1300, 50 * Q0, Q0); sd = np.ones(T)
    elif mode == "coloured":
        T = 4000; q = np.full(T, Q0); sd = np.ones(T)
    else:
        raise ValueError(mode)
    dt = np.ones(T)
    if mode == "gappy":
        dt = rng.integers(1, 41, size=T).astype(float)
        cum = np.cumsum(dt); up = np.clip((cum - 20000) / 40000, 0, 1); dn = np.clip((cum - 60000) / 40000, 0, 1)
        sd = 8.0 ** (up - dn)
    x = np.cumsum(rng.normal(0, np.sqrt(q * dt)))
    if mode == "coloured":
        e = np.empty(T); e[0] = rng.normal(); rho = 0.8
        for k in range(1, T): e[k] = rho * e[k - 1] + math.sqrt(1 - rho * rho) * rng.normal()
        v = e
    else:
        v = rng.normal(0, sd)
    return dict(y=x + v, x=x, q=q, r=sd ** 2, dt=dt)

def kalman(y, q, r, dt):
    m, P = y[0], r[0]; out = np.empty(len(y)); out[0] = m; ll = 0.0
    for k in range(1, len(y)):
        P = P + q[k] * dt[k]; S = P + r[k]; K = P / S; e = y[k] - m
        ll += -0.5 * (math.log(2 * math.pi * S) + e * e / S)
        m = m + K * e; P = (1 - K) * P; out[k] = m
    return out, ll

def run_filter(d, kw):
    f = LucidFilter(**kw); T = len(d["y"]); est = np.empty(T); t0 = time.perf_counter()
    if np.all(d["dt"] == 1.0):
        est = np.asarray(f.filter(d["y"].reshape(-1, 1)).mean).reshape(-1)
    else:
        t = np.cumsum(d["dt"])
        for k in range(T): est[k] = f.update(np.array([d["y"][k]]), t=float(t[k])).mean[0]
    return dict(est=est, ll=f.loglik, dt=time.perf_counter() - t0, v=None if f._drift is None else f._drift.v)

MODES = ["ramp", "flicker 25", "flicker 100", "flicker 400", "outliers", "stale", "manoeuvre", "coloured", "gappy"]
want = [m for m in MODES if not sys.argv[1:] or m.split()[0] in sys.argv[1:] or m.replace(" ", "") in sys.argv[1:]]
BURN, BLK = 100, 200
def rm(e, sl=slice(BURN, None)): return float(np.sqrt(np.mean(e[sl] ** 2)))
print(f"{'mode':12s} {'fixed KF':>9s} {'A default':>10s} {'S None':>8s} {'S/A':>6s} {'t(S-A)':>7s}   {'worst blk A':>11s} {'worst blk S':>11s}   {'dloglik/step':>12s} {'t':>6s}   {'rate frac':>9s}  extra")
for mode in want:
    rows = []
    for seed in SEEDS:
        d = make(mode, seed)
        ko, _ = kalman(d["y"], d["q"], d["r"], d["dt"])
        kf, _ = kalman(d["y"], np.full_like(d["q"], d["q"][0]), np.full_like(d["r"], d["r"][0]), d["dt"])
        A = cached(f"0012{TAG}_{mode.replace(' ', '')}_{seed}_A", lambda: run_filter(d, {}))
        S = cached(f"0012{TAG}_{mode.replace(' ', '')}_{seed}_S", lambda: run_filter(d, dict(forget=None)))
        eo, ef, ea, es = ko - d["x"], kf - d["x"], A["est"] - d["x"], S["est"] - d["x"]
        T = len(eo); blocks = range(BURN, T - BLK + 1, BLK)
        wb = lambda e: max(rm(e, slice(b, b + BLK)) / rm(eo, slice(b, b + BLK)) for b in blocks)
        row = dict(fixed=rm(ef) / rm(eo), A=rm(ea) / rm(eo), S=rm(es) / rm(eo), wbA=wb(ea), wbS=wb(es),
                   dll=(S["ll"] - A["ll"]) / T, frac=math.exp(S["v"]), fin=bool(np.all(np.isfinite(es))))
        if mode == "stale":
            row["after"] = (rm(ea, slice(10000, 10300)) / rm(eo, slice(10000, 10300)), rm(es, slice(10000, 10300)) / rm(eo, slice(10000, 10300)))
            row["calm"] = (rm(ea, slice(2000, 10000)) / rm(eo, slice(2000, 10000)), rm(es, slice(2000, 10000)) / rm(eo, slice(2000, 10000)))
        rows.append(row)
    g = lambda k: np.array([r[k] for r in rows])
    dl = np.log(g("S")) - np.log(g("A")); t_r = dl.mean() / (dl.std(ddof=1) / math.sqrt(len(dl)))
    t_l = g("dll").mean() / (g("dll").std(ddof=1) / math.sqrt(len(rows)))
    extra = ""
    if mode == "stale":
        extra = (f"300 steps after the change A {np.mean([r['after'][0] for r in rows]):.3f} / S {np.mean([r['after'][1] for r in rows]):.3f};"
                 f" calm A {np.mean([r['calm'][0] for r in rows]):.4f} / S {np.mean([r['calm'][1] for r in rows]):.4f}")
    if not all(r["fin"] for r in rows): extra += "  NON-FINITE OUTPUT"
    print(f"{mode:12s} {g('fixed').mean():9.3f} {g('A').mean():10.3f} {g('S').mean():8.3f} {math.exp(dl.mean()):6.3f} {t_r:+7.2f}   "
          f"{g('wbA').max():11.2f} {g('wbS').max():11.2f}   {g('dll').mean():+12.5f} {t_l:+6.2f}   {np.median(g('frac')):9.3f}  {extra}", flush=True)
