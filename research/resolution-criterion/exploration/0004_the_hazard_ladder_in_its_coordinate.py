"""0004: the hazard ladder in its own coordinate.

A hazard rung is used by the filter as a DIFFUSION RATE: the offset walker (`_MeanChannel`) and the departure
walker (`_Departure`) each drift at q = rho * (class size)^2 per step, observed in the noise the column sits in
(the top class is by construction that noise, so at the top class the signal-to-noise ratio per step is rho).
A random walk of drift q observed in noise sigma^2 is, differenced, MA(1) with theta = 1 - K, K the steady gain:
    P-/sigma^2 = (rho + sqrt(rho^2 + 4 rho))/2,   K = P-/(P- + sigma^2),   t = arccos(1 - K)
and t is the arclength of the Whittle metric -- the same coordinate as the split ladder, with per-step Fisher 1.
Part 1 prints the shipped ladder in t; part 2 races the shipped ladder against the uniform-t ladder on the
dynamics-learning 0009 rig (both worlds, 20 seeds).  Run from the repo root: python 0004... [nseeds]"""
import math, sys, time, types
import numpy as np
sys.path.insert(0, ".")
SRC = "lucid/filter/lucid.py"
def K_of(rho):
    Pm = (rho + math.sqrt(rho * rho + 4 * rho)) / 2.0; return Pm / (Pm + 1.0)
def rho_of_t(t):
    K = 1 - math.cos(t); Pm = K / (1 - K); return Pm * Pm / (Pm + 1)
def ladder_t(mem=1000.0, c=1.5):
    step = c * math.sqrt(2.0 / mem); top = math.acos(1 - K_of(0.5)); J = int(math.ceil(top / step))
    tt = (np.arange(J) + 0.5) * top / J
    return tuple(sorted((rho_of_t(t) for t in tt), reverse=True))
def load(name, hazards=None):
    src = open(SRC).read()
    if hazards is not None:
        src = src.replace("_HAZARDS = tuple(0.5 * math.exp(-_HAZARD_GAP * j) for j in range(6))",
                          "_HAZARDS = %r" % (tuple(float(h) for h in hazards),))
    mod = types.ModuleType("lh_" + name); mod.__file__ = SRC; sys.modules[mod.__name__] = mod
    exec(compile(src, SRC, "exec"), mod.__dict__); return mod
# the 0009 rig, verbatim, parametrised by module
T, TSTAR, BURN = 2500, 1500, 300; A0, A1 = 0.90, 0.55; QS, RS = 0.3, 0.5
def series(seed, change=True):
    rng = np.random.default_rng(seed); x = 0.0; xs = np.empty(T); ys = np.empty((T, 2))
    for t in range(T):
        a = A1 if (change and t >= TSTAR) else A0
        x = a * x + QS * rng.standard_normal(); xs[t] = x; ys[t] = x + RS * np.sqrt(2.0) * rng.standard_normal(2)
    return xs, ys
def run(mod, seed, change=True):
    xs, ys = series(seed, change)
    f = mod.LucidFilter(dynamics=[[A0]], H=[[1.0], [1.0]], process=[[QS ** 2]], measurement=[2 * RS ** 2, 2 * RS ** 2],
                        faults=True, phis=(0.70, 0.95), ss=(0.30, 0.80))
    r = f.filter(ys); err = (r.mean[:, 0] - xs) ** 2
    out = dict(rmse_calm=np.sqrt(err[BURN:TSTAR].mean()), false=float((r.fault[BURN:TSTAR] > 0.5).mean()), hz_calm=float(r.hazard[TSTAR - 1]))
    if change:
        cross = np.flatnonzero(r.fault[TSTAR:] > 0.5); out["delay"] = float(cross[0]) if cross.size else np.inf
        out["rmse_rec"] = np.sqrt(err[TSTAR:TSTAR + 200].mean()); out["rmse_set"] = np.sqrt(err[TSTAR + 200:].mean()); out["hz_end"] = float(r.hazard[-1])
    return out
def run_recurrent(mod, seed, period=150, nev=8):
    rng = np.random.default_rng(seed); Tr = 400 + period * nev; x = 0.0; xs = np.empty(Tr); ys = np.empty((Tr, 2)); marks = []; a = A0
    for t in range(Tr):
        if t >= 400 and (t - 400) % period == 0:
            a = A1 if a == A0 else A0; marks.append(t)
        x = a * x + QS * rng.standard_normal(); xs[t] = x; ys[t] = x + RS * np.sqrt(2.0) * rng.standard_normal(2)
    f = mod.LucidFilter(dynamics=[[A0]], H=[[1.0], [1.0]], process=[[QS ** 2]], measurement=[2 * RS ** 2, 2 * RS ** 2],
                        faults=True, phis=(0.70, 0.95), ss=(0.30, 0.80))
    r = f.filter(ys); err = (r.mean[:, 0] - xs) ** 2; delays = []
    for tm in marks[::2]:
        w = r.fault[tm:tm + period]; c = np.flatnonzero(w > 0.5); delays.append(float(c[0]) if c.size else float(period))
    return dict(delay_late=float(np.mean(delays[2:])), delay_first=delays[0], rmse=np.sqrt(err[400:].mean()), hz_end=float(r.hazard[-1]))
def agg(rows, key):
    v = np.array([r[key] for r in rows if key in r]); v = v[np.isfinite(v)]
    return (v.mean(), v.std() / max(np.sqrt(len(v)), 1)) if v.size else (np.inf, 0.0)
# --- part 1: the shipped ladder, seen in t
ship = [0.5 * math.exp(-1.5 * j) for j in range(6)]; ts = [math.acos(1 - K_of(r)) for r in ship]
mem = 1000.0; blur = math.sqrt(2 / mem)
print("shipped rungs rho :", ["%.2e" % r for r in ship]); print("  gain K          :", ["%.4f" % K_of(r) for r in ship])
print("  arclength t     :", ["%.3f" % t for t in ts])
print("  gaps in blurs   :", ["%.1f" % ((ts[i] - ts[i + 1]) / blur) for i in range(5)], " bottom rung to rho=0: %.1f blurs" % (ts[-1] / blur))
for Kv in (0.05, 0.3, 0.7, 0.95):   # the Bernoulli identity: d arccos(1-K)/dK == d 2 asin sqrt(K/2)/dK
    h = Kv / 2; d = 1e-6
    print("  K=%.2f  dt/dK=%.5f  d[2 asin sqrt(K/2)]/dK=%.5f" % (Kv, (math.acos(1 - Kv - d) - math.acos(1 - Kv)) / d,
          (2 * math.asin(math.sqrt(h + d / 2)) - 2 * math.asin(math.sqrt(h))) / d))
ns = int(sys.argv[1]) if len(sys.argv) > 1 else 20
arms = [("shipped log-rho, 6", load("ship")), ("arclength t, 16", load("t16", ladder_t())),
        ("arclength t, c=3 (8)", load("t8", ladder_t(c=3.0)))]
print("derived ladder:", ["%.2e" % r for r in ladder_t()])
hdr = f"{'arm':>22} | {'delay':>12} | {'false%':>7} | {'calm rmse':>16} | {'recov':>7} | {'settled':>8} | {'hz(calm)':>9} | {'hz(end)':>9} | s"
print(hdr, flush=True)
for name, mod in arms:
    t0 = time.time()
    ch = [run(mod, 100 + s) for s in range(ns)]; nc = [run(mod, 900 + s, change=False) for s in range(ns)]
    d, dse = agg(ch, "delay"); fp = np.mean([r["false"] for r in ch] + [r["false"] for r in nc])
    cr, crse = agg(nc, "rmse_calm"); rr, _ = agg(ch, "rmse_rec"); sr, _ = agg(ch, "rmse_set"); hzc, _ = agg(nc, "hz_calm"); hze, _ = agg(ch, "hz_end")
    print(f"{name:>22} | {d:6.1f} ± {dse:4.1f} | {100 * fp:6.2f}% | {cr:.5f} ± {crse:.5f} | {rr:7.4f} | {sr:8.4f} | {hzc:9.3g} | {hze:9.3g} | {time.time() - t0:.0f}", flush=True)
print("fault-RICH world:")
for name, mod in arms:
    rows = [run_recurrent(mod, 500 + s_) for s_ in range(ns)]
    d0, d0se = agg(rows, "delay_first"); dl, dlse = agg(rows, "delay_late"); rm, _ = agg(rows, "rmse"); hz, _ = agg(rows, "hz_end")
    print(f"{name:>22} | first {d0:5.1f} ± {d0se:3.1f} | late {dl:5.1f} ± {dlse:3.1f} | rmse {rm:8.4f} | hz {hz:9.3g}", flush=True)
