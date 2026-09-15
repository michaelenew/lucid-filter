"""0005: the split ladder in its own coordinate -- (1) exact Whittle KL between adjacent rungs of the shipped
ladder, both directions; (2) AUD-5: code length vs rung spacing c on the scalar hero rig, paired 12 seeds.
Run from the repo root."""
import math, sys, time, types
import numpy as np
sys.path.insert(0, ".")
SRC = "lucid/filter/lucid.py"
def load(name, c=None):
    src = open(SRC).read()
    if c is not None:
        src = src.replace("    step = _GAP_FACTOR * math.sqrt(2.0 / mem)", "    step = %r * math.sqrt(2.0 / mem)" % c)
    mod = types.ModuleType("ls_" + name); mod.__file__ = SRC; sys.modules[mod.__name__] = mod
    exec(compile(src, SRC, "exec"), mod.__dict__); return mod
A = load("ship")
odds = A._rung_odds(0.999); K = (np.sqrt(odds * odds + 4 * odds) - odds) / 2.0   # invert odds = K^2/(1-K)
th = 1 - K; t = np.arccos(1 - K)
D = lambda a, b: 0.5 * (a - b) ** 2 / (1 - b * b)   # exact Whittle KL, MA(1) unit innovations, true theta=a vs alt theta=b
step = t[1] - t[0]; tgt = 0.5 * step * step
up = [D(th[i], th[i + 1]) for i in range(len(th) - 1)]     # lower-K rung true, next rung up the alternative
dn = [D(th[i + 1], th[i]) for i in range(len(th) - 1)]
print("shipped split ladder: %d rungs, t in [%.4f, %.4f], step %.5f, K %.5f..%.5f" % (len(K), t[0], t[-1], step, K[0], K[-1]))
print("target 0.5 step^2 = %.4e" % tgt)
print("KL/target, lower rung true:", " ".join("%.3f" % (v / tgt) for v in up))
print("KL/target, upper rung true:", " ".join("%.3f" % (v / tgt) for v in dn))
print("geometric mean / target  :", " ".join("%.3f" % (math.sqrt(a * b) / tgt) for a, b in zip(up, dn)))
print("same ladder, spacing in log-odds:", " ".join("%.2f" % v for v in np.diff(np.log(odds))))
# (2) AUD-5: code length and RMSE vs c on the scalar hero rig, paired 12 seeds
N, JA, JU, NA = 900, 380, 9.0, 600
mods = [(c, load("c%d" % int(100 * c), c)) for c in (3.0, 1.5, 1.0, 0.75)]
print("AUD-5 sweep: c, rungs, total loglik (mean over 12 seeds), jump/steady/C RMSE, ms/step")
for c, mod in mods:
    J = len(mod._rung_odds(0.999)); ll = []; rows = {k: [] for k in ("jump", "steady", "C")}; tt = 0.0
    for seed in range(11, 23):
        rng = np.random.default_rng(seed); th_ = np.cumsum(rng.normal(0, math.sqrt(0.02), N)); th_[JA:] += JU
        sd = np.where(np.arange(N) < NA, 1.0, 3.0); y = th_ + rng.normal(0, sd)
        t0 = time.perf_counter(); r = mod.LucidFilter().filter(y.reshape(-1, 1)); tt += time.perf_counter() - t0
        m = np.asarray(r.mean).reshape(-1); ll.append(float(r.loglik))
        for k, sl in (("jump", slice(JA, JA + 40)), ("steady", slice(80, JA)), ("C", slice(NA + 40, N))):
            rows[k].append(np.mean((m[sl] - th_[sl]) ** 2))
    print(f"  c={c:4.2f} rungs={J:3d} loglik {np.mean(ll):9.2f} ± {np.std(ll)/math.sqrt(12):5.2f}  "
          + "  ".join(f"{k} {math.sqrt(np.mean(v)):.4f}" for k, v in rows.items()) + f"  {1e3*tt/(12*N):.2f} ms/step", flush=True)
