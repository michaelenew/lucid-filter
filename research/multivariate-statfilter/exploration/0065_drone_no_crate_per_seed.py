"""The drone's no-crate control (dynamics-learning 0008), per seed: RMSE / oracle, worst position error and where,
per-phase ratios, patience by phase.   python 0065_drone_no_crate_per_seed.py <lucid.py source|tree>"""
import sys, types, importlib.util, os, time
import numpy as np
sys.path.insert(0, ".")
src = sys.argv[1]
if src != "tree":
    import lucid.filter, lucid as _top
    mod = types.ModuleType("lucid.filter.lucid"); mod.__file__ = "lucid/filter/lucid.py"; mod.__package__ = "lucid.filter"
    sys.modules["lucid.filter.lucid"] = mod
    exec(compile(open(src).read(), "lucid/filter/lucid.py", "exec"), mod.__dict__)
    lucid.filter.lucid = mod
    for name in ("LucidFilter", "LucidResult", "LucidStep"):
        if hasattr(mod, name): setattr(lucid.filter, name, getattr(mod, name)); setattr(_top, name, getattr(mod, name))
rig = "research/dynamics-learning/exploration/0008_drone3d_payload.py"
sys.path.insert(0, os.path.dirname(rig)); sys.path.insert(0, "research/dynamics-learning/scripts")
spec = importlib.util.spec_from_file_location("p08", rig); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
R = m.R
print("source", os.path.basename(src), "| memory ladder:", hasattr(sys.modules["lucid.filter.lucid"], "_memory_rungs"), flush=True)
POS = np.arange(0, 3)
def phase_of(k):
    for nm, a, b in R.PHASES:
        if a <= k < b: return f"{nm}@{k-a}"
    return "?"
for sd in range(m.NS_CALM):
    t0 = time.time()
    d = m.run(sd, False)
    est, X, orc = d["r"].mean, d["X"], d["oracle"]
    sl = np.zeros(R.T, bool); sl[200:] = True
    ratio = m.rmse(est, X, sl, POS) / m.rmse(orc, X, sl, POS)
    err = np.linalg.norm(est[:, POS] - X[:, POS], axis=1); eo = np.linalg.norm(orc[:, POS] - X[:, POS], axis=1); k = int(np.argmax(err[200:])) + 200
    per = "  ".join(f"{nm} {m.rmse(est, X, m.mask_of([(a + 120, b)]), POS) / m.rmse(orc, X, m.mask_of([(a + 120, b)]), POS):.2f}" for nm, a, b in R.PHASES if b - a > 200)
    flagged = float(np.mean(d["r"].fault[200:] > 0.5))
    mem = getattr(d["r"], "memory", None); pat = getattr(d["r"], "patience", None)
    if pat is not None:
        pw = "  ".join(f"{nm}@{np.nanmean(pat[a + 120:b]):.2f}" for nm, a, b in R.PHASES if b - a > 200)
        ms = np.asarray(d["r"].measurement_scale); ps = np.asarray(d["r"].process_scale)
        mp = [(a, b) for nm, a, b in R.PHASES if nm == "MULTIPATH"][0]
        print(f"   patience by phase: {pw} | in MULTIPATH: sensor log-scales (mean over channels 0-2 = GPS pos?) {ms[mp[0]+120:mp[1], :3].mean(0).round(2)} others {ms[mp[0]+120:mp[1], 3:].mean().round(2)} | process log-scale mean {ps[mp[0]+120:mp[1]].mean():.2f}", flush=True)
    print(f"seed {sd}: RMSE/oracle {ratio:.3f} | worst {err[k]:.3f} m at {k} ({phase_of(k)}), oracle there {eo[k]:.3f} | flagged {100*flagged:.2f}% | {per}"
          + (f" | memory median {np.nanmedian(mem):.0f}" if mem is not None else "") + f" | {time.time()-t0:.0f}s", flush=True)
