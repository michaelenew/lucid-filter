"""Chain repro with v5 (IMM state mixing at the switching rate): J = 2 and 5, 3 seeds, vs v4 (no mixing)."""
import os, sys, types, math, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "research/multivariate-statfilter/scripts"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arm5dof as AR
import importlib; v5_patch = importlib.import_module('0061_state_mixing_patch')
src2 = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "0061_chain_repro.py")).read()
head = src2[:src2.index("ns = int(sys.argv[1])")]; ns = {}; exec(compile(head, "r2", "exec"), ns)
F, G, H, Q0, R0, T, sim, oracle, PH = ns["F"], ns["G"], ns["H"], ns["Q0"], ns["R0"], ns["T"], ns["sim"], ns["oracle"], ns["PH"]
def load5(J):
    src = v5_patch.patched_source(J); mod = types.ModuleType("lv5_%d" % J); mod.__file__ = "lucid/filter/lucid.py"; sys.modules[mod.__name__] = mod
    exec(compile(src, "lucid/filter/lucid.py", "exec"), mod.__dict__); return mod
for J in (2, 5):
    for tag, mod in (("v4 no mixing", ns["load"](J)), ("v5 IMM mixing", load5(J))):
        res = {nm: [] for nm, _, _ in PH}; orc = {nm: [] for nm, _, _ in PH}
        for seed in range(3):
            X, Y = sim(seed); o = oracle(Y); f = mod.LucidFilter(dynamics=F, H=H, process=Q0, measurement=R0); est = np.empty((T, 3))
            for t in range(T): est[t] = f.update(Y[t]).mean
            for nm, a, bb in PH:
                res[nm].append(np.sqrt(np.mean((est[a + 40:bb, 0] - X[a + 40:bb, 0]) ** 2))); orc[nm].append(np.sqrt(np.mean((o[a + 40:bb, 0] - X[a + 40:bb, 0]) ** 2)))
        print(f"J={J} {tag:>14}: " + "  ".join(f"{nm} {np.mean(res[nm]) / np.mean(orc[nm]):.3f}" for nm, _, _ in PH) + "   per-seed POTFAIL " + " ".join(f"{r/o_:.2f}" for r, o_ in zip(res['POTFAIL'], orc['POTFAIL'])), flush=True)
