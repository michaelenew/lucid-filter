"""The compact ladder for twice-read modes: a process eigenmode at its floor (base share < 1/2) read by
several sensors gets a complete ladder of SNR hypotheses x = q/r_eff, uniform in the gain arclength
t = arccos(1 - K) on [0, pi/2], as bank members (the caltrop star of `_split_star`, centre = the supplied
base); the mode's axis leaves the scale walk (walk_axes).  Sensors keep their walks.
    python probe_modeladder.py {budget|<per>} {arm|async|scalar} [seed]"""
import math, os, sys, time, types, importlib.util
import numpy as np
sys.path.insert(0, ".")
SRC = "lucid/filter/lucid.py"
PER, RIG = sys.argv[1], sys.argv[2]
HELPERS = '''

def _mode_groups(eng, I):
    """Twice-read modes at their floor: active, read by the sensors, not a singly-read pair, base share < 1/2."""
    single = {k for k, _, _ in eng._groups}
    g = np.sqrt(np.maximum(2.0 * (np.diag(I)), 0.0))
    out = []
    for k in range(eng.n):
        if k in single or not eng.active[k] or g[k] >= 0.5:
            continue
        hv = eng.HV[:, k]
        w = float(((hv * hv) / eng.rho).sum())
        if w <= 0.0:
            continue
        out.append((k, 1.0 / w))                      # r_eff: the noise the mode is read in, unit gain
    return out


def _mode_rungs(forget):
    """SNR rungs x = q/r_eff, uniform in the gain arclength on [0, pi/2] at the split ladder's spacing."""
    mem = min(1.0 / (1.0 - forget), _LADDER_MEM) if forget < 1.0 else _LADDER_MEM
    step = _GAP_FACTOR * math.sqrt(2.0 / mem)
    J = int(math.ceil((0.5 * math.pi) / step))
    t = (np.arange(J) + 0.5) * (0.5 * math.pi) / J
    K = 1.0 - np.cos(t)
    P = K / np.maximum(1.0 - K, 1e-12)
    return P * P / (P + 1.0)


def _mode_star(xs, n_modes, per_override=None):
    """Centre = the supplied base (None); arms = one mode moved to a rung, the `_split_star` budget."""
    if n_modes == 0:
        return [None]
    arms = max(len(xs) - 1, 1)
    per = max(arms // n_modes, 2) if per_override is None else per_override
    idx = np.unique(np.round(np.linspace(0, len(xs) - 1, per + 1)).astype(int))
    out = [None]
    for g in range(n_modes):
        for j in idx:
            out.append((g, float(xs[j])))
    return out
'''
def load(variant):
    src = open(SRC).read()
    if variant == "C":
        src = src.replace("\n\n# AUDIT[derived] exact null flow: dQ = -dR, the level set of the total (sequence-demix 0001).",
                          HELPERS + "\n\n# AUDIT[derived] exact null flow: dQ = -dR, the level set of the total (sequence-demix 0001).")
        old = "        bases = [_apply_split(probe, v) for v in self.split_arr]\n"
        per = "None" if PER == "budget" else PER
        new = old + '''        modes = _mode_groups(probe, _scale_fisher(probe, *probe._balanced_base(), probe._fisher_Si))
        self.mode_groups = modes
        mstar = _mode_star(_mode_rungs(self.forget), len(modes), %s)
        mb = []
        for mv in mstar:
            for (bq, br) in bases:
                if mv is None:
                    mb.append((bq, br)); continue
                g, x = mv; k, reff = modes[g]
                lam = np.linalg.eigvalsh(bq)[np.argsort(np.linalg.eigvalsh(bq))]  # placeholder, replaced below
                Vb, lb = probe.V, probe.lam.copy()
                lb[k] = x * reff
                mb.append((Vb @ np.diag(lb) @ Vb.T, br))
        bases = mb
        self._mode_walk = None
        if modes:
            self._mode_walk = np.ones(probe.D, bool)
            for k, _ in modes:
                self._mode_walk[k] = False
        self.mode_star = mstar
''' % per
        assert old in src; src = src.replace(old, new)
        old2 = "                    e = _WalkEngine(bq, br, Hm, Fs, Bs, ph, sv, fisher_Si=Si_c, prop=pr)\n"
        assert old2 in src
        src = src.replace(old2, "                    e = _WalkEngine(bq, br, Hm, Fs, Bs, ph, sv, fisher_Si=Si_c, prop=pr, walk_axes=self._mode_walk)\n")
    mod = types.ModuleType("lml_" + variant); mod.__file__ = SRC; sys.modules[mod.__name__] = mod
    exec(compile(src, SRC, "exec"), mod.__dict__); return mod
A = load("A"); C = load("C")
np.seterr(all="ignore")
if RIG == "scalar":
    N, JA, JU, NA = 900, 380, 9.0, 600
    for seed in (11, 12, 13):
        rng = np.random.default_rng(seed); th = np.cumsum(rng.normal(0, math.sqrt(0.02), N)); th[JA:] += JU
        sd = np.where(np.arange(N) < NA, 1.0, 3.0); y = th + rng.normal(0, sd)
        ma = np.asarray(A.LucidFilter().filter(y.reshape(-1, 1)).mean).reshape(-1)
        fc = C.LucidFilter(); mc = np.asarray(fc.filter(y.reshape(-1, 1)).mean).reshape(-1)
        print(f"scalar seed {seed}: mode groups {fc.mode_groups} star {len(fc.mode_star)}  max|A-C| = {np.abs(ma-mc).max():.3e}", flush=True)
elif RIG == "async":
    spec = importlib.util.spec_from_file_location("rig", "research/pointwise-streaming/exploration/0005_the_asynchronous_rig.py")
    rig = importlib.util.module_from_spec(spec); spec.loader.exec_module(rig)
    for nm, mod in (("A", A), ("C", C)):
        whole = []; hot = []
        for seed in (0, 10, 20, 30, 40):
            stream = rig.simulate(seed); truth = np.array([s[3] for s in stream]); times = np.array([s[1] for s in stream])
            F, Q = rig.nominal_model(); f = mod.LucidFilter(dynamics=F, H=rig.H, process=Q, measurement=rig.SIGMA ** 2, timestep=rig.NOMINAL)
            if nm == "C" and seed == 0: print("async mode groups", f.mode_groups, "star", len(f.mode_star), "members", len(f._members), flush=True)
            est = np.empty((len(stream), 2))
            for k, (i, t, val, _tr, _sd) in enumerate(stream): est[k] = f.observe(i, val, t=t).mean
            err = est[:, 0] - truth[:, 0]; j, t0, t1, fac = rig.FAIL
            hw = (times >= t0 + 1) & (times < t1); ww = (times >= 1) & (times < rig.DURATION)
            whole.append(np.sqrt(np.mean(err[ww] ** 2))); hot.append(np.sqrt(np.mean(err[hw] ** 2)))
        print(f"ASYNC {nm}: whole {np.mean(whole)/0.0345:.2f}x  hot {np.mean(hot)/0.0241:.2f}x  (worst hot {max(hot)/0.0241:.2f}x)", flush=True)
else:
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    sys.path.insert(0, "research/multivariate-statfilter/scripts"); import arm5dof as AR
    spec = importlib.util.spec_from_file_location("p54", "research/multivariate-statfilter/exploration/0054_physical_sensors.py")
    m54 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m54)
    jstd, pot, acc = m54.schedule(); U, S_, Y = AR.simulate(seed, jstd, pot, acc)
    orc = AR.kalman(U, Y, [j ** 2 * (AR.B @ AR.B.T) for j in jstd], [np.concatenate([[pot[k, j] ** 2, acc[k, j] ** 2, acc[k, j] ** 2] for j in range(AR.NJ)]) for k in range(m54.T)])
    Pt = m54.tip(S_); Po = m54.tip(orc); W = m54.windows()
    for nm, mod in (("A", A), ("C", C)):
        f = mod.LucidFilter(dynamics=AR.F, control=AR.B, H=AR.measure, process=AR.Q0, measurement=AR.R0)
        if nm == "C":
            print("arm mode groups:", [(k, "%.2e" % r) for k, r in f.mode_groups], "star", len(f.mode_star), "members", len(f._members),
                  "nodes/engine", f._members[0]._G, flush=True)
        t0 = time.perf_counter(); est = np.asarray(f.filter(Y, U).mean); dt = time.perf_counter() - t0
        if np.all(np.isfinite(est)):
            Pl = m54.tip(est)
            print(f"ARM seed{seed} {nm} ({1e3*dt/m54.T:.0f} ms/step): " + "  ".join(f"{k} {m54.rms(Pl,Pt,m54.mask(sp))/m54.rms(Po,Pt,m54.mask(sp)):.2f}" for k, sp in W.items()), flush=True)
        else:
            print(f"ARM seed{seed} {nm}: NaN", flush=True)
