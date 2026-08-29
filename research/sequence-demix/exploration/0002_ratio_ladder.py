"""Probe 0002 -- race the candidates on the hero rig, told nothing.

0001 established the geometry: in one channel the scale-Fisher is EXACTLY rank 1, its null
direction is the split at fixed total, and the walk's per-step score carries no information along
it.  So the split has to be carried by something that is not a per-step score.

Candidate A (`LadderFilter`) -- **bank the split, walk the total**.  The ladder is a BANK
dimension, not a statistic: the filter's members are crossed with a ladder of splits, so every
rung is a complete anchored filter of its own and the existing bank machinery carries it.

  * A GROUP is a process eigenmode read by EXACTLY ONE sensor.  Then `dS_xi` and `dS_eta` are
    proportional as matrices, the 2x2 scale-Fisher block is exactly rank 1, and only the SUM of
    the two contributions to that channel's `S` is identifiable per step -- Proposition 1 in
    coordinates.  Both members must carry information: an axis whose own scale-Fisher is
    numerically zero is not half a confound, it is nothing.  On the 5-DOF arm no group qualifies
    (every jerk mode is read by its pot as well as its accelerometer), so the ladder never
    switches on there: gate 2 by construction, at zero cost.
  * RUNGS are placed by their consequence, not by an offset from the supplied base.  All a split
    does is set the filter's gain; the per-step divergence between two gains is `0.5 dt^2` in the
    arclength `t = arccos(1 - K)` (MA(1) Whittle, see `_rung_gains`), and `t` runs over the
    BOUNDED interval `[0, pi/2]`.  A grid spaced at `1.5 sqrt(2 (1 - forget))` -- the engine's
    Sparrow factor on the resolution the bank's own memory can support -- therefore covers every
    possible split, completely, with ~24 rungs and no span constant.  Told nothing means told
    nothing: no rung refers to the supplied base.
  * Each rung is a full `_WalkEngine`: it runs its OWN filter, so a rung with too much process
    chases sensor noise and pays for it in its own predictive likelihood -- the per-node MEANS
    that de-mix (0053 section 1), with no EMA and no whiteness statistic.  Rung weights are bank
    weights, so they accumulate on the `forget` timescale (0053 lesson b), and a rung is an
    absolute hypothesis that never moves (lesson a).
  * The walk inside a rung still takes its per-axis step, whose null component is not information.
    It is allowed as a TRANSIENT -- it is what absorbs a level jump -- but reverts to the rung's
    anchor at the class's own rate `phi`, at fixed total.  Nothing accumulates that the data
    cannot support (lesson c: no member can wander off its hypothesis).

    python 0002_ratio_ladder.py [nodrift|nolimit]
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

from lucid import LucidFilter                                             # noqa: E402
from lucid.statfilter.lucid import _WalkEngine, _logsumexp, _rung_odds     # noqa: E402

# the box these control experiments were measured on -- pinned so the numbers in 0002.md stay
# reproducible if the shipped default moves (it since has; see 0005)
_BOX = {"phis": (0.70, 0.85, 0.95), "ss": (0.20, 0.30, 0.45, 0.60, 0.80)}
_RANK_TOL = 1e-8        # numerical rank tolerance -- the order the engine already uses for
                        # structural activation (`hv_norm > 1e-8`); not a tuned constant
_SPARROW = 1.5          # the engine's own resolution factor (_GAP_FACTOR, finding 11)

SEED, N, JUMP_AT, JUMP, NOISE_AT = 11, 900, 380, 9.0, 600
Q_TRUE, S2_A, S2_C = 0.02, 1.0, 9.0
A, C = slice(60, JUMP_AT), slice(NOISE_AT, N)


def hero_series(seed=SEED):
    rng = np.random.default_rng(seed)
    theta = np.cumsum(rng.normal(0.0, np.sqrt(Q_TRUE), N))
    theta[JUMP_AT:] += JUMP
    sd = np.where(np.arange(N) < NOISE_AT, np.sqrt(S2_A), np.sqrt(S2_C))
    return theta, theta + rng.normal(0.0, sd)


def kalman(y, Q, R):
    m, P = y[0], R
    mean = np.empty_like(y); var = np.empty_like(y)
    for t, v in enumerate(y):
        P += Q
        S = P + R
        K = P / S
        m += K * (v - m)
        P = (1.0 - K) * P
        mean[t], var[t] = m, P
    return mean, var


# ------------------------------------------------------------------ the rung grid
# ------------------------------------------- structure: where is the walk blind?
# ===================================================================== candidate A
class _LadderEngine(_WalkEngine):
    """A bank member with the CONTROLS this probe needs, on top of the shipped engine's ladder.

    The engine already anchors a member at a rung and reverts the walk's null excursion at the
    class rate; the two knobs here exist only to measure what happens when that is changed:
    ``revert`` (``"class"``, ``"hard"`` = at once, ``None`` = never) and ``memoryless`` (drop the
    window posterior on a group's process axis, sensor axis, both, or neither).  Both are filed
    negatives -- see 0002 §3 and 0004 §1 -- and neither is a setting the public filter offers.
    """

    def __init__(self, Q0, R0, H, F, B, phi, s, groups=(), lo=0.0, revert="class",
                 memoryless="none"):
        super().__init__(Q0, R0, H, F, B, phi, s, groups=groups, anchor_lo=lo)
        self._revert = {"class": float(phi), "hard": 0.0}.get(revert, None)
        self._memoryless = memoryless or "none"        # none | both | proc | sens

    def update(self, y, u=None):
        step = super().update(y, u=u)
        if self._groups and self._memoryless != "none":
            # A group's two axes span the split, and the split has ZERO per-step Fisher, so a
            # window posterior on them can accumulate something the data never said.  Measured:
            # it does not -- the window's memory carries the part of the split that genuinely
            # moves inside a run, and dropping it is worse in every form (0004 §1).
            want = {"both": (True, True), "proc": (True, False), "sens": (False, True)}
            wp, wm = want[self._memoryless]
            for (k, i, _h2) in self._groups:
                for ax, on in ((k, wp), (self.n + i, wm)):
                    if on and ax in self._axwin:
                        self._pi_ax[self._act.index(ax)] = self._w1
        return step


class LadderFilter(LucidFilter):
    """`LucidFilter` with the probe's controls on every member.  Same public API, same defaults."""

    def __init__(self, *a, **kw):
        rungs = kw.pop("rungs", True)
        revert = kw.pop("revert", "class")
        memoryless = kw.pop("memoryless", "none")
        super().__init__(*a, **kw)
        e0 = self._members[0]
        Q0 = e0.V @ np.diag(e0.lam) @ e0.V.T
        groups = self.groups if rungs else []
        los = self.split_arr if (rungs and self.groups) else np.zeros(1)
        phis = sorted(set(self.phi_arr.tolist()))
        ss = sorted(set(self.s_arr.tolist()))
        mem, pa, sa, la = [], [], [], []
        for p in phis:
            for sv in ss:
                for lo in los:
                    mem.append(_LadderEngine(Q0, e0.rho, e0.H, e0.F, e0.B, p, sv, groups=groups,
                                             lo=float(lo), revert=revert, memoryless=memoryless))
                    pa.append(p); sa.append(sv); la.append(float(lo))
        self._members = mem
        self.phi_arr, self.s_arr, self.lo_arr = np.array(pa), np.array(sa), np.array(la)
        self.groups = groups
        self.reset()

    def verdict(self):
        """Bank-posterior split, as log-odds."""
        if not self.groups:
            return float("nan")
        w = np.exp(self._logw - _logsumexp(self._logw))
        a = np.empty(len(self._members)); b = np.empty(len(self._members))
        for mi, mem in enumerate(self._members):
            tot, lo = mem._group_read(mem.mu)[0]
            a[mi] = tot / (1.0 + math.exp(-min(max(lo, -80.0), 80.0)))
            b[mi] = tot - a[mi]
        return math.log(max(float(w @ a), 1e-300)) - math.log(max(float(w @ b), 1e-300))


# ===================================================================== the race
def rmse(e, th, s):
    return float(np.sqrt(np.mean((e[s] - th[s]) ** 2)))


def calib(e, v, th, s):
    return float(np.mean((e[s] - th[s]) ** 2 / v[s]))


def rise(est, th, start, jump, frac=0.1):
    err = np.abs(est[start:] - th[start:])
    b = np.flatnonzero(err < frac * abs(jump))
    return int(b[0]) if b.size else len(err)


def report(name, m, v, th, kA, kC, elapsed=None, extra=""):
    ra, rc = rmse(m, th, A) / kA, rmse(m, th, C) / kC
    ok = "PASS" if (ra <= 1.10 and rc <= 1.05 and rise(m, th, JUMP_AT, JUMP) <= 4
                    and 0.6 <= calib(m, v, th, A) <= 1.5
                    and 0.6 <= calib(m, v, th, C) <= 1.5) else "    "
    print(f"  {ok} {name:24s} ssRMSE {rmse(m, th, A):.4f} ({ra:5.3f}x)   "
          f"C {rmse(m, th, C):.4f} ({rc:5.3f}x)   rise {rise(m, th, JUMP_AT, JUMP):2d}   "
          f"calib A {calib(m, v, th, A):.2f} C {calib(m, v, th, C):.2f}"
          + (f"   {1e3 * elapsed / N:.1f} ms/step" if elapsed else "") + extra)
    return ra, rc


def scales(r, tag):
    ps, ms = r.process_scale[:, 0], r.measurement_scale[:, 0]
    out = []
    for nm, sl, qt, rt in (("A", A, Q_TRUE, S2_A), ("C", C, Q_TRUE, S2_C)):
        Qh, Rh = np.exp(ps[sl]).mean(), np.exp(ms[sl]).mean()
        out.append(f"{nm}: Q={Qh:.4f} R={Rh:.4f} ratio={Qh / Rh:.5f} (truth {qt / rt:.5f})")
    print(f"     {tag}  " + "   ".join(out))


def main():
    th, y = hero_series()
    kal_m, kal_v = kalman(y, Q_TRUE, S2_A)
    kA, kC = rmse(kal_m, th, A), rmse(kal_m, th, C)
    t, K, odds = _rung_gains(0.999)
    print(f"hero rig: seed {SEED}, Q={Q_TRUE}, R {S2_A}->{S2_C}, jump {JUMP} at {JUMP_AT}")
    print(f"  oracle-tuned Kalman: ssRMSE {kA:.4f}  regime-C RMSE {kC:.4f}  "
          f"rise {rise(kal_m, th, JUMP_AT, JUMP)}  calib C {calib(kal_m, kal_v, th, C):.2f}")
    print(f"  GATES: ssRMSE <= 1.10x   C <= 1.05x   rise <= 4   calib in [0.6, 1.5]")
    print(f"  retired FITTED filter (the bar): ss 1.056x  C 1.012x  rise 1  calib C 0.66")
    print(f"  rung grid (forget=0.999): {len(t)} rungs, gains {K[0]:.4f}..{K[-1]:.4f}, "
          f"log-odds {math.log(odds[0]):+.1f}..{math.log(odds[-1]):+.1f} "
          f"(truth A {math.log(Q_TRUE / S2_A):+.2f}, C {math.log(Q_TRUE / S2_C):+.2f})")
    print()

    t0 = time.time(); r = LucidFilter(**_BOX).filter(y[:, None]); t1 = time.time()
    report("shipped LucidFilter()", r.mean[:, 0], r.var[:, 0, 0], th, kA, kC, t1 - t0)
    scales(r, "shipped")

    print("  (the shipped (phi, s) box these controls were measured on is pinned in `_BOX`)")
    variants = (("A: ladder (class revert)", {}),
                ("A: ladder (hard project)", {"revert": "hard"}),
                ("A: ladder (free drift)", {"revert": None}))
    for tag, kw in variants:
        t0 = time.time(); f = LadderFilter(**_BOX, **kw)
        r2 = f.filter(y[:, None]); t1 = time.time()
        report(tag, r2.mean[:, 0], r2.var[:, 0, 0], th, kA, kC, t1 - t0,
               extra=f"   verdict {f.verdict():+.2f}")
        scales(r2, tag.split(": ")[1])
        if not kw:
            print(f"     groups {f.groups}   members {len(f._members)} "
                  f"({len(f.rung_lo)} rungs x 15 (phi,s))")


if __name__ == "__main__":
    main()
