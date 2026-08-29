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
from lucid.statfilter.lucid import _WalkEngine, _logsumexp                # noqa: E402

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
def _rung_gains(forget):
    """Rung positions, as gains, covering every possible split at the bank's own resolution.

    A split only ever acts through the filter's steady-state gain `K`.  A local-level filter run
    at gain `K` models its differenced data as MA(1) with `theta = 1 - K`, so the per-step
    Kullback-Leibler divergence between two splits is (Whittle)

        D = 0.5 log[(1 - 2 th th' + th'^2) / (1 - th^2)]  ->  0.5 (dt)^2,
        dt = d th / sqrt(1 - th^2),   t = arccos(1 - K)  in  [0, pi/2].

    The whole space of splits is an interval of arclength pi/2.  Two rungs are resolvable when
    the evidence the bank can hold -- `1/(1 - forget)` steps -- separates them by order one nat:
    `N 0.5 dt^2 = 1`, i.e. `dt = sqrt(2 (1 - forget))`.  Spacing at the engine's Sparrow factor
    above that limit gives a grid with no dead zone and no span constant: it is COMPLETE.
    """
    step = _SPARROW * math.sqrt(2.0 * (1.0 - forget))
    J = int(math.ceil((0.5 * math.pi) / step))
    t = (np.arange(J) + 0.5) * (0.5 * math.pi) / J
    K = 1.0 - np.cos(t)
    return t, K, K * K / np.maximum(1.0 - K, 1e-12)      # arclength, gain, odds a/b


# ------------------------------------------- structure: where is the walk blind?
def split_groups(eng):
    """(process axis, sensor axis, (H v)^2) triples whose split no per-step score can carry."""
    I = steady_fisher_full(eng)
    dg = np.diag(I)
    info = dg > _RANK_TOL * dg.max()
    out = []
    for k in range(eng.n):
        if not (eng.active[k] and info[k]):
            continue
        hv = np.abs(eng.HV[:, k])
        nz = np.flatnonzero(hv > _RANK_TOL * hv.max())
        if nz.size != 1:
            continue                          # read by more than one sensor: not degenerate
        i = int(nz[0])
        if not info[eng.n + i]:
            continue
        sub = I[np.ix_([k, eng.n + i], [k, eng.n + i])]
        w = np.linalg.eigvalsh(sub / np.sqrt(np.outer(np.diag(sub), np.diag(sub))))
        if w[0] < _RANK_TOL * w[-1]:
            out.append((k, i, float(eng.HV[i, k] ** 2)))
    return out


def steady_fisher_full(eng):
    H, F, n = eng.H, eng.F, eng.n
    P = np.eye(n) * (eng.lam.max() + eng.rho.max())
    Q0, R0 = eng._Q_of(np.zeros(n)), eng._R_of(np.zeros(eng.m))
    for _ in range(400):
        Pp = F @ P @ F.T + Q0
        K = Pp @ H.T @ np.linalg.inv(H @ Pp @ H.T + R0)
        P = Pp - K @ H @ Pp
    Pp = F @ P @ F.T + Q0
    Si = np.linalg.inv(H @ Pp @ H.T + R0)
    dS = eng._dS_list(np.zeros(eng.D))
    I = np.empty((eng.D, eng.D))
    for a in range(eng.D):
        SdA = Si @ dS[a]
        for b in range(a, eng.D):
            I[a, b] = I[b, a] = 0.5 * float(np.trace(SdA @ Si @ dS[b]))
    return I


# ===================================================================== candidate A
class _LadderEngine(_WalkEngine):
    """One bank member: the caltrop engine anchored at a fixed split of every confounded group."""

    def __init__(self, Q0, R0, H, F, B, phi, s, groups=(), lo=0.0, revert="class"):
        self._groups = list(groups)
        self._anchor_lo = float(lo)
        self._revert = revert          # "class": at rate phi;  "hard": at once;  None: free
        super().__init__(Q0, R0, H, F, B, phi, s)

    # the two coordinates of a group: its contribution to S (identifiable) and its log-odds (not)
    def _read(self, mu):
        r = []
        for (k, i, h2) in self._groups:
            a = self.lam[k] * h2 * math.exp(min(mu[k], 60.0))
            b = self.rho[i] * math.exp(min(mu[self.n + i], 60.0))
            r.append((a + b, math.log(max(a, 1e-300)) - math.log(max(b, 1e-300))))
        return r

    def _write(self, mu, tots, los):
        out = mu.copy()
        for gi, (k, i, h2) in enumerate(self._groups):
            lo = float(np.clip(los[gi], -80.0, 80.0))
            a = tots[gi] / (1.0 + math.exp(-lo))
            b = tots[gi] - a
            out[k] = math.log(max(a, 1e-300) / (self.lam[k] * h2))
            out[self.n + i] = math.log(max(b, 1e-300) / self.rho[i])
        return out

    def reset(self, mean=None, scale=None):
        super().reset(mean, scale)
        if self._groups:                       # start ON the rung, at the supplied base total
            tots = [t for t, _ in self._read(self.mu)]
            self.mu = self._write(self.mu, tots, [self._anchor_lo] * len(self._groups))
        return self

    def update(self, y, u=None):
        step = super().update(y, u=u)
        if self._groups and self._revert is not None:
            # The per-axis Newton walk steps by `score/info`, which on the process axis is ~1/Q
            # and on the sensor axis ~1/R: when Q << R that step is almost entirely along the
            # NULL direction, where the per-step score carries no information at all.  It is an
            # artefact of taking a per-axis Newton step against a singular Fisher, and it
            # systematically blames the smaller variance -- exactly the wrong reflex when a
            # SENSOR degrades.  The walk keeps the identifiable part; the null part goes back to
            # this member's hypothesis, at the total the walk just established.
            tots, los = zip(*self._read(self.mu))
            rate = self.phi if self._revert == "class" else 0.0
            back = [self._anchor_lo + rate * (lo - self._anchor_lo) for lo in los]
            self.mu = self._write(self.mu, list(tots), back)
        return step


class LadderFilter(LucidFilter):
    """`LucidFilter` with the split laddered across the bank.  Same public API, same defaults."""

    def __init__(self, *a, **kw):
        rungs = kw.pop("rungs", True)
        revert = kw.pop("revert", "class")
        super().__init__(*a, **kw)
        e0 = self._members[0]
        Q0 = e0.V @ np.diag(e0.lam) @ e0.V.T
        groups = split_groups(e0) if rungs else []
        self.groups = groups
        t, K, odds = _rung_gains(self.forget)
        self.rung_t, self.rung_lo = t, np.log(odds)
        mem, phis, ss, los = [], [], [], []
        for p, sv in zip(self.phi_arr, self.s_arr):
            for lo in (self.rung_lo if groups else [0.0]):
                mem.append(_LadderEngine(Q0, e0.rho, e0.H, e0.F, e0.B, p, sv,
                                         groups=groups, lo=lo, revert=revert))
                phis.append(p); ss.append(sv); los.append(lo)
        self._members = mem
        self.phi_arr, self.s_arr = np.array(phis), np.array(ss)
        self.lo_arr = np.array(los)
        self.reset()

    def verdict(self):
        """Bank-posterior mean split, in the arclength coordinate the ladder is uniform in."""
        if not self.groups:
            return float("nan")
        w = np.exp(self._logw - _logsumexp(self._logw))
        K = np.interp(self.lo_arr, self.rung_lo, 1.0 - np.cos(self.rung_t))
        Kh = float(w @ K)
        return math.log(Kh * Kh / max(1.0 - Kh, 1e-12))


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

    t0 = time.time(); r = LucidFilter().filter(y[:, None]); t1 = time.time()
    report("shipped LucidFilter()", r.mean[:, 0], r.var[:, 0, 0], th, kA, kC, t1 - t0)
    scales(r, "shipped")

    variants = (("A: ladder (class revert)", {}),
                ("A: ladder (hard project)", {"revert": "hard"}),
                ("A: ladder (free drift)", {"revert": None}))
    for tag, kw in variants:
        t0 = time.time(); f = LadderFilter(**kw); r2 = f.filter(y[:, None]); t1 = time.time()
        report(tag, r2.mean[:, 0], r2.var[:, 0, 0], th, kA, kC, t1 - t0,
               extra=f"   verdict {f.verdict():+.2f}")
        scales(r2, tag.split(": ")[1])
        if not kw:
            print(f"     groups {f.groups}   members {len(f._members)} "
                  f"({len(f.rung_lo)} rungs x 15 (phi,s))")


if __name__ == "__main__":
    main()
