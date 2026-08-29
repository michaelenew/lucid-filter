"""LucidFilter -- the single public filter.

You supply the *dynamics* (a linear ODE `F` with an optional forcing bias `B u`, default: none ->
a random-walk level) and the *observation* (`H`, default: identity).  Everything about the *noise* it
infers online: which process eigenmode is drifting, which sensor is glitching, and how far -- by
WALKING a per-component log-scale grid with unbounded reach, and it does not even take the AR(1) class
`(phi, s)` -- it runs a small bank across a broad `(phi, s)` box and lets the data average it out
(the flat identification ridge integrates away; only `forget`, a ~1000-step weight memory, remains, and
tracking is identical for any value near 1).

    theta_t = F theta_{t-1} + B u_t + w_t,   w_t ~ N(0, Q(t))
    y_t     = H theta_t          + v_t,      v_t ~ N(0, R(t))
    Q(t) = V diag(lam_k e^{xi_k(t)}) V^T,     R(t) = diag(rho_i e^{eta_i(t)})     (per-component scales, walked)

Configure by the give-what-you-know / infer-the-rest rule -- for each input pass an explicit value
("I know this"), a null/zero ("there is none"), or leave the default:

    LucidFilter(dynamics=F, H=H)                 # you know the dynamics and the sensors
    LucidFilter()                                # scalar random-walk level, direct observation
    LucidFilter(dynamics=F, process=Q0, ...)     # you also know the base noise magnitudes

Everything is vector; a scalar problem is length 1.

**The dynamics channel.**  `dynamics=None` learns `F` (and `B`) online from the random-walk prior;
`dynamics=F0, faults=rho` says the supplied dynamics may CHANGE -- a payload attached to a drone, a
tire blown out -- and the filter detects the change and recovers the new dynamics without a refit, a
threshold, or a fitted constant.  It is realised as a state augmentation `(x, g)` with
`F = F0 + sum_j g_j A_j`, so the noise machinery above runs on top of it unchanged, which is what
separating a wrong `F` from elevated `Q` requires: the two compete as hypotheses under a live noise
walk rather than through a bolted-on whiteness statistic.  The bank carries the nominal member
forever (a false detection then costs ~nothing), optional named fault `anchors` (the fastest
detector when the failure modes can be named), and a departure walker whose variance is bounded at
the class cap and re-priced to it when a fault is confirmed -- bounded, never frozen.  The one
labeled prior is the fault hazard `rho`; every gain, drift, cap and restart width follows from it
and the class size, and the detection delay it buys is derived (`log(1/rho) / KL-rate`), not tuned.
The derivations and the measured acceptance results are `research/dynamics-learning/SUMMARY.md`.

This is a benchmark toy: the RMSE for a given amount of supplied knowledge is the bound a real
implementation can aim at.  The mechanism (per-component walk, axial GPB1, structural axis
activation, finding-18 loop) is the parameter-free `WalkingVectorFilter` (now moved to research/ as
a specimen), lifted with a supplied `F` and wrapped in the `(phi, s)` bank of `WalkingBank` -- with
one structural change: the specimen's exact tensor-product scale grid is ``(2K+1)**(n+m)`` nodes,
EXPONENTIAL in the component count (a 5-DOF arm is out of reach), and is retired to research as the
theory-only reference.  The engine here evaluates the CALTROP axial star instead (research 0013):
the window centre plus an axial window per active axis, ``1 + 2K * (#active axes)`` nodes -- LINEAR
in `n + m`, so the whole filter is polynomial-time.  The star does not represent the joint scale
density (no corner nodes); it locates its peak by per-axis walking, which 0013 validates as matching
the exact grid for state tracking at linear cost.

The walk is the right instrument for every scale direction the one-step likelihood can see, and
there is one it cannot.  Where a process eigenmode is read by exactly ONE sensor, the two scale
derivatives are proportional as matrices, so the likelihood sees only the SUM of their
contributions to that channel and the SPLIT between them is invisible at every step, at every
operating point -- Proposition 1 in coordinates (research/sequence-demix/0001).  The engine finds
those pairs structurally and carries the split as a dimension of the BANK instead: a ladder of
anchored hypotheses, each a complete filter, so the sequence evidence reaches it through the
member's own mean (a rung with too much process chases sensor noise and pays for it in its own
predictive likelihood) and its weight accumulates on the `forget` timescale.  The rungs are placed
by their consequence rather than by an offset from the supplied base, which makes the ladder
COMPLETE -- see `_rung_odds`.  A structure where every process mode is read more than once has no
such pair, gets no ladder, and costs exactly what it did before.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["LucidFilter", "LucidStep", "LucidResult"]

_LOG2PI = math.log(2.0 * math.pi)
_GAP_FACTOR = 1.5           # grid spacing gap = 1.5 s (Sparrow resolution limit, finding 11)
_SPAN_S = 3.0               # window half-span in units of s (support budget -> node count)
_RIDGE = 1e-4               # Fisher stabiliser
_PHIS = (0.70, 0.85, 0.95)                 # default (phi, s) box for the bank -- a broad range, not a
_SS = (0.20, 0.40, 0.80, 1.60, 3.20)       #   fitted value; the data down-weights the unsupported corners.
                                           #   Geometric, and it has to reach: `s` is the SD of a
                                           #   LOG variance, so the top of the box is the largest
                                           #   scale change the window can represent in one step
                                           #   (`3 s` of half-span).  A box ending at 0.8 tops out
                                           #   at a factor of 11, which is smaller than the regime
                                           #   changes in this repository's own rigs.
_HAZARD = 1e-4              # default fault rate: ~1 dynamics fault per 10,000 steps.  A LABELED
                            # prior of the same standing as `forget`, not a tuning constant: it is
                            # the operating point on the false-alarm/delay frontier, and the delay
                            # it buys is derived, log(1/rho) / KL-rate (research 0001).
_RANK_TOL = 1e-8            # numerical rank tolerance (the order used for structural activation)
_LADDER_MEM = 1000.0        # node budget for the split ladder, in the same sense as _SPAN_S: the
                            # finest grid the engine will build is the one a thousand-step memory
                            # supports (24 rungs).  A longer `forget` still sharpens the bank's
                            # weights; it does not buy a finer ladder, and `forget = 1` would ask
                            # for an infinite one.


# --------------------------------------------------- the per-step-blind directions
def _steady_Si(eng, lam, rho):
    """The steady-state innovation precision at the base ``(lam, rho)`` -- the one Riccati solve.

    Everything the Fisher needs from the recursion is in here, and it depends only on
    ``(F, H, cap)`` and the base MATRICES -- not on the class ``(phi, s)``, and not on which
    split hypothesis a member carries, because it is evaluated at the split-BALANCED base and
    balancing undoes every split exactly (a split moves along ``dQ = -dR`` at fixed totals).
    So one spec needs ONE solve however many members it has; `LucidFilter` computes it once per
    spec and hands it to every member.
    """
    H, F, n = eng.H, eng.F, eng.n
    P = np.eye(n) * (lam.max() + rho.max())
    Q0 = eng.V @ np.diag(lam) @ eng.V.T
    R0 = np.diag(rho)
    for _ in range(400):
        Pp = eng._cap_P(F @ P @ F.T + Q0)
        K = Pp @ H.T @ np.linalg.inv(H @ Pp @ H.T + R0)
        P = Pp - K @ H @ Pp
    Pp = eng._cap_P(F @ P @ F.T + Q0)
    return np.linalg.inv(H @ Pp @ H.T + R0)


def _scale_fisher(eng, lam, rho, Si):
    """Full per-step scale-Fisher ``I_ab = 0.5 tr(Si dS_a Si dS_b)`` at the steady state.

    ONE solve answers both questions a member has to ask about the model it runs: the DIAGONAL is
    the characteristic Fisher that sets the walk's drift, and the off-diagonals are where a
    degenerate pair shows itself.  It is evaluated at the base ``(lam, rho)`` handed in -- the
    split-balanced one -- so that neither answer depends on which split hypothesis the member
    happens to be carrying.  The structure of a model is not a function of the hypothesis under
    test, and read at an extreme rung it would not even look like itself.  ``Si`` is the base's
    steady-state innovation precision from `_steady_Si`.
    """
    n, m = eng.n, eng.m
    dS = [lam[k] * np.outer(eng.HV[:, k], eng.HV[:, k]) for k in range(n)]
    for i in range(m):
        E = np.zeros((m, m)); E[i, i] = rho[i]
        dS.append(E)
    I = np.empty((eng.D, eng.D))
    for a in range(eng.D):
        SdA = Si @ dS[a]
        for b in range(a, eng.D):
            I[a, b] = I[b, a] = 0.5 * float(np.trace(SdA @ Si @ dS[b]))
    return I


def _split_groups(eng, I):
    """Pairs whose SPLIT no per-step score can ever carry -- Proposition 1, in coordinates.

    ``dS_xi_k = lam_k e^xi (H v_k)(H v_k)^T`` and ``dS_eta_i = rho_i e^eta E_ii`` are proportional
    as matrices exactly when ``H v_k`` lies along ``e_i``: a process eigenmode read by ONE sensor.
    Then the 2x2 scale-Fisher block is exactly rank 1, the one-step likelihood sees only the SUM of
    the two contributions to ``S_ii``, and the split between them is invisible at every step, at
    every operating point (research 0001).  Both axes must carry non-negligible information: an
    axis whose own scale-Fisher is numerically zero is not half of a confound, it is nothing --
    which is what keeps a ridge-regularised ``Q0``'s null modes out.

    Returns ``(process axis, sensor axis, (H v)_i^2)`` triples.  ``I`` is the scale-Fisher of
    `_scale_fisher`, already computed for the walk's drift, so this costs a look and no solve.
    """
    dg = np.diag(I)
    info = dg > _RANK_TOL * dg.max()
    out = []
    for k in range(eng.n):
        if not (eng.active[k] and info[k]):
            continue
        hv = np.abs(eng.HV[:, k])
        nz = np.flatnonzero(hv > _RANK_TOL * hv.max())
        if nz.size != 1:
            continue
        i = int(nz[0])
        if not info[eng.n + i]:
            continue
        sub = I[np.ix_([k, eng.n + i], [k, eng.n + i])]
        w = np.linalg.eigvalsh(sub / np.sqrt(np.outer(np.diag(sub), np.diag(sub))))
        if w[0] < _RANK_TOL * w[-1]:
            out.append((k, i, float(eng.HV[i, k] ** 2)))
    return out


def _apply_split(eng, lo_vec):
    """Divide each confounded pair's noise between process and sensor at a FIXED TOTAL.

    The null direction of the per-step scale-Fisher integrates to ``dQ = -dR`` (research 0001), so
    this moves exactly along the coordinate the one-step likelihood cannot see and leaves
    everything it can see alone.  Returns a ``(Q0, R0)`` base, which is the only thing a member
    ever needs to be told: it reads its own anchor back off it.
    """
    lam, rho = eng.lam.copy(), eng.rho.copy()
    for g, (k, i, h2) in enumerate(eng._groups):
        a, b = lam[k] * h2, rho[i]
        tot = a + b
        lo = float(np.clip(lo_vec[g], -80.0, 80.0))
        a2 = tot / (1.0 + math.exp(-lo))
        lam[k] = max(a2, 1e-300) / h2
        rho[i] = max(tot - a2, 1e-300)
    return eng.V @ np.diag(lam) @ eng.V.T, rho


def _split_star(los, n_pairs):
    """The split hypotheses: one reference vector, plus one pair moved off it at a time.

    Enumerating a rung for every pair jointly is ``J ** n_pairs``, exponential in the pair count,
    which is the same wall the scale grid hits and is answered the same way -- the caltrop star of
    research 0013: the centre plus an axial arm per axis.  With a single pair the star IS the whole
    ladder, so nothing about the one-pair case is a special case of anything; with none it is the
    empty vector and the bank is what it was.

    The star is linear in the pair count, and linear is still a multiplier on a bank of complete
    filters: five pairs at full resolution is 116 vectors, and a pots-only 5-DOF arm measured 1740
    members and 1.5 s/step.  So the NODE BUDGET is what one pair's resolution costs, and several
    pairs share it -- the same "support budget -> node count" trade `_SPAN_S` already makes, and
    it needs no constant of its own because the budget is the resolution `forget` supports.  The
    grid stays COMPLETE over `[0, pi/2]` at every pair count; what degrades is its resolution, and
    that is the honest thing to give up when there is more to resolve and the same budget to do it
    with.

    The reference is the middle of the arclength grid -- the agnostic split, the one the ladder's
    own uniform prior puts its mass around.  It is not the supplied base: no rung refers to that.
    """
    if n_pairs == 0:
        return np.zeros((1, 0))
    arms = max(len(los) - 1, 1)                       # the budget: one pair's own arm count
    per = max(arms // n_pairs, 2)                     # shared out, never below a usable ladder
    idx = np.unique(np.round(np.linspace(0, len(los) - 1, per + 1)).astype(int))
    c = int(idx[len(idx) // 2])
    ref = np.full(n_pairs, los[c])
    out = [ref]
    for g in range(n_pairs):
        for j in idx:
            if j == c:
                continue
            v = ref.copy()
            v[g] = los[j]
            out.append(v)
    return np.array(out)


def _rung_odds(forget):
    """The ladder of splits: complete, at the bank's own resolution, with no span constant.

    A split acts only through the filter's gain ``K``.  A local-level filter run at gain ``K``
    models its differenced data as MA(1) with ``theta = 1 - K``, so the per-step Kullback-Leibler
    divergence between two splits is (Whittle)

        D = 0.5 log[(1 - 2 th th' + th'^2) / (1 - th^2)]  ->  0.5 (dt)^2,
        dt = d th / sqrt(1 - th^2),   t = arccos(1 - K)  in  [0, pi/2].

    The entire space of splits is therefore an interval of arclength ``pi/2``.  Two rungs are
    resolvable when the evidence the bank can hold -- ``1/(1 - forget)`` steps -- separates them
    by order one nat, i.e. ``dt = sqrt(2 (1 - forget))``; spacing them at the grid's own Sparrow
    factor above that limit leaves no dead zone.  The memory entering that resolution is capped at
    ``_LADDER_MEM``, which is a node budget and not a statistical claim.  The result COVERS EVERY POSSIBLE SPLIT with a
    couple of dozen rungs, and no rung refers to the supplied base: told nothing means told
    nothing.
    """
    mem = min(1.0 / (1.0 - forget), _LADDER_MEM) if forget < 1.0 else _LADDER_MEM
    step = _GAP_FACTOR * math.sqrt(2.0 / mem)
    J = int(math.ceil((0.5 * math.pi) / step))
    t = (np.arange(J) + 0.5) * (0.5 * math.pi) / J
    K = 1.0 - np.cos(t)
    return K * K / np.maximum(1.0 - K, 1e-12)


# --------------------------------------------------------------------- results
@dataclass
class LucidStep:
    """What the filter knows after one vector observation (bank model-averaged)."""

    mean: np.ndarray               #: posterior state mean (n,)
    var: np.ndarray                #: posterior state covariance (n, n)
    innovation: np.ndarray         #: y_t - H (F prev_mean + B u)  (m,)
    loglik: float                  #: mixture predictive log-density of y_t
    process_scale: np.ndarray      #: per process-eigenmode log-scale (n,)
    measurement_scale: np.ndarray  #: per-sensor log-scale (m,)
    dynamics: np.ndarray = None    #: posterior-mean ``F`` (n, n) -- ``None`` when supplied fixed
    control: np.ndarray = None     #: posterior-mean ``B`` (n, p) -- ``None`` when fixed or absent
    fault: float = 0.0             #: posterior probability the dynamics have left the NOMINAL --
    #: which is the supplied ``F`` under ``faults=``, and the random walk ``F = I`` under
    #: ``dynamics=None`` (where it therefore reads "the dynamics are not a random walk")

    @property
    def scale(self) -> np.ndarray:
        return np.concatenate([self.process_scale, self.measurement_scale])


@dataclass
class LucidResult:
    """Batch output.  ``mean`` (T, n); ``var`` (T, n, n)."""

    mean: np.ndarray
    var: np.ndarray
    innovation: np.ndarray
    process_scale: np.ndarray
    measurement_scale: np.ndarray
    loglik: float = 0.0
    dynamics: np.ndarray = None    #: (T, n, n) learned ``F``, or ``None`` when supplied fixed
    control: np.ndarray = None     #: (T, n, p) learned ``B``, or ``None``
    fault: np.ndarray = None       #: (T,) posterior probability of a dynamics fault

    def __len__(self) -> int:
        return len(self.mean)


def _logsumexp(a: np.ndarray) -> float:
    m = float(np.max(a))
    return m + math.log(float(np.sum(np.exp(a - m))))


# ------------------------------------------------------- the per-(phi,s) engine
class _WalkEngine:
    """One bank member: the per-component walking CALTROP filter WITH supplied dynamics F (+ forcing B u).

    Same model and walk as the research `WalkingVectorFilter`, with the state prediction being the
    supplied linear dynamics ``mpred = F m + B u``, ``Ppred = F P F^T + Q`` (F defaults to the
    identity -> random walk) -- but the scale posterior lives on the caltrop star (research 0013),
    not the exact tensor grid: the window centre ``mu`` plus one axial window per active axis,
    ``1 + 2K * r`` nodes where the tensor grid is ``(2K+1)**r`` (exponential in the active-axis
    count ``r``; theory-only).  Per-axis window posteriors carry the AR(1) memory (propagated
    through the 1-D kernel, reweighted by the per-node likelihood); the state KF is GPB1-collapsed
    over the star as the evidence-weighted mixture of the axial windows; and the centre walks by
    the finding-18 loop per axis (score/Fisher averaged over the axial profile), so reach stays
    unbounded.  Axes are active by STRUCTURAL observability (research 0024): a process eigenmode
    is live iff it carries base variance and is seen by ``H``; a sensor is always live.  The
    delocalisation the 0010 spectral freeze prevented is bounded instead of frozen out: ``q_mu``'s
    Fisher is floored at the 0010 threshold and the walk covariance is capped at the window.
    Parameter-free: gain ``K* = (1-phi)/4``, drift ``q_mu``, floor and cap are all derived from
    the class ``(phi, s)``; no EMA, no tuned constant.
    """

    def __init__(self, Q0, R0, H, F, B, phi, s, walk_axes=None, cap=None, group_class=None,
                 fisher_Si=None):
        self._revert = float(phi)         # rate the walk's null excursion returns to that
                                          # hypothesis; the class's own persistence.  Named so
                                          # research can vary it -- both bounds are load-bearing
                                          # (research/sequence-demix/0002 §3), and neither end is
                                          # a setting anyone should reach for.
        # ``group_class`` optionally gives a confounded group's two axes their OWN (phi, s) --
        # ``((phi_P, s_P), (phi_M, s_M))``.  A shared class cannot express what a confounded pair
        # needs: a level jump wants a process window that reaches a long way, and a sensor that
        # degrades wants that same window not to HOLD what it reached, while the sensor's own
        # window must hold.  Reach and persistence pull opposite ways on one axis and the same
        # way on two.  This is the structure `fit()` found for the retired filter on this very
        # rig -- phi_P ~ 0 with s_P = 3.69, phi_M = 0.93 with s_M = 1.62 -- and it is the one
        # piece of it the shipped bank could not represent.
        self._group_class = group_class
        n = Q0.shape[0]
        lam, V = np.linalg.eigh(Q0)
        self.n, self.m = n, R0.size
        self.D = n + self.m
        self.V, self.lam, self.rho, self.H = V, lam, R0, H
        self.F = F
        self.B = B
        self.p = 0 if B is None else B.shape[1]
        self.HV = H @ V
        self.phi, self.s = float(phi), float(s)
        self._cap = None if cap is None else np.asarray(cap, float)
        self._dyn = None            # optional (mean, u) -> (Jacobian, predicted mean) callable
        self._hook = None           # optional mean -> (H Jacobian, predicted measurement).
        # ``self.H`` stays the CHARACTERISTIC linearisation whatever the hook does: the
        # structural questions -- which axes are observable, which pairs are confounded, what
        # the steady-state scale-Fisher is -- are facts about the model, not about where the
        # state happens to be this step, and are answered once at the origin (the same
        # convention `_as_base` uses for a moving F).  The live Jacobian is used where it
        # actually belongs: the innovation, its covariance, and dS/dxi.
        # ACTIVATE structurally-observable axes -- NOT by the delocalisation floor (research 0024,
        # ported from AdaptiveKalmanFilter): a quiet process mode next to a loud sensor would be
        # frozen and then coast rigidly on a wrong velocity when that sensor is down-weighted --
        # the process+pot runaway.  A process eigenmode is live iff it carries base variance AND
        # is seen by H; a sensor is always live.  The delocalisation the 0010 freeze prevented is
        # bounded instead: q_mu's Fisher is floored at the 0010 threshold and the walk covariance
        # is capped at the window (Var(mu) <= L^2 -- the 0010 localisation condition as a bound).
        hv_norm = np.linalg.norm(self.HV, axis=0)
        self.active = np.ones(self.D, dtype=bool)
        for k in range(n):
            self.active[k] = self.lam[k] > 1e-12 * self.lam.max() and hv_norm[k] > 1e-8
        if walk_axes is not None:
            # Axes whose scale is a CLASS COMMITMENT, not a live noise scale (the dynamics
            # channel's theta block: its drift is q_theta = cap * rho from the fault class).
            # This bounds nothing about theta itself -- theta is a state with a live gain and a
            # cap, never frozen (research 0003); only the log-scale WALK of its drift is off.
            self.active &= np.asarray(walk_axes, bool)
        # Every member answers the same two questions about the model IT runs, and answers them
        # the same way: which of my scale axes are confounded in pairs, and what split does my
        # own base already sit at?  That base is where the walk's null excursion returns to, so a
        # member is anchored by construction rather than by being told -- which is what makes the
        # rule uniform across the bank, whatever dynamics hypothesis a member carries and whether
        # or not its state has been augmented.
        lam_b, rho_b = self._balanced_base()
        if fisher_Si is None:
            fisher_Si = _steady_Si(self, lam_b, rho_b)
        self._fisher_Si = fisher_Si
        _I = _scale_fisher(self, lam_b, rho_b, fisher_Si)
        self._groups = tuple(_split_groups(self, _I))
        self._anchor_lo = np.array([lo for _, lo in self._group_read(np.zeros(self.D))])
        self.phi_ax = np.full(self.D, self.phi)
        self.s_ax = np.full(self.D, self.s)
        if group_class is not None:
            for (k, i, _h2) in self._groups:
                (self.phi_ax[k], self.s_ax[k]) = group_class[0]
                (self.phi_ax[n + i], self.s_ax[n + i]) = group_class[1]
        self.gap = _GAP_FACTOR * self.s_ax
        self._Kstar = (1.0 - self.phi_ax) / 4.0
        # The DIAGONAL of that same solve is the characteristic Fisher, which sets `q_mu`, the
        # drift of the scale WALK.  The walk's business is the identifiable directions, not the
        # split, so taking it at the balanced point is what keeps the bank's hypothesis out of
        # the walk's tuning -- measured worth 1.230x -> 1.138x on regime C and 0.06 of
        # calibration headroom (`research/sequence-demix/0002`).
        self._Ichar = np.diag(_I) + _RIDGE
        self._Ifloor = (1.0 - self.phi_ax) / (4.0 * (_SPAN_S * self.s_ax) ** 2)
        self._qmu = self._Kstar ** 2 / (np.maximum(self._Ichar, self._Ifloor)
                                        * (1.0 - self._Kstar))
        self._Pmu_cap = (_SPAN_S * self.s_ax) ** 2
        self._build_window()
        self.reset()

    # -- caltrop star window (the 1-D pieces are unchanged from WalkingVectorFilter) --
    def _build_window(self):
        # One window per axis.  Every axis keeps the same NODE COUNT (so the axial posteriors stay
        # one rectangular array) and differs only in spacing, prior and kernel -- all three read
        # off that axis's own class.
        K = int(math.ceil(_SPAN_S / _GAP_FACTOR))
        node = np.arange(-K, K + 1)
        self._nn, self._c = node.size, K
        off = self.gap[:, None] * node[None, :]
        w1 = np.exp(-0.5 * (off / self.s_ax[:, None]) ** 2)
        w1 /= w1.sum(1, keepdims=True)
        nu = np.maximum(self.s_ax ** 2 * (1.0 - self.phi_ax ** 2), 1e-12)
        T1 = np.exp(np.clip(-0.5 * (off[:, None, :] - self.phi_ax[:, None, None]
                                    * off[:, :, None]) ** 2 / nu[:, None, None], -700.0, 700.0))
        T1 /= T1.sum(2, keepdims=True)
        self._off, self._w1, self._T1 = off, w1, T1
        self._act = [int(k) for k in np.flatnonzero(self.active)]
        # star node table: node 0 is the centre (all axes at mu); then, per active axis, the
        # 2K off-centre offsets along that axis alone.  1 + 2K*r nodes -- linear in the axes.
        arm = np.delete(np.arange(self._nn), self._c)
        self._star_axis = np.concatenate(
            [np.full(1, -1)] + [np.full(self._nn - 1, k) for k in self._act]).astype(int)
        self._star_off = np.concatenate(
            [np.zeros(1)] + [off[k][arm] for k in self._act])
        self._G = self._star_axis.size
        self._axwin = {}
        for i, k in enumerate(self._act):
            w = np.empty(self._nn, dtype=int)
            w[self._c] = 0
            w[arm] = 1 + i * (self._nn - 1) + np.arange(self._nn - 1)
            self._axwin[k] = w

    def _star_QR(self):
        """(Q_g, R_g) at every star node: the centre pair plus a per-node one-axis change."""
        n = self.n
        Qc = self.V @ np.diag(self.lam * np.exp(np.clip(self.mu[:n], -60, 60))) @ self.V.T
        rc = self.rho * np.exp(np.clip(self.mu[n:], -60, 60))
        Qg = np.repeat(Qc[None], self._G, 0)
        rg = np.repeat(rc[None], self._G, 0)
        for g in range(1, self._G):
            k = int(self._star_axis[g]); o = float(self._star_off[g])
            if k < n:
                dlam = self.lam[k] * (math.exp(min(self.mu[k] + o, 60.0))
                                      - math.exp(min(self.mu[k], 60.0)))
                Qg[g] += dlam * np.outer(self.V[:, k], self.V[:, k])
            else:
                i = k - n
                rg[g, i] = self.rho[i] * math.exp(min(self.mu[k] + o, 60.0))
        Rg = np.zeros((self._G, self.m, self.m))
        Rg[:, np.arange(self.m), np.arange(self.m)] = rg
        return Qg, Rg

    def _star_weights(self, pi_ax, alpha=None):
        """One distribution over the star: the axial windows mixed with weights ``alpha``
        (uniform when no evidence).  The shared centre accumulates every axis's centre mass."""
        w = np.zeros(self._G)
        r = len(self._act)
        if r == 0:
            w[0] = 1.0
            return w
        a = np.full(r, 1.0 / r) if alpha is None else alpha
        for i, k in enumerate(self._act):
            w[self._axwin[k]] += a[i] * pi_ax[i]
        return w

    def _wmean(self, pi_ax):
        """Posterior mean window offset per component (frozen axes sit at 0)."""
        w = np.zeros(self.D)
        for i, k in enumerate(self._act):
            w[k] = float(pi_ax[i] @ self._off[k])
        return w

    def _H_at(self, mean):
        """(live Jacobian, predicted measurement) -- the constant map unless a hook is set."""
        if self._hook is None:
            return self.H, self.H @ mean
        out = self._hook(mean)
        if isinstance(out, tuple):
            Hk, yp = out
            Hk = np.atleast_2d(np.asarray(Hk, float))
            return Hk, np.asarray(yp, float)
        Hk = np.atleast_2d(np.asarray(out, float))
        return Hk, Hk @ mean

    def _dS_axis(self, k, HV=None):
        """dS/dxi_k at each of axis k's window nodes (dS_k depends only on the k-coordinate)."""
        a = np.exp(np.minimum(self.mu[k] + self._off[k], 60.0))
        if k < self.n:
            hv = (self.HV if HV is None else HV)[:, k]
            return (self.lam[k] * a)[:, None, None] * np.outer(hv, hv)[None]
        out = np.zeros((self._nn, self.m, self.m))
        i = k - self.n
        out[:, i, i] = self.rho[i] * a
        return out

    def _cap_P(self, P):
        """Bound a state's variance at its class cap by symmetric row/column scaling.

        A BOUND, never a freeze: the gain stays live, so later evidence still moves the
        component (research 0003 measured the latched-freeze alternative at 20x worse)."""
        if self._cap is None:
            return P
        d = np.diag(P)
        over = d > self._cap
        if not over.any():
            return P
        sc = np.ones_like(d)
        sc[over] = np.sqrt(self._cap[over] / np.maximum(d[over], 1e-300))
        return P * np.outer(sc, sc)

    def _Q_of(self, xi):
        return self.V @ np.diag(self.lam * np.exp(np.clip(xi, -60, 60))) @ self.V.T

    def _R_of(self, eta):
        return np.diag(self.rho * np.exp(np.clip(eta, -60, 60)))

    def _balanced_base(self):
        """This member's base with every confounded pair's noise divided evenly.

        The split-agnostic point of the same base -- the same total in every channel, half of each
        confounded pair's share on each side of it.  A pair is a process eigenmode whose ``H v``
        lies along one sensor axis, which is a fact about ``H`` and needs no prior knowledge of
        the pairs; the rest of the test needs the Fisher this base is for.
        """
        lam, rho = self.lam.copy(), self.rho.copy()
        for k in range(self.n):
            hv = np.abs(self.HV[:, k])
            if hv.max() <= 0.0:
                continue
            nz = np.flatnonzero(hv > _RANK_TOL * hv.max())
            if nz.size != 1:
                continue
            i, h2 = int(nz[0]), float(self.HV[nz[0], k] ** 2)
            tot = lam[k] * h2 + rho[i]
            lam[k], rho[i] = 0.5 * tot / h2, 0.5 * tot
        return lam, rho

    # -- a confounded group's two coordinates: its contribution to S (identifiable per step)
    #    and its log-odds (in the exact null space of the per-step scale-Fisher) --
    def _group_read(self, mu):
        out = []
        for (k, i, h2) in self._groups:
            a = self.lam[k] * h2 * math.exp(min(mu[k], 60.0))
            b = self.rho[i] * math.exp(min(mu[self.n + i], 60.0))
            out.append((a + b, math.log(max(a, 1e-300)) - math.log(max(b, 1e-300))))
        return out

    def _group_write(self, mu, tots, los):
        """Put each group's total back with the given log-odds -- the exact null flow.

        The null direction is ``(R, -Q)`` up to scale at every operating point, and integrating
        that direction field gives ``da = -db``: the null manifold is the LEVEL SET OF THE TOTAL.
        So sliding the log-odds and handing the total straight back moves exactly along the
        coordinate the one-step likelihood cannot see, and touches nothing that it can.
        """
        out = mu.copy()
        for gi, (k, i, h2) in enumerate(self._groups):
            lo = float(np.clip(los[gi], -80.0, 80.0))
            a = tots[gi] / (1.0 + math.exp(-lo))
            b = tots[gi] - a
            out[k] = math.log(max(a, 1e-300) / (self.lam[k] * h2))
            out[self.n + i] = math.log(max(b, 1e-300) / self.rho[i])
        return out

    def reset(self, mean=None, scale=None):
        self._pi_ax = None
        self._m = None if mean is None else np.asarray(mean, float)
        self._P = None
        self.mu = np.zeros(self.D) if scale is None else np.asarray(scale, float).copy()
        self._Pmu = self.s_ax ** 2
        self.loglik = 0.0
        return self

    def reprice(self, idx):
        """A detected jump re-prices ignorance: variance back to the class cap on ``idx``,
        cross-covariances cleared (research 0003).  The estimate itself is kept -- the data
        moves it at cap gain.  A fault is ONE event, so this fires on EVERY departure axis,
        excited or not: an unexcited axis then stands at honest cap width instead of
        reporting a stale coefficient it has no evidence for."""
        if self._P is None or self._cap is None or len(idx) == 0:
            return
        self._P[idx, :] = 0.0
        self._P[:, idx] = 0.0
        self._P[idx, idx] = self._cap[idx]

    def update(self, y, u=None):
        n, m, H = self.n, self.m, self.H
        y = np.atleast_1d(np.asarray(y, dtype=float))
        r = len(self._act)
        Qg, Rg = self._star_QR()
        if self._pi_ax is None:
            self._pi_ax = self._w1[self._act].copy()
            if self._m is None:
                self._m = (np.linalg.lstsq(H, y, rcond=None)[0]
                           if np.all(np.isfinite(y)) else np.zeros(n))
            if self._P is None:
                self._P = self._cap_P(
                    np.eye(n) * float(Rg.reshape(self._G, -1).max()
                                      + Qg.reshape(self._G, -1).max()) * n)
        pi_ax = np.einsum("ai,aij->aj", self._pi_ax, self._T1[self._act])
        # The dynamics are a CALLABLE when they are learned or nonlinear: it returns the
        # linearisation (for the covariance) and the predicted mean (from f, not the
        # Jacobian -- they differ once the transition depends on the state).  Real F/B
        # arrive linearised per operating point, so this is the general path.
        if self._dyn is None:
            F = self.F
            mpred = F @ self._m + ((self.B @ u) if self.B is not None else 0.0)
        else:
            F, mpred = self._dyn(self._m, u)
        FPFt = F @ self._P @ F.T
        if not np.all(np.isfinite(y)):
            self._pi_ax = pi_ax
            w = self._star_weights(pi_ax)
            self._P = self._cap_P(FPFt + np.einsum("g,gij->ij", w, Qg))
            self._m = mpred
            wmean = self._wmean(pi_ax)
            return LucidStep(self._m.copy(), self._P.copy(), np.full(m, np.nan), 0.0,
                             self.mu[:n] + wmean[:n], self.mu[n:] + wmean[n:])
        Ppred = FPFt[None] + Qg
        # The measurement map is a CALLABLE whenever the sensors are not a fixed linear
        # functional of the state -- which is every inertial sensor on a moving linkage: what
        # a link-mounted gyro reads is the whole chain below it, through axes that rotate with
        # the state.  It returns the linearisation (for the covariance and for dS/dxi) and the
        # predicted measurement (from h, not the Jacobian -- they differ once h is nonlinear).
        H, ypred = self._H_at(mpred)
        HV = self.HV if self._hook is None else H @ self.V
        e = y - ypred
        PHt = np.einsum("gij,kj->gik", Ppred, H)
        S = np.einsum("ij,gjk->gik", H, PHt) + Rg
        Si = np.linalg.inv(S)
        sgn, logdet = np.linalg.slogdet(S)
        maha = np.einsum("i,gij,j->g", e, Si, e)
        lg = -0.5 * (m * _LOG2PI + logdet + maha)
        if r:
            logZ = np.empty(r)
            for i, k in enumerate(self._act):
                lgi = lg[self._axwin[k]]
                mi = float(lgi.max())
                wk = pi_ax[i] * np.exp(lgi - mi)
                Zi = float(wk.sum())
                pi_ax[i] = wk / Zi
                logZ[i] = mi + math.log(Zi)
            mz = float(logZ.max())
            aw = np.exp(logZ - mz)
            ll = mz + math.log(float(aw.mean()))
            alpha = aw / float(aw.sum())
        else:
            ll = float(lg[0])
            alpha = None
        pi = self._star_weights(pi_ax, alpha)
        K = np.einsum("gik,gkl->gil", PHt, Si)
        Kbar = np.einsum("g,gil->il", pi, K)
        m_new = mpred + Kbar @ e
        mpost = mpred[None] + np.einsum("gil,l->gi", K, e)
        dm = mpost - m_new
        HPp = np.einsum("ij,gjk->gik", H, Ppred)
        Ppost = Ppred - np.einsum("gil,glk->gik", K, HPp)   # K(H Ppred): n^2 m per node, not n^3
        P_new = np.einsum("g,gij->ij", pi, Ppost) + np.einsum("g,gi,gj->ij", pi, dm, dm)
        P_new = self._cap_P(0.5 * (P_new + P_new.T))
        # finding-18 walk per active axis, score/Fisher averaged over that axis's window
        # posterior only (the caltrop: dS_k depends only on the k-coordinate, so the axial
        # profile carries the axis's evidence at linear cost).
        for i, k in enumerate(self._act):
            idx = self._axwin[k]
            dpk = self._dS_axis(k, HV)
            Sik = Si[idx]
            Sie = np.einsum("gij,j->gi", Sik, e)
            score_g = 0.5 * (np.einsum("gi,gij,gj->g", Sie, dpk, Sie)
                             - np.einsum("gij,gji->g", Sik, dpk))
            SidS = np.einsum("gij,gjk->gik", Sik, dpk)
            info_g = 0.5 * np.einsum("gij,gji->g", SidS, SidS)
            info = float(pi_ax[i] @ info_g) + _RIDGE
            grad = float(pi_ax[i] @ score_g)
            K_mu = self._Pmu[k] / (self._Pmu[k] + 1.0 / info)
            self.mu[k] += float(np.clip(K_mu * (grad / info), -self.gap[k], self.gap[k]))
            self._Pmu[k] = min((1.0 - K_mu) * self._Pmu[k] + self._qmu[k], self._Pmu_cap[k])
        self._pi_ax, self._m, self._P = pi_ax, m_new, P_new
        if self._groups and self._revert is not None:
            # The per-axis Newton walk steps by ``score/info``, which is ~1/Q on a process axis
            # and ~1/R on a sensor axis; where the two are confounded, that step is almost
            # entirely along the NULL direction, in which the score carries no information at
            # all.  It is an artefact of taking a per-axis step against a singular Fisher, and it
            # systematically blames the smaller variance -- the wrong reflex when a sensor
            # degrades.  The excursion is allowed, because it is what absorbs a level jump, but
            # it is a TRANSIENT and not a verdict: it reverts to this member's hypothesis at the
            # class's own rate ``phi``, at the total the walk just established.  The verdict is
            # the bank's, on the ``forget`` timescale (research 0053's lesson b).
            tots, los = zip(*self._group_read(self.mu))
            back = [a + self._revert * (lo - a) for a, lo in zip(self._anchor_lo, los)]
            self.mu = self._group_write(self.mu, list(tots), back)
        self.loglik += ll
        wmean = self._wmean(pi_ax)
        return LucidStep(m_new.copy(), P_new.copy(), e.copy(), ll,
                         self.mu[:n] + wmean[:n], self.mu[n:] + wmean[n:])


# ------------------------------------------------- stacked execution of the bank
def _bank_key(f):
    """Members that can run as one stacked recursion: same shapes, same index tables, same
    model objects -- they differ only in parameters and state."""
    return (type(f).update is _WalkEngine.update, f.n, f.m, f.D, f._G,
            tuple(f._act), tuple((k, i) for k, i, _ in f._groups),
            f._dyn is None, id(f.H), id(f.F), f._hook is None,
            None if f._cap is None else id(f._cap), f._revert is None)


class _LoopBank:
    """Fallback executor: members whose class overrides ``update`` (research subclasses) run
    looped, exactly as before.  Same stacked outputs as `_EngineBank`, so the mixing above is
    one code path."""

    def __init__(self, members):
        self.members = members

    def update(self, y, u=None):
        st = [f.update(y, u=u) for f in self.members]
        return (np.stack([t.mean for t in st]), np.stack([t.var for t in st]),
                np.stack([t.innovation for t in st]), np.array([t.loglik for t in st]),
                np.stack([t.process_scale for t in st]),
                np.stack([t.measurement_scale for t in st]))


class _EngineBank:
    """Stacked execution of ``_WalkEngine.update`` over structurally identical members.

    The split ladder multiplies the member count, and almost all of a member's per-step cost is
    numpy dispatch on tiny arrays, not arithmetic (measured on the scalar hero rig: 360 members,
    135 ms/step, ~140k interpreter/numpy calls per step).  Members of one dynamics spec share
    every shape and every index table -- they differ only in parameters and state -- so the same
    recursion runs ONCE with a leading member axis.  Identical math step for step, pinned by
    ``test_bank_matches_the_looped_members``; each member's state arrays are views into the
    stack (all bank updates are in-place), so everything a member exposes -- ``mu``, ``_P``,
    ``reprice`` -- stays live for the dynamics channel and for research instrumentation.
    """

    def __init__(self, members):
        e0 = members[0]
        self.members = members
        M = len(members)
        self.M, self.n, self.m, self.D = M, e0.n, e0.m, e0.D
        self._nn, self._c, self._G = e0._nn, e0._c, e0._G
        self._act, self._axwin = list(e0._act), e0._axwin
        self.H, self.B, self.F = e0.H, e0.B, e0.F
        self._dyns = None if e0._dyn is None else [f._dyn for f in members]
        self._hooks = None if e0._hook is None else [f._hook for f in members]
        self._cap = e0._cap
        self._groups = e0._groups
        st = lambda name: np.ascontiguousarray(np.stack([getattr(f, name) for f in members]))
        self.V, self.lam, self.rho, self.HV = st("V"), st("lam"), st("rho"), st("HV")
        self.gap, self._qmu, self._Pmu_cap = st("gap"), st("_qmu"), st("_Pmu_cap")
        self._off, self._w1 = st("_off"), st("_w1")
        self._T1a = np.ascontiguousarray(np.stack([f._T1[f._act] for f in members]))
        self._star_off = st("_star_off")
        sa = e0._star_axis
        self._gp = np.flatnonzero((sa >= 0) & (sa < self.n))
        self._kp = sa[self._gp]
        self._gs = np.flatnonzero(sa >= self.n)
        self._is = sa[self._gs] - self.n
        self._Vout = np.einsum("bik,bjk->bkij", self.V, self.V)
        self._Hout = np.einsum("bik,bjk->bkij", self.HV, self.HV)
        ng = len(self._groups)
        self._h2 = (np.stack([[h for _, _, h in f._groups] for f in members])
                    if ng else np.zeros((M, 0)))
        self._anchor = st("_anchor_lo") if ng else np.zeros((M, 0))
        self._revert = np.array([f._revert for f in members], float)
        # state, stacked -- and handed back to the members as views
        self.mu, self._Pmu = st("mu"), st("_Pmu")
        self._m = np.zeros((M, self.n))
        self._P = np.zeros((M, self.n, self.n))
        self._pi = np.zeros((M, len(self._act), self._nn))
        self._ll = np.zeros(M)
        self._fresh = True
        for j, f in enumerate(members):
            f.mu, f._Pmu = self.mu[j], self._Pmu[j]
            f._m, f._P, f._pi_ax = self._m[j], self._P[j], self._pi[j]

    def _cap_P(self, P):
        if self._cap is None:
            return P
        d = np.einsum("bii->bi", P)
        sc = np.ones_like(d)
        cap = np.broadcast_to(self._cap, d.shape)
        over = d > cap
        sc[over] = np.sqrt(cap[over] / np.maximum(d[over], 1e-300))
        P *= sc[:, :, None] * sc[:, None, :]
        return P

    def _star_QR(self):
        n, M, G = self.n, self.M, self._G
        le = self.lam * np.exp(np.clip(self.mu[:, :n], -60, 60))
        Qc = np.einsum("bk,bkij->bij", le, self._Vout)
        rc = self.rho * np.exp(np.clip(self.mu[:, n:], -60, 60))
        Qg = np.repeat(Qc[:, None], G, 1)
        rg = np.repeat(rc[:, None], G, 1)
        if self._gp.size:
            kp = self._kp
            mu_k = self.mu[:, kp]
            dlam = self.lam[:, kp] * (
                np.exp(np.minimum(mu_k + self._star_off[:, self._gp], 60.0))
                - np.exp(np.minimum(mu_k, 60.0)))
            Qg[:, self._gp] += dlam[:, :, None, None] * self._Vout[:, kp]
        if self._gs.size:
            rg[:, self._gs, self._is] = self.rho[:, self._is] * np.exp(
                np.minimum(self.mu[:, n + self._is] + self._star_off[:, self._gs], 60.0))
        Rg = np.zeros((M, G, self.m, self.m))
        Rg[:, :, np.arange(self.m), np.arange(self.m)] = rg
        return Qg, Rg

    def _star_w(self, pi, alpha):
        r = len(self._act)
        w = np.zeros((self.M, self._G))
        if r == 0:
            w[:, 0] = 1.0
            return w
        a = np.full((self.M, r), 1.0 / r) if alpha is None else alpha
        for i, k in enumerate(self._act):
            w[:, self._axwin[k]] += a[:, i, None] * pi[:, i]
        return w

    def _wmean(self, pi):
        w = np.zeros((self.M, self.D))
        for i, k in enumerate(self._act):
            w[:, k] = np.einsum("bn,bn->b", pi[:, i], self._off[:, k])
        return w

    def _dS_axis(self, k, Hout=None):
        a = np.exp(np.minimum(self.mu[:, k, None] + self._off[:, k], 60.0))
        if k < self.n:
            Ho = self._Hout if Hout is None else Hout
            return (self.lam[:, k, None] * a)[:, :, None, None] * Ho[:, k, None]
        out = np.zeros((self.M, self._nn, self.m, self.m))
        i = k - self.n
        out[:, :, i, i] = self.rho[:, i, None] * a
        return out

    def _revert_groups(self):
        n = self.n
        for gi, (k, i, _h) in enumerate(self._groups):
            h2 = self._h2[:, gi]
            a = self.lam[:, k] * h2 * np.exp(np.minimum(self.mu[:, k], 60.0))
            b = self.rho[:, i] * np.exp(np.minimum(self.mu[:, n + i], 60.0))
            tot = a + b
            lo = np.log(np.maximum(a, 1e-300)) - np.log(np.maximum(b, 1e-300))
            anc = self._anchor[:, gi]
            back = np.clip(anc + self._revert * (lo - anc), -80.0, 80.0)
            a2 = tot / (1.0 + np.exp(-back))
            b2 = tot - a2
            self.mu[:, k] = np.log(np.maximum(a2, 1e-300) / (self.lam[:, k] * h2))
            self.mu[:, n + i] = np.log(np.maximum(b2, 1e-300) / self.rho[:, i])

    def update(self, y, u=None):
        M, n, m, H = self.M, self.n, self.m, self.H
        y = np.atleast_1d(np.asarray(y, dtype=float))
        ok = bool(np.all(np.isfinite(y)))
        r = len(self._act)
        Qg, Rg = self._star_QR()
        if self._fresh:
            self._pi[:] = self._w1[:, self._act]
            self._m[:] = (np.linalg.lstsq(H, y, rcond=None)[0] if ok else np.zeros(n))
            scal = (Rg.reshape(M, -1).max(1) + Qg.reshape(M, -1).max(1)) * n
            self._P[:] = np.eye(n)[None] * scal[:, None, None]
            self._cap_P(self._P)
            self._fresh = False
        pi = np.einsum("bai,baij->baj", self._pi, self._T1a)
        if self._dyns is None:
            F = np.broadcast_to(self.F, (M, n, n))
            mpred = self._m @ self.F.T + ((self.B @ u) if self.B is not None else 0.0)
        else:
            F = np.empty((M, n, n)); mpred = np.empty((M, n))
            for j in range(M):
                F[j], mpred[j] = self._dyns[j](self._m[j], u)
        FPFt = np.einsum("bij,bjk,blk->bil", F, self._P, F)
        if not ok:
            self._pi[:] = pi
            w = self._star_w(pi, None)
            self._P[:] = FPFt + np.einsum("bg,bgij->bij", w, Qg)
            self._cap_P(self._P)
            self._m[:] = mpred
            sc = self.mu + self._wmean(pi)
            return (self._m.copy(), self._P.copy(), np.full((M, m), np.nan),
                    np.zeros(M), sc[:, :n], sc[:, n:])
        Ppred = FPFt[:, None] + Qg
        # The live measurement map, per member (each evaluates its own mean).  With no hook
        # this is exactly the shared-H arithmetic it always was, contraction order included --
        # a state-dependent H costs nothing where there is none.
        if self._hooks is None:
            Hb, Hout = None, None
            e = y[None] - mpred @ H.T
            PHt = np.einsum("bgij,kj->bgik", Ppred, H)
            S = np.einsum("ij,bgjk->bgik", H, PHt) + Rg
        else:
            Hb = np.empty((M, m, n)); yp = np.empty((M, m))
            for j in range(M):
                Hb[j], yp[j] = self.members[j]._H_at(mpred[j])
            HbV = np.einsum("bij,bjk->bik", Hb, self.V)
            Hout = np.einsum("bik,bjk->bkij", HbV, HbV)
            e = y[None] - yp
            PHt = np.einsum("bgij,bkj->bgik", Ppred, Hb)
            S = np.einsum("bij,bgjk->bgik", Hb, PHt) + Rg
        Si = np.linalg.inv(S)
        _, logdet = np.linalg.slogdet(S)
        maha = np.einsum("bi,bgij,bj->bg", e, Si, e)
        lg = -0.5 * (m * _LOG2PI + logdet + maha)
        if r:
            logZ = np.empty((M, r))
            for a, k in enumerate(self._act):
                lgi = lg[:, self._axwin[k]]
                mi = lgi.max(1)
                wk = pi[:, a] * np.exp(lgi - mi[:, None])
                Zi = wk.sum(1)
                pi[:, a] = wk / Zi[:, None]
                logZ[:, a] = mi + np.log(Zi)
            mz = logZ.max(1)
            aw = np.exp(logZ - mz[:, None])
            ll = mz + np.log(aw.mean(1))
            alpha = aw / aw.sum(1, keepdims=True)
        else:
            ll = lg[:, 0].copy()
            alpha = None
        w = self._star_w(pi, alpha)
        K = np.einsum("bgik,bgkl->bgil", PHt, Si)
        Kbar = np.einsum("bg,bgil->bil", w, K)
        m_new = mpred + np.einsum("bil,bl->bi", Kbar, e)
        mpost = mpred[:, None] + np.einsum("bgil,bl->bgi", K, e)
        dm = mpost - m_new[:, None]
        HPp = (np.einsum("ij,bgjk->bgik", H, Ppred) if Hb is None
               else np.einsum("bij,bgjk->bgik", Hb, Ppred))
        Ppost = Ppred - np.einsum("bgil,bglk->bgik", K, HPp)
        P_new = (np.einsum("bg,bgij->bij", w, Ppost)
                 + np.einsum("bg,bgi,bgj->bij", w, dm, dm))
        P_new = self._cap_P(0.5 * (P_new + np.swapaxes(P_new, 1, 2)))
        for a, k in enumerate(self._act):
            idx = self._axwin[k]
            dpk = self._dS_axis(k, Hout)
            Sik = Si[:, idx]
            Sie = np.einsum("bgij,bj->bgi", Sik, e)
            score = 0.5 * (np.einsum("bgi,bgij,bgj->bg", Sie, dpk, Sie)
                           - np.einsum("bgij,bgji->bg", Sik, dpk))
            SidS = np.einsum("bgij,bgjk->bgik", Sik, dpk)
            info_g = 0.5 * np.einsum("bgij,bgji->bg", SidS, SidS)
            info = np.einsum("bg,bg->b", pi[:, a], info_g) + _RIDGE
            grad = np.einsum("bg,bg->b", pi[:, a], score)
            Kmu = self._Pmu[:, k] / (self._Pmu[:, k] + 1.0 / info)
            self.mu[:, k] += np.clip(Kmu * (grad / info), -self.gap[:, k], self.gap[:, k])
            self._Pmu[:, k] = np.minimum((1.0 - Kmu) * self._Pmu[:, k] + self._qmu[:, k],
                                         self._Pmu_cap[:, k])
        self._pi[:] = pi
        self._m[:] = m_new
        self._P[:] = P_new
        if self._groups:
            self._revert_groups()
        self._ll += ll
        sc = self.mu + self._wmean(pi)
        return (m_new, P_new, e, ll, sc[:, :n], sc[:, n:])


# ------------------------------------------------------- the dynamics channel
class _Departure:
    """One learned-dynamics hypothesis: ``F(g) = F0 + sum_j g_j A_j``, ``B(g) = B0 + sum_j g_j C_j``.

    Realised as a STATE AUGMENTATION.  With the augmented state ``(x, g)``, the augmented
    observation ``[H | 0]`` and the augmented transition

        [[ F(g),  d(F(g)x + B(g)u)/dg ],
         [ 0,     I                   ]]

    the departure channel is just more state, so the whole noise machinery above (the caltrop
    scale walk, the structural activation, the bank) runs on top of it unchanged -- which is what
    the record demands: the Q<->F confound is split by per-hypothesis MEANS competing under a
    LIVE noise walk (research 0002), not by a whiteness statistic bolted on the side.

    Three commitments, all from the fault class, none tuned:

    * ``g``'s drift is ``q_g = sigma^2 rho`` (a jump of class size ``sigma`` at hazard ``rho``
      has that mean square per step) and its variance is capped at ``sigma^2`` -- **bounded,
      never frozen**: the gain stays live so an axis the data cannot see today still moves when
      excitation arrives (research 0003 measured the latched-freeze alternative at 20x worse).
    * ``g``'s scale axes are excluded from the noise walk: their drift is a class commitment,
      not a live noise scale.  (Structural, per 0024 -- ``[H | 0]`` cannot see ``g`` directly,
      so the engine's own activation rule already reaches this conclusion; the mask makes it
      exact under eigenvalue degeneracy.)
    * ``sigma = 1`` in CLASS UNITS, where a direction is scaled so that a unit coefficient
      moves that part of the dynamics by about its own magnitude.  That is the only scale-free
      statement of "how big is a fault", and the only dimensionally sound one -- see the
      comment in ``__init__``.
    """

    def __init__(self, base, dirs, rho, n, p):
        self.base, self.n, self.p = base, n, p
        # A direction may itself be a callable of the state: on a real vehicle the direction a
        # physical parameter pushes in ROTATES with the operating point (a wheel radius acts
        # along the heading; a drone's mass acts along the tilted thrust axis).  Such a
        # direction is scaled ONCE, at the origin, so its coefficient keeps a fixed meaning
        # rather than drifting with the state.
        self._dirs = [d if callable(d) else (lambda x, _d=d: _d) for d in dirs]
        self.moving = any(callable(d) for d in dirs)
        self.k = len(self._dirs)
        ref = [self._pair(f(np.zeros(n)), n, p) for f in self._dirs]
        # CLASS SIZE.  A unit coefficient must mean "this part of the dynamics changed by
        # about its own magnitude" -- the only scale-free statement available, and the only
        # one that is dimensionally sound: F's entries are O(1) for a stable discrete
        # transition, but B's are in the input's units and can be arbitrarily small (a 50 Hz
        # differential drive has |B| ~ 5e-3).  Normalising directions to unit Frobenius norm
        # instead would put the true coefficient orders of magnitude inside the prior and the
        # departure would never move.
        F_rep, B_rep = base(np.zeros(n))
        fn = float(np.linalg.norm(F_rep)) or 1.0
        bn = float(np.linalg.norm(B_rep)) if B_rep is not None else 0.0
        bn = bn or fn
        div = []
        for a, c in ref:
            aN, cN = float(np.linalg.norm(a)), float(np.linalg.norm(c))
            dn = math.hypot(aN, cN)
            if dn == 0.0:
                raise ValueError("a departure direction must be non-zero")
            div.append(dn * dn / math.hypot(aN * fn, cN * bn))
        self._div = np.array(div)
        self.A = np.stack([a for a, _ in ref]) / self._div[:, None, None]
        self.C = np.stack([c for _, c in ref]) / self._div[:, None, None]
        self.na = n + self.k
        sig2 = np.ones(self.k)                      # directions are in class units, so 1
        self.cap = np.concatenate([np.full(n, np.inf), sig2])
        self.q_g = sig2 * rho
        self.gidx = np.arange(n, self.na)

    @staticmethod
    def _pair(d, n, p):
        A, C = d if isinstance(d, tuple) else (d, None)
        A = np.zeros((n, n)) if A is None else np.atleast_2d(np.asarray(A, float))
        C = (np.zeros((n, max(p, 1))) if C is None
             else np.atleast_2d(np.asarray(C, float)))
        return A, C

    def at(self, x):
        """The departure directions at the operating point ``x``."""
        if not self.moving:
            return self.A, self.C
        ref = [self._pair(f(x), self.n, self.p) for f in self._dirs]
        return (np.stack([a for a, _ in ref]) / self._div[:, None, None],
                np.stack([c for _, c in ref]) / self._div[:, None, None])

    def augment(self, Q0, R0, H):
        """(Q0_aug, R0, H_aug, walk_axes) -- walk_axes in EIGENMODE space, so it is exact even
        if a departure's drift coincides with a process eigenvalue."""
        Qa = np.zeros((self.na, self.na))
        Qa[:self.n, :self.n] = Q0
        Qa[self.gidx, self.gidx] = self.q_g
        Ha = np.zeros((H.shape[0], self.na))
        Ha[:, :self.n] = H
        _, V = np.linalg.eigh(Qa)                   # the engine repeats this deterministically
        walk = np.ones(self.na + H.shape[0], bool)
        walk[:self.na] = np.linalg.norm(V[self.n:], axis=0) < 0.5
        return Qa, R0, Ha, walk

    def dynamics_of(self, x, g):
        """The dynamics as currently believed, at the operating point ``x``."""
        F0, B0 = self.base(x)
        if not self.k:
            return F0, B0
        A, C = self.at(x)
        F = F0 + np.einsum("j,jab->ab", g, A)
        return F, (None if B0 is None else B0 + np.einsum("j,jab->ab", g, C))

    def callable_for(self):
        """The (mean, u) -> (Jacobian, predicted mean) hook the engine calls each step."""
        n, k = self.n, self.k

        def _dyn(m, u):
            x, g = m[:n], m[n:]
            F, B = self.dynamics_of(x, g)
            mp = np.empty_like(m)
            mp[:n] = F @ x + (B @ u if B is not None else 0.0)
            mp[n:] = g
            J = np.zeros((self.na, self.na))
            J[:n, :n] = F
            J[n:, n:] = np.eye(k)
            if k:
                A, C = self.at(x)
                J[:n, n:] = np.einsum("jab,b->aj", A, x)
                if B is not None:
                    J[:n, n:] += np.einsum("jab,b->aj", C, np.atleast_1d(u))
            return J, mp

        return _dyn


def _as_base(dynamics, B, n):
    """Normalise supplied dynamics to ``base(x) -> (F, B)``, plus a constant linearisation.

    A callable is the general robotics case: real ``F``/``B`` arrive linearised per operating
    point, so the dynamics cannot be a constant matrix and the block structure cannot be
    precomputed.  It is called with the current state estimate and may return ``F`` or
    ``(F, B)``.  The returned constant is the linearisation at the origin -- the characteristic
    transition the steady-state Fisher scale is computed from, not a model commitment.
    """
    def base(x):
        out = dynamics(x)
        F_, B_ = out if isinstance(out, tuple) else (out, B)
        return (np.atleast_2d(np.asarray(F_, float)),
                None if B_ is None else np.atleast_2d(np.asarray(B_, float)))
    return base, base(np.zeros(n))[0]


def _as_hbase(H):
    """Normalise a callable measurement map to ``h(x) -> (Jacobian, predicted measurement)``.

    Accepts either return shape: a bare Jacobian, for a map that is linear with
    state-dependent coefficients (``h(x) = H(x) x`` -- a link-mounted gyro reading a chain of
    joint rates through rotating axes), or an ``(H, y)`` pair when ``h`` is genuinely
    nonlinear and the predicted measurement is not the Jacobian applied to the mean (the same
    distinction `_as_base` draws for the transition).
    """
    def hbase(x):
        out = H(x)
        if isinstance(out, tuple):
            Hj, yp = out
            return np.atleast_2d(np.asarray(Hj, float)), np.asarray(yp, float)
        Hj = np.atleast_2d(np.asarray(out, float))
        return Hj, Hj @ np.asarray(x, float)
    return hbase


def _augment_hook(hbase, n, k):
    """The same measurement map on a departure walker's augmented state ``(x, g)``.

    ``g`` is not measured, so the augmented Jacobian is ``[H(x) | 0]`` -- and the map is
    linearised at the walker's own ``x``, not at the caller's, which is the whole point of
    carrying it per member.
    """
    def hook(ma):
        Hj, yp = hbase(ma[:n])
        return np.hstack([Hj, np.zeros((Hj.shape[0], k))]), yp
    return hook


def _fixed_hook(base):
    """Engine hook for supplied-but-moving dynamics (no departure channel)."""
    def _dyn(m, u):
        F, B = base(m)
        return F, F @ m + (B @ u if B is not None else 0.0)
    return _dyn


def _basis(n, p, departures):
    """Departure directions as (A, C) pairs, unit-Frobenius.

    Default (research mechanism (c), the full walk): every elementary entry of F, and of B when
    a control map is supplied.  Supplying ``departures`` is mechanism (b) -- the low-rank
    physical channel (a drone's added mass moves F AND B through one coefficient), which is the
    production form and is `n**2 / len(departures)` times cheaper.  Directions are returned raw;
    ``_Departure`` puts them in class units.
    """
    if departures is None:
        out = []
        for i in range(n):
            for j in range(n):
                A = np.zeros((n, n)); A[i, j] = 1.0
                out.append((A, np.zeros((n, max(p, 1)))))
        for i in range(n):
            for j in range(p):
                C = np.zeros((n, p)); C[i, j] = 1.0
                out.append((np.zeros((n, n)), C))
        return out
    out = []
    for d in departures:
        if callable(d):
            out.append(d)                       # a state-dependent direction, scaled in
            continue                            # _Departure by its norm at the origin
        A, C = d if isinstance(d, tuple) else (d, None)
        A = np.atleast_2d(np.asarray(A, float))
        if A.shape != (n, n):
            raise ValueError(f"each departure direction must be ({n}, {n})")
        C = (np.zeros((n, max(p, 1))) if C is None
             else np.atleast_2d(np.asarray(C, float)))
        if p and C.shape != (n, p):
            raise ValueError(f"each departure control direction must be ({n}, {p})")
        out.append((A, C))
    return out


# ------------------------------------------------------------- the public filter
class LucidFilter:
    """The single public filter: supply dynamics + observation, it infers all the noise online.

    Parameters (each: explicit value, or a null/default for "none", or the give-what-you-know rule)
    ----------
    dynamics : (n, n) array, 0, or None
        The linear state dynamics ``F``.  ``0`` (default) means no dynamics -> a random-walk level.
        ``None`` means **learn them**: the prior is the random walk (``F = I``) and the filter
        recovers ``F`` (and ``B``) online.  See *The dynamics channel* below.
    faults : float or True, optional
        Turn on the dynamics channel around a SUPPLIED ``F``: the value is the hazard ``rho``,
        the per-step probability that the dynamics change (``True`` -> ``1e-4``, about one fault
        per 10,000 steps).  Implied by ``dynamics=None``.  This is a labeled prior of the same
        standing as ``forget``, not a tuning constant -- it is the operating point on the
        false-alarm/delay frontier, and the detection delay it buys is derived,
        ``log(1/rho) / KL-rate``.
    departures : sequence, optional
        The directions the dynamics may move along: each an ``(n, n)`` matrix, or an
        ``(A, C)`` pair when the same physical parameter moves ``F`` and ``B`` together (a
        drone's added mass does).  Default: every entry of ``F`` (and of ``B``) -- fully general
        and ``n**2`` coefficients.  Supplying the physical directions is the production form and
        is far cheaper.
    anchors : sequence, optional
        Named fault hypotheses -- each an ``(n, n)`` ``F`` or an ``(F, B)`` pair -- carried as
        their own full filters.  When the failure modes are nameable (blown-left, blown-right,
        payload-attached) this is the fastest detector there is; the walker still refines
        whatever the anchors only bracket.
    control : (n, p) array, optional
        Forcing/bias map ``B`` for a known input ``u`` (the ODE bias).  If given, ``update``/``filter``
        require ``u``/``U``.
    H : (m, n) array or callable, optional
        Measurement matrix.  Defaults to the identity.  A **callable** of the state is the
        general sensing case, and the one every inertial sensor on a moving linkage needs:
        what a link-mounted gyro reads is the whole chain below it, through axes that rotate
        with the state, so ``H`` has to be linearised at each step exactly as ``F`` does.  It
        is called with the predicted mean and returns either the Jacobian, or an
        ``(H, y_predicted)`` pair when the map is genuinely nonlinear (a rate-squared term,
        say) and ``h(x)`` is not ``H(x) x``.  Then ``n`` cannot be read off ``H``, so supply
        it through ``dynamics``, ``process`` or ``n``.
    process, measurement : arrays, optional
        Base noise magnitudes ``Q0`` (n, n, PD) and ``R0`` (m, diagonal).  Default to identity/unit --
        the walk breathes around them with unbounded reach, so a rough base is fine.
    n : int, optional
        State dimension, when it cannot be inferred from ``dynamics``/``process``/``H`` (default 1).
    phis, ss : sequences, optional  
        The ``(phi, s)`` box the bank averages over.  Defaults to a broad dead-zone-free range;
        not a fitted value.  ``forget`` (default 0.999) is the weight memory -- the one residual.
    """

    def __init__(self, dynamics=0, control=None, H=None, process=None, measurement=None,
                 n=None, faults=None, departures=None, anchors=None,
                 phis=_PHIS, ss=_SS, forget=0.999):
        learn = dynamics is None or faults is not None
        if faults is None or faults is True:
            rho = _HAZARD
        else:
            rho = float(faults)
            if not 0.0 < rho < 1.0:
                raise ValueError("faults (the hazard rho) must lie in (0, 1)")
        if anchors is not None and not learn:
            learn = True                        # named fault hypotheses imply a fault class
        h_moving = callable(H)
        hbase = _as_hbase(H) if h_moving else None
        H = None if (H is None or h_moving) else np.atleast_2d(np.asarray(H, float))
        B = None if control is None else np.atleast_2d(np.asarray(control, float))
        moving = callable(dynamics)
        Fm = (None if (moving or np.ndim(dynamics) == 0)
              else np.atleast_2d(np.asarray(dynamics, float)))
        proc = None if process is None else np.atleast_2d(np.asarray(process, float))
        # resolve n
        if Fm is not None:
            n = Fm.shape[0]
        elif proc is not None:
            n = proc.shape[0]
        elif H is not None:
            n = H.shape[1]
        elif B is not None:
            n = B.shape[0]
        elif n is None:
            n = 1
        if moving:
            base, F = _as_base(dynamics, B, n)
        else:
            F = np.eye(n) if Fm is None else Fm
            base = None
        if F.shape != (n, n):
            raise ValueError(f"dynamics must be ({n}, {n})"
                             + (" -- the callable returned the wrong shape" if moving else ""))
        # The characteristic linearisation of a moving H, at the origin -- the same convention
        # `_as_base` uses for a moving F.  Everything structural is answered from it.
        Hm = hbase(np.zeros(n))[0] if h_moving else (np.eye(n) if H is None else H)
        m = Hm.shape[0]
        if Hm.shape != (m, n):
            raise ValueError(f"H must be (m, {n})"
                             + (" -- the callable returned the wrong shape" if h_moving else ""))
        Q0 = np.eye(n) if proc is None else proc
        if Q0.shape != (n, n) or not np.allclose(Q0, Q0.T, atol=1e-10):
            raise ValueError("process must be a square symmetric (n, n) matrix")
        if measurement is None:
            R0 = np.ones(m)
        else:
            R0 = np.asarray(measurement, float)
            R0 = np.diag(R0) if R0.ndim == 2 else R0
        if R0.size != m or np.any(R0 <= 0):
            raise ValueError(f"measurement must have {m} positive entries")
        if B is not None and B.shape[0] != n:
            raise ValueError(f"control must have {n} rows")
        if not 0.0 < forget <= 1.0:
            raise ValueError("forget must lie in (0, 1]")

        self.n, self.m, self.D = n, m, n + m
        self.p = 0 if B is None else B.shape[1]
        self.B = B
        self.forget = float(forget)
        # Which directions can no per-step score ever carry?  Where a process eigenmode is read
        # by exactly one sensor, the two scale derivatives are proportional as matrices and only
        # their SUM is identifiable per step (research/sequence-demix 0001), so the split is
        # carried as a dimension of the BANK: every member is a complete filter anchored at one
        # rung of the ladder, the evidence reaches it through its own MEAN (a rung with too much
        # process chases sensor noise and pays for it in its own predictive likelihood), and its
        # weight accumulates on the `forget` timescale.  No EMA, no whiteness statistic, and no
        # rung refers to the supplied base.  A structure with no such pair -- any rig where every
        # process mode is read by more than one sensor -- gets no ladder and no extra cost.
        #
        # The ladder is a third bank dimension, INSIDE the cell index, so the dynamics channel's
        # own reshape(n_dynamics, n_cells) and its hazard kernel are untouched: a fault is a
        # change of dynamics, a split is a property of the noise, and mixing one does not mix the
        # other.  It is crossed into the hypotheses that share the caller's own state space --
        # the nominal dynamics and any named anchors; the departure WALKERS carry an augmented
        # state whose axes are not the caller's, and are left unladdered (see the SUMMARY opens).
        #
        # Several pairs would make the joint set of split vectors exponential, so the ladder is
        # enumerated the way this filter enumerates every other scale grid: as the CALTROP STAR
        # (research 0013) -- one reference vector, plus one pair moved off it at a time.  That is
        # ``1 + G (J - 1)`` vectors, LINEAR in the pair count, and with ONE pair it is the whole
        # ladder exactly, and with none it is a single empty vector and the bank is what it was.
        # The reference is the middle of the arclength grid, the agnostic split; no rung refers to
        # the supplied base.
        #
        # A split vector is applied to the BASE `(Q0, R0)` a member is built from, not carried
        # beside it.  That is what it is -- a division of a channel's noise between the process
        # and the sensor at a fixed total -- and it keeps the bank uniform: a member reads its own
        # anchor off its own base, so the same construction reaches the augmented state of a
        # departure walker through `augment` without anything having to know it is one.
        probe = _WalkEngine(Q0, R0, Hm, F, B, phis[0], ss[0])
        self.groups = probe._groups
        self.split_arr = _split_star(np.log(_rung_odds(self.forget)), len(self.groups))
        bases = [_apply_split(probe, v) for v in self.split_arr]
        self.phi_arr = np.array([ph for ph in phis for _ in ss for _ in bases], float)
        self.s_arr = np.array([sv for _ in phis for sv in ss for _ in bases], float)
        cells = [(ph, sv, bq, br) for ph in phis for sv in ss for (bq, br) in bases]

        # -------- the dynamics hypotheses (the ladder of research/dynamics-learning) --------
        # The NOMINAL member is always present and never leaves: it is the hedge that makes a
        # false detection cost ~nothing, which is in turn what makes the fast end of the
        # detection frontier affordable (research 0001).  Named ANCHORS come next -- when the
        # failure modes can be named, they are the fastest detector there is (0005).  The
        # WALKER refines whatever the anchors only bracket: detection degrades gracefully under
        # a mis-specified anchor set, recovery does not (0001 s4).
        const = base if base is not None else (lambda x, _F=F, _B=B: (_F, _B))
        specs = [(base, F, B, None)]
        for a in (anchors or []):
            Fa, Ba = a if isinstance(a, tuple) else (a, B)
            Fa = np.atleast_2d(np.asarray(Fa, float))
            if Fa.shape != (n, n):
                raise ValueError(f"each anchor must be ({n}, {n})")
            specs.append((None, Fa,
                          None if Ba is None else np.atleast_2d(np.asarray(Ba, float)), None))
        if learn:
            specs.append((base, F, B,
                          _Departure(const, _basis(n, self.p, departures),
                                     rho, n, self.p)))

        self._members, self._pidx, self._specs = [], [], specs
        for bs, Fs, Bs, dep in specs:
            if dep is None:
                Si_c = probe._fisher_Si if Fs is F else None
                eng = []
                for ph, sv, bq, br in cells:
                    e = _WalkEngine(bq, br, Hm, Fs, Bs, ph, sv, fisher_Si=Si_c)
                    Si_c = e._fisher_Si
                    eng.append(e)
                if bs is not None:
                    for e in eng:
                        e._dyn = _fixed_hook(bs)
                if hbase is not None:
                    for e in eng:
                        e._hook = hbase
                self._members += eng
                self._pidx += [np.arange(n)] * len(cells)
                continue
            _, _, Ha, walk = dep.augment(Q0, R0, Hm)   # the shape; the bases follow per cell
            xmode = np.flatnonzero(walk[:dep.na])
            if xmode.size != n:
                raise RuntimeError("augmented process eigenbasis did not separate")
            Fa = np.eye(dep.na)
            Fa[:n, :n] = Fs                     # the Jacobian at g = 0, x = 0: the
            Ba = (None if Bs is None                     # characteristic linearisation the
                  else np.vstack([Bs, np.zeros((dep.k, self.p))]))   # steady Fisher wants
            Si_c = None
            for ph, sv, bq, br in cells:   # the split rides into the augmentation with the base
                Qa, Ra, _Ha, _w = dep.augment(bq, br, Hm)
                e = _WalkEngine(Qa, Ra, Ha, Fa, Ba, ph, sv, walk_axes=walk, cap=dep.cap,
                                fisher_Si=Si_c)
                Si_c = e._fisher_Si
                e._dyn = dep.callable_for()
                if hbase is not None:
                    e._hook = _augment_hook(hbase, n, dep.k)
                self._members.append(e)
                self._pidx.append(xmode)
        self._nd, self._nc = len(specs), len(cells)
        k = self._nd
        # The fault class's kernel: a uniform-leak chain with probability rho per step of
        # leaving the current dynamics hypothesis (Shiryaev's rule for a jump process).  It
        # mixes WITHIN each (phi, s) cell -- a dynamics fault does not change the noise class,
        # so the joint kernel over the two nuisances is their product.
        self._Md = (np.eye(k) * (1.0 - rho * k / (k - 1)) + np.ones((k, k)) * (rho / (k - 1))
                    if k > 1 else np.ones((1, 1)))
        self._learn, self.hazard = learn, rho
        # report the dynamics whenever they are not a fixed matrix the caller already has
        self._report = learn or base is not None
        self.reset()

    def reset(self):
        for f in self._members:
            f.reset()
        # Group structurally-identical members into stacked executors -- same recursion, one
        # leading member axis (see `_EngineBank`).  Grouping is by structure, NOT by position:
        # `eigh` orders a base's eigenmodes by value, so rungs of one spec can carry the same
        # physics under permuted labels, and grouping contiguously would shatter the bank into
        # slivers (measured: 225 banks on a five-pair rig).  Each bank scatters its outputs
        # back through the member indices it holds, so the member ORDER the dynamics channel's
        # reshape depends on is untouched.  A member whose class overrides ``update``
        # (research subclasses) runs looped, unchanged.
        order, index = [], {}
        for i, f in enumerate(self._members):
            key = (_bank_key(f), tuple(np.asarray(self._pidx[i]).tolist()))
            if key not in index:
                index[key] = len(order)
                order.append((key, []))
            order[index[key]][1].append(i)
        banks = []
        for (key, _), idx in order:
            mem = [self._members[i] for i in idx]
            bank = _EngineBank(mem) if key[0] else _LoopBank(mem)
            bank.pidx = self._pidx[idx[0]]
            bank.idx = np.array(idx)
            banks.append(bank)
        self._banks = banks
        self._logw = np.zeros(len(self._members))
        self.loglik = 0.0
        self._alarm = False
        return self

    def _hazard_mix(self, logw):
        """Propagate the bank prior through the fault class's kernel."""
        W = np.exp(logw - float(logw.max())).reshape(self._nd, self._nc)
        out = np.log(np.maximum(self._Md.T @ W, 1e-300)).ravel()
        return out - _logsumexp(out)

    def _dynamics_mean(self, post):
        """Posterior-mean (F, B) across the bank -- fixed members contribute their own."""
        W = post.reshape(self._nd, self._nc)
        Fh = np.zeros((self.n, self.n))
        Bh = None if self.B is None else np.zeros((self.n, self.p))
        for d, (bs, Fs, Bs, dep) in enumerate(self._specs):
            if dep is None and bs is None:
                w = float(W[d].sum())
                Fh += w * Fs
                if Bh is not None and Bs is not None:
                    Bh += w * Bs
                continue
            for c in range(self._nc):
                e = self._members[d * self._nc + c]
                x = np.zeros(self.n) if e._m is None else e._m[:self.n]
                if dep is None:
                    Fg, Bg = bs(x)
                else:
                    g = np.zeros(dep.k) if e._m is None else e._m[self.n:]
                    Fg, Bg = dep.dynamics_of(x, g)
                Fh += W[d, c] * Fg
                if Bh is not None and Bg is not None:
                    Bh += W[d, c] * Bg
        return Fh, Bh

    def _reprice(self):
        """Fire the shared-event restart on every walker (research 0003)."""
        for d, (_, _, _, dep) in enumerate(self._specs):
            if dep is None:
                continue
            for c in range(self._nc):
                self._members[d * self._nc + c].reprice(dep.gidx)

    def update(self, y, u=None) -> LucidStep:
        if self.B is not None and u is None:
            raise ValueError(f"this filter has a control input; pass u (length {self.p})")
        if self.B is None and u is not None:
            raise ValueError("filter has no control map; do not pass u")
        M = len(self._members)
        prior = self._logw - _logsumexp(self._logw)
        if self._nd > 1:
            prior = self._hazard_mix(prior)
        n = self.n
        mn = np.empty((M, n)); vr = np.empty((M, n, n)); inn = np.empty((M, self.m))
        llv = np.empty(M); psc = np.empty((M, n)); msc = np.empty((M, self.m))
        for bank in self._banks:
            bm, bv, bi, bl, bp, bms = bank.update(y, u=u)
            ix = bank.idx
            mn[ix] = bm[:, :n]; vr[ix] = bv[:, :n, :n]; inn[ix] = bi
            llv[ix] = bl; psc[ix] = bp[:, bank.pidx]; msc[ix] = bms
        yv = np.atleast_1d(np.asarray(y, float))
        if np.all(np.isfinite(yv)):
            bank_ll = _logsumexp(prior + llv)
            self._logw = self.forget * prior + llv
        else:
            bank_ll = 0.0
            self._logw = prior
        post = np.exp(self._logw - _logsumexp(self._logw))
        mean = post @ mn
        dmn = mn - mean
        var = (np.einsum("b,bij->ij", post, vr)
               + np.einsum("b,bi,bj->ij", post, dmn, dmn))
        ps = post @ psc
        ms = post @ msc
        innov = post @ inn
        self.loglik += bank_ll
        if not self._report:
            return LucidStep(mean, var, innov, bank_ll, ps, ms)
        Fh, Bh = self._dynamics_mean(post)
        # The fault readout is a marginal of the posterior -- the filter itself never
        # thresholds, it mixes.  Its RISING EDGE re-prices the walkers' ignorance: a jump has
        # just been confirmed, so what the walkers think they know about the departure is
        # priced back to the class prior (research 0003).
        fault = float(1.0 - post.reshape(self._nd, self._nc)[0].sum())
        alarm = fault > 0.5
        if alarm and not self._alarm:
            self._reprice()
        self._alarm = alarm
        return LucidStep(mean, var, innov, bank_ll, ps, ms, Fh, Bh, fault)

    def filter(self, Y, U=None) -> LucidResult:
        Y = np.atleast_2d(np.asarray(Y, float))
        if Y.ndim != 2 or Y.shape[1] != self.m:
            raise ValueError(f"Y must be (T, {self.m})")
        if self.B is not None:
            if U is None:
                raise ValueError(f"this filter has control input; pass U of shape ({Y.shape[0]}, {self.p})")
            U = np.atleast_2d(np.asarray(U, float))
        self.reset()
        T = Y.shape[0]
        mean = np.empty((T, self.n)); var = np.empty((T, self.n, self.n))
        inn = np.empty((T, self.m)); ps = np.empty((T, self.n)); ms = np.empty((T, self.m))
        live = self._report
        dyn = np.empty((T, self.n, self.n)) if live else None
        ctl = np.empty((T, self.n, self.p)) if live and self.B is not None else None
        flt = np.empty(T) if live else None
        total = 0.0
        for i, row in enumerate(Y):
            st = self.update(row, None if U is None else U[i])
            mean[i] = st.mean; var[i] = st.var; inn[i] = st.innovation
            ps[i] = st.process_scale; ms[i] = st.measurement_scale; total += st.loglik
            if live:
                dyn[i] = st.dynamics; flt[i] = st.fault
                if ctl is not None:
                    ctl[i] = st.control
        return LucidResult(mean=mean, var=var, innovation=inn,
                           process_scale=ps, measurement_scale=ms, loglik=total,
                           dynamics=dyn, control=ctl, fault=flt)

    def loglik_of(self, Y, U=None) -> float:
        return self.filter(Y, U).loglik
