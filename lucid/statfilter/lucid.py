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

**The input is a stream of events, not a matrix of rows.**  The general thing that
happens is that ONE sensor reports, at some time:

    f.observe(sensor, value, t=timestamp)     # one (sensor, timestamp, value) point

Sensors are not assumed to share a schedule and the gaps are not assumed equal.  A
partly-observed row is a first-class input (``NaN`` = that sensor did not report, and the
present ones are sub-selected out of ``H`` and ``R`` rather than the row being discarded);
a fully synchronous row at a fixed rate is the special case, ``filter(Y)``, and runs the
arithmetic it always did.  ``timestep`` fixes the time unit: everything supplied about the
model and every class timescale is per NOMINAL STEP, and an event ``a = dt / timestep``
steps after the last takes each of them to that power -- ``F(a) = exp(a log F)``,
``Q -> Q a``, ``phi -> phi**a``, ``forget -> forget**a``, ``rho -> 1 - (1-rho)**a``.
``R`` alone is not scaled: a measurement variance belongs to the reading, not to the gap
before it.  See ``research/pointwise-streaming/SUMMARY.md``.

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
_SS = (0.20, 0.30, 0.45, 0.60, 0.80)       #   fitted value; the data down-weights the unsupported corners
_HAZARD = 1e-4              # default fault rate: ~1 dynamics fault per 10,000 steps.  A LABELED
                            # prior of the same standing as `forget`, not a tuning constant: it is
                            # the operating point on the false-alarm/delay frontier, and the delay
                            # it buys is derived, log(1/rho) / KL-rate (research 0001).


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
    time: float = math.nan         #: the filter clock after this event (see ``timestep``)

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
    time: np.ndarray = None        #: (T,) the filter clock at each event
    sensor: np.ndarray = None      #: (T,) which sensor each event carried -- streams only

    def __len__(self) -> int:
        return len(self.mean)


def _logsumexp(a: np.ndarray) -> float:
    m = float(np.max(a))
    return m + math.log(float(np.sum(np.exp(a - m))))


# ------------------------------------------------------ the elapsed-time map
# A supplied ``F`` is the ONE-NOMINAL-STEP sampling of a fixed continuous generator
# ``A = log F``; over an elapsed ``a`` nominal steps the transition is ``exp(a A)``.  The
# three primitives below are the numpy-only realisation (the package has no scipy
# dependency): scaling-and-squaring for ``exp``, Denman-Beavers for the square root, and
# inverse scaling-and-squaring for ``log``.  They are exercised only when a stream actually
# carries a non-nominal gap -- ``a == 1`` and ``a == 0`` short-circuit to ``F`` and ``I``.


def _expm(M):
    """Matrix exponential by scaling and squaring with a Taylor core."""
    M = np.asarray(M, float)
    nrm = float(np.abs(M).sum(1).max())
    if nrm == 0.0:
        return np.eye(M.shape[0])
    k = max(0, int(math.ceil(math.log2(nrm / 0.5))))
    A = M / (2.0 ** k)
    E = np.eye(A.shape[0])
    term = E
    for j in range(1, 32):              # ||A|| <= 1/2: the tail is below double precision
        term = term @ A / j
        E = E + term
        if np.abs(term).sum(1).max() < 1e-18:
            break
    for _ in range(k):
        E = E @ E
    return E


def _sqrtm(A):
    """Principal matrix square root (Denman-Beavers), for matrices with no eigenvalue on
    the closed negative real axis -- which is exactly the condition for ``log A`` to be
    real, so a failure here and a failure there are the same failure."""
    Y = np.array(A, float)
    Z = np.eye(A.shape[0])
    for _ in range(64):
        Yn = 0.5 * (Y + np.linalg.inv(Z))
        Zn = 0.5 * (Z + np.linalg.inv(Y))
        done = np.abs(Yn - Y).max() <= 1e-14 * max(1.0, float(np.abs(Yn).max()))
        Y, Z = Yn, Zn
        if done:
            break
    return Y


def _logm(F):
    """Principal matrix logarithm by inverse scaling and squaring.

    Repeated square roots pull ``F`` towards the identity until the ``log(I + X)`` series
    converges fast, then the result is scaled back.  Defective matrices are fine -- the
    constant-velocity transition ``[[1, dt], [0, 1]]`` is the common one and has no
    eigenbasis at all, which is why this is not an eigendecomposition.
    """
    n = F.shape[0]
    I = np.eye(n)
    A = np.array(F, float)
    k = 0
    while np.abs(A - I).sum(1).max() > 0.25 and k < 60:
        A = _sqrtm(A)
        k += 1
    X = A - I
    L = np.zeros_like(X)
    term = I
    for j in range(1, 48):
        term = term @ X
        L = L + ((-1.0) ** (j + 1) / j) * term
        if np.abs(term).sum(1).max() < 1e-18:
            break
    return (2.0 ** k) * L


class _Propagator:
    """The elapsed-time transition and forcing map of a FIXED one-nominal-step ``F``.

    ``F`` is read as the ``a = 1`` sampling of a fixed generator ``A = log F``, so over an
    elapsed ``a`` nominal steps

        F(a) = exp(a A),     Phi(a) = int_0^a exp(A tau) dtau,

    both read off one exponential of ``a * [[A, I], [0, 0]]``.  The supplied ``B`` is the
    ONE-STEP forcing map (``x_t = F x_{t-1} + B u_t``), so its continuous counterpart is
    ``Phi(1)^-1 B`` and the elapsed map is ``Phi(a) Phi(1)^-1 B`` -- which is exactly ``B``
    at ``a = 1`` and ``0`` at ``a = 0``, i.e. continuous THROUGH the nominal step rather
    than agreeing with it only there.

    The factorisation is LAZY: a filter that is never handed a non-nominal gap never
    computes a logarithm, and one whose ``F`` has no real logarithm only finds out (with an
    error naming the fix) if it is actually asked to propagate over a non-nominal gap.
    """

    def __init__(self, F):
        self.F = np.asarray(F, float)
        self.n = self.F.shape[0]
        self._I = np.eye(self.n)
        self._Z = np.zeros((self.n, self.n))
        self._trivial = np.array_equal(self.F, self._I)
        self._A = None
        self._Pinv = None
        self._key = None
        self._val = None

    def _factor(self):
        if self._A is not None:
            return
        try:
            A = _logm(self.F)
            ok = np.all(np.isfinite(A)) and np.allclose(_expm(A), self.F,
                                                        rtol=1e-6, atol=1e-8)
        except np.linalg.LinAlgError:
            ok = False
        if not ok:
            raise ValueError(
                "these dynamics have no real continuous-time generator, so there is no "
                "transition over a non-nominal time step to compute (a negative real "
                "eigenvalue is the usual cause -- no continuous system samples to that).  "
                "Either sample uniformly, or pass `dynamics` as a callable of the state, "
                "which is propagated by the first-order elapsed map instead.")
        self._A = A
        aug = np.zeros((2 * self.n, 2 * self.n))
        aug[:self.n, :self.n] = A
        aug[:self.n, self.n:] = self._I
        self._aug = aug
        Phi1 = _expm(aug)[:self.n, self.n:]
        try:
            self._Pinv = np.linalg.solve(Phi1, self._I)
        except np.linalg.LinAlgError:               # int_0^1 exp(A t) dt singular: only
            self._Pinv = None                       # when A has a 2*pi*i*k eigenvalue

    def at(self, a):
        """``(F(a), Psi(a))`` with ``mpred = F(a) x + Psi(a) B u``."""
        if a == 1.0:
            return self.F, self._I
        if a == 0.0:
            return self._I, self._Z
        if self._trivial:
            return self._I, a * self._I
        if a == self._key:
            return self._val
        self._factor()
        M = _expm(a * self._aug)
        Fa, Phi = M[:self.n, :self.n], M[:self.n, self.n:]
        out = (Fa, a * self._I if self._Pinv is None else Phi @ self._Pinv)
        self._key, self._val = a, out
        return out


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

    def __init__(self, Q0, R0, H, F, B, phi, s, walk_axes=None, cap=None, prop=None):
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
        self.gap = _GAP_FACTOR * self.s
        self._Kstar = (1.0 - self.phi) / 4.0
        self._cap = None if cap is None else np.asarray(cap, float)
        self._dyn = None            # optional (mean, u) -> (Jacobian, predicted mean) callable
        self._prop = prop if prop is not None else _Propagator(F)
        self._Tcache = {}           # AR(1) scale kernel per elapsed time (5x5 -- tiny)
        self._Ichar = self._steady_fisher()
        self._Ifloor = (1.0 - self.phi) / (4.0 * (_SPAN_S * self.s) ** 2)
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
        self._qmu = self._Kstar ** 2 / (np.maximum(self._Ichar, self._Ifloor)
                                        * (1.0 - self._Kstar))
        self._Pmu_cap = (_SPAN_S * self.s) ** 2
        self._build_window()
        self.reset()

    # -- caltrop star window (the 1-D pieces are unchanged from WalkingVectorFilter) --
    def _build_window(self):
        K = int(math.ceil(_SPAN_S / _GAP_FACTOR))
        off1 = self.gap * np.arange(-K, K + 1)
        self._off1, self._nn, self._c = off1, off1.size, K
        w1 = np.exp(-0.5 * (off1 / self.s) ** 2); w1 /= w1.sum()
        nu = max(self.s * self.s * (1.0 - self.phi ** 2), 1e-12)
        T1 = np.exp(np.clip(-0.5 * (off1[None, :] - self.phi * off1[:, None]) ** 2 / nu, -700.0, 700.0))
        T1 /= T1.sum(1, keepdims=True)
        self._w1, self._T1 = w1, T1
        self._act = [int(k) for k in np.flatnonzero(self.active)]
        # star node table: node 0 is the centre (all axes at mu); then, per active axis, the
        # 2K off-centre offsets along that axis alone.  1 + 2K*r nodes -- linear in the axes.
        arm = np.delete(np.arange(self._nn), self._c)
        self._star_axis = np.concatenate(
            [np.full(1, -1)] + [np.full(self._nn - 1, k) for k in self._act]).astype(int)
        self._star_off = np.concatenate(
            [np.zeros(1)] + [off1[arm] for _ in self._act])
        self._G = self._star_axis.size
        self._axwin = {}
        for i, k in enumerate(self._act):
            w = np.empty(self._nn, dtype=int)
            w[self._c] = 0
            w[arm] = 1 + i * (self._nn - 1) + np.arange(self._nn - 1)
            self._axwin[k] = w

    def _star_QR(self, a=1.0):
        """(Q_g, r_g) at every star node: the centre pair plus a per-node one-axis change.

        ``Q`` accumulates over ELAPSED TIME and so carries the factor ``a``; ``r`` does not
        -- a measurement variance is a property of the reading, not of the gap before it.
        ``r_g`` is returned as the (G, m) diagonal, so a partial event can sub-select
        sensors out of it without ever forming the (m, m) block.
        """
        n = self.n
        Qc = self.V @ np.diag(a * self.lam * np.exp(np.clip(self.mu[:n], -60, 60))) @ self.V.T
        rc = self.rho * np.exp(np.clip(self.mu[n:], -60, 60))
        Qg = np.repeat(Qc[None], self._G, 0)
        rg = np.repeat(rc[None], self._G, 0)
        for g in range(1, self._G):
            k = int(self._star_axis[g]); o = float(self._star_off[g])
            if k < n:
                dlam = a * self.lam[k] * (math.exp(min(self.mu[k] + o, 60.0))
                                          - math.exp(min(self.mu[k], 60.0)))
                Qg[g] += dlam * np.outer(self.V[:, k], self.V[:, k])
            else:
                i = k - n
                rg[g, i] = self.rho[i] * math.exp(min(self.mu[k] + o, 60.0))
        return Qg, rg

    def _kernel(self, a):
        """The scale class's AR(1) transition over ``a`` nominal steps.

        ``xi`` is AR(1) with persistence ``phi`` and stationary sd ``s`` PER NOMINAL STEP,
        i.e. an Ornstein-Uhlenbeck process sampled at the nominal rate; over ``a`` steps its
        persistence is ``phi**a`` and its innovation variance ``s^2 (1 - phi^2a)``.  At
        ``a = 0`` that is the identity (nothing moves between two readings at one instant);
        at ``a = 1`` it is the kernel built once at construction, bit for bit.
        """
        if a == 1.0:
            return self._T1
        T = self._Tcache.get(a)
        if T is None:
            pa = self.phi ** a
            nu = max(self.s * self.s * (1.0 - pa * pa), 1e-12)
            off1 = self._off1
            T = np.exp(np.clip(-0.5 * (off1[None, :] - pa * off1[:, None]) ** 2 / nu,
                               -700.0, 700.0))
            T /= T.sum(1, keepdims=True)
            if len(self._Tcache) < 512:
                self._Tcache[a] = T
        return T

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
            w[k] = float(pi_ax[i] @ self._off1)
        return w

    def _dS_axis(self, k, obs, a=1.0):
        """dS/dxi_k at each of axis k's window nodes, over the sensors ``obs`` this event
        carried (dS_k depends only on the k-coordinate).

        A process mode's entry carries the elapsed factor ``a`` because ``Q`` does; a
        sensor's does not.  Either can come back identically zero -- a mode this event's
        sensors cannot see, or a sensor that did not read -- and a zero here is what tells
        ``update`` to let that axis DRIFT rather than update it on no evidence.

        The ``a`` passed for a process mode is the LIVE process time (``_aQ``), not the
        gap since the last event.  The two differ only at a zero gap, and there the
        distinction is the whole ball game: this score is the local one, keeping ``Q``'s
        own dependence on ``xi`` and dropping the prior covariance's, and at a zero gap
        the dropped term is the ONLY term.  Zeroing the score there would hand every
        process-scale axis's evidence to whichever sensor happened to follow the gap and
        discard what the other sensors at that instant say about the same ``Q`` -- which
        is measurably what it did (research 0003).  Carrying the live process time keeps
        the leading term instead, so ``m`` readings of one instant weigh on the process
        scale whether they arrive as a row or as ``m`` points.
        """
        e = np.exp(np.minimum(self.mu[k] + self._off1, 60.0))
        if k < self.n:
            hv = self.HV[obs, k]
            return (a * self.lam[k] * e)[:, None, None] * np.outer(hv, hv)[None]
        out = np.zeros((self._nn, obs.size, obs.size))
        i = k - self.n
        j = int(np.searchsorted(obs, i))
        if j < obs.size and obs[j] == i:
            out[:, j, j] = self.rho[i] * e
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

    def _dS_list(self, scale):
        out = []
        for k in range(self.n):
            hv = self.HV[:, k]
            out.append(self.lam[k] * math.exp(min(scale[k], 60)) * np.outer(hv, hv))
        for i in range(self.m):
            E = np.zeros((self.m, self.m)); E[i, i] = self.rho[i] * math.exp(min(scale[self.n + i], 60))
            out.append(E)
        return out

    def _steady_fisher(self):
        H, F, n, m = self.H, self.F, self.n, self.m
        P = np.eye(n) * (self.lam.max() + self.rho.max())
        Q0, R0 = self._Q_of(np.zeros(n)), self._R_of(np.zeros(m))
        for _ in range(400):
            Ppred = self._cap_P(F @ P @ F.T + Q0)
            S = H @ Ppred @ H.T + R0
            K = Ppred @ H.T @ np.linalg.inv(S)
            P = Ppred - K @ H @ Ppred
        Ppred = self._cap_P(F @ P @ F.T + Q0)
        Si = np.linalg.inv(H @ Ppred @ H.T + R0)
        dS = self._dS_list(np.zeros(self.D))
        return np.array([0.5 * np.trace(Si @ d @ Si @ d) for d in dS]) + _RIDGE

    def reset(self, mean=None, scale=None):
        self._pi_ax = None
        self._aQ = 0.0              # the live process time: see _dS_axis
        self._m = None if mean is None else np.asarray(mean, float)
        self._P = None
        self.mu = np.zeros(self.D) if scale is None else np.asarray(scale, float).copy()
        self._Pmu = np.full(self.D, self.s * self.s)
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

    def update(self, y, u=None, a=1.0):
        """One EVENT: the sensors of ``y`` that are finite, ``a`` nominal steps after the last.

        ``a = 1`` with an all-finite ``y`` is the classical synchronous step and runs the
        identical arithmetic.  Otherwise:

        * **Partial.** The sensors that read are sub-selected out of ``H`` and ``r`` -- the
          correction, the predictive density and the scale evidence are all over that subset
          alone, and the absent sensors are never imputed.  Cost falls with the subset:
          ``G (2 n^2 m_o + 2 n m_o^2 + m_o^3)``, so a single-sensor event has no ``m^3`` term
          at all.
        * **Elapsed.** ``Q`` and the walk's drift accumulate over ``a``; the scale class's
          AR(1) kernel is taken to the power ``a``; the state transition is the propagator's
          ``F(a)``.  ``a = 0`` -- two sensors reporting the same instant -- correctly moves
          nothing but the correction.

        An axis with no evidence in this event DRIFTS: its window relaxes through the kernel
        and its walk covariance grows by ``q_mu * a``, which is the honest statement that
        the filter learned nothing about that scale.  It is not frozen and not updated on
        noise (research 0003's floor-and-cap, never freeze, applied one level down).
        """
        n, m, H = self.n, self.m, self.H
        y = np.atleast_1d(np.asarray(y, dtype=float))
        obs = np.flatnonzero(np.isfinite(y))
        mo = obs.size
        r = len(self._act)
        # the model uses the true gap; the process-scale SCORE uses the live process
        # time, which differs from it only at a zero gap (see _dS_axis)
        aQ = a if a > 0.0 else self._aQ
        self._aQ = aQ
        Qg, rg = self._star_QR(a)
        if self._pi_ax is None:
            self._pi_ax = np.tile(self._w1, (r, 1))
            if self._m is None:
                self._m = (np.linalg.lstsq(H[obs], y[obs], rcond=None)[0]
                           if mo else np.zeros(n))
            if self._P is None:
                Q1, r1 = (Qg, rg) if a == 1.0 else self._star_QR(1.0)
                self._P = self._cap_P(
                    np.eye(n) * float(r1.max() + Q1.reshape(self._G, -1).max()) * n)
        pi_ax = self._pi_ax @ self._kernel(a)
        # The dynamics are a CALLABLE when they are learned or nonlinear: it returns the
        # linearisation (for the covariance) and the predicted mean (from f, not the
        # Jacobian -- they differ once the transition depends on the state).  Real F/B
        # arrive linearised per operating point, so this is the general path.
        if self._dyn is None:
            F, Psi = self._prop.at(a)
            mpred = F @ self._m + ((Psi @ (self.B @ u)) if self.B is not None else 0.0)
        else:
            F1, mp1 = self._dyn(self._m, u)
            if a == 1.0:
                F, mpred = F1, mp1
            else:
                # A per-step linearisation has no fixed generator to exponentiate -- it is
                # only ever the truth AT the nominal step -- so the elapsed map is the
                # first-order one, exact at a = 0 and a = 1 and correct to first order in
                # the generator between them.  That is the regime a linearised transition
                # lives in anyway (F ~ I at any sampling fine enough to be asynchronous).
                F = np.eye(F1.shape[0]) + a * (F1 - np.eye(F1.shape[0]))
                mpred = self._m + a * (mp1 - self._m)
        FPFt = F @ self._P @ F.T
        if mo == 0:
            self._pi_ax = pi_ax
            w = self._star_weights(pi_ax)
            self._P = self._cap_P(FPFt + np.einsum("g,gij->ij", w, Qg))
            self._m = mpred
            for k in self._act:                     # nothing seen: every scale drifts
                self._Pmu[k] = min(self._Pmu[k] + self._qmu[k] * a, self._Pmu_cap)
            wmean = self._wmean(pi_ax)
            return LucidStep(self._m.copy(), self._P.copy(), np.full(m, np.nan), 0.0,
                             self.mu[:n] + wmean[:n], self.mu[n:] + wmean[n:])
        Hs = H if mo == m else H[obs]
        ys = y if mo == m else y[obs]
        Ppred = FPFt[None] + Qg
        e = ys - Hs @ mpred
        PHt = np.einsum("gij,kj->gik", Ppred, Hs)
        S = np.einsum("ij,gjk->gik", Hs, PHt)
        S[:, np.arange(mo), np.arange(mo)] += rg[:, obs]
        Si = np.linalg.inv(S)
        sgn, logdet = np.linalg.slogdet(S)
        maha = np.einsum("i,gij,j->g", e, Si, e)
        lg = -0.5 * (mo * _LOG2PI + logdet + maha)
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
        HPp = np.einsum("ij,gjk->gik", Hs, Ppred)
        Ppost = Ppred - np.einsum("gil,glk->gik", K, HPp)   # K(H Ppred): n^2 m per node, not n^3
        P_new = np.einsum("g,gij->ij", pi, Ppost) + np.einsum("g,gi,gj->ij", pi, dm, dm)
        P_new = self._cap_P(0.5 * (P_new + P_new.T))
        # finding-18 walk per active axis, score/Fisher averaged over that axis's window
        # posterior only (the caltrop: dS_k depends only on the k-coordinate, so the axial
        # profile carries the axis's evidence at linear cost).  An axis this event carries
        # no evidence for -- an unobserved sensor, or a mode none of the reporting sensors
        # sees -- has dS = 0 identically, and only drifts.
        for i, k in enumerate(self._act):
            idx = self._axwin[k]
            dpk = self._dS_axis(k, obs, aQ if k < n else a)
            if not dpk.any():
                self._Pmu[k] = min(self._Pmu[k] + self._qmu[k] * a, self._Pmu_cap)
                continue
            Sik = Si[idx]
            Sie = np.einsum("gij,j->gi", Sik, e)
            score_g = 0.5 * (np.einsum("gi,gij,gj->g", Sie, dpk, Sie)
                             - np.einsum("gij,gji->g", Sik, dpk))
            SidS = np.einsum("gij,gjk->gik", Sik, dpk)
            info_g = 0.5 * np.einsum("gij,gji->g", SidS, SidS)
            info = float(pi_ax[i] @ info_g) + _RIDGE
            grad = float(pi_ax[i] @ score_g)
            K_mu = self._Pmu[k] / (self._Pmu[k] + 1.0 / info)
            self.mu[k] += float(np.clip(K_mu * (grad / info), -self.gap, self.gap))
            self._Pmu[k] = min((1.0 - K_mu) * self._Pmu[k] + self._qmu[k] * a, self._Pmu_cap)
        self._pi_ax, self._m, self._P = pi_ax, m_new, P_new
        self.loglik += ll
        wmean = self._wmean(pi_ax)
        innov = e
        if mo != m:
            innov = np.full(m, np.nan)
            innov[obs] = e
        return LucidStep(m_new.copy(), P_new.copy(), innov.copy(), ll,
                         self.mu[:n] + wmean[:n], self.mu[n:] + wmean[n:])


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
    H : (m, n) array, optional
        Measurement matrix.  Defaults to the identity.
    process, measurement : arrays, optional
        Base noise magnitudes ``Q0`` (n, n, PD) and ``R0`` (m, diagonal).  Default to identity/unit --
        the walk breathes around them with unbounded reach, so a rough base is fine.
    n : int, optional
        State dimension, when it cannot be inferred from ``dynamics``/``process``/``H`` (default 1).
    phis, ss : sequences, optional  
        The ``(phi, s)`` box the bank averages over.  Defaults to a broad dead-zone-free range;
        not a fitted value.  ``forget`` (default 0.999) is the weight memory -- the one residual.
    timestep : float, optional
        How long ONE NOMINAL STEP is, in whatever units the timestamps are in (default 1.0,
        i.e. time is counted in steps).  Everything supplied about the model -- ``dynamics``,
        ``process`` -- and every class timescale -- ``phis``, ``ss``, ``forget``, ``faults``
        -- is per nominal step; ``timestep`` is what lets an event carry a real ``t`` or
        ``dt``.  Sampling at 100 Hz with timestamps in seconds: ``timestep=0.01``.

    Events
    ------
    ``observe(sensor, value, t=)`` is one ``(sensor, timestamp, value)`` point and is the
    general input; ``stream(points)`` runs a whole stream of them.  ``update(y, t=)`` takes a
    length-``m`` row in which ``NaN`` means *that sensor did not report*, and ``filter(Y, t=)``
    a batch of such rows.  Absent sensors are sub-selected out of ``H`` and ``R`` -- never
    imputed, never a reason to discard the sensors that did report.  With no clock and no
    absences this is the uniform, fully-observed filter, unchanged.
    """

    def __init__(self, dynamics=0, control=None, H=None, process=None, measurement=None,
                 n=None, faults=None, departures=None, anchors=None,
                 phis=_PHIS, ss=_SS, forget=0.999, timestep=1.0):
        learn = dynamics is None or faults is not None
        if faults is None or faults is True:
            rho = _HAZARD
        else:
            rho = float(faults)
            if not 0.0 < rho < 1.0:
                raise ValueError("faults (the hazard rho) must lie in (0, 1)")
        if anchors is not None and not learn:
            learn = True                        # named fault hypotheses imply a fault class
        H = None if H is None else np.atleast_2d(np.asarray(H, float))
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
        Hm = np.eye(n) if H is None else H
        m = Hm.shape[0]
        if Hm.shape != (m, n):
            raise ValueError(f"H must be (m, {n})")
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
        timestep = float(timestep)
        if not timestep > 0.0:
            raise ValueError("timestep (the duration of one nominal step) must be positive")

        self.n, self.m, self.D = n, m, n + m
        self.p = 0 if B is None else B.shape[1]
        self.B = B
        self.forget = float(forget)
        self.timestep = timestep
        self._Mdcache = {}
        self.phi_arr = np.array([ph for ph in phis for _ in ss], float)
        self.s_arr = np.array([sv for _ in phis for sv in ss], float)
        cells = [(ph, sv) for ph in phis for sv in ss]

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
                pr = _Propagator(Fs)        # one generator per hypothesis, shared by its
                eng = [_WalkEngine(Q0, R0, Hm, Fs, Bs, ph, sv, prop=pr)   # (phi, s) cells
                       for ph, sv in cells]
                if bs is not None:
                    for e in eng:
                        e._dyn = _fixed_hook(bs)
                self._members += eng
                self._pidx += [np.arange(n)] * len(cells)
                continue
            Qa, Ra, Ha, walk = dep.augment(Q0, R0, Hm)
            xmode = np.flatnonzero(walk[:dep.na])
            if xmode.size != n:
                raise RuntimeError("augmented process eigenbasis did not separate")
            Fa = np.eye(dep.na)
            Fa[:n, :n] = Fs                     # the Jacobian at g = 0, x = 0: the
            Ba = (None if Bs is None                     # characteristic linearisation the
                  else np.vstack([Bs, np.zeros((dep.k, self.p))]))   # steady Fisher wants
            for ph, sv in cells:
                e = _WalkEngine(Qa, Ra, Ha, Fa, Ba, ph, sv, walk_axes=walk, cap=dep.cap)
                e._dyn = dep.callable_for()
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
        self._logw = np.zeros(len(self._members))
        self.loglik = 0.0
        self._alarm = False
        self._t = None
        return self

    # ------------------------------------------------------------------ the clock
    @property
    def time(self):
        """The filter clock -- the timestamp of the last event, ``None`` before the first."""
        return self._t

    def _elapsed(self, t, dt):
        """Advance the clock and return the gap in NOMINAL STEPS.

        Everything the caller supplied about the model -- ``dynamics``, ``process`` -- and
        every class timescale -- ``phis``, ``ss``, ``forget``, ``faults`` -- is per nominal
        step, and ``timestep`` says how long one of those is in the caller's time units.
        Supplying neither ``t`` nor ``dt`` advances exactly one nominal step, which is the
        uniform-sampling filter this one generalises.
        """
        prev = self._t
        if t is not None and dt is not None:
            raise ValueError("pass t (an absolute timestamp) or dt (an elapsed time), not both")
        if t is not None:
            now = float(t)
            a = 1.0 if prev is None else (now - prev) / self.timestep
        elif dt is not None:
            a = float(dt) / self.timestep
            now = (0.0 if prev is None else prev) + float(dt)
        else:
            a = 1.0
            now = (0.0 if prev is None else prev) + self.timestep
        if not a >= 0.0:
            raise ValueError(
                f"time went backwards ({a * self.timestep:.6g} before the last event): a "
                "stream must arrive in non-decreasing timestamp order (equal timestamps, "
                "for sensors that read the same instant, are fine)")
        self._t = now
        return a

    def _hazard_kernel(self, a):
        """The fault class's mixing kernel over ``a`` nominal steps: a hazard ``rho`` per
        nominal step is ``1 - (1 - rho)**a`` over the gap."""
        if a == 1.0:
            return self._Md
        M = self._Mdcache.get(a)
        if M is None:
            k = self._nd
            if k > 1:
                rho_a = -math.expm1(a * math.log1p(-self.hazard))
                M = (np.eye(k) * (1.0 - rho_a * k / (k - 1))
                     + np.ones((k, k)) * (rho_a / (k - 1)))
            else:
                M = np.ones((1, 1))
            if len(self._Mdcache) < 512:
                self._Mdcache[a] = M
        return M

    def _hazard_mix(self, logw, a=1.0):
        """Propagate the bank prior through the fault class's kernel."""
        W = np.exp(logw - float(logw.max())).reshape(self._nd, self._nc)
        out = np.log(np.maximum(self._hazard_kernel(a).T @ W, 1e-300)).ravel()
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

    def update(self, y, u=None, t=None, dt=None) -> LucidStep:
        """One event: the readings in ``y`` that are finite, at time ``t`` (or ``dt`` after
        the last event; neither means one nominal step, the uniform case).

        ``y`` is always length ``m`` -- the full sensor suite -- and a ``NaN`` entry means
        *that sensor did not report at this instant*, which is the ordinary condition of a
        multi-rate sensor set, not an exception.  Absent sensors are sub-selected out of
        ``H`` and ``R``; they are never imputed and they never enter the likelihood.
        """
        if self.B is not None and u is None:
            raise ValueError(f"this filter has a control input; pass u (length {self.p})")
        if self.B is None and u is not None:
            raise ValueError("filter has no control map; do not pass u")
        a = self._elapsed(t, dt)
        M = len(self._members)
        prior = self._logw - _logsumexp(self._logw)
        if self._nd > 1:
            prior = self._hazard_mix(prior, a)
        steps = [f.update(y, u=u, a=a) for f in self._members]
        ll = np.array([st.loglik for st in steps])
        yv = np.atleast_1d(np.asarray(y, float))
        if np.any(np.isfinite(yv)):
            bank_ll = _logsumexp(prior + ll)
            # ``forget`` is a memory PER NOMINAL STEP, so over a gap of ``a`` it is
            # ``forget**a`` -- the bank's weight memory is a duration, not a count of events.
            self._logw = (self.forget ** a) * prior + ll
        else:
            bank_ll = 0.0
            self._logw = prior
        post = np.exp(self._logw - _logsumexp(self._logw))
        n = self.n
        mn = [st.mean[:n] for st in steps]
        mean = sum(post[i] * mn[i] for i in range(M))
        var = sum(post[i] * (steps[i].var[:n, :n] + np.outer(mn[i] - mean, mn[i] - mean))
                  for i in range(M))
        ps = sum(post[i] * steps[i].process_scale[self._pidx[i]] for i in range(M))
        ms = sum(post[i] * steps[i].measurement_scale for i in range(M))
        innov = sum(post[i] * steps[i].innovation for i in range(M))
        self.loglik += bank_ll
        if not self._report:
            return LucidStep(mean, var, innov, bank_ll, ps, ms, time=self._t)
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
        return LucidStep(mean, var, innov, bank_ll, ps, ms, Fh, Bh, fault, self._t)

    def observe(self, sensor, value, t=None, dt=None, u=None) -> LucidStep:
        """One ``(sensor, timestamp, value)`` point -- the filter's most general input.

        This is the streaming form: sensors are not assumed to share a schedule, so the
        general thing that happens is that ONE of them reports.  A synchronous row is the
        special case where ``m`` points share a timestamp, and feeding it as ``m`` points at
        one ``t`` is legal and means the same thing (``dt = 0`` between them moves no state
        and accumulates no process noise -- only the corrections land).

            f = LucidFilter(H=H, measurement=R0, timestep=0.01)   # 100 Hz nominal
            for sensor, t, value in stream:
                st = f.observe(sensor, value, t=t)
                st.mean          # the state estimate as of t

        ``sensor`` may also be a sequence of indices with a matching sequence of values,
        for a device that reports several channels together.
        """
        y = np.full(self.m, np.nan)
        idx = np.atleast_1d(np.asarray(sensor))
        if idx.size and (idx.min() < -self.m or idx.max() >= self.m):
            raise ValueError(f"sensor index out of range for m = {self.m}")
        y[idx] = np.atleast_1d(np.asarray(value, float))
        return self.update(y, u=u, t=t, dt=dt)

    def _times(self, T, t, dt):
        """Per-event ``(t, dt)`` arguments for a batch of ``T`` events."""
        if t is not None and dt is not None:
            raise ValueError("pass t (absolute timestamps) or dt (elapsed times), not both")
        if t is not None:
            t = np.atleast_1d(np.asarray(t, float)).ravel()
            if t.size != T:
                raise ValueError(f"t must have {T} entries, one per event")
            return [(float(v), None) for v in t]
        if dt is not None:
            d = np.atleast_1d(np.asarray(dt, float)).ravel()
            if d.size == 1:
                d = np.repeat(d, T)
            if d.size != T:
                raise ValueError(f"dt must be a scalar or have {T} entries")
            return [(None, float(v)) for v in d]
        return [(None, None)] * T

    def filter(self, Y, U=None, t=None, dt=None) -> LucidResult:
        """Filter a batch of synchronous rows.  ``Y`` is (T, m); a ``NaN`` entry is a sensor
        that did not report on that row, and a row may be partly or wholly absent.

        ``t`` gives the (T,) absolute timestamps and ``dt`` the elapsed times (a scalar for a
        uniform but non-nominal rate); neither means one nominal step per row.
        """
        Y = np.atleast_2d(np.asarray(Y, float))
        if Y.ndim != 2 or Y.shape[1] != self.m:
            raise ValueError(f"Y must be (T, {self.m})")
        if self.B is not None:
            if U is None:
                raise ValueError(f"this filter has control input; pass U of shape ({Y.shape[0]}, {self.p})")
            U = np.atleast_2d(np.asarray(U, float))
        self.reset()
        T = Y.shape[0]
        when = self._times(T, t, dt)
        mean = np.empty((T, self.n)); var = np.empty((T, self.n, self.n))
        inn = np.empty((T, self.m)); ps = np.empty((T, self.n)); ms = np.empty((T, self.m))
        clock = np.empty(T)
        live = self._report
        dyn = np.empty((T, self.n, self.n)) if live else None
        ctl = np.empty((T, self.n, self.p)) if live and self.B is not None else None
        flt = np.empty(T) if live else None
        total = 0.0
        for i, row in enumerate(Y):
            ti, di = when[i]
            st = self.update(row, None if U is None else U[i], t=ti, dt=di)
            mean[i] = st.mean; var[i] = st.var; inn[i] = st.innovation
            ps[i] = st.process_scale; ms[i] = st.measurement_scale; total += st.loglik
            clock[i] = st.time
            if live:
                dyn[i] = st.dynamics; flt[i] = st.fault
                if ctl is not None:
                    ctl[i] = st.control
        return LucidResult(mean=mean, var=var, innovation=inn,
                           process_scale=ps, measurement_scale=ms, loglik=total,
                           dynamics=dyn, control=ctl, fault=flt, time=clock)

    def stream(self, points, U=None) -> LucidResult:
        """Filter a stream of ``(sensor, timestamp, value)`` points -- one sensor at a time.

        ``points`` is any iterable of triples, or an array with those three columns.
        Timestamps must be non-decreasing; equal ones mean "the same instant".  ``U``, if
        the filter has a control map, is the known forcing at each point.

        The result is per POINT: ``mean[i]`` is the state as of point ``i``, ``sensor[i]``
        says which sensor it was and ``time[i]`` when, and ``innovation[i]`` is ``NaN``
        everywhere except that sensor.
        """
        pts = list(points)
        T = len(pts)
        if self.B is not None:
            if U is None:
                raise ValueError(f"this filter has control input; pass U of shape ({T}, {self.p})")
            U = np.atleast_2d(np.asarray(U, float))
        self.reset()
        mean = np.empty((T, self.n)); var = np.empty((T, self.n, self.n))
        inn = np.empty((T, self.m)); ps = np.empty((T, self.n)); ms = np.empty((T, self.m))
        clock = np.empty(T); which = np.empty(T, dtype=int)
        live = self._report
        dyn = np.empty((T, self.n, self.n)) if live else None
        ctl = np.empty((T, self.n, self.p)) if live and self.B is not None else None
        flt = np.empty(T) if live else None
        total = 0.0
        for i, pt in enumerate(pts):
            try:
                sensor, when, value = pt
            except (TypeError, ValueError):
                raise ValueError(
                    f"point {i} is not a (sensor, timestamp, value) triple: {pt!r}") from None
            if float(sensor) != int(sensor):
                raise ValueError(
                    f"point {i} has a non-integer sensor index ({sensor!r}) -- a point is "
                    "(sensor, timestamp, value), in that order")
            st = self.observe(int(sensor), float(value), t=float(when),
                              u=None if U is None else U[i])
            mean[i] = st.mean; var[i] = st.var; inn[i] = st.innovation
            ps[i] = st.process_scale; ms[i] = st.measurement_scale; total += st.loglik
            clock[i] = st.time; which[i] = int(sensor)
            if live:
                dyn[i] = st.dynamics; flt[i] = st.fault
                if ctl is not None:
                    ctl[i] = st.control
        return LucidResult(mean=mean, var=var, innovation=inn,
                           process_scale=ps, measurement_scale=ms, loglik=total,
                           dynamics=dyn, control=ctl, fault=flt, time=clock, sensor=which)

    def loglik_of(self, Y, U=None, t=None, dt=None) -> float:
        return self.filter(Y, U, t=t, dt=dt).loglik
