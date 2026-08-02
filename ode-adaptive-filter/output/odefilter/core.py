"""Adaptive filter for a process locally described by a linear ODE.

The model, in full:

    x_t = alpha . (x_{t-1}, ..., x_{t-p}) + w_t,   w_t ~ N(0, Q  * exp(lamP_t))
    y_t = x_t + v_t,                               v_t ~ N(0, S2 * exp(lamM_t))
    lam^c_t = phi_c lam^c_{t-1} + sqrt(nu_c) z_t,  c in {P, M}
    s_c^2   = nu_c / (1 - phi_c^2)

A scalar process whose evolution is a linear recurrence, observed with noise,
where both noise scales are themselves log-AR(1).  Everything below the first
line is the parent filter unchanged; `alpha` is what this adds.

Why a recurrence and not a derivative vector.  A second-order linear ODE with a
constant offset has solution space span{1, e^{l1 t}, e^{l2 t}}, so a uniformly
sampled solution is annihilated by (z-1)(z-z1)(z-z2): an order-3 homogeneous
recurrence with one root pinned at 1.  **The constant offset is a root at z = 1,
not an extra state.**  The lag vector and the derivative vector are related by a
fixed invertible integer matrix, so they carry the same information; carrying
lags means no finite difference is ever formed, and `derivatives()` converts the
posterior on request.

**The parent filter is this one's p = 1, alpha = 1 face**, exactly.

Learned parameters: alpha (p of them), Q, S2, phi_P, phi_M, s_P, s_M, and the
dynamics channel phi_A, s_A.  All are learned by maximum marginal likelihood.
`order` and `order_A` are quadrature resolutions -- compute budgets, not tuning
parameters.

**alpha is estimated once and then TRACKED, not held fixed.**  The dynamics
channel grids a scalar g on

    alpha(g) = (1 - g) * (1, 0, ..., 0) + g * alpha

and evolves it as an AR(1) with learned persistence phi_A and learned spread
s_A -- the same machinery as the two noise channels, one level up.  g = 1 is
the fitted dynamics; **g = 0 is exactly the parent's local-level model**, so
"the dynamics have stopped governing" is a member of the family with its own
likelihood rather than an absence of evidence, and the filter reverts to it on
affirmative evidence and comes back when the evidence does.  g > 1 is the other
direction: more persistent than fitted, which is what a forecast that decays
too fast needs.  `Step.dynamics` reports the posterior mean of g.

With s_A = 0 the channel collapses to a single node and the recursion is
bit-for-bit what it was before the channel existed.

Two diagnostics come out for free, and they are orthogonal by construction
(measured in exploration/0025):

    innovation MEAN transient   an event: a one-off disturbance.  Any such
                                event, in any direction, IS process noise --
                                the filter absorbs it and it leaves no trace.
    innovation WHITENESS        a parameter change: the dynamics themselves
                                moved.  Leaves no mean signature at all and a
                                permanent one in the lag-1 autocorrelation.

`Step.whiteness` reports the second, and is a cheap always-on residual check
that needs no grid: a correctly specified filter emits white innovations, so
sustained departure from zero means `alpha` no longer fits.  It is a smoke
alarm rather than a controller -- being cumulative, it cannot come back down.
Acting on the signal is the dynamics channel's job, and `Step.dynamics` is the
quantity that both detects and reverts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field

import numpy as np

__all__ = ["OdeFilter", "Params", "FilterResult", "Step"]

_LOG2PI = math.log(2.0 * math.pi)


class _Numerical(ArithmeticError):
    """Raised when a parameter vector drives the recursion out of range."""


def _companion(alpha) -> np.ndarray:
    a = np.asarray(alpha, dtype=float)
    p = a.size
    F = np.zeros((p, p))
    F[0] = a
    if p > 1:
        F[1:, :-1] = np.eye(p - 1)
    return F


def _radius(alpha) -> float:
    """Spectral radius of the companion matrix of `alpha`."""
    r = np.roots(np.concatenate([[1.0], -np.asarray(alpha, dtype=float)]))
    return float(np.max(np.abs(r))) if r.size else 0.0


# --------------------------------------------------------------------- params
@dataclass(frozen=True)
class Params:
    """The learned numbers.

    alpha      the recurrence coefficients, x_t = sum_i alpha_i x_{t-i}.  Their
               characteristic polynomial's roots are the ODE's modes: a root at
               1 is a constant offset, a complex pair is an oscillator.
    Q, s2      median (geometric-mean) variance of process and measurement
               noise.  As in the parent, these are medians rather than means so
               that they stay separated from s_P and s_M.
    phi_P/M    persistence of each noise channel's log-scale, in [0, 1).
    s_P/M      log-SD of each channel's scale.  Zero means homoscedastic.
    """

    alpha: tuple
    Q: float
    s2: float
    phi_P: float = 0.0
    phi_M: float = 0.0
    s_P: float = 0.0
    s_M: float = 0.0
    phi_A: float = 0.0
    s_A: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "alpha", tuple(float(a) for a in self.alpha))
        if len(self.alpha) < 1:
            raise ValueError("alpha must have at least one entry")
        if not (self.Q > 0 and self.s2 > 0):
            raise ValueError("Q and s2 must be positive")
        for n in ("phi_P", "phi_M", "phi_A"):
            if not 0.0 <= getattr(self, n) < 1.0:
                raise ValueError(f"{n} must lie in [0, 1)")
        for n in ("s_P", "s_M", "s_A"):
            if getattr(self, n) < 0.0:
                raise ValueError(f"{n} must be non-negative")

    def alpha_at(self, g: float) -> np.ndarray:
        """The dynamics with a fraction `g` of the fitted departure in force.

        alpha(g) = (1-g) * alpha_FLAT + g * alpha, on the straight line through
        the fitted dynamics and the degenerate member alpha_FLAT = (1,0,...,0).

        **g = 0 is exactly the parent's local-level model**, so "the dynamics
        have stopped governing" is a member of the family with a likelihood of
        its own, not an absence of evidence.  g = 1 is the fitted dynamics and
        g > 1 extrapolates away from flat -- which is the direction that matters
        when a fitted alpha is too damped and its forecasts decay too fast.

        The line is clipped where it would leave the unit disc, since an
        explosive alpha is not a hypothesis about anything.
        """
        flat = np.zeros(self.p)
        flat[0] = 1.0
        a = np.asarray(self.alpha, dtype=float)
        cand = (1.0 - g) * flat + g * a
        if _radius(cand) <= 1.0 + 1e-9:
            return cand
        # An explosive BASE alpha must stay explosive: the likelihood's -inf
        # guard is what stops an unconstrained search wandering outside the
        # unit disc, and clipping here would flatten the surface it needs.
        # Only the channel's own excursions are pulled back, toward g = 1.
        if _radius(a) > 1.0 + 1e-9:
            return cand
        lo, hi = 1.0, g                               # lo is known stable
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if _radius((1.0 - mid) * flat + mid * a) <= 1.0 + 1e-9:
                lo = mid
            else:
                hi = mid
        return (1.0 - lo) * flat + lo * a

    @property
    def p(self) -> int:
        return len(self.alpha)

    @property
    def roots(self) -> np.ndarray:
        """Roots of z^p - alpha_1 z^{p-1} - ... - alpha_p: the ODE's modes."""
        return np.roots(np.concatenate([[1.0], -np.asarray(self.alpha)]))

    @property
    def companion(self) -> np.ndarray:
        p = self.p
        F = np.zeros((p, p))
        F[0] = self.alpha
        if p > 1:
            F[1:, :-1] = np.eye(p - 1)
        return F

    def memory(self) -> float:
        """1 / (1 - |z|max): the horizon over which alpha affects a forecast.

        Infinite when a root sits on the unit circle, which is what a constant
        offset is.  This is the number of steps of genuine predictive power.
        """
        r = float(np.max(np.abs(self.roots)))
        # A fitted offset root lands near 1 but never on it, so this is often
        # very large rather than infinite; that is the honest reading.
        return math.inf if r >= 1.0 else 1.0 / (1.0 - r)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["alpha"] = list(self.alpha)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        return cls(**d)

    def _vec(self) -> np.ndarray:
        return np.concatenate([
            np.asarray(self.alpha, dtype=float),
            [math.log(self.Q), math.log(self.s2),
             _logit(self.phi_P), _logit(self.phi_M),
             math.log(max(self.s_P, 1e-6)), math.log(max(self.s_M, 1e-6)),
             _logit(self.phi_A), math.log(max(self.s_A, 1e-6))]])

    @classmethod
    def _from_vec(cls, v: np.ndarray, p: int) -> "Params":
        return cls(alpha=tuple(v[:p]), Q=math.exp(v[p]), s2=math.exp(v[p + 1]),
                   phi_P=_expit(v[p + 2]), phi_M=_expit(v[p + 3]),
                   s_P=math.exp(v[p + 4]), s_M=math.exp(v[p + 5]),
                   phi_A=_expit(v[p + 6]), s_A=math.exp(v[p + 7]))


def _logit(x: float) -> float:
    x = min(max(x, 1e-9), 1.0 - 1e-9)
    return math.log(x / (1.0 - x))


def _expit(z: float) -> float:
    # branch to stay finite at large |z|; an unconstrained search reaches there
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


# -------------------------------------------------------------------- results
@dataclass
class Step:
    """Everything known after one observation."""

    mean: float                  #: posterior mean of x_t
    var: float                   #: posterior variance of x_t
    innovation: float            #: y_t - prior mean
    loglik: float                #: log predictive density of y_t

    # amplitude conservation: these three sum to 1
    share_prior: float           #: attributed to already being wrong about x
    share_process: float         #: attributed to the process genuinely moving
    share_measurement: float     #: attributed to measurement noise

    # the parent's four mode coordinates, in log-scale nats
    process_anomaly: float
    process_regime: float
    measurement_anomaly: float
    measurement_regime: float

    #: running lag-1 innovation autocorrelation.  Zero when the model fits;
    #: sustained departure means `alpha` no longer does.  See module docstring.
    whiteness: float

    #: posterior mean of g, the fraction of the fitted dynamics currently in
    #: force.  1 = the fitted ODE, 0 = flat (the parent's model exactly),
    #: above 1 = more persistent than fitted.  Constant at 1 when s_A = 0.
    dynamics: float = 1.0

    @property
    def process_scale(self) -> float:
        return self.process_anomaly + self.process_regime

    @property
    def measurement_scale(self) -> float:
        return self.measurement_anomaly + self.measurement_regime


@dataclass
class FilterResult:
    """Batch output.  Every field is an array of length n."""

    mean: np.ndarray
    var: np.ndarray
    innovation: np.ndarray
    share_prior: np.ndarray
    share_process: np.ndarray
    share_measurement: np.ndarray
    process_anomaly: np.ndarray
    process_regime: np.ndarray
    measurement_anomaly: np.ndarray
    measurement_regime: np.ndarray
    whiteness: np.ndarray
    dynamics: np.ndarray
    state_mean: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    state_cov: np.ndarray = field(default_factory=lambda: np.empty((0, 0, 0)))
    loglik: float = 0.0

    @property
    def process_scale(self) -> np.ndarray:
        """E[lamP | data]; equals process_anomaly + process_regime."""
        return self.process_anomaly + self.process_regime

    @property
    def measurement_scale(self) -> np.ndarray:
        """E[lamM | data]; equals measurement_anomaly + measurement_regime."""
        return self.measurement_anomaly + self.measurement_regime

    @property
    def modes(self) -> np.ndarray:
        """(n, 4) array of the signed mode coordinates: PA, PR, MA, MR."""
        return np.column_stack([self.process_anomaly, self.process_regime,
                                self.measurement_anomaly,
                                self.measurement_regime])

    def __len__(self) -> int:
        return len(self.mean)


# ------------------------------------------------------------------- the grid
def _gauss_hermite(n: int):
    z, w = np.polynomial.hermite_e.hermegauss(n)
    return z, w / w.sum()


def _chain(phi: float, s: float, n: int):
    """Quadrature grid for a stationary AR(1) log-scale, and its kernel."""
    z, w = _gauss_hermite(n)
    lam = s * z
    if not s > 0.0 or s * s <= 0.0:
        return np.zeros(n), w, np.tile(w, (n, 1))
    nu = max(s * s * (1.0 - phi * phi), 1e-12)
    ex = (-0.5 * (lam[None, :] - phi * lam[:, None]) ** 2 / nu
          + 0.5 * lam[None, :] ** 2 / (s * s))
    T = w[None, :] * np.exp(np.clip(ex, -700.0, 700.0))
    T /= T.sum(1, keepdims=True)
    return lam, w, T


# ------------------------------------------------- the same grid, for a batch
# The recursion is sequential in t and cannot be vectorised over time.  It can be
# vectorised over PARAMETER VECTORS, and that is nearly free: the per-step cost is
# dominated by numpy dispatch and by the (nA, p, p) contractions, not by the batch
# axis, so widening every array from (G,) to (B, G) costs far less than running it
# B times.  The parent workstream measured the same thing first and acted on it
# (adaptive-random-walk-filter, SPEED-001/002); this is that finding carried over
# to a filter with a matrix state and a third channel.
#
# The caps below are the batch's equivalents of the -60/+60 clips in the scalar
# path: they keep exp() and the logit finite where an unconstrained search would
# not.  They sit far outside any estimate the model can meaningfully produce.
_LOGIT_CAP = 14.0                       # |logit phi| <= 14  <=>  phi within 8e-7 of an end
_LOG_S_CAP = math.log(20.0)             # s <= 20
_LOG_S_FLOOR = math.log(1e-6)           # s >= 1e-6, so no channel ever collapses


def _chain_batch(phi: np.ndarray, s: np.ndarray, n: int):
    """:func:`_chain` for a batch of (phi, s); returns (B, n) lam and (B, n, n) T.

    ``s`` is floored by the caller at 1e-6 rather than allowed to be zero, so the
    ``s = 0`` branch of :func:`_chain` -- which collapses the channel to a single
    node -- has no batch analogue and is not needed: at s = 1e-6 every node
    carries lam within 5e-6 of zero and the two agree to ~1e-11.
    """
    z, w = _gauss_hermite(n)
    lam = s[:, None] * z
    nu = np.maximum(s * s * (1.0 - phi * phi), 1e-12)[:, None, None]
    ex = (-0.5 * (lam[:, None, :] - phi[:, None, None] * lam[:, :, None]) ** 2 / nu
          + 0.5 * lam[:, None, :] ** 2 / (s * s)[:, None, None])
    T = w * np.exp(np.clip(ex, -700.0, 700.0))
    T /= T.sum(2, keepdims=True)
    return lam, np.broadcast_to(w, lam.shape), T


def _alpha_line(a: np.ndarray, flat: np.ndarray, g: float) -> np.ndarray:
    """:meth:`Params.alpha_at` without needing a Params.  Same clipping rule."""
    cand = (1.0 - g) * flat + g * a
    if _radius(cand) <= 1.0 + 1e-9 or _radius(a) > 1.0 + 1e-9:
        return cand
    lo, hi = 1.0, g
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _radius((1.0 - mid) * flat + mid * a) <= 1.0 + 1e-9:
            lo = mid
        else:
            hi = mid
    return (1.0 - lo) * flat + lo * a


def _loglik_batch(y: np.ndarray, V: np.ndarray, p: int, order: int,
                  order_A: int) -> np.ndarray:
    """Marginal log-likelihood of ``y`` at every unconstrained vector in ``V``.

    ``V`` is (B, p + 8) in the coordinates of :meth:`Params._vec`.  The recursion
    is the one in :meth:`OdeFilter.update`, carried out for all B vectors at once
    and producing only the likelihood -- no shares, no mode coordinates, no Step
    objects -- because that is all a fit needs.  Vectors that drive the recursion
    out of range score ``-inf``, matching the scalar path's ``_Numerical`` guard.
    """
    V = np.atleast_2d(np.asarray(V, dtype=float))
    B = V.shape[0]

    # A channel pinned off across the WHOLE batch does not need a grid: at
    # s <= 1e-6 every node carries lam within 5e-6 of zero, so collapsing it to
    # one node changes the likelihood in the eleventh figure and divides the
    # state count by `order`.  fit() pins channels off for entire stages, so
    # this is not a rare case -- it is most of the fit.  Without it, stage 2
    # would carry 75 states to represent one.
    nP = order if np.any(V[:, p + 4] > _LOG_S_FLOOR) else 1
    nM = order if np.any(V[:, p + 5] > _LOG_S_FLOOR) else 1
    nA = order_A if np.any(V[:, p + 7] > _LOG_S_FLOOR) else 1
    G = nA * nP * nM

    alpha = V[:, :p]
    Q = np.exp(np.clip(V[:, p], -700.0, 700.0))
    S2 = np.exp(np.clip(V[:, p + 1], -700.0, 700.0))
    cap = _LOGIT_CAP + 1.0
    phP = 1.0 / (1.0 + np.exp(-np.clip(V[:, p + 2], -cap, cap)))
    phM = 1.0 / (1.0 + np.exp(-np.clip(V[:, p + 3], -cap, cap)))
    phA = 1.0 / (1.0 + np.exp(-np.clip(V[:, p + 6], -cap, cap)))
    lo, hi = _LOG_S_FLOOR - 1.0, _LOG_S_CAP + 1.0
    sP = np.exp(np.clip(V[:, p + 4], lo, hi))
    sM = np.exp(np.clip(V[:, p + 5], lo, hi))
    sA = np.exp(np.clip(V[:, p + 7], lo, hi))

    lamP, wP, TP = _chain_batch(phP, sP, nP)
    lamM, wM, TM = _chain_batch(phM, sM, nM)
    lamA, wA, TA = _chain_batch(phA, sA, nA)

    # joint grid, ordered exactly as _build does it: A outermost, then P, then M
    LP = np.tile(np.repeat(lamP, nM, axis=1), (1, nA))
    LM = np.tile(lamM, (1, nP * nA))
    nPM = nP * nM
    TPM = (TP[:, :, None, :, None] * TM[:, None, :, None, :]).reshape(B, nPM, nPM)
    T = (TA[:, :, None, :, None] * TPM[:, None, :, None, :]).reshape(B, G, G)
    pi = ((wA[:, :, None] * (wP[:, :, None] * wM[:, None, :]).reshape(B, 1, nPM))
          .reshape(B, G).copy())
    Aidx = np.repeat(np.arange(nA), nPM)

    # one companion matrix per (batch element, dynamics node)
    flat = np.zeros(p)
    flat[0] = 1.0
    gs = 1.0 + lamA
    Fs = np.empty((B, nA, p, p))
    for b in range(B):
        for j in range(nA):
            Fs[b, j] = _companion(_alpha_line(alpha[b], flat, float(gs[b, j])))

    Qg = Q[:, None] * np.exp(np.clip(LP, -60.0, 60.0))
    Rg = S2[:, None] * np.exp(np.clip(LM, -60.0, 60.0))

    y0 = float(y[0]) if np.isfinite(y[0]) else 0.0
    m = np.full((B, p), y0)
    P = (np.eye(p) * p)[None] * (Rg.max(1) + Qg.max(1))[:, None, None]
    ll = np.zeros(B)
    bad = np.zeros(B, dtype=bool)
    e1 = np.zeros(p)
    e1[0] = 1.0

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for v in y:
            pi = np.einsum("bi,bij->bj", pi, T)
            mj = np.einsum("bjar,br->bja", Fs, m)
            Aj = np.einsum("bjar,brs,bjcs->bjac", Fs, P, Fs)
            a00 = Aj[:, :, 0, 0]

            if not np.isfinite(v):                       # missing: propagate only
                piA = pi.reshape(B, nA, -1).sum(2)
                m_new = np.einsum("bj,bja->ba", piA, mj)
                P = np.einsum("bj,bjac->bac", piA, Aj)
                P[:, 0, 0] += (pi * Qg).sum(1)
                dm = mj - m_new[:, None, :]
                P = P + np.einsum("bj,bja,bjc->bac", piA, dm, dm)
                m = m_new
                continue

            S = a00[:, Aidx] + Qg + Rg
            step_bad = ~np.all(np.isfinite(S), axis=1) | np.any(S <= 0.0, axis=1)
            bad |= step_bad
            S = np.where(bad[:, None], 1.0, S)

            eA = v - mj[:, :, 0]
            e_all = eA[:, Aidx]
            lg = -0.5 * (np.log(S) + e_all * e_all / S)
            lg = np.where(np.isfinite(lg), lg, -np.inf)
            mx = lg.max(1)
            w = pi * np.exp(lg - mx[:, None])
            Z = w.sum(1)
            ll += np.where(bad, 0.0, np.log(Z) + mx - 0.5 * _LOG2PI)
            pi = w / Z[:, None]

            row = Aj[:, :, :, 0][:, Aidx, :] + Qg[:, :, None] * e1
            K = row / S[:, :, None]
            mm = mj[:, Aidx, :] + K * e_all[:, :, None]
            m_new = np.einsum("bg,bga->ba", pi, mm)

            piA = pi.reshape(B, nA, -1).sum(2)
            Pn = np.einsum("bj,bjac->bac", piA, Aj)
            Pn[:, 0, 0] += (pi * Qg).sum(1)
            Pn -= np.einsum("bg,bga,bgc->bac", pi, K, row)
            dm = mm - m_new[:, None, :]
            Pn += np.einsum("bg,bga,bgc->bac", pi, dm, dm)

            # a diverged element must not poison the shared arithmetic
            if bad.any():
                m_new = np.where(bad[:, None], 0.0, m_new)
                Pn = np.where(bad[:, None, None], np.eye(p), Pn)
                pi = np.where(bad[:, None], 1.0 / G, pi)
            m, P = m_new, Pn

    return np.where(bad, -np.inf, ll)


# ------------------------------ the s = 0 face, with sigma^2 concentrated out
# On the face s_P = s_M = s_A = 0 the grid collapses to one state and the model is
# an ordinary Kalman filter for a known recurrence.  There the recursion is
# homogeneous of degree 1 in the noise scale: writing Q = q * S2 and P_t = S2 * p_t
# leaves every gain, and hence every innovation, a function of the RATIO q alone.
# So S2 is concentrated out in closed form and the face is a one-parameter
# problem.  That matters here more than it did in the parent, because Q from
# moments is the badly conditioned quantity in this filter (see _moment_noises:
# 151x amplification), and this replaces the 13-pass likelihood scan that existed
# to paper over it with an exact optimum costing one batched pass.
def _face_scan(y: np.ndarray, alpha: np.ndarray, qs: np.ndarray):
    """Profile the face at ratios ``qs``; returns (S2, loglik) per ratio."""
    qs = np.atleast_1d(np.asarray(qs, dtype=float))
    B, p = qs.size, alpha.size
    F = _companion(alpha)
    m = np.full((B, p), float(y[0]) if np.isfinite(y[0]) else 0.0)
    P = (np.eye(p) * p)[None] * (1.0 + qs)[:, None, None]
    acc = np.zeros(B)                        # sum e^2 / Stilde
    lsum = np.zeros(B)                       # sum log Stilde
    cnt = 0
    e1 = np.zeros(p)
    e1[0] = 1.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for v in y:
            mj = m @ F.T
            Aj = np.einsum("ar,brs,cs->bac", F, P, F)
            Aj[:, 0, 0] += qs
            if not np.isfinite(v):
                m, P = mj, Aj
                continue
            S = Aj[:, 0, 0] + 1.0
            e = v - mj[:, 0]
            acc += e * e / S
            lsum += np.log(S)
            row = Aj[:, :, 0]
            K = row / S[:, None]
            m = mj + K * e[:, None]
            P = Aj - K[:, :, None] * row[:, None, :]
            cnt += 1
    if cnt == 0:
        raise ValueError("no finite observations")
    # An explosive alpha can drive acc to zero or to a non-finite value; that
    # ratio simply has no profile optimum, so it is refused rather than warned
    # about.
    ok = np.isfinite(acc) & (acc > 0.0) & np.isfinite(lsum)
    S2 = np.where(ok, acc / cnt, 1.0)
    ll = -0.5 * (cnt * (_LOG2PI + np.log(S2) + 1.0) + np.where(ok, lsum, 0.0))
    return S2, np.where(ok, ll, -np.inf)


_Q_SCAN = np.logspace(-8.0, 3.0, 45)         # q is a ratio, so it needs no scaling


def _face_optimum(y: np.ndarray, alpha: np.ndarray) -> tuple[float, float]:
    """(Q, S2) maximising the face likelihood at a fixed ``alpha``."""
    S2s, lls = _face_scan(y, alpha, _Q_SCAN)
    i = int(np.argmax(lls))
    lo = math.log(_Q_SCAN[max(i - 1, 0)])
    hi = math.log(_Q_SCAN[min(i + 1, _Q_SCAN.size - 1)])
    q = float(_Q_SCAN[i])
    if hi > lo:
        from scipy.optimize import minimize_scalar
        r = minimize_scalar(lambda z: -_face_scan(y, alpha, math.exp(z))[1][0],
                            bounds=(lo, hi), method="bounded",
                            options=dict(xatol=1e-4))
        if np.isfinite(r.fun):
            q = math.exp(float(r.x))
    S2 = float(_face_scan(y, alpha, q)[0][0])
    return max(q * S2, 1e-300), max(S2, 1e-300)


# ------------------------------------------------------------ the start screen
# The persistence grid the old fit() scanned, now crossed with the scale of each
# channel rather than fixing both at one value, and evaluated in one batch: two
# hundred starts cost about ten evaluations, so the screen is far wider than the
# two hand-picked starts it replaces and still cheaper than them.
#
# The scale grid has to reach past 1.  The old pair (0.03, 0.6) was chosen on
# smooth synthetic data where the process-scale channel fits dead; on daily
# crypto log-price it fits to s_P = 1.24, outside the old grid entirely, and a
# screen that cannot propose the answer cannot rank it.
_PHI_GRID = (0.02, 0.25, 0.5, 0.75, 0.95)
_S_P_GRID = (0.03, 0.2, 0.6, 1.2)
_S_M_GRID = (0.03, 0.6)
_QUIET = 0.1                    # divides "no scale structure" starts from the rest
_SCREEN_KEEP = 2                # distinct starts kept per half of that split
_PHI_A_GRID = (0.5, 0.9, 0.99)
_S_A_GRID = (0.05, 0.15, 0.6)

#: Ceiling on the fit objective, in nats per point.  Where the recursion
#: overflows the scalar path returns -inf and a search cannot back out of an
#: infinity, so the surface is flattened at a value no real fit approaches.
_BAD_NLL = 1e4


def difference_matrix(p: int) -> np.ndarray:
    """D with D[i, j] = (-1)^j C(i, j): lag vector -> (x, Dx, D^2 x, ...).

    An involution, so it is its own inverse.  |det| = 1, so the two bases carry
    identical information and the conversion neither creates nor destroys any.
    """
    D = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1):
            D[i, j] = ((-1.0) ** j) * math.comb(i, j)
    return D


# ----------------------------------------------------------------- the filter
class OdeFilter:
    """Adaptive filter for a noisily observed linear recurrence.

    Parameters
    ----------
    params : Params, optional
        Omit and call :meth:`fit`.
    order : int
        Quadrature nodes per noise channel; the joint grid has ``order**2``
        states.  A numerical resolution, not a model choice.
    order_A : int
        Quadrature nodes for the dynamics channel.  It multiplies the cost of
        every step and of every likelihood evaluation, so the default is lower
        than ``order``: g is one smooth scalar and does not need as fine a
        grid.  Ignored entirely when ``s_A = 0``.
    """

    def __init__(self, params: Params | None = None, order: int = 5,
                 order_A: int = 3):
        if order < 3:
            raise ValueError("order must be at least 3")
        if order_A < 3:
            raise ValueError("order_A must be at least 3")
        self.params = params
        self.order = int(order)
        self.order_A = int(order_A)
        self._built = None
        self.reset()

    def __repr__(self) -> str:
        if self.params is None:
            return f"OdeFilter(unfitted, order={self.order})"
        p = self.params
        return (f"OdeFilter(alpha={np.round(p.alpha, 4).tolist()}, "
                f"Q={p.Q:.4g}, s2={p.s2:.4g}, order={self.order})")

    def to_dict(self) -> dict:
        if self.params is None:
            raise ValueError("filter is not fitted")
        return {"params": self.params.to_dict(), "order": self.order,
                "order_A": self.order_A}

    @classmethod
    def from_dict(cls, d: dict) -> "OdeFilter":
        return cls(Params.from_dict(d["params"]), order=int(d.get("order", 5)),
                   order_A=int(d.get("order_A", 3)))

    # ------------------------------------------------------------------ grid
    def _build(self):
        key = (self.params, self.order, self.order_A)
        if self._built is not None and self._built[0] == key:
            return self._built[1]
        pr, n = self.params, self.order
        lamP, wP, TP = _chain(pr.phi_P, pr.s_P, n)
        lamM, wM, TM = _chain(pr.phi_M, pr.s_M, n)
        # the dynamics channel: g = 1 + lamA, so g = 1 is the fitted alpha and
        # g = 0 is FLAT.  With s_A = 0 it collapses to a single node and the
        # recursion is bit-for-bit what it was before this channel existed.
        if pr.s_A > 0.0:
            nA = self.order_A
            lamA, wA, TA = _chain(pr.phi_A, pr.s_A, nA)
        else:
            nA, lamA, wA, TA = 1, np.zeros(1), np.ones(1), np.ones((1, 1))
        nN = n * n
        gd = {"LP": np.tile(np.repeat(lamP, n), nA),
              "LM": np.tile(np.tile(lamM, n), nA),
              "LA": np.repeat(lamA, nN),
              "T": np.kron(TA, np.kron(TP, TM)),
              "pi0": np.kron(wA, np.kron(wP, wM)),
              "Aidx": np.repeat(np.arange(nA), nN),
              "starts": np.arange(nA) * nN,
              "nA": nA,
              "gs": 1.0 + lamA}
        gd["Fs"] = np.stack([_companion(pr.alpha_at(float(v)))
                             for v in gd["gs"]])
        gd["Qg"] = pr.Q * np.exp(np.clip(gd["LP"], -60.0, 60.0))
        gd["Rg"] = pr.s2 * np.exp(np.clip(gd["LM"], -60.0, 60.0))
        self._built = (key, gd)
        return gd

    # ------------------------------------------------------------- streaming
    def reset(self) -> "OdeFilter":
        """Clear the streaming state.  Returns self, so it chains."""
        self._pi = None
        self._m = None
        self._P = None
        self._prev_lamP = 0.0
        self._prev_lamM = 0.0
        self._loglik = 0.0
        self._e_prev = 0.0
        self._ee = 0.0
        self._e2 = 0.0
        self._nw = 0
        return self

    def update(self, y: float) -> Step:
        """Absorb one observation.  NaN is treated as missing."""
        if self.params is None:
            raise ValueError("filter is not fitted; call fit() or pass params")
        g = self._build()
        p = self.params.p
        if self._pi is None:
            self._pi = g["pi0"].copy()
            y0 = float(y) if np.isfinite(y) else 0.0
            self._m = np.full(p, y0)
            self._P = np.eye(p) * float(g["Rg"].max() + g["Qg"].max()) * p

        pi = self._pi @ g["T"]
        Fs, Aidx, starts = g["Fs"], g["Aidx"], g["starts"]
        mj = Fs @ self._m                              # (nA, p) per dynamics node
        Aj = Fs @ self._P @ Fs.transpose(0, 2, 1)      # (nA, p, p)
        a00, a0 = Aj[:, 0, 0], Aj[:, :, 0]

        if not np.isfinite(y):                    # missing observation
            self._pi = pi
            piA = np.add.reduceat(pi, starts)
            m_new = piA @ mj
            Pbar = np.einsum("j,jab->ab", piA, Aj)
            Pbar[0, 0] += float(pi @ g["Qg"])
            dm = mj - m_new
            Pbar += np.einsum("j,ja,jb->ab", piA, dm, dm)
            self._m, self._P = m_new, Pbar
            lamP, lamM = float(pi @ g["LP"]), float(pi @ g["LM"])
            st = Step(float(m_new[0]), float(Pbar[0, 0]), math.nan, 0.0,
                      1.0, 0.0, 0.0,
                      lamP - self.params.phi_P * self._prev_lamP,
                      self.params.phi_P * self._prev_lamP,
                      lamM - self.params.phi_M * self._prev_lamM,
                      self.params.phi_M * self._prev_lamM,
                      self._whiteness(), 1.0 + float(pi @ g["LA"]))
            self._prev_lamP, self._prev_lamM = lamP, lamM
            return st

        Qg, Rg = g["Qg"], g["Rg"]
        S = a00[Aidx] + Qg + Rg                   # predictive variance per node
        # An unconstrained search reaches explosive alpha, where P overflows and
        # S goes non-finite or non-positive.  Signal it rather than emitting a
        # nan log-likelihood the optimiser would then chase.
        if not np.all(np.isfinite(S)) or np.any(S <= 0.0):
            raise _Numerical("non-positive predictive variance")
        eA = float(y) - mj[:, 0]                  # innovation per dynamics node
        e_all = eA[Aidx]
        # the reported innovation is against the PRIOR mixture mean, which is
        # what a caller means by "how surprised were we"
        e = float(y) - float(pi @ mj[Aidx, 0])
        lg = -0.5 * (np.log(S) + e_all * e_all / S)
        mx = float(lg.max())
        w = pi * np.exp(lg - mx)
        Z = float(w.sum())
        ll = math.log(Z) + mx - 0.5 * _LOG2PI
        S_pred = float(pi @ (S + (mj[Aidx, 0] - (pi @ mj[Aidx, 0])) ** 2))
        pi = w / Z

        e1 = np.zeros(p)
        e1[0] = 1.0
        row = a0[Aidx] + Qg[:, None] * e1         # (G, p): the prior's row 0
        K = row / S[:, None]
        mm = mj[Aidx] + K * e_all[:, None]        # (G, p): per-node posterior mean
        m_new = pi @ mm

        # collapse (GPB1): mean conditional covariance + spread of the means.
        # The means now differ across dynamics nodes as well as noise nodes,
        # so the spread term carries the dynamics disagreement too.
        piA = np.add.reduceat(pi, starts)
        Pbar = np.einsum("j,jab->ab", piA, Aj)
        Pbar[0, 0] += float(pi @ Qg)
        Pbar -= np.einsum("g,ga,gb->ab", pi, K, row)
        dm = mm - m_new
        Pbar += np.einsum("g,ga,gb->ab", pi, dm, dm)

        lamP, lamM = float(pi @ g["LP"]), float(pi @ g["LM"])
        st = Step(
            mean=float(m_new[0]), var=float(Pbar[0, 0]), innovation=e, loglik=ll,
            share_prior=float(pi @ (a00[Aidx] / S)),
            share_process=float(pi @ (Qg / S)),
            share_measurement=float(pi @ (Rg / S)),
            process_anomaly=lamP - self.params.phi_P * self._prev_lamP,
            process_regime=self.params.phi_P * self._prev_lamP,
            measurement_anomaly=lamM - self.params.phi_M * self._prev_lamM,
            measurement_regime=self.params.phi_M * self._prev_lamM,
            whiteness=self._accum_whiteness(e, S_pred),
            dynamics=1.0 + float(pi @ g["LA"]),
        )
        self._pi, self._m, self._P = pi, m_new, Pbar
        self._prev_lamP, self._prev_lamM = lamP, lamM
        self._loglik += ll
        return st

    # -------------------------------------------------------- the diagnostic
    def _accum_whiteness(self, e: float, S: float) -> float:
        r = e / math.sqrt(max(S, 1e-300))
        self._ee += r * self._e_prev
        self._e2 += r * r
        self._nw += 1
        self._e_prev = r
        return self._whiteness()

    def _whiteness(self) -> float:
        if self._nw < 3 or self._e2 <= 0.0:
            return 0.0
        return float(self._ee / self._e2)

    # ------------------------------------------------------------ forecasting
    def predict(self, horizon: int = 1) -> tuple[float, float]:
        """Mean and variance of x_{t+h} given everything so far.

        This is where alpha earns its keep: tracking error is nearly blind to
        the dynamics, forecast error is not (exploration/0006).
        """
        if self._pi is None:
            raise ValueError("nothing observed yet")
        g = self._build()
        Fs, starts = g["Fs"], g["starts"]
        pi, m, P = self._pi, self._m.copy(), self._P.copy()
        for _ in range(int(horizon)):
            pi = pi @ g["T"]
            piA = np.add.reduceat(pi, starts)
            mj = Fs @ m
            Aj = Fs @ P @ Fs.transpose(0, 2, 1)
            m = piA @ mj
            P = np.einsum("j,jab->ab", piA, Aj)
            dm = mj - m
            P += np.einsum("j,ja,jb->ab", piA, dm, dm)
            P[0, 0] += float(pi @ g["Qg"])
        return float(m[0]), float(P[0, 0])

    def predict_mixture(self, horizon: int = 1, observation: bool = True):
        """The predictive distribution, as a mixture rather than as two numbers.

        Returns ``(w, mean, var)``, each of length ``order**2 * nA``: the
        forecast is ``sum_g w_g N(mean_g, var_g)``.  With ``observation=True``
        the measurement noise of each node is included, so this is the
        distribution of ``y_{t+h}``; otherwise it is the state's.

        :meth:`predict` returns this mixture's mean and its total variance --
        exactly, to 1e-10, which the test suite checks -- and **that reduction
        is lossy in a way that matters.**  Two numbers are a Gaussian, and where
        the scale channels are alive this mixture is not one: it is
        right-skewed in ``S``, so its mean variance is far larger than its
        typical one (a factor ``exp(s_P^2)`` between ``E[S]`` and
        ``1/E[1/S]`` in the limit where the state is known, measured at 2-4.3
        on daily crypto), and its tails are fat where a Gaussian's are not.

        ``E[S]`` remains the right quantity for anything second-moment: a
        mean-square risk target, or a Kelly fraction, which to leading order is
        ``mu/E[S]`` and **not** ``mu*E[1/S]`` -- see
        ``crypto-predictivity/0005``, where the tempting Jensen argument for the
        latter is checked against the exact Kelly root and refused.  What the
        reduction genuinely costs is the shape: on daily crypto the mixture
        scores 0.07-0.16 nats/point above its own Gaussian summary, which is
        more than any effect that workstream measured on the forecast mean.

        So this returns the mixture, and the caller takes the functional its
        decision needs -- a quantile, a tail probability, the density itself.
        """
        if self._pi is None:
            raise ValueError("nothing observed yet")
        h = int(horizon)
        if h < 1:
            raise ValueError("horizon must be at least 1")
        g = self._build()
        Fs, starts = g["Fs"], g["starts"]
        pi, m, P = self._pi, self._m.copy(), self._P.copy()
        for _ in range(h - 1):                 # collapse all but the last step
            pi = pi @ g["T"]
            piA = np.add.reduceat(pi, starts)
            mj = Fs @ m
            Aj = Fs @ P @ Fs.transpose(0, 2, 1)
            m = piA @ mj
            P = np.einsum("j,jab->ab", piA, Aj)
            dm = mj - m
            P += np.einsum("j,ja,jb->ab", piA, dm, dm)
            P[0, 0] += float(pi @ g["Qg"])
        pi = pi @ g["T"]
        Aidx = g["Aidx"]
        mj = Fs @ m
        Aj = Fs @ P @ Fs.transpose(0, 2, 1)
        var = Aj[:, 0, 0][Aidx] + g["Qg"]
        if observation:
            var = var + g["Rg"]
        return pi.copy(), mj[Aidx, 0].copy(), var

    def derivatives(self) -> tuple[np.ndarray, np.ndarray]:
        """The current posterior in (x, Dx, D^2 x, ...) coordinates.

        A fixed invertible integer change of basis, so nothing is created or
        lost; the growth of the diagonal is the noise amplification of
        differencing, reported rather than incurred.
        """
        if self._m is None:
            raise ValueError("nothing observed yet")
        D = difference_matrix(self.params.p)
        return D @ self._m, D @ self._P @ D.T

    # ----------------------------------------------------------------- batch
    def loglik(self, y) -> float:
        return self._run(np.asarray(y, dtype=float), want=False)

    def filter(self, y) -> FilterResult:
        return self._run(np.asarray(y, dtype=float), want=True)

    def _run(self, y: np.ndarray, want: bool):
        if self.params is None:
            raise ValueError("filter is not fitted; call fit() or pass params")
        if y.ndim != 1 or y.size == 0:
            raise ValueError("y must be a non-empty 1-D array")
        saved = (self._pi, self._m, self._P, self._prev_lamP, self._prev_lamM,
                 self._loglik, self._e_prev, self._ee, self._e2, self._nw)
        try:
            self.reset()
            if not want:
                total = 0.0
                try:
                    for v in y:
                        total += self.update(v).loglik
                except _Numerical:
                    return -np.inf
                return total
            n, p = y.size, self.params.p
            cols = ("mean", "var", "innovation", "share_prior", "share_process",
                    "share_measurement", "process_anomaly", "process_regime",
                    "measurement_anomaly", "measurement_regime", "whiteness",
                    "dynamics")
            out = {c: np.empty(n) for c in cols}
            sm = np.empty((n, p))
            sc = np.empty((n, p, p))
            total = 0.0
            for i, v in enumerate(y):
                st = self.update(v)
                for c in cols:
                    out[c][i] = getattr(st, c)
                sm[i], sc[i] = self._m, self._P
                total += st.loglik
            return FilterResult(loglik=total, state_mean=sm, state_cov=sc, **out)
        finally:
            (self._pi, self._m, self._P, self._prev_lamP, self._prev_lamM,
             self._loglik, self._e_prev, self._ee, self._e2, self._nw) = saved

    # ------------------------------------------------------------------- fit
    @classmethod
    def fit(cls, y, p: int = 3, order: int = 5, max_iter: int = 400,
            scales: bool = True, dynamics: bool = True,
            order_A: int = 3) -> "OdeFilter":
        """Learn every parameter from a series and return a fitted filter.

        ``p`` is the order of the recurrence: 3 is a second-order ODE plus a
        constant offset, which is the class this filter targets.  ``p`` is a
        modelling commitment, not a tuning parameter -- and because each root of
        the characteristic polynomial is a channel, choosing ``p`` is the same
        act as counting channels.

        ``scales=False`` pins the two log-scale channels off, giving an ordinary
        (non-adaptive) recurrence filter.  Useful as a baseline and much faster.
        ``dynamics=False`` additionally pins the dynamics channel off, giving
        a static ``alpha``.
        """
        f = cls(order=order, order_A=order_A)
        f.fit_(y, p=p, max_iter=max_iter, scales=scales, dynamics=dynamics)
        return f

    def fit_(self, y, p: int = 3, max_iter: int = 400,
             scales: bool = True, dynamics: bool = True) -> "OdeFilter":
        """Fit in place.  Returns self.

        Staged, because a nine-dimensional search from one start is not
        reliable.  Every stage is organised around the one measured fact about
        this recursion: it is dispatch- and contraction-bound, so B parameter
        vectors cost far less than B evaluations (:func:`_loglik_batch`).  Start
        screens and finite-difference gradients are therefore batches rather
        than loops, and the search itself is a quasi-Newton method on a surface
        that is smooth in these coordinates.

        stage 0   alpha by instrumental variables, using lags p+1..3p as
                  instruments.  Regressing on observed lags is errors-in-
                  variables and does not merely attenuate the dynamics -- it
                  deletes the oscillation outright (exploration/0002).  Lagging
                  past the order annihilates the measurement noise exactly, for
                  every (Q, S2), without needing either.
        stage 1   (Q, S2) exactly, by concentrating S2 out of the s = 0 face and
                  optimising the one ratio that is left (:func:`_face_optimum`).
                  This replaces both the moment estimate -- whose Q carries a
                  151x error amplification, see :func:`_moment_noises` -- and the
                  13-pass likelihood scan that existed to paper over it.
        stage 2   alpha, Q, S2 by maximum likelihood with the scales off.
        stage 3   the two scale channels: one batched screen over the
                  persistence grid crossed with four splits of the scale between
                  the channels, then ML on the four, then a joint polish.
        stage 4   the dynamics channel (phi_A, s_A), screened the same way, then
                  a final polish.  This one is fitted last because it is the
                  only channel whose grid multiplies the cost of every
                  likelihood evaluation.

        All of it is scaffolding: the estimate is the maximum likelihood
        estimate however it is reached.  ``max_iter`` is a compute budget.
        """
        from scipy.optimize import minimize

        y = np.asarray(y, dtype=float)
        good = y[np.isfinite(y)]
        if good.size < 8 * p:
            raise ValueError(f"need at least {8 * p} finite observations")
        d = np.diff(good)
        g0 = float(np.mean(d * d))
        if not g0 > 0:
            raise ValueError("series is constant; nothing to fit")
        n = max(y.size, 1)
        order, order_A = self.order, self.order_A

        # The objective, for a whole batch of vectors at once.  Nothing here
        # excludes an explosive alpha by fiat -- the likelihood is perfectly
        # computable at many of them and merely bad, and it is the likelihood's
        # job to say so (exploration/0038).  The only intervention is a ceiling,
        # which flattens the region where the recursion overflows outright and
        # the scalar path returns -inf.  It is a numerical guard: it can only
        # ever make a hopeless point look no worse than other hopeless points.
        def obj(V):
            ll = _loglik_batch(y, np.atleast_2d(V), p, order, order_A)
            return np.minimum(np.where(np.isfinite(ll), -ll / n, np.inf), _BAD_NLL)

        lg = math.log(g0)
        bounds = ([(-10.0, 10.0)] * p
                  + [(lg - 80.0, lg + 8.0)] * 2
                  + [(-_LOGIT_CAP, _LOGIT_CAP)] * 2
                  + [(_LOG_S_FLOOR, _LOG_S_CAP)] * 2
                  + [(-_LOGIT_CAP, _LOGIT_CAP), (_LOG_S_FLOOR, _LOG_S_CAP)])
        h = 1e-3

        def opt(v0, idx, maxit):
            """ML over the coordinates ``idx``, gradient by one batched pass."""
            idx = list(idx)
            st = np.zeros((2 * len(idx) + 1, v0.size))
            for k, i in enumerate(idx):
                st[1 + 2 * k, i] = h
                st[2 + 2 * k, i] = -h

            def fg(vs):
                v = v0.copy()
                v[idx] = vs
                f = obj(v + st)
                return f[0], (f[1::2] - f[2::2]) / (2.0 * h)

            r = minimize(fg, v0[idx], jac=True, method="L-BFGS-B",
                         bounds=[bounds[i] for i in idx],
                         options=dict(maxiter=int(maxit), ftol=1e-12, gtol=1e-8))
            out = v0.copy()
            out[idx] = np.clip(r.x, *zip(*[bounds[i] for i in idx]))
            return out, float(obj(out)[0])

        # stage 0 / 1 -- closed-form start
        a0 = _iv_alpha(good, p)
        Q0, s20 = _face_optimum(y, a0)
        off = math.log(1e-6)
        base = np.concatenate([a0, [math.log(Q0), math.log(s20),
                                    _logit(0.5), _logit(0.5), off, off,
                                    _logit(0.9), off]])
        base = np.clip(base, *zip(*bounds))

        # stage 2 -- alpha, Q, S2 with the scales pinned off
        full, bestf = opt(base, range(p + 2), max_iter)

        if not scales:
            self.params = Params._from_vec(full, p)
            self._built = None
            return self.reset()

        # stage 3 -- the scale channels, from one batched screen.
        #
        # Q is the MEDIAN process variance, so switching a log-scale channel on
        # at fixed mean variance moves the median by exp(-s^2/2).  A screen that
        # varies s while holding Q at the homoscedastic fit is therefore not
        # comparing candidates at anything like their own optima, and its ranking
        # is close to meaningless: uncorrected, its best start on BTC scored
        # 2348.7 nats where the corrected one scores 2405.8, and following the
        # uncorrected ranking landed the whole fit in a local optimum 10 nats
        # worse.  The correction costs nothing and is not a tuning choice -- it
        # is the definition of the parameter.
        sidx = [p, p + 2, p + 3, p + 4, p + 5]
        screen = []
        for pp in _PHI_GRID:
            for pm in _PHI_GRID:
                for sp in _S_P_GRID:
                    for sm in _S_M_GRID:
                        v = full.copy()
                        v[p] -= 0.5 * sp * sp
                        v[p + 2:p + 6] = [_logit(pp), _logit(pm),
                                          math.log(sp), math.log(sm)]
                        screen.append(v)
        V = np.clip(np.array(screen), *zip(*bounds))
        val = obj(V)
        loud = V[:, p + 4:p + 6].max(1) > math.log(_QUIET)
        best, bestf = full, bestf
        for m in (~loud, loud):
            # Keep a few starts from each half of the quiet/loud split rather
            # than the single winner, deduplicated by score: where a channel is
            # inert -- and on this data the measurement channel is -- dozens of
            # screen points are the same point, and keeping "the top three"
            # would keep one point three times.
            where = np.flatnonzero(m)
            if where.size == 0:
                continue
            kept, seen = [], []
            for i in where[np.argsort(val[where])]:
                if any(abs(val[i] - s) <= 1e-9 * max(abs(s), 1.0) for s in seen):
                    continue
                seen.append(val[i])
                kept.append(i)
                if len(kept) == _SCREEN_KEEP:
                    break
            for i in kept:
                v, f = opt(V[i], sidx, max_iter)
                if f < bestf:
                    best, bestf = v, f

        full, bestf2 = opt(best, range(p + 6), 2 * max_iter)
        if bestf2 > bestf:
            full = best
        else:
            bestf = bestf2

        if not dynamics:
            self.params = Params._from_vec(full, p)
            self._built = None
            return self.reset()

        # stage 4 -- the dynamics channel.  Screened persistent, because a
        # change in the dynamics is a regime by construction (0025): it has no
        # first-moment signature at all, so the impulsive end of this channel
        # is a different object from the impulsive end of the noise channels.
        aidx = [p + 6, p + 7]
        screen = []
        for pa in _PHI_A_GRID:
            for sa in _S_A_GRID:
                v = full.copy()
                v[aidx] = [_logit(pa), math.log(sa)]
                screen.append(v)
        V = np.clip(np.array(screen), *zip(*bounds))
        val = obj(V)
        cand, f = opt(V[int(np.argmin(val))], aidx, max_iter)
        if f < bestf:
            full, bestf = cand, f

        # and a final polish over everything at once.  The staging exists to
        # find the basin, not to define the estimate: the estimate is the
        # maximum over all nine coordinates jointly, and only a joint step can
        # trade the dynamics channel off against the noise channels it competes
        # with for the same innovation.
        cand, f = opt(full, range(p + 8), 2 * max_iter)
        if f < bestf:
            full, bestf = cand, f

        self.params = Params._from_vec(full, p)
        self._built = None
        return self.reset()


# --------------------------------------------------------- closed-form starts
def _iv_alpha(y: np.ndarray, p: int, extra: int | None = None) -> np.ndarray:
    """alpha by instrumental variables, instruments at lags p+1 .. p+m.

    The residual  y_t - sum_i alpha_i y_{t-i}  involves the measurement noise
    only at times t, t-1, ..., t-p, so every observation at lag >= p+1 is
    uncorrelated with it, for every (Q, S2), without stationarity.  That is the
    exact analogue of the parent's "increments annihilate the level".

    **``m > p`` is a precondition, not a dial.**  At the just-identified m = p the
    fit diverges -- Q comes out at 409 against a truth of 1 -- while m = 2p and
    m = 4p agree to 0.003 (exploration/0028).  So the over-identified case is
    required rather than merely preferred, and a series too short to supply it is
    an error rather than something to degrade quietly into.
    """
    m = extra if extra is not None else 2 * p
    if m <= p:
        raise ValueError(f"need more than {p} instruments, got {m}")
    lo = p + m
    if y.size <= lo + 4 * p:
        m = (y.size - p - 4 * p) // 2
        if m <= p:
            raise ValueError(
                f"series of {y.size} points is too short for p = {p}: "
                f"instrumental variables needs more than p instruments")
        lo = p + m
    idx = np.arange(lo, y.size)
    Y = y[idx]
    W = np.column_stack([y[idx - i] for i in range(1, p + 1)])
    Z = np.column_stack([y[idx - p - j] for j in range(1, m + 1)])
    Wh = Z @ np.linalg.lstsq(Z, W, rcond=None)[0]
    a = np.linalg.lstsq(Wh, Y, rcond=None)[0]
    if not np.all(np.isfinite(a)):
        a = np.zeros(p)
        a[0] = 1.0
    return a


def _moment_noises(y: np.ndarray, a: np.ndarray) -> tuple[float, float]:
    """(Q, S2) in closed form from the residual autocovariances.

    With r_t = y_t - sum_i a_i y_{t-i} = w_t + v_t - sum_i a_i v_{t-i},

        gamma_0 = Q + S2 (1 + |a|^2),
        gamma_k = S2 (-a_k + sum_j a_j a_{j+k}),  k >= 1

    so the lags k >= 1 identify S2 without Q, and gamma_0 then gives Q.  The
    direct analogue of the parent's gamma_0 = Q + 2 S2.

    **S2 comes out well; Q does not, and the reason is structural.**  Q is only
    gamma_0 - S2 |c|^2, and for a smooth process |c|^2 is large: at the target
    class's own parameters |c|^2 = 16.8, so with Q = 1 and S2 = 9, Q is 0.66% of
    gamma_0 and the error amplification is |c|^2 S2 / Q = 151.  Measured over
    six seeds at n = 6000, S2 recovers to 8.5-9.0 against a truth of 9.0 while Q
    lands at 3.0-5.2 against a truth of 1.0.  A 2% error in S2 is a 300% error
    in Q.

    The smoother the process the worse this gets, which is the same trade the
    differencing law (1 - rho_1) describes in exploration/0011.  So the returned
    Q is a scale hint only; `fit_` scans over it by likelihood rather than
    believing it, exactly as the parent's stage 0 does.
    """
    p = len(a)
    lags = np.arange(1, p + 1)
    idx = np.arange(p, y.size)
    r = y[idx] - np.column_stack([y[idx - i] for i in lags]) @ a
    r = r - r.mean()
    g = np.array([float(np.mean(r[:len(r) - k] * r[k:])) for k in range(p + 1)])

    c = np.concatenate([[1.0], -a])
    coef = np.array([float(np.sum(c[:len(c) - k] * c[k:])) for k in range(p + 1)])
    num = float(np.sum(coef[1:] * g[1:]))
    den = float(np.sum(coef[1:] ** 2))
    s2 = num / den if den > 1e-12 else 0.0
    s2 = min(max(s2, 1e-8 * abs(g[0]) + 1e-12), 0.98 * abs(g[0]) / coef[0])
    Q = g[0] - s2 * coef[0]
    if not Q > 0:
        Q = 0.05 * abs(g[0]) + 1e-12
    return float(Q), float(s2)
