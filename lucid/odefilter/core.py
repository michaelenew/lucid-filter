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

For TIME-VARYING known dynamics -- a robotics loop re-linearising around the
operating point each step -- pass a ``linearized_dynamics`` callable
(``state -> p x p`` transition) to ``fit`` or the constructor.  The dynamics are
then the caller's, evaluated at the running state estimate each step (EKF-style);
``alpha`` and the dynamics channel are not fitted, and only the NOISE class is
inferred -- give the filter what you know (the dynamics), it infers what you don't
(the live noise).  A constant callable returning ``companion(alpha)`` reduces to
fixing ``alpha`` exactly.

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


def _pin_maps(p: int, d: int):
    """The exact linear map from the free coefficients to the pinned alpha.

    Pinning d roots at z = 1 writes the characteristic polynomial as

        z^p - sum_i alpha_i z^{p-i}  =  (z - 1)^d (z^m - sum_j beta_j z^{m-j})

    with m = p - d free coefficients beta.  Polynomial multiplication is linear
    in the coefficients, so alpha = base + beta @ M with (base, M) fixed
    integer arrays; this returns them.  The constraint therefore costs nothing
    per evaluation and holds exactly, to the last bit the convolution keeps.

    d = 1 is the constant offset; d = 2 is the linear offset -- a level whose
    RATE of change is part of the state, which is what a climbing or declining
    bias is.  m = 0 (all roots pinned) is legal: alpha is then the binomial
    row of (z - 1)^d and nothing about the dynamics is searched.
    """
    if not 0 <= d <= p:
        raise ValueError("unit_roots must lie in [0, p]")
    u = np.array([math.comb(d, k) * (-1.0) ** k for k in range(d + 1)])
    m = p - d
    U = np.zeros((m + 1, p + 1))
    for i in range(m + 1):
        U[i, i:i + d + 1] = u
    return -U[0, 1:], U[1:, 1:]


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
    unit_roots how many of alpha's roots are PINNED at z = 1.  0 (the default)
               means every root was free -- the current, weaker assumption.
               1 pins the constant offset; 2 pins the linear offset, so a
               climbing or declining bias is part of the state rather than
               something a free root has to crawl toward.  alpha always stores
               the full p coefficients, pinned factor multiplied in, so the
               recursion never sees the constraint -- only the fit does.
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
    unit_roots: int = 0

    def __post_init__(self):
        object.__setattr__(self, "alpha", tuple(float(a) for a in self.alpha))
        object.__setattr__(self, "unit_roots", int(self.unit_roots))
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
        if not 0 <= self.unit_roots <= len(self.alpha):
            raise ValueError("unit_roots must lie in [0, p]")
        if self.unit_roots:
            # the claim is exact by construction (the fit multiplies the pinned
            # factor in), so a violation means alpha and unit_roots came from
            # different places; say so rather than filtering under a false label
            c = np.concatenate([[1.0], -np.asarray(self.alpha)])
            scale = float(np.abs(c).sum())
            for _ in range(self.unit_roots):
                c = np.cumsum(c)                # synthetic division by (z - 1)
                if abs(c[-1]) > 1e-8 * scale:   # the remainder, = poly at z = 1
                    raise ValueError(
                        "alpha does not carry the claimed unit roots")
                c = c[:-1]

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
        """The unconstrained fit coordinates: the FREE coefficients, then the
        eight noise coordinates.  With unit_roots = d the free coefficients are
        the quotient polynomial's beta (p - d of them), recovered by synthetic
        division; :meth:`_from_vec` inverts this exactly."""
        c = np.concatenate([[1.0], -np.asarray(self.alpha)])
        for _ in range(self.unit_roots):
            c = np.cumsum(c)[:-1]               # divide out one (z - 1)
        return np.concatenate([
            -c[1:],
            [math.log(self.Q), math.log(self.s2),
             _logit(self.phi_P), _logit(self.phi_M),
             math.log(max(self.s_P, 1e-6)), math.log(max(self.s_M, 1e-6)),
             _logit(self.phi_A), math.log(max(self.s_A, 1e-6))]])

    @classmethod
    def _from_vec(cls, v: np.ndarray, p: int, unit_roots: int = 0) -> "Params":
        m = p - unit_roots
        if unit_roots:
            base, M = _pin_maps(p, unit_roots)
            alpha = tuple(base + np.asarray(v[:m]) @ M)
        else:
            alpha = tuple(v[:p])
        return cls(alpha=alpha, Q=math.exp(v[m]), s2=math.exp(v[m + 1]),
                   phi_P=_expit(v[m + 2]), phi_M=_expit(v[m + 3]),
                   s_P=math.exp(v[m + 4]), s_M=math.exp(v[m + 5]),
                   phi_A=_expit(v[m + 6]), s_A=math.exp(v[m + 7]),
                   unit_roots=unit_roots)


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

    #: variance of the one-step predictive distribution for y_t, mixture spread
    #: included.  `innovation` and this are the forecast that `loglik` scores,
    #: so a caller scoring forecasts as distributions needs it (see 0034).
    pred_var: float = math.nan

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
    pred_var: np.ndarray = field(default_factory=lambda: np.empty(0))
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
_GAP_FACTOR = 1.5   # node spacing = 1.5 * s: the resolution limit, no dead zone
                    # (research/adaptive-grid finding 11).  Uniform, not Gauss-
                    # Hermite -- GH optimises quadrature accuracy, the wrong
                    # criterion for representing a log-scale; its non-uniform nodes
                    # over-resolve the centre and leave an edge gap > 1.5 s.


def _uniform_grid(n: int):
    """Centred uniform node offsets (units of s) and the stationary weights."""
    z = np.arange(n, dtype=float) - (n - 1) / 2.0
    w = np.exp(-0.5 * (_GAP_FACTOR * z) ** 2)
    return z, w / w.sum()


def _chain(phi: float, s: float, n: int):
    """Quadrature grid for a stationary AR(1) log-scale, and its kernel.

    Nodes are uniform at the resolution spacing ``1.5 s`` (finding 11); weights are
    the stationary density and the transition is the exact AR(1) Gaussian kernel.
    """
    z, w = _uniform_grid(n)
    lam = (_GAP_FACTOR * s) * z
    if not s > 0.0 or s * s <= 0.0:
        return np.zeros(n), w, np.tile(w, (n, 1))
    nu = max(s * s * (1.0 - phi * phi), 1e-12)
    ex = -0.5 * (lam[None, :] - phi * lam[:, None]) ** 2 / nu
    T = np.exp(np.clip(ex, -700.0, 700.0))
    T /= T.sum(1, keepdims=True)
    return lam, w, T


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


# ============================================================================
#                       the same recursion, for a batch
# ============================================================================
# The recursion is sequential in t and cannot be vectorised over time.  It can
# be vectorised over PARAMETER VECTORS, and here that pays even better than it
# does in the parent: the parent's per-node state is a scalar, this one's is a
# p-vector and a p x p covariance, so a step is a handful of einsums whose cost
# is dominated by numpy dispatch rather than by arithmetic.  Widening every
# array from (G, ...) to (B, G, ...) leaves the number of dispatches unchanged.
#
# Everything fit_ does -- start screens, finite-difference gradients -- is
# therefore organised as batches rather than as points.
#
# The three caps are the batch's equivalents of the -60/+60 clip in _build and
# the 1e-6 floors in Params._vec: they keep exp() and the logit finite.  They
# sit far outside any estimate this model can meaningfully produce.
_LOGIT_CAP = 14.0                       # |logit phi| <= 14 <=> phi in [8e-7, 1-8e-7]
_LOG_S_CAP = math.log(20.0)             # s <= 20
_LOG_S_FLOOR = math.log(1e-6)           # s >= 1e-6


def _chain_batch(phi: np.ndarray, s: np.ndarray, n: int):
    """:func:`_chain` for a batch of (phi, s); returns (B, n) lam, (B, n) w, (B, n, n) T."""
    z, w = _uniform_grid(n)
    lam = s[:, None] * (_GAP_FACTOR * z)
    nu = np.maximum(s * s * (1.0 - phi * phi), 1e-12)[:, None, None]
    ex = -0.5 * (lam[:, None, :] - phi[:, None, None] * lam[:, :, None]) ** 2 / nu
    T = np.exp(np.clip(ex, -700.0, 700.0))
    T /= T.sum(2, keepdims=True)
    return lam, np.broadcast_to(w, lam.shape), T


def _kron_batch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise Kronecker product of two stacks of square matrices."""
    b, i, j = A.shape
    _, k, l = B.shape
    return (A[:, :, None, :, None] * B[:, None, :, None, :]).reshape(b, i * k, j * l)


def _unpack(V: np.ndarray, p: int, unit_roots: int = 0):
    """The nine coordinates of a batch of unconstrained vectors, clipped.

    With ``unit_roots = d`` each row carries the p - d free coefficients and the
    pinned factor (z - 1)^d is multiplied in here -- one matrix product for the
    whole batch, since the map is linear (:func:`_pin_maps`).
    """
    m = p - unit_roots

    def ph(col):
        return 1.0 / (1.0 + np.exp(-np.clip(V[:, col], -_LOGIT_CAP - 1.0,
                                            _LOGIT_CAP + 1.0)))

    def sc(col):
        return np.exp(np.clip(V[:, col], _LOG_S_FLOOR - 1.0, _LOG_S_CAP + 1.0))

    if unit_roots:
        base, M = _pin_maps(p, unit_roots)
        alpha = base + V[:, :m] @ M
    else:
        alpha = V[:, :p]
    return (alpha, np.exp(np.clip(V[:, m], -700.0, 700.0)),
            np.exp(np.clip(V[:, m + 1], -700.0, 700.0)),
            ph(m + 2), ph(m + 3), sc(m + 4), sc(m + 5), ph(m + 6), sc(m + 7))


def _alpha_at_batch(alpha: np.ndarray, gs: np.ndarray) -> np.ndarray:
    """(B, nA, p) stack of :meth:`Params.alpha_at`, one row per parameter vector.

    The bisection in the scalar version only runs when a node leaves the unit
    disc, which is rare, so this loops over the (B, nA) pairs rather than
    vectorising a branch that almost never fires.  It is called once per
    likelihood evaluation, not once per step.
    """
    B, p = alpha.shape
    nA = gs.shape[1]
    out = np.empty((B, nA, p))
    flat = np.zeros(p)
    flat[0] = 1.0
    for b in range(B):
        a = alpha[b]
        explosive = _radius(a) > 1.0 + 1e-9
        for j in range(nA):
            g = float(gs[b, j])
            cand = (1.0 - g) * flat + g * a
            if explosive or _radius(cand) <= 1.0 + 1e-9:
                out[b, j] = cand
                continue
            lo, hi = 1.0, g                       # lo is known stable
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if _radius((1.0 - mid) * flat + mid * a) <= 1.0 + 1e-9:
                    lo = mid
                else:
                    hi = mid
            out[b, j] = (1.0 - lo) * flat + lo * a
    return out


def _grid_batch(V: np.ndarray, p: int, order: int, order_A: int, with_A: bool,
                unit_roots: int = 0):
    """The batched equivalent of :meth:`OdeFilter._build`."""
    alpha, Q, S2, phP, phM, sP, sM, phA, sA = _unpack(V, p, unit_roots)
    B, n = V.shape[0], order
    lamP, wP, TP = _chain_batch(phP, sP, n)
    lamM, wM, TM = _chain_batch(phM, sM, n)
    if with_A:
        nA = order_A
        lamA, wA, TA = _chain_batch(phA, sA, nA)
    else:                                   # exactly the s_A = 0 collapse
        nA = 1
        lamA = np.zeros((B, 1))
        wA = np.ones((B, 1))
        TA = np.ones((B, 1, 1))
    nN = n * n
    LP = np.tile(np.repeat(lamP, n, axis=1), (1, nA))
    LM = np.tile(lamM, (1, n * nA))
    LA = np.repeat(lamA, nN, axis=1)
    T = _kron_batch(TA, _kron_batch(TP, TM))
    pi0 = (wA[:, :, None] * (wP[:, :, None] * wM[:, None, :]).reshape(B, nN)[:, None, :]
           ).reshape(B, nA * nN)
    Fs = np.stack([_companion(a) for row in _alpha_at_batch(alpha, 1.0 + lamA)
                   for a in row]).reshape(B, nA, p, p)
    return dict(
        T=T, pi0=pi0, nA=nA, LA=LA,
        Qg=Q[:, None] * np.exp(np.clip(LP, -60.0, 60.0)),
        Rg=S2[:, None] * np.exp(np.clip(LM, -60.0, 60.0)),
        Fs=Fs, Aidx=np.repeat(np.arange(nA), nN))


def _loglik_batch(y: np.ndarray, V: np.ndarray, p: int, order: int,
                  order_A: int = 3, with_A: bool = True,
                  unit_roots: int = 0) -> np.ndarray:
    """Marginal log-likelihood of ``y`` at every unconstrained vector in ``V``.

    ``V`` is (B, p - unit_roots + 8) in the coordinates of
    :meth:`Params._vec`: the free coefficients, then the noise block.  The
    recursion is the one in :meth:`OdeFilter.update` -- per-node covariances,
    mixed by the chain's own kernel -- carried out for all B vectors at once;
    at B = 1 it agrees with :meth:`OdeFilter.loglik` to floating-point
    round-off.  No Step objects, no shares, no whiteness -- that is all a fit
    needs.

    ``with_A=False`` pins the dynamics channel off exactly (nA = 1), which is
    what the earlier passes of a fit want and is `order_A` times cheaper.

    A row whose parameters drive the recursion out of range gets -inf, the
    batched equivalent of :class:`_Numerical` reaching ``_run``'s guard.  Rows
    are independent -- every operation below contracts within its own ``b`` --
    so one dead row cannot poison the others.
    """
    V = np.atleast_2d(V)
    B = V.shape[0]
    g = _grid_batch(V, p, order, order_A, with_A, unit_roots)
    T, Qg, Rg, Aidx = g["T"], g["Qg"], g["Rg"], g["Aidx"]
    Fg = g["Fs"][:, Aidx]                                 # (B, G, p, p)
    G = Qg.shape[1]

    pi = g["pi0"].copy()
    y0 = float(y[0]) if np.isfinite(y[0]) else 0.0
    m = np.full((B, G, p), y0)
    P = (np.eye(p)[None, None]
         * ((Rg.max(1) + Qg.max(1)) * p)[:, None, None, None]
         * np.ones((1, G, 1, 1)))
    ll = np.zeros(B)
    dead = np.zeros(B, dtype=bool)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for v in y:
            pi_pred = np.maximum(np.einsum("bi,bij->bj", pi, T), 1e-300)
            mu = pi[:, :, None] * T / pi_pred[:, None, :]

            m0 = np.einsum("bij,bix->bjx", mu, m)
            dmix = m[:, :, None, :] - m0[:, None, :, :]
            P0 = (np.einsum("bij,bixz->bjxz", mu, P)
                  + np.einsum("bij,bijx,bijz->bjxz", mu, dmix, dmix))

            mp = np.einsum("bgxz,bgz->bgx", Fg, m0)
            Ap = np.einsum("bgxw,bgwv,bgzv->bgxz", Fg, P0, Fg)
            Ap[:, :, 0, 0] += Qg
            S = Ap[:, :, 0, 0] + Rg

            if not np.isfinite(v):                        # propagate only
                pi, m, P = pi_pred, mp, Ap
            else:
                bad = ~np.isfinite(S).all(1) | (S <= 0.0).any(1)
                dead |= bad
                S = np.where(np.isfinite(S) & (S > 0.0), S, 1.0)
                e = v - mp[:, :, 0]
                lg = -0.5 * (np.log(S) + e * e / S)
                mx = lg.max(1)
                w = pi_pred * np.exp(lg - mx[:, None])
                Z = w.sum(1)
                ll += np.log(Z) + mx - 0.5 * _LOG2PI

                K = Ap[:, :, :, 0] / S[:, :, None]
                m = mp + K * e[:, :, None]
                P = Ap - K[:, :, :, None] * Ap[:, :, None, 0, :]
                pi = w / Z[:, None]

            if dead.any():
                m = np.where(dead[:, None, None], y0, m)
                P = np.where(dead[:, None, None, None], np.eye(p), P)
                pi = np.where(dead[:, None], 1.0 / G, pi)

    return np.where(dead | ~np.isfinite(ll), -np.inf, ll)


# ============================================================================
#                  the s = 0 face, with S2 concentrated out
# ============================================================================
# Where s_P = s_M = s_A = 0 every quadrature node carries lam = 0, the grid
# collapses to one state, and the model is an ordinary linear-Gaussian state
# space: a bare p x p Kalman filter with no mixture at all.  Two exact facts
# then reduce that face further:
#
#   * the recursion is homogeneous of degree 1 in S2, so with P_t = S2 * Ptil_t
#     and Q = S2 * q the gains -- and hence the innovations -- depend on the
#     ratio q alone;
#   * S2 is therefore concentrated out in closed form, leaving a profile
#     likelihood in (alpha, log q) with one fewer coordinate and no grid.
#
# This is where the old stages 0, 1b and 2 lived, and they paid the full
# order^2 grid for a face on which the grid is a single point repeated.

def _face_profile(y: np.ndarray, alpha: np.ndarray, q: float):
    """(S2, profile loglik, standardised innovations) at ratio q = Q/S2."""
    p = len(alpha)
    F = _companion(alpha)
    Pt = np.eye(p) * (1.0 + q) * p              # Ptil_0, matching update()
    y0 = float(y[0]) if math.isfinite(float(y[0])) else 0.0
    m = np.full(p, y0)                          # ditto
    acc = 0.0                                   # sum e^2 / Stil
    lsum = 0.0                                  # sum log Stil
    cnt = 0
    u = np.full(y.size, np.nan)
    for i in range(y.size):
        mj = F @ m
        Aj = F @ Pt @ F.T
        Aj[0, 0] += q
        S = Aj[0, 0] + 1.0
        if not (math.isfinite(S) and S > 0.0):
            return math.nan, -math.inf, u
        v = y[i]
        if not math.isfinite(v):                # missing: propagate, no correction
            m, Pt = mj, Aj
            continue
        e = v - mj[0]
        acc += e * e / S
        lsum += math.log(S)
        u[i] = e / math.sqrt(S)
        K = Aj[:, 0] / S
        m = mj + K * e
        Pt = Aj - np.outer(K, Aj[0, :])
        cnt += 1
    if cnt < 2 or not acc > 0.0:
        return math.nan, -math.inf, u
    s2 = acc / cnt
    ll = -0.5 * (cnt * (_LOG2PI + math.log(s2) + 1.0) + lsum)
    u /= math.sqrt(s2)
    # The diffuse start puts p * (1 + q) on the diagonal, so the first few
    # innovations are dominated by not knowing where the state is rather than by
    # the noise.  They are legitimate likelihood terms but useless as a scale
    # statistic, so _moment_scale does not see them.
    u[:p] = np.nan
    return s2, ll, u


_Q_SCAN = np.logspace(-7.0, 3.0, 41)            # q is a ratio, so it needs no scaling


def _face_optimum(y: np.ndarray, b0: np.ndarray, max_iter: int = 400,
                  p: int | None = None, unit_roots: int = 0):
    """Maximise the concentrated profile over the free coefficients and log q.

    ``b0`` is the start in the FREE coefficients: alpha itself when nothing is
    pinned, the quotient polynomial's beta when ``unit_roots > 0`` (the pinned
    factor is multiplied in before every profile evaluation, so the search
    simply never leaves the constraint surface).  Returns (free, Q, S2, u).
    """
    from scipy.optimize import minimize

    if unit_roots:
        base_, M_ = _pin_maps(p, unit_roots)

        def full(b):
            return base_ + np.asarray(b, dtype=float) @ M_
    else:
        def full(b):
            return np.asarray(b, dtype=float)

    lls = [_face_profile(y, full(b0), q)[1] for q in _Q_SCAN]
    lq = math.log(float(_Q_SCAN[int(np.argmax(lls))]))

    def nll(v):
        s2, ll, _ = _face_profile(y, full(v[:-1]), math.exp(min(v[-1], 700.0)))
        return math.inf if not math.isfinite(ll) else -ll

    r = minimize(nll, np.concatenate([b0, [lq]]), method="Nelder-Mead",
                 options=dict(maxiter=int(max_iter), xatol=1e-4, fatol=1e-6))
    v = r.x if math.isfinite(r.fun) else np.concatenate([b0, [lq]])
    free, q = v[:-1], math.exp(min(v[-1], 700.0))
    s2, ll, u = _face_profile(y, full(free), q)
    if not math.isfinite(ll):                   # the scan point is known good
        free, q = np.asarray(b0, dtype=float), math.exp(lq)
        s2, ll, u = _face_profile(y, full(free), q)
    return np.asarray(free, dtype=float), q * s2, s2, u


_VAR_LOG_CHI2 = math.pi ** 2 / 2.0              # Var(log z^2), z ~ N(0,1), exact


def _moment_scale(u: np.ndarray):
    """(s, phi) for the log-scale, read off the face's residuals in one pass.

    If a log-scale channel is present it survives into the homoscedastic fit's
    standardised innovations, where ``log u_t^2 = log z_t^2 + lam_t`` with the
    two terms independent.  Var(log z^2) is exactly pi^2/2, so the lag-0 and
    lag-1 autocovariances of log u^2 give the log-scale's variance and its
    persistence directly.  It cannot say WHICH channel carries them -- both
    inflate an innovation -- so it supplies a magnitude and a persistence, and
    the start screen decides the split.
    """
    g = np.log(np.maximum(u[np.isfinite(u)] ** 2, 1e-12))
    if g.size < 3:
        return 0.0, 0.5
    g = g - g.mean()
    c0 = float(g @ g) / g.size
    c1 = float(g[1:] @ g[:-1]) / g.size
    var = c0 - _VAR_LOG_CHI2
    if var <= 1e-3:
        return 0.0, 0.5
    return math.sqrt(var), float(min(max(c1 / var, 1e-3), 0.98))


# ------------------------------------------------------------- the start screen
_PHI_GRID = (0.02, 0.25, 0.5, 0.75, 0.95)
_S_SPLITS = ((0.03, 0.03), (0.5, 0.5), (0.03, 0.5), (0.5, 0.03))
_QUIET = 0.1                        # divides "no scale structure" starts from the rest


def _bounds(p: int, g0: float):
    """Box for the search, in the unconstrained coordinates.

    A numerical guard, not a prior.  Q and S2 cannot exceed the data's own
    residual scale by more than a rounding margin; the six channel coordinates
    are the caps of :func:`_loglik_batch`; and alpha is boxed only widely
    enough to keep a diverging search from overflowing -- the unit-disc
    question is left to the likelihood, which answers it with -inf.

    The alpha box is set from the largest coefficient a stable alpha can carry:
    the extreme case is a p-fold root at 1, whose coefficients are binomial, so
    four times the central binomial coefficient is generous by construction.
    """
    lg = math.log(max(g0, 1e-300))
    alpha_span = max(8.0, 4.0 * math.comb(p, p // 2))
    return ([(-alpha_span, alpha_span)] * p
            + [(lg - 30.0, lg + 5.0), (lg - 30.0, lg + 5.0),
               (-_LOGIT_CAP, _LOGIT_CAP), (-_LOGIT_CAP, _LOGIT_CAP),
               (_LOG_S_FLOOR, _LOG_S_CAP), (_LOG_S_FLOOR, _LOG_S_CAP),
               (-_LOGIT_CAP, _LOGIT_CAP), (_LOG_S_FLOOR, _LOG_S_CAP)])


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

    The recursion carries one (m, P) PER NODE, mixed by the chain's own
    transition kernel before each time update (standard IMM).  The filter
    originally shipped with a shared-covariance collapse (GPB1) as the
    default and grew this recursion as an option; `oracle-gap`
    measured the collapse's cost -- the likelihood goes flat along the ridge
    Q e^{s_P^2/2} = const (relief 0.0022 nats/pt against 0.0101), the
    s_P = 0 boundary becomes self-confirming, and a forced process-scale
    channel stops at 80% of an oracle's gap where per-node covariances reach
    89.5% -- so the collapse was removed and this is now the only recursion.
    The two agree to machine precision when s_P = s_M = s_A = 0, which is
    also where this filter still reduces exactly to the parent; with a live
    scale channel the parent (GPB1 by construction) and this filter share
    the model and differ by the collapse, this one keeping strictly more of
    the evidence.
    """

    def __init__(self, params: Params | None = None, order: int = 5,
                 order_A: int = 3, linearized_dynamics=None):
        if order < 3:
            raise ValueError("order must be at least 3")
        if order_A < 3:
            raise ValueError("order_A must be at least 3")
        self.params = params
        self.order = int(order)
        self.order_A = int(order_A)
        self._built = None
        # supplied-dynamics mode: a callable state -> (p x p) transition, evaluated
        # at the running state estimate each step (EKF-style).  The dynamics are then
        # KNOWN, not fitted or ranged; only the noise scales are inferred.  None keeps
        # the ordinary fit-and-track-alpha behaviour.
        self.linearized_dynamics = linearized_dynamics
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
        # the linearized_dynamics callable is not serialisable; re-attach it after
        # from_dict (the fitted NOISE class is what persists).
        return {"params": self.params.to_dict(), "order": self.order,
                "order_A": self.order_A}

    @classmethod
    def from_dict(cls, d: dict) -> "OdeFilter":
        # a stored "collapse" key from the two-recursion era is ignored: the
        # parameters mean the same thing and this recursion evaluates them
        # with strictly more of the evidence
        return cls(Params.from_dict(d["params"]), order=int(d.get("order", 5)),
                   order_A=int(d.get("order_A", 3)))

    # ------------------------------------------- supplied-dynamics (noise-only) fit
    @classmethod
    def _fit_noise_only(cls, y, linearized_dynamics, p: int, order: int,
                        scales: bool, max_iter: int) -> "OdeFilter":
        """Learn only the NOISE class; the dynamics come from the callable.

        The transition each step is ``linearized_dynamics(state_estimate)`` (EKF-style),
        so ``alpha`` and the dynamics channel are not fitted.  A plain local
        optimisation of the exact filter likelihood over ``(Q, s2)`` and, when
        ``scales``, the AR(1) log-scale class ``(phi_P, s_P, phi_M, s_M)`` -- simpler
        than the batched :meth:`fit`; near ``s = 0`` the spread estimate is ill-posed
        (Fisher information vanishes at zero spread), so read a small fitted ``s`` as
        cheap insurance, not a finding.
        """
        from scipy.optimize import minimize
        y = np.asarray(y, dtype=float)
        p = int(p)
        v0 = max(float(np.var(np.diff(y))), 1e-3)

        def build(v):
            params = Params(alpha=tuple([0.0] * p), Q=math.exp(v[0]), s2=math.exp(v[1]),
                            phi_P=_expit(v[2]), phi_M=_expit(v[3]),
                            s_P=(math.exp(v[4]) if scales else 0.0),
                            s_M=(math.exp(v[5]) if scales else 0.0), phi_A=0.0, s_A=0.0)
            return cls(params, order=order, linearized_dynamics=linearized_dynamics)

        def neg(v):
            try:
                L = build(v).loglik(y)
            except _Numerical:
                return 1e18
            return -L if np.isfinite(L) else 1e18

        start = np.array([math.log(0.5 * v0), math.log(0.5 * v0),
                          _logit(0.9), _logit(0.9), math.log(0.3), math.log(0.3)])
        res = minimize(neg, start, method="Nelder-Mead",
                       options={"maxiter": max_iter, "xatol": 1e-3, "fatol": 1e-3})
        return build(res.x)

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

    def update(self, y: float, F=None) -> Step:
        """Absorb one observation.  NaN is treated as missing.

        When the filter carries a ``linearized_dynamics`` callable, the ``p x p``
        transition for this step is computed from it at the current state estimate
        (no per-step argument needed).  ``F`` is an optional low-level override that
        forces the transition for this one step; normally leave it None.
        """
        if self.params is None:
            raise ValueError("filter is not fitted; call fit() or pass params")
        p = self.params.p
        g = self._build()
        if self._pi is None:
            self._pi = g["pi0"].copy()
            y0 = float(y) if np.isfinite(y) else 0.0
            G = g["Qg"].size
            self._m = np.full((G, p), y0)
            self._P = np.tile(np.eye(p) * float(g["Rg"].max()
                                                + g["Qg"].max()) * p,
                              (G, 1, 1))
        if F is None and self.linearized_dynamics is not None:
            # EKF-style: linearise around the current collapsed state estimate.
            xhat = self._pi @ self._m
            F = self.linearized_dynamics(xhat)
        if F is not None:
            F = np.asarray(F, dtype=float)
            if F.shape != (p, p):
                raise ValueError(f"transition must have shape ({p}, {p}), got {F.shape}")
            if not np.all(np.isfinite(F)):
                raise ValueError("transition must be finite")
        return self._update_imm(y, g, p, F)

    def _update_imm(self, y: float, g: dict, p: int, F=None) -> Step:
        """One step of the recursion: per-node covariances, standard IMM.

        Each node keeps its own (m, P), mixed across nodes by the kernel
        before the time update.  The accumulated history that separates scale
        hypotheses -- what a shared-covariance collapse erases -- therefore
        survives.  At s_P = s_M = s_A = 0 the grid is one node and this is an
        ordinary Kalman filter step exactly.
        """
        T, Qg, Rg, Fs, Aidx = g["T"], g["Qg"], g["Rg"], g["Fs"], g["Aidx"]
        pi = self._pi
        pi_pred = np.maximum(pi @ T, 1e-300)
        mu = pi[:, None] * T / pi_pred[None, :]        # P(came from i | now j)

        m0 = np.einsum("ij,ix->jx", mu, self._m)       # mixed state per node
        dmix = self._m[:, None, :] - m0[None, :, :]
        P0 = (np.einsum("ij,ixz->jxz", mu, self._P)
              + np.einsum("ij,ijx,ijz->jxz", mu, dmix, dmix))

        # supplied dynamics: one F for every node this step; else the fitted/
        # tracked companion per dynamics-channel node.
        Fg = np.broadcast_to(F, (Qg.size, p, p)) if F is not None else Fs[Aidx]
        mp = np.einsum("gxz,gz->gx", Fg, m0)
        Aj = np.einsum("gxw,gwv,gzv->gxz", Fg, P0, Fg)
        a00 = Aj[:, 0, 0].copy()                       # prior part, pre-Q
        Aj[:, 0, 0] += Qg
        S = Aj[:, 0, 0] + Rg

        if not np.isfinite(y):                    # missing: mix and propagate
            self._pi, self._m, self._P = pi_pred, mp, Aj
            lamP = float(pi_pred @ g["LP"])
            lamM = float(pi_pred @ g["LM"])
            mbar = float(pi_pred @ mp[:, 0])
            vbar = float(pi_pred @ (Aj[:, 0, 0] + (mp[:, 0] - mbar) ** 2))
            st = Step(mbar, vbar, math.nan, 0.0, 1.0, 0.0, 0.0,
                      lamP - self.params.phi_P * self._prev_lamP,
                      self.params.phi_P * self._prev_lamP,
                      lamM - self.params.phi_M * self._prev_lamM,
                      self.params.phi_M * self._prev_lamM,
                      self._whiteness(), 1.0 + float(pi_pred @ g["LA"]))
            self._prev_lamP, self._prev_lamM = lamP, lamM
            return st

        if not np.all(np.isfinite(S)) or np.any(S <= 0.0):
            raise _Numerical("non-positive predictive variance")
        e = float(y) - mp[:, 0]
        ybar = float(pi_pred @ mp[:, 0])
        e_rep = float(y) - ybar
        lg = -0.5 * (np.log(S) + e * e / S)
        mx = float(lg.max())
        w = pi_pred * np.exp(lg - mx)
        Z = float(w.sum())
        ll = math.log(Z) + mx - 0.5 * _LOG2PI
        S_pred = float(pi_pred @ (S + (mp[:, 0] - ybar) ** 2))
        pi = w / Z

        K = Aj[:, :, 0] / S[:, None]
        m_new = mp + K * e[:, None]
        P_new = Aj - K[:, :, None] * Aj[:, None, 0, :]

        mbar = float(pi @ m_new[:, 0])
        vbar = float(pi @ (P_new[:, 0, 0] + (m_new[:, 0] - mbar) ** 2))
        lamP, lamM = float(pi @ g["LP"]), float(pi @ g["LM"])
        st = Step(
            mean=mbar, var=vbar, innovation=e_rep, loglik=ll,
            share_prior=float(pi @ (a00 / S)),
            share_process=float(pi @ (Qg / S)),
            share_measurement=float(pi @ (Rg / S)),
            process_anomaly=lamP - self.params.phi_P * self._prev_lamP,
            process_regime=self.params.phi_P * self._prev_lamP,
            measurement_anomaly=lamM - self.params.phi_M * self._prev_lamM,
            measurement_regime=self.params.phi_M * self._prev_lamM,
            whiteness=self._accum_whiteness(e_rep, S_pred),
            dynamics=1.0 + float(pi @ g["LA"]),
            pred_var=S_pred,
        )
        self._pi, self._m, self._P = pi, m_new, P_new
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
        Fg = g["Fs"][g["Aidx"]]
        pi, m, P = self._pi, self._m.copy(), self._P.copy()
        for _ in range(int(horizon)):
            pi_pred = np.maximum(pi @ g["T"], 1e-300)
            mu = pi[:, None] * g["T"] / pi_pred[None, :]
            m0 = np.einsum("ij,ix->jx", mu, m)
            dmix = m[:, None, :] - m0[None, :, :]
            P0 = (np.einsum("ij,ixz->jxz", mu, P)
                  + np.einsum("ij,ijx,ijz->jxz", mu, dmix, dmix))
            m = np.einsum("gxz,gz->gx", Fg, m0)
            P = np.einsum("gxw,gwv,gzv->gxz", Fg, P0, Fg)
            P[:, 0, 0] += g["Qg"]
            pi = pi_pred
        mbar = float(pi @ m[:, 0])
        vbar = float(pi @ (P[:, 0, 0] + (m[:, 0] - mbar) ** 2))
        return mbar, vbar

    def derivatives(self) -> tuple[np.ndarray, np.ndarray]:
        """The current posterior in (x, Dx, D^2 x, ...) coordinates.

        A fixed invertible integer change of basis, so nothing is created or
        lost; the growth of the diagonal is the noise amplification of
        differencing, reported rather than incurred.  The per-node mixture is
        collapsed here, for reporting only -- the recursion itself never sees
        this collapse.
        """
        if self._m is None:
            raise ValueError("nothing observed yet")
        D = difference_matrix(self.params.p)
        pi = self._pi
        m = pi @ self._m
        dm = self._m - m
        P = (np.einsum("g,gxz->xz", pi, self._P)
             + np.einsum("g,gx,gz->xz", pi, dm, dm))
        return D @ m, D @ P @ D.T

    # ----------------------------------------------------------------- batch
    def loglik(self, y, Fs=None) -> float:
        return self._run(np.asarray(y, dtype=float), want=False, Fs=Fs)

    def filter(self, y, Fs=None) -> FilterResult:
        """Batch run.  When the filter carries a ``linearized_dynamics`` callable the
        per-step transition is taken from it automatically; ``Fs`` (length-T sequence
        of ``p x p`` transitions) is an optional low-level override."""
        return self._run(np.asarray(y, dtype=float), want=True, Fs=Fs)

    def _run(self, y: np.ndarray, want: bool, Fs=None):
        if self.params is None:
            raise ValueError("filter is not fitted; call fit() or pass params")
        if y.ndim != 1 or y.size == 0:
            raise ValueError("y must be a non-empty 1-D array")
        if Fs is not None:
            Fs = np.asarray(Fs, dtype=float)
            if Fs.shape != (y.size, self.params.p, self.params.p):
                raise ValueError(f"Fs must have shape ({y.size}, {self.params.p}, "
                                 f"{self.params.p}), got {Fs.shape}")
        saved = (self._pi, self._m, self._P, self._prev_lamP, self._prev_lamM,
                 self._loglik, self._e_prev, self._ee, self._e2, self._nw)
        try:
            self.reset()
            if not want:
                total = 0.0
                try:
                    for i, v in enumerate(y):
                        total += self.update(v, None if Fs is None else Fs[i]).loglik
                except _Numerical:
                    return -np.inf
                return total
            n, p = y.size, self.params.p
            cols = ("mean", "var", "innovation", "share_prior", "share_process",
                    "share_measurement", "process_anomaly", "process_regime",
                    "measurement_anomaly", "measurement_regime", "whiteness",
                    "dynamics", "pred_var")
            out = {c: np.empty(n) for c in cols}
            sm = np.empty((n, p))
            sc = np.empty((n, p, p))
            total = 0.0
            for i, v in enumerate(y):
                st = self.update(v, None if Fs is None else Fs[i])
                for c in cols:
                    out[c][i] = getattr(st, c)
                # collapse the per-node mixture for reporting only
                mbar = self._pi @ self._m
                dmix = self._m - mbar
                sm[i] = mbar
                sc[i] = (np.einsum("g,gxz->xz", self._pi, self._P)
                         + np.einsum("g,gx,gz->xz", self._pi, dmix, dmix))
                total += st.loglik
            return FilterResult(loglik=total, state_mean=sm, state_cov=sc, **out)
        finally:
            (self._pi, self._m, self._P, self._prev_lamP, self._prev_lamM,
             self._loglik, self._e_prev, self._ee, self._e2, self._nw) = saved

    # ------------------------------------------------------------------- fit
    @classmethod
    def fit(cls, y, p: int = 3, order: int = 5, max_iter: int = 400,
            scales: bool = True, dynamics: bool = True,
            order_A: int = 3, unit_roots: int = 0,
            linearized_dynamics=None) -> "OdeFilter":
        """Learn every parameter from a series and return a fitted filter.

        ``linearized_dynamics`` — a callable ``state -> (p x p)`` transition — turns
        this into supplied-dynamics mode: the dynamics are KNOWN (the caller's model,
        linearised at the running state estimate each step, EKF-style), so ``alpha``
        and the dynamics channel are NOT fitted; only the noise class is learned (the
        whole point of this workstream: the user supplies what they know, the dynamics,
        and the filter infers what they don't, the live noise).  ``p`` must then be
        given to match the callable's matrix size; ``dynamics``/``unit_roots`` are
        ignored.  The returned filter carries the callable — call ``update``/``filter``
        with no per-step dynamics argument.

        ``p`` is the order of the recurrence: 3 is a second-order ODE plus a
        constant offset, which is the class this filter targets.  ``p`` is a
        modelling commitment, not a tuning parameter -- and because each root of
        the characteristic polynomial is a channel, choosing ``p`` is the same
        act as counting channels.

        ``unit_roots`` pins that many roots at z = 1 exactly, and fits only the
        quotient polynomial's p - unit_roots coefficients.  0 (the default)
        lets every root float -- the weaker and safer assumption, and the
        behaviour this filter always had.  1 asserts the constant offset;
        2 asserts a LINEAR offset -- a climbing or declining bias whose rate is
        part of the state.  A free fit cannot reliably find that bias: a
        maximum-likelihood unit root lands just inside the circle, and a root
        at 1 - eps forecasts a drift that decays instead of one that continues
        (exploration/0040).  Pinning is a hypothesis, not an assumption -- fit
        with unit_roots at both values and compare the same prequential
        likelihood this filter uses everywhere else.  This is the internal
        form of "fit the differenced series", and strictly dominates it:
        differencing pushes iid measurement noise out of the model class
        (it becomes MA(1)); pinning leaves it alone.

        ``scales=False`` pins the two log-scale channels off, giving an ordinary
        (non-adaptive) recurrence filter.  Useful as a baseline and much faster.
        ``dynamics=False`` additionally pins the dynamics channel off, giving
        a static ``alpha``.

        The search runs on the per-node-covariance likelihood -- the one with
        curvature along the Q e^{s_P^2/2} ridge.  Under the removed GPB1
        collapse the fitted (Q, s_P) split was decided by the optimiser's
        path rather than by the data (oracle-gap/0004, 0006), which is
        how fitted process-scale channels ended up dead exactly when they
        were needed.  Caution: near s_P = 0 the point estimate is ill-posed
        under either likelihood (the Fisher information in a spread parameter
        vanishes at zero spread), so small fitted s_P values should be read
        as "cheap insurance", not as findings (oracle-gap/0009).
        """
        if linearized_dynamics is not None:
            return cls._fit_noise_only(y, linearized_dynamics, p=p, order=order,
                                       scales=scales, max_iter=max_iter)
        f = cls(order=order, order_A=order_A)
        f.fit_(y, p=p, max_iter=max_iter, scales=scales, dynamics=dynamics,
               unit_roots=unit_roots)
        return f

    def fit_(self, y, p: int = 3, max_iter: int = 400,
             scales: bool = True, dynamics: bool = True,
             unit_roots: int = 0) -> "OdeFilter":
        """Fit in place.  Returns self.

        Staged, because a nine-dimensional search from one start is not
        reliable.  Every stage is organised around the one measured fact about
        this recursion: it is dispatch-bound, so B parameter vectors cost far
        less than B evaluations (:func:`_loglik_batch`).  Nothing here is a
        tuning choice -- the estimate is the maximum likelihood estimate however
        it is reached, and ``max_iter`` is a compute budget.

        pass 0   alpha by instrumental variables, using lags p+1..3p as
                 instruments.  Regressing on observed lags is errors-in-
                 variables and does not merely attenuate the dynamics -- it
                 deletes the oscillation outright (exploration/0002).  Lagging
                 past the order annihilates the measurement noise exactly, for
                 every (Q, S2), without needing either.  A start, not an
                 estimate;
        pass 1   the optimum on the s_P = s_M = s_A = 0 face.  That face is an
                 ordinary linear-Gaussian state space -- a bare p x p Kalman
                 filter, no mixture at all -- on which S2 concentrates out in
                 closed form, so only (alpha, log q) is searched.  This replaces
                 the old stages 0, 1b and 2, which paid the full order^2 grid
                 for a face on which the grid is one point repeated;
        pass 2   the magnitude and persistence of whatever log-scale structure
                 is left in that fit's residuals, read off their log-squares in
                 one pass (:func:`_moment_scale`);
        pass 3   ONE batched evaluation of ~100 starts -- a 5x5 grid over the
                 two persistences crossed with four splits of the scale between
                 the channels, plus what pass 2 proposed;
        pass 4   maximum likelihood by L-BFGS-B, its gradient taken by central
                 differences batched into a single evaluation.  Run from the
                 best quiet start and the best volatile one, better likelihood
                 kept.  Done in the six noise coordinates first and only then
                 jointly with alpha: pass 1 has already put alpha at its exact
                 optimum on the face, and leaving it out makes the subspace both
                 better conditioned and cheaper per gradient.  The dynamics
                 channel is pinned off throughout, exactly, so every evaluation
                 here is order_A times cheaper than one that carries it;
        pass 5   the dynamics channel: a batched screen over (phi_A, s_A), a
                 polish in those two coordinates alone, and a joint polish only
                 if the channel has already earned its place.  It is fitted last
                 because it is the only channel whose grid multiplies the cost
                 of every likelihood evaluation, and it is accepted only if it
                 beats the pass-4 optimum on the same likelihood.

        The dynamics channel is started persistent, because a change in the
        dynamics is a regime by construction (0025): it has no first-moment
        signature at all, so the impulsive end of this channel is a different
        object from the impulsive end of the noise channels.

        With ``unit_roots = d`` every pass runs unchanged in the p - d free
        coefficients: the pinned factor (z - 1)^d is a fixed linear map
        multiplied in at each likelihood evaluation (:func:`_pin_maps`), so the
        search simply cannot leave the constraint surface, and d = 0 is
        bit-for-bit the unconstrained fit.
        """
        y = np.asarray(y, dtype=float)
        finite = np.isfinite(y)
        good = y[finite]
        if good.size < 8 * p:
            raise ValueError(f"need at least {8 * p} finite observations")
        n = max(int(finite.sum()), 1)
        off = math.log(1e-6)
        d = int(unit_roots)
        m = p - d                                # free coefficients
        if d:
            pin_base, pin_M = _pin_maps(p, d)    # validates d in [0, p]

        # pass 0 -- the free coefficients by instrumental variables, and the
        # residual scale that sets the search box.  With d roots pinned the
        # quotient dynamics live on the d-times differenced series -- that is
        # what pinning MEANS -- so the IV start is taken there.  Differencing
        # thickens the measurement noise, but this is a start, not an estimate;
        # the likelihood that polishes it runs on the level, in class.
        if d:
            good_d = np.diff(y, d)
            good_d = good_d[np.isfinite(good_d)]
            b0 = _iv_alpha(good_d, m) if m else np.zeros(0)
            a0 = pin_base + b0 @ pin_M
        else:
            b0 = a0 = _iv_alpha(good, p)
        idx = np.arange(p, good.size)
        r0 = good[idx] - np.column_stack([good[idx - i]
                                          for i in range(1, p + 1)]) @ a0
        g0 = float(np.mean(r0 * r0))
        if not g0 > 0:
            raise ValueError("series has no residual variation; nothing to fit")
        bounds = _bounds(m, g0)
        lo, hi = np.array(bounds).T

        # pass 1 -- the s = 0 face, with S2 concentrated out
        free, Q0, s20, resid = _face_optimum(y, b0, max_iter, p, d)
        base = np.concatenate([free, [math.log(Q0), math.log(s20),
                                      _logit(0.5), _logit(0.5), off, off,
                                      _logit(0.9), off]])
        base = np.clip(base, lo, hi)

        if not scales:
            self.params = Params._from_vec(base, p, d)
            self._built = None
            return self.reset()

        # pass 2 -- what the face left behind
        s_hat, phi_hat = _moment_scale(resid)

        # pass 3 -- one batched screen over the starts
        starts = []
        for pp in _PHI_GRID:
            for pm in _PHI_GRID:
                for sp, sm in _S_SPLITS:
                    v = base.copy()
                    v[m + 2], v[m + 3] = _logit(pp), _logit(pm)
                    v[m + 4], v[m + 5] = math.log(sp), math.log(sm)
                    starts.append(v)
        if s_hat > 0.0:
            lz, lp = math.log(s_hat), _logit(phi_hat)
            for sp, sm in ((lz, lz), (lz, off), (off, lz)):
                v = base.copy()
                v[m + 2], v[m + 3] = lp, lp
                v[m + 4], v[m + 5] = sp, sm
                starts.append(v)
        V = np.clip(np.array(starts), lo, hi)
        val = _loglik_batch(y, V, p, self.order, self.order_A, with_A=False,
                            unit_roots=d)
        loud = V[:, m + 4:m + 6].max(1) > math.log(_QUIET)
        chosen = [V[np.argmax(np.where(msk, val, -np.inf))]
                  for msk in (~loud, loud) if msk.any()]

        # pass 4 -- maximum likelihood, in the cheap subspace first.
        # The six noise coordinates alone are both better conditioned than the
        # full set (alpha's curvature is orders of magnitude away from log Q's,
        # and pass 1 already put alpha at its exact optimum on the face) and
        # cheaper per gradient, because the stencil is 2*6+1 rows rather than
        # 2*(p+6)+1.  The joint step afterwards then starts from a point alpha
        # barely has to move from.
        noise = list(range(m, m + 6))            # Q, S2, phi_P, phi_M, s_P, s_M
        full, _ = self._polish(y, chosen, noise, bounds, n, p,
                              max_iter, with_A=False, unit_roots=d)
        full, _ = self._polish(y, [full], list(range(m + 6)), bounds, n, p,
                               max_iter, with_A=False, unit_roots=d)

        if not dynamics:
            self.params = Params._from_vec(full, p, d)
            self._built = None
            return self.reset()

        # pass 5 -- the dynamics channel.  Same shape as pass 4 and for the same
        # reason, except that here the cheap subspace is also the only one the
        # old fit ever moved: (phi_A, s_A) is 5 stencil rows against 2p + 17.
        starts = []
        for pa in (0.5, 0.9, 0.98):
            for sa in (0.05, 0.15, 0.6):
                v = full.copy()
                v[m + 6], v[m + 7] = _logit(pa), math.log(sa)
                starts.append(v)
        W = np.clip(np.array(starts), lo, hi)
        valA = _loglik_batch(y, W, p, self.order, self.order_A, with_A=True,
                             unit_roots=d)

        # the reference: the pass-4 optimum scored on the SAME likelihood, the
        # one that carries the channel's grid.  The channel is accepted only if
        # it beats this.
        off_v = full.copy()
        off_v[m + 7] = _LOG_S_FLOOR
        ref = float(_loglik_batch(y, off_v[None, :], p, self.order,
                                  self.order_A, with_A=True, unit_roots=d)[0]) / n

        best_d, bestd = self._polish(y, [W[int(np.argmax(valA))]],
                                     [m + 6, m + 7], bounds, n, p,
                                     max_iter, with_A=True, unit_roots=d)
        if -bestd > ref:
            best_d, bestd = self._polish(y, [best_d], list(range(m + 8)),
                                         bounds, n, p, max_iter, with_A=True,
                                         unit_roots=d)
            if -bestd > ref:
                full = best_d

        self.params = Params._from_vec(full, p, d)
        self._built = None
        return self.reset()

    def _polish(self, y, starts, act, bounds, n, p, max_iter, with_A,
                unit_roots=0):
        """L-BFGS-B over the coordinates in ``act``, gradient batched.

        The gradient is a central difference over ``act`` only, and the whole
        stencil -- centre plus 2 * len(act) offsets -- goes to
        :func:`_loglik_batch` as one call.  A non-finite centre means the
        parameters left the range the recursion can represent; the step is then
        reported as a large finite value with no gradient, which stops L-BFGS-B
        rather than sending it chasing a NaN, and the start it came from is
        kept.  Returns (best vector, best objective).
        """
        from scipy.optimize import minimize

        d = len(act)
        h = 1e-4
        stencil = np.zeros((2 * d + 1, len(bounds)))
        eye = np.eye(len(bounds))[act]
        stencil[1::2] = h * eye
        stencil[2::2] = -h * eye
        sub_bounds = [bounds[i] for i in act]

        def fg(vs, v0):
            v = v0.copy()
            v[act] = vs
            ll = _loglik_batch(y, v + stencil, p, self.order, self.order_A,
                               with_A=with_A, unit_roots=unit_roots)
            if not np.isfinite(ll[0]):
                return 1e10, np.zeros(d)
            up, dn = ll[1::2], ll[2::2]
            ok = np.isfinite(up) & np.isfinite(dn)
            gr = np.zeros(d)                    # a coordinate whose step leaves
            np.subtract(up, dn, out=gr, where=ok)   # the range gets no gradient
            return -ll[0] / n, -gr / (2.0 * h * n)

        best_vec, best_f = None, np.inf
        for start in starts:
            f0 = fg(start[act], start)[0]
            r = minimize(fg, start[act], args=(start,), jac=True,
                         method="L-BFGS-B", bounds=sub_bounds,
                         options=dict(maxiter=int(max_iter), ftol=1e-12,
                                      gtol=1e-7))
            cand, fval = (r.x, r.fun) if r.fun < f0 else (start[act], f0)
            if fval < best_f:
                best_vec = start.copy()
                best_vec[act] = cand
                best_f = fval
        return best_vec, best_f


# --------------------------------------------------------- closed-form starts
def _iv_alpha(y: np.ndarray, p: int, extra: int | None = None) -> np.ndarray:
    """alpha by instrumental variables, instruments at lags p+1 .. p+m.

    The residual  y_t - sum_i alpha_i y_{t-i}  involves the measurement noise
    only at times t, t-1, ..., t-p, so every observation at lag >= p+1 is
    uncorrelated with it, for every (Q, S2), without stationarity.  That is the
    exact analogue of the parent's "increments annihilate the level".
    """
    m = extra if extra is not None else 2 * p
    lo = p + m
    if y.size <= lo + 4 * p:
        m = max(p, (y.size - p - 4 * p) // 2)
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
