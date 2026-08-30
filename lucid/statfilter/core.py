"""Adaptive local-level filter with learned noise structure.

The model, in full:

    theta_t = theta_{t-1} + w_t,   w_t ~ N(0, Q  * exp(lamP_t))
    x_t     = theta_t     + v_t,   v_t ~ N(0, S2 * exp(lamM_t))
    lam^c_t = phi_c * lam^c_{t-1} + sqrt(nu_c) z_t,   c in {P, M}
    s_c^2   = nu_c / (1 - phi_c^2)                     (stationary variance)

A level performing a random walk, observed with noise, where *both* noise
scales are themselves log-AR(1) processes.  That single structure covers the
four ways a series can deviate, as two channels crossed with the two ends of
each channel's own autocorrelation:

                       phi -> 0 (impulsive)      phi -> 1 (persistent)
    process channel    a jump in the level       a change in drift rate
    measurement chan.  an outlier                a change in noise level

There is no event detection, no changepoint test, no gate, and no threshold
anywhere in this file.  Deviations are not detected; they are *allocated*, as
continuous per-step quantities that are always defined.

Six parameters -- Q, S2, phi_P, phi_M, s_P, s_M -- and all six are learned from
the data by maximum marginal likelihood.  Nothing is tuned by hand.

Q and S2 are MEDIAN (geometric-mean) variances.  Parameterising by the mean of
the log is what keeps them separated from s_P and s_M; centring the multiplier
so that E[exp(lam)] = 1 instead makes log Q and s^2/2 exactly confounded and the
fit runs away along that ridge.

Inference is an exact forward recursion over a uniform quadrature grid on the
joint log-scale (nodes spaced at the resolution limit ``1.5 s`` so there is no
dead zone -- see ``_chain``), with the level posterior collapsed to a single
Gaussian per step (GPB1).  That collapse is the one approximation in the method.

Two conservation laws hold per step, by construction rather than by rule:

    amplitude:  the innovation splits three ways, coefficients summing to 1
                  "I was already wrong about theta" + "the level really moved"
                  + "that was measurement noise"
    scale:      each channel's log-scale splits into a part carried over from
                before (a persistent, regime-like contribution) and a part new
                at t (an impulsive, anomaly-like contribution)

Basic use:

    from statfilter import AdaptiveFilter

    f = AdaptiveFilter.fit(x)          # learn the six parameters
    r = f.filter(x)                    # r.mean, r.var, r.process_anomaly, ...

    f.reset()                          # then stream:
    for value in stream:
        step = f.update(value)
        print(step.mean, step.measurement_regime)
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass, asdict, field

import numpy as np

try:                                    # the compiled recursion, when built
    import lucid_kernel as _kernel      # see lucid_kernel/README.md
except ImportError:                     # pragma: no cover - optional
    try:                                # installed as `lucid.odefilter` etc.
        from .. import lucid_kernel as _kernel
    except ImportError:
        _kernel = None

__all__ = ["AdaptiveFilter", "Params", "FilterResult", "Step"]

_LOG2PI = math.log(2.0 * math.pi)


# --------------------------------------------------------------------- params
@dataclass(frozen=True)
class Params:
    """The six learned numbers.

    Q, s2      median (geometric-mean) variance of the process and measurement
               noise.  With s_P = s_M = 0 the model is the ordinary local-level
               model and these are its Q and sigma^2 exactly.
    phi_P/M    persistence of each channel's log-scale, in [0, 1).  Near 0 the
               channel produces isolated spikes; near 1 it drifts.  Only
               meaningful when the corresponding s is greater than zero -- the
               persistence of a scale that does not vary is undefined.
    s_P/M      log-SD of each channel's scale.  Zero means homoscedastic; this
               is the coordinate that says whether there is any scale structure
               to speak of, and it is the more reliably estimated of the two.
    """

    Q: float
    s2: float
    phi_P: float = 0.0
    phi_M: float = 0.0
    s_P: float = 0.0
    s_M: float = 0.0

    def __post_init__(self):
        if not (self.Q > 0 and self.s2 > 0):
            raise ValueError("Q and s2 must be positive")
        for name in ("phi_P", "phi_M"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must lie in [0, 1)")
        for name in ("s_P", "s_M"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def snr(self) -> float:
        """q = Q / s2, the signal-to-noise ratio of the level model."""
        return self.Q / self.s2

    @property
    def gain(self) -> float:
        """Steady-state Kalman gain of the homoscedastic part: K^2 s2 + K Q = Q."""
        q = self.snr
        return (-q + math.sqrt(q * q + 4.0 * q)) / 2.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        return cls(**d)

    # internal: the unconstrained vector the optimiser works in
    def _vec(self) -> np.ndarray:
        return np.array([math.log(self.Q), math.log(self.s2),
                         _logit(self.phi_P), _logit(self.phi_M),
                         math.log(max(self.s_P, 1e-6)),
                         math.log(max(self.s_M, 1e-6))])

    @classmethod
    def _from_vec(cls, v: np.ndarray) -> "Params":
        return cls(Q=math.exp(v[0]), s2=math.exp(v[1]),
                   phi_P=_expit(v[2]), phi_M=_expit(v[3]),
                   s_P=math.exp(v[4]), s_M=math.exp(v[5]))


def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return math.log(p / (1.0 - p))


def _expit(z: float) -> float:
    # Branch to stay finite at large |z|.  The unconstrained Nelder-Mead search
    # in fit_ can push the logit far past the +-20.7 that _logit ever produces,
    # and 1/(1+exp(-z)) raises OverflowError below z = -709.
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


# --------------------------------------------------------------------- results
@dataclass
class Step:
    """Everything the filter knows after one observation."""

    mean: float                 #: posterior mean of the level
    var: float                  #: posterior variance of the level
    innovation: float           #: x_t - prior mean
    loglik: float               #: log predictive density of x_t

    # amplitude conservation -- these three sum to 1
    share_prior: float          #: attributed to already being wrong about theta
    share_process: float        #: attributed to the level genuinely moving
    share_measurement: float    #: attributed to measurement noise

    # the four signed mode coordinates, in log-scale nats
    process_anomaly: float      #: new-at-t process-scale excess
    process_regime: float       #: carried-over process-scale level
    measurement_anomaly: float  #: new-at-t measurement-scale excess
    measurement_regime: float   #: carried-over measurement-scale level

    @property
    def process_scale(self) -> float:
        """E[lamP | data]; equals process_anomaly + process_regime."""
        return self.process_anomaly + self.process_regime

    @property
    def measurement_scale(self) -> float:
        """E[lamM | data]; equals measurement_anomaly + measurement_regime."""
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
    loglik: float = 0.0

    @property
    def process_scale(self) -> np.ndarray:
        return self.process_anomaly + self.process_regime

    @property
    def measurement_scale(self) -> np.ndarray:
        return self.measurement_anomaly + self.measurement_regime

    @property
    def modes(self) -> np.ndarray:
        """(n, 4) array of the signed mode coordinates: PA, PR, MA, MR."""
        return np.column_stack([self.process_anomaly, self.process_regime,
                                self.measurement_anomaly, self.measurement_regime])

    def __len__(self) -> int:
        return len(self.mean)


# ------------------------------------------------------------------- the grid
_GAP_FACTOR = 1.5   # node spacing = 1.5 * s: the resolution limit, no dead zone
                    # (research/adaptive-grid finding 11).  Uniform spacing, not
                    # Gauss-Hermite -- GH optimises quadrature accuracy of a smooth
                    # integrand, the wrong criterion for *representing* a log-scale;
                    # its non-uniform nodes over-resolve the centre and leave an
                    # edge gap > 1.5 s (1.73 s at order 3, 1.50 s at order 5).


def _uniform_grid(n: int):
    """Centred uniform node offsets (in units of s) and the stationary weights.

    Nodes sit at ``1.5 * s * z`` with ``z`` a centred integer ladder, so the
    spacing is exactly the resolution limit ``1.5 s`` and the span is
    ``+-(n-1)/2 * 1.5 s`` of the stationary law.  The weights are the stationary
    normal density on those nodes -- independent of ``s`` because the grid scales
    with ``s`` -- so this is the discrete stationary law of the AR(1) log-scale.
    """
    z = np.arange(n, dtype=float) - (n - 1) / 2.0
    w = np.exp(-0.5 * (_GAP_FACTOR * z) ** 2)
    return z, w / w.sum()


def _chain(phi: float, s: float, n: int):
    """Quadrature grid for a stationary AR(1) log-scale, and its transition matrix.

    Nodes are **uniform at the resolution spacing** ``1.5 s`` (finding 11: the
    dead-zone-free spacing); the weights are the stationary density on those nodes
    and the transition is the exact AR(1) Gaussian kernel, row-normalised.  The
    only choice is ``n``, a resolution (span ``+-(n-1)/2 * 1.5 s``).
    """
    z, w = _uniform_grid(n)
    lam = (_GAP_FACTOR * s) * z
    # s * s underflows to 0.0 for s below about 1e-162, and the unconstrained
    # search in fit_ does reach there, so guard on the square rather than on s
    # itself -- otherwise the 1/nu below produces nan and poisons the fit.
    if not s > 0.0 or s * s <= 0.0:
        return np.zeros(n), w, np.tile(w, (n, 1))
    nu = max(s * s * (1.0 - phi * phi), 1e-12)
    ex = -0.5 * (lam[None, :] - phi * lam[:, None]) ** 2 / nu
    T = np.exp(np.clip(ex, -700.0, 700.0))
    T /= T.sum(1, keepdims=True)
    return lam, w, T


# ------------------------------------------------- the same grid, for a batch
# The recursion is sequential in t and cannot be vectorised over time.  It can be
# vectorised over PARAMETER VECTORS, and that turns out to be nearly free: the
# per-step cost is dominated by numpy dispatch, not arithmetic, so widening every
# array from (G,) to (B, G) costs far less than running it B times.  Measured on
# a 1200-point series at order 5: B = 13 costs 1.4x one evaluation, B = 100 costs
# 3.8x.  Everything fit() does -- start scans, finite-difference gradients -- is
# therefore organised as batches rather than points.  See exploration/scripts/
# SPEED-002-batch-scaling.py.
#
# The three caps below are the batch's equivalents of the -60/+60 and 1e-6 clips
# in the scalar path: they keep exp() and the logit finite.  They sit far outside
# any estimate the model can meaningfully produce (phi to within 1e-6 of 0 or 1,
# a log-scale SD of 20 nats) and are checked against the probe battery for
# bindingness in SPEED-006.
_LOGIT_CAP = 14.0                       # |logit phi| <= 14  <=>  phi in [8e-7, 1-8e-7]
_LOG_S_CAP = math.log(20.0)             # s <= 20
_LOG_S_FLOOR = math.log(1e-6)           # s >= 1e-6


def _chain_batch(phi: np.ndarray, s: np.ndarray, n: int):
    """:func:`_chain` for a batch of (phi, s); returns (B, n) lam and (B, n, n) T."""
    z, w = _uniform_grid(n)
    lam = s[:, None] * (_GAP_FACTOR * z)
    nu = np.maximum(s * s * (1.0 - phi * phi), 1e-12)[:, None, None]
    ex = -0.5 * (lam[:, None, :] - phi[:, None, None] * lam[:, :, None]) ** 2 / nu
    T = np.exp(np.clip(ex, -700.0, 700.0))
    T /= T.sum(2, keepdims=True)
    return lam, np.broadcast_to(w, lam.shape), T


@functools.lru_cache(maxsize=None)
def _batch_verified(order: int):
    """Whether the compiled recursion has been checked against NumPy here.

    The parent's step is elementwise work plus three reductions, none of which
    einsum gets a say in, so there is nothing to choose between -- but the
    check is run anyway, because "identical" is a claim about the machine the
    code is on and not only about the code.  Cached per grid size.
    """
    if _kernel is None or not _kernel.available():
        return False
    ext = _kernel.ext()
    rng = np.random.default_rng(90210)
    B, n = 5, 96
    V = np.array([-7.0, -9.0, 0.0, 0.0, math.log(0.3), math.log(0.3)]) \
        + 0.4 * rng.standard_normal((B, 6))
    xv = np.cumsum(0.03 * rng.standard_normal(n))
    xv[n // 3] = np.nan
    Q, S2 = np.exp(V[:, 0]), np.exp(V[:, 1])
    phP = 1.0 / (1.0 + np.exp(-V[:, 2]))
    phM = 1.0 / (1.0 + np.exp(-V[:, 3]))
    sP, sM = np.exp(V[:, 4]), np.exp(V[:, 5])
    lamP, wP, TP = _chain_batch(phP, sP, order)
    lamM, wM, TM = _chain_batch(phM, sM, order)
    LP = np.repeat(lamP, order, axis=1)
    LM = np.tile(lamM, (1, order))
    T = (TP[:, :, None, :, None] * TM[:, None, :, None, :]).reshape(
        B, order * order, order * order)
    pi = (wP[:, :, None] * wM[:, None, :]).reshape(B, order * order)
    Qg = Q[:, None] * np.exp(np.clip(LP, -60.0, 60.0))
    Rg = S2[:, None] * np.exp(np.clip(LM, -60.0, 60.0))
    args = (np.ascontiguousarray(xv), np.ascontiguousarray(T),
            np.ascontiguousarray(pi), np.ascontiguousarray(Qg),
            np.ascontiguousarray(Rg))
    return _kernel.verify(("stat", int(order)),
                          lambda: ext.stat_loglik_batch(*args),
                          lambda: _batch_numpy(xv, T, pi, Qg, Rg),
                          candidates=((),)) is not None


def _loglik_batch(x: np.ndarray, V: np.ndarray, order: int) -> np.ndarray:
    """Marginal log-likelihood of ``x`` at every unconstrained vector in ``V``.

    ``V`` is (B, 6) in the coordinates of :meth:`Params._vec`.  The recursion is
    the one in :meth:`AdaptiveFilter.update`, carried out for all B vectors at
    once; at B = 1 it agrees with :meth:`AdaptiveFilter.loglik` bit for bit.
    Only the likelihood is produced -- no shares, no mode coordinates, no Step
    objects -- because that is all a fit needs.
    """
    V = np.atleast_2d(V)
    B, n = V.shape[0], order
    Q = np.exp(V[:, 0])
    S2 = np.exp(V[:, 1])
    phP = 1.0 / (1.0 + np.exp(-np.clip(V[:, 2], -_LOGIT_CAP - 1.0, _LOGIT_CAP + 1.0)))
    phM = 1.0 / (1.0 + np.exp(-np.clip(V[:, 3], -_LOGIT_CAP - 1.0, _LOGIT_CAP + 1.0)))
    sP = np.exp(np.clip(V[:, 4], _LOG_S_FLOOR - 1.0, _LOG_S_CAP + 1.0))
    sM = np.exp(np.clip(V[:, 5], _LOG_S_FLOOR - 1.0, _LOG_S_CAP + 1.0))

    lamP, wP, TP = _chain_batch(phP, sP, n)
    lamM, wM, TM = _chain_batch(phM, sM, n)
    LP = np.repeat(lamP, n, axis=1)                     # joint grid: P varies slowly
    LM = np.tile(lamM, (1, n))
    T = (TP[:, :, None, :, None] * TM[:, None, :, None, :]).reshape(B, n * n, n * n)
    pi = (wP[:, :, None] * wM[:, None, :]).reshape(B, n * n)

    Qg = Q[:, None] * np.exp(np.clip(LP, -60.0, 60.0))
    Rg = S2[:, None] * np.exp(np.clip(LM, -60.0, 60.0))

    if _batch_verified(order):
        out = _kernel.ext().stat_loglik_batch(
            np.ascontiguousarray(x, dtype=float), np.ascontiguousarray(T),
            np.ascontiguousarray(pi), np.ascontiguousarray(Qg),
            np.ascontiguousarray(Rg))
        if out is not None:
            return out
    return _batch_numpy(x, T, pi, Qg, Rg)


def _batch_numpy(x, T, pi, Qg, Rg):
    """The recursion in NumPy.  The kernel is checked against this, and this
    is what runs when there is no kernel to check."""
    B = Qg.shape[0]
    pi = pi.copy()
    QR = Qg + Rg
    x0 = float(x[0])
    m = np.full(B, x0 if np.isfinite(x0) else 0.0)
    P = Rg.max(1) + Qg.max(1)                           # the diffuse start of update()
    ll = np.zeros(B)
    for v in x:
        pi = np.einsum("bi,bij->bj", pi, T)
        if not np.isfinite(v):                          # missing: propagate only
            P = P + (pi * Qg).sum(1)
            continue
        Pp = P[:, None] + Qg
        S = P[:, None] + QR
        e = v - m
        lg = -0.5 * (np.log(S) + (e * e)[:, None] / S)
        mx = lg.max(1)
        w = pi * np.exp(lg - mx[:, None])
        Z = w.sum(1)
        ll += np.log(Z) + mx - 0.5 * _LOG2PI
        pi = w / Z[:, None]
        K = Pp / S
        Kbar = (pi * K).sum(1)
        m = m + Kbar * e
        P = ((pi * ((1.0 - K) * Pp)).sum(1)
             + e * e * (pi * (K - Kbar[:, None]) ** 2).sum(1))
    return ll


# ----------------------------------------------- the s = 0 face, in closed form
# Where s_P = s_M = 0 every quadrature node carries lam = 0, the grid collapses to
# one state, and the model is the plain local-level model.  Two exact facts then
# reduce that face from a two-parameter search to a one-parameter one:
#
#   * the scalar recursion is homogeneous of degree 1 in sigma^2, so with
#     P_t = sigma^2 p_t and Q = sigma^2 q the gains -- and hence the innovations
#     -- depend on the ratio q alone;
#   * sigma^2 is therefore concentrated out in closed form, leaving a profile
#     likelihood in log q that a scalar Python loop evaluates ~100x faster than
#     one pass over the 25-state grid.
#
# Derivation and numerical check: exploration/scripts/SPEED-004.

def _face_profile(x: np.ndarray, q: float):
    """(sigma^2, profile loglik, standardised innovations) at ratio q = Q/sigma^2."""
    p = 1.0 + q                                 # P_0 / sigma^2, matching update()
    m = float(x[0]) if math.isfinite(x[0]) else 0.0          # ditto
    acc = 0.0                                   # sum e^2 / Stil
    lsum = 0.0                                  # sum log Stil
    cnt = 0
    u = np.empty(x.size)
    for i in range(x.size):
        v = x[i]
        if not math.isfinite(v):
            p += q                              # missing: propagate, do not correct
            u[i] = math.nan
            continue
        S = p + q + 1.0
        e = v - m
        acc += e * e / S
        lsum += math.log(S)
        u[i] = e / math.sqrt(S)
        K = (p + q) / S
        m += K * e
        p = (1.0 - K) * (p + q)
        cnt += 1
    s2 = acc / cnt
    ll = -0.5 * (cnt * (_LOG2PI + math.log(s2) + 1.0) + lsum)
    u /= math.sqrt(s2)
    if math.isfinite(x[0]):
        u[0] = math.nan     # m_0 = x_0, so this innovation is zero by construction
    return s2, ll, u


_Q_SCAN = np.logspace(-7.0, 3.0, 41)            # q is a ratio, so this needs no scaling


def _face_optimum(x: np.ndarray):
    """Maximise the concentrated profile over q.  Returns (Q, sigma^2, u)."""
    lls = [_face_profile(x, q)[1] for q in _Q_SCAN]
    i = int(np.argmax(lls))
    lo = math.log(_Q_SCAN[max(i - 1, 0)])
    hi = math.log(_Q_SCAN[min(i + 1, _Q_SCAN.size - 1)])
    if hi > lo:
        from scipy.optimize import minimize_scalar
        r = minimize_scalar(lambda z: -_face_profile(x, math.exp(z))[1],
                            bounds=(lo, hi), method="bounded",
                            options=dict(xatol=1e-4))
        q = math.exp(float(r.x))
    else:
        q = float(_Q_SCAN[i])
    s2, _, u = _face_profile(x, q)
    return q * s2, s2, u


_VAR_LOG_CHI2 = math.pi ** 2 / 2.0              # Var(log z^2), z ~ N(0,1), exact


def _moment_scale(u: np.ndarray):
    """(s, phi) for the log-scale, read off the face's residuals in one pass.

    If a log-scale channel is present it survives into the homoscedastic fit's
    standardised innovations, where ``log u_t^2 = log z_t^2 + lam_t`` with the two
    terms independent.  Var(log z^2) is exactly pi^2/2, so the lag-0 and lag-1
    autocovariances of log u^2 give the log-scale's variance and its persistence
    directly.  It cannot say which channel carries them -- both inflate an
    innovation -- so it supplies a magnitude and a persistence, and the start
    scan decides the split.
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


# ------------------------------------------------------------ the start screen
# The same 5x5 persistence grid the previous fit() scanned, now crossed with four
# ways of splitting the scale between the channels rather than fixing both at
# 0.6.  A hundred starts in one batched evaluation cost about four, so the screen
# is wider than before and still an order of magnitude cheaper.
_PHI_GRID = (0.02, 0.25, 0.5, 0.75, 0.95)
_S_SPLITS = ((0.03, 0.03), (0.6, 0.6), (0.03, 0.6), (0.6, 0.03))
_QUIET = 0.1                            # divides "no scale structure" starts from the rest


def _bounds(g0: float):
    """Box for the search, in the unconstrained coordinates.

    A numerical guard, not a prior: sigma^2 and Q cannot exceed the data's own
    gamma_0 = E[(x_t - x_{t-1})^2] by more than a rounding margin, and the other
    four are the caps of :func:`_loglik_batch`.  SPEED-006 checks that no
    coordinate of any battery fit lands on a bound.
    """
    lg = math.log(g0)
    return [(lg - 30.0, lg + 5.0), (lg - 30.0, lg + 5.0),
            (-_LOGIT_CAP, _LOGIT_CAP), (-_LOGIT_CAP, _LOGIT_CAP),
            (_LOG_S_FLOOR, _LOG_S_CAP), (_LOG_S_FLOOR, _LOG_S_CAP)]


# ------------------------------------------------------------------ the filter
class AdaptiveFilter:
    """Adaptive local-level filter with learned process and measurement noise.

    Parameters
    ----------
    params : Params, optional
        The six model numbers.  Omit and call :meth:`fit` to learn them.
    order : int
        Quadrature nodes per channel; the joint grid has ``order**2`` states.
        This is a numerical resolution, not a model choice.  The default of 5
        is what the published probe battery used.  Higher orders resolve the
        likelihood surface better at quadratic cost -- 7 or 9 is a reasonable
        upgrade if fitted volatilities matter more than speed.

    Notes
    -----
    The filter is stateful when streaming.  :meth:`filter` is stateless with
    respect to the object (it starts from a fresh state and leaves the object's
    streaming state untouched); :meth:`update` advances the streaming state.
    """

    def __init__(self, params: Params | None = None, order: int = 5):
        if order < 3:
            raise ValueError("order must be at least 3")
        self.params = params
        self.order = int(order)
        self._built = None
        self.reset()

    # ---------------------------------------------------------------- plumbing
    def __repr__(self) -> str:
        if self.params is None:
            return f"AdaptiveFilter(unfitted, order={self.order})"
        p = self.params
        return (f"AdaptiveFilter(Q={p.Q:.4g}, s2={p.s2:.4g}, "
                f"phi_P={p.phi_P:.3f}, s_P={p.s_P:.3f}, "
                f"phi_M={p.phi_M:.3f}, s_M={p.s_M:.3f}, order={self.order})")

    def to_dict(self) -> dict:
        if self.params is None:
            raise ValueError("filter is not fitted")
        return {"params": self.params.to_dict(), "order": self.order}

    @classmethod
    def from_dict(cls, d: dict) -> "AdaptiveFilter":
        return cls(Params.from_dict(d["params"]), order=int(d.get("order", 5)))

    def _build(self):
        """Precompute the grid.  Cached against (params, order)."""
        key = (self.params, self.order)
        if self._built is not None and self._built[0] == key:
            return self._built[1]
        p, n = self.params, self.order
        lamP, wP, TP = _chain(p.phi_P, p.s_P, n)
        lamM, wM, TM = _chain(p.phi_M, p.s_M, n)
        g = {
            "LP": np.repeat(lamP, n),
            "LM": np.tile(lamM, n),
            "T": np.kron(TP, TM),
            "pi0": np.kron(wP, wM),
        }
        g["Qg"] = p.Q * np.exp(np.clip(g["LP"], -60.0, 60.0))
        g["Rg"] = p.s2 * np.exp(np.clip(g["LM"], -60.0, 60.0))
        g["QR"] = g["Qg"] + g["Rg"]
        self._built = (key, g)
        return g

    # ------------------------------------------------------------- streaming
    def reset(self, mean: float | None = None, var: float | None = None) -> "AdaptiveFilter":
        """Clear the streaming state.  Returns self, so it chains."""
        self._pi = None
        self._m = mean
        self._P = var
        self._prev_lamP = 0.0
        self._prev_lamM = 0.0
        self._loglik = 0.0
        return self

    def update(self, x: float) -> Step:
        """Absorb one observation and return everything known after it.

        A NaN observation is treated as missing: the state is propagated but not
        corrected, which is the right behaviour for gaps in a series.
        """
        if self.params is None:
            raise ValueError("filter is not fitted; call fit() or pass params")
        g = self._build()
        if self._pi is None:
            self._pi = g["pi0"].copy()
            if self._m is None:
                self._m = float(x) if np.isfinite(x) else 0.0
            if self._P is None:
                # Diffuse start at the model's own scale: one observation's worth
                # of measurement noise plus the widest process step on the grid.
                self._P = float(g["Rg"].max() + g["Qg"].max())

        pi = self._pi @ g["T"]

        if not np.isfinite(x):                       # missing observation
            self._pi = pi
            self._P = float(self._P + pi @ g["Qg"])
            lamP, lamM = float(pi @ g["LP"]), float(pi @ g["LM"])
            step = Step(self._m, self._P, math.nan, 0.0,
                        1.0, 0.0, 0.0,
                        lamP - self.params.phi_P * self._prev_lamP,
                        self.params.phi_P * self._prev_lamP,
                        lamM - self.params.phi_M * self._prev_lamM,
                        self.params.phi_M * self._prev_lamM)
            self._prev_lamP, self._prev_lamM = lamP, lamM
            return step

        P = self._P
        S = P + g["QR"]                              # predictive variance per state
        e = float(x) - self._m
        lg = -0.5 * (np.log(S) + e * e / S)
        mx = float(lg.max())
        w = pi * np.exp(lg - mx)
        Z = float(w.sum())
        ll = math.log(Z) + mx - 0.5 * _LOG2PI
        pi = w / Z

        K = (P + g["Qg"]) / S
        Kbar = float(pi @ K)
        m_new = self._m + Kbar * e
        # collapse (GPB1): the mixture variance is the mean conditional variance
        # plus the spread of the conditional means, and mi - mbar = (K - Kbar) e
        P_new = float(pi @ ((1.0 - K) * (P + g["Qg"]))
                      + e * e * (pi @ (K - Kbar) ** 2))

        lamP, lamM = float(pi @ g["LP"]), float(pi @ g["LM"])
        share_prior = float(pi @ (P / S))
        share_process = float(pi @ (g["Qg"] / S))
        share_measurement = float(pi @ (g["Rg"] / S))

        step = Step(
            mean=m_new, var=P_new, innovation=e, loglik=ll,
            share_prior=share_prior,
            share_process=share_process,
            share_measurement=share_measurement,
            process_anomaly=lamP - self.params.phi_P * self._prev_lamP,
            process_regime=self.params.phi_P * self._prev_lamP,
            measurement_anomaly=lamM - self.params.phi_M * self._prev_lamM,
            measurement_regime=self.params.phi_M * self._prev_lamM,
        )
        self._pi, self._m, self._P = pi, m_new, P_new
        self._prev_lamP, self._prev_lamM = lamP, lamM
        self._loglik += ll
        return step

    def predict(self, horizon: int = 1) -> tuple[float, float]:
        """Mean and variance of theta_{t+h} given everything so far.

        The level is a random walk, so the forecast mean is flat and the variance
        grows by the expected process variance per step.
        """
        if self._pi is None:
            raise ValueError("nothing observed yet")
        g = self._build()
        pi, var = self._pi, self._P
        for _ in range(int(horizon)):
            pi = pi @ g["T"]
            var = var + float(pi @ g["Qg"])
        return self._m, var

    # ----------------------------------------------------------------- batch
    def loglik(self, x) -> float:
        """Exact marginal log-likelihood of a series.  Does not touch state."""
        return self._run(np.asarray(x, dtype=float), want=False)

    def filter(self, x) -> FilterResult:
        """Run over a whole series from a fresh state.  Does not touch state."""
        return self._run(np.asarray(x, dtype=float), want=True)

    def _run(self, x: np.ndarray, want: bool, criterion: str = "loglik"):
        if self.params is None:
            raise ValueError("filter is not fitted; call fit() or pass params")
        if x.ndim != 1 or x.size == 0:
            raise ValueError("x must be a non-empty 1-D array")
        saved = (self._pi, self._m, self._P, self._prev_lamP,
                 self._prev_lamM, self._loglik)
        try:
            self.reset()
            if not want:
                if criterion == "pem":
                    # mean squared one-step innovation.  Negated so that, like
                    # the log-likelihood, larger is better and fit_ can maximise
                    # either without knowing which it has.
                    total = 0.0
                    for v in x:
                        e = self.update(v).innovation
                        total += e * e
                    return -total / max(x.size, 1)
                total = 0.0
                for v in x:
                    total += self.update(v).loglik
                return total
            n = x.size
            cols = ("mean", "var", "innovation", "share_prior", "share_process",
                    "share_measurement", "process_anomaly", "process_regime",
                    "measurement_anomaly", "measurement_regime")
            out = {c: np.empty(n) for c in cols}
            total = 0.0
            for i, v in enumerate(x):
                st = self.update(v)
                for c in cols:
                    out[c][i] = getattr(st, c)
                total += st.loglik
            return FilterResult(loglik=total, **out)
        finally:
            (self._pi, self._m, self._P, self._prev_lamP,
             self._prev_lamM, self._loglik) = saved

    # ------------------------------------------------------------------- fit
    @classmethod
    def fit(cls, x, order: int = 5, max_iter: int = 500,
            criterion: str = "loglik") -> "AdaptiveFilter":
        """Learn all six parameters from a series and return a fitted filter.

        ``criterion`` selects what is optimised.  ``"loglik"`` (the default)
        maximises the predictive log-likelihood.  ``"pem"`` minimises the mean
        squared one-step innovation; both cost one filter pass, so neither is
        cheaper.

        **``"pem"`` is not recommended for the full six-parameter fit.**  The
        squared innovation depends on the parameters only through the predicted
        mean, so it is nearly blind to any direction that moves the predictive
        variance without moving the gain, and the search drifts along it.
        Measured over four regimes with a true ``s2`` of 1.0, ``"pem"`` recovered
        1.10, 2.02, 9.21 and 6.58 against ``"loglik"``'s 0.96, 1.03, 1.04 and
        1.02, and inflated ``s_M`` everywhere -- to 1.54 on data with no scale
        variation at all.  Tracking MSE was only about 1% worse, so the damage is
        to the parameters rather than to the estimate, but the parameters are
        what callers read.  It is exposed for comparison work; see
        ``optimality-proof/0027``.
        """
        f = cls(order=order)
        f.fit_(x, max_iter=max_iter, criterion=criterion)
        return f

    def fit_(self, x, max_iter: int = 500,
             criterion: str = "loglik") -> "AdaptiveFilter":
        """Fit in place.  Returns self.

        Staged, because a six-dimensional search from one start is not reliable.
        Every stage is organised around the one measured fact about this
        recursion: it is dispatch-bound, so B parameter vectors cost far less
        than B evaluations (:func:`_loglik_batch`).  Nothing here is a tuning
        choice -- the estimate is the maximum likelihood estimate however it is
        reached, and ``max_iter`` is a compute budget.

        pass 1   the exact optimum on the s_P = s_M = 0 face.  That face is the
                 plain local-level model, where sigma^2 concentrates out in
                 closed form and only the ratio q = Q/sigma^2 is searched --
                 a scalar recursion, ~100x cheaper than one pass over the grid,
                 and an exact optimum rather than a scan's best point;
        pass 2   the magnitude and persistence of whatever log-scale structure
                 is left in that fit's residuals, read off their log-squares in
                 one pass (:func:`_moment_scale`);
        pass 3   ONE batched evaluation of ~100 starts -- a 5x5 grid over the two
                 persistences crossed with four splits of the scale between the
                 channels, plus what pass 2 proposed.  Costs about four
                 evaluations, and replaces a 25-point scan that cost 25;
        pass 4   full 6-D maximum likelihood by L-BFGS-B, its gradient taken by
                 central differences batched into a single evaluation.  The
                 surface is smooth in these coordinates, so a quasi-Newton method
                 gets there in tens of gradients where Nelder-Mead needed ~500
                 function values per start.  Run from the best quiet start and
                 the best volatile one, better likelihood kept, as before.

        The passes above are the ``"loglik"`` path.  ``criterion="pem"`` has no
        batched kernel, so it keeps the older scan-and-simplex search; see
        :meth:`fit` for why it is a comparison tool only.
        """
        from scipy.optimize import minimize          # only needed for fitting

        x = np.asarray(x, dtype=float)
        finite = np.isfinite(x)
        good = x[finite]
        if good.size < 10:
            raise ValueError("need at least 10 finite observations to fit")
        d = np.diff(good)
        g0 = float(np.mean(d * d))
        if not g0 > 0:
            raise ValueError("series is constant; nothing to fit")
        n = max(int(finite.sum()), 1)

        if criterion not in ("loglik", "pem"):
            raise ValueError("criterion must be 'loglik' or 'pem'")

        if criterion == "pem":
            # No batched kernel computes the innovation criterion, so the PEM
            # fit keeps the scan-and-simplex search.  It is a comparison tool,
            # not a production path; see :meth:`fit`.
            def ll(vec) -> float:
                try:
                    self.params = Params._from_vec(vec)
                except (ValueError, OverflowError):
                    return -np.inf
                return self._run(x, want=False, criterion=criterion)

            # stage 0 -- Q, with s2 = (gamma_0 - Q)/2
            cand = g0 * np.logspace(-5.0, -0.05, 25)
            base = [np.array([math.log(Qc), math.log((g0 - Qc) / 2.0),
                              0.0, 0.0, math.log(1e-3), math.log(1e-3)])
                    for Qc in cand]
            Q0 = cand[int(np.argmax([ll(v) for v in base]))]
            s20 = (g0 - Q0) / 2.0

            # stage 0.5 -- the two persistences
            grid = [0.02, 0.25, 0.5, 0.75, 0.95]
            best_ph, best_v = (0.0, 0.0), -np.inf
            for pp in grid:
                for pm in grid:
                    v = np.array([math.log(Q0), math.log(s20), _logit(pp),
                                  _logit(pm), math.log(0.6), math.log(0.6)])
                    val = ll(v)
                    if val > best_v:
                        best_ph, best_v = (_logit(pp), _logit(pm)), val

            # stage 1 -- full search from that start
            best_vec, best_f = None, np.inf
            for s0 in (0.03, 0.6):
                start = np.array([math.log(Q0), math.log(s20), best_ph[0],
                                  best_ph[1], math.log(s0), math.log(s0)])
                r = minimize(lambda v: -ll(v) / n, start, method="Nelder-Mead",
                             options=dict(maxiter=int(max_iter), xatol=2e-3,
                                          fatol=1e-5))
                if r.fun < best_f:
                    best_vec, best_f = r.x, r.fun

            self.params = Params._from_vec(best_vec)
            self._built = None
            self.reset()
            return self

        # pass 1 -- the s = 0 face, exactly
        Q0, s20, resid = _face_optimum(x)
        lQ0, ls20 = math.log(Q0), math.log(s20)

        # pass 2 -- what the face left behind
        s_hat, phi_hat = _moment_scale(resid)

        # pass 3 -- one batched screen over the starts
        starts = [[lQ0, ls20, _logit(pp), _logit(pm), math.log(sp), math.log(sm)]
                  for pp in _PHI_GRID for pm in _PHI_GRID for sp, sm in _S_SPLITS]
        if s_hat > 0.0:
            lz, lp, tiny = math.log(s_hat), _logit(phi_hat), math.log(1e-3)
            starts += [[lQ0, ls20, lp, lp, lz, lz],
                       [lQ0, ls20, lp, lp, lz, tiny],
                       [lQ0, ls20, lp, lp, tiny, lz]]
        V = np.clip(np.array(starts), *zip(*_bounds(g0)))
        val = _loglik_batch(x, V, self.order)
        loud = V[:, 4:].max(1) > math.log(_QUIET)
        chosen = [V[np.argmax(np.where(m, val, -np.inf))]
                  for m in (~loud, loud) if m.any()]

        # pass 4 -- full ML, gradients batched
        bounds = _bounds(g0)
        h = 1e-4
        stencil = np.zeros((13, 6))
        stencil[1::2] = h * np.eye(6)
        stencil[2::2] = -h * np.eye(6)

        def fg(v):
            ll = _loglik_batch(x, v + stencil, self.order)
            return -ll[0] / n, -(ll[1::2] - ll[2::2]) / (2.0 * h * n)

        best_vec, best_f = None, np.inf
        for start in chosen:
            r = minimize(fg, start, jac=True, method="L-BFGS-B", bounds=bounds,
                         options=dict(maxiter=int(max_iter), ftol=1e-12, gtol=1e-7))
            if r.fun < best_f:
                best_vec, best_f = r.x, r.fun

        self.params = Params._from_vec(best_vec)
        self._built = None
        self.reset()
        return self
