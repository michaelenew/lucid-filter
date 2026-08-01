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

Learned parameters: alpha (p of them), Q, S2, phi_P, phi_M, s_P, s_M.  All are
learned by maximum marginal likelihood.  `order` is a quadrature resolution --
a compute budget, not a tuning parameter.

Two diagnostics come out for free, and they are orthogonal by construction
(measured in exploration/0025):

    innovation MEAN transient   an event: a one-off disturbance.  Any such
                                event, in any direction, IS process noise --
                                the filter absorbs it and it leaves no trace.
    innovation WHITENESS        a parameter change: the dynamics themselves
                                moved.  Leaves no mean signature at all and a
                                permanent one in the lag-1 autocorrelation.

`Step.whiteness` reports the second.  A correctly specified filter emits white
innovations, so sustained departure from zero is the signal that `alpha` no
longer fits -- the one thing this filter models but does not yet adapt.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field

import numpy as np

__all__ = ["OdeFilter", "Params", "FilterResult", "Step"]

_LOG2PI = math.log(2.0 * math.pi)


class _Numerical(ArithmeticError):
    """Raised when a parameter vector drives the recursion out of range."""


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

    def __post_init__(self):
        object.__setattr__(self, "alpha", tuple(float(a) for a in self.alpha))
        if len(self.alpha) < 1:
            raise ValueError("alpha must have at least one entry")
        if not (self.Q > 0 and self.s2 > 0):
            raise ValueError("Q and s2 must be positive")
        for n in ("phi_P", "phi_M"):
            if not 0.0 <= getattr(self, n) < 1.0:
                raise ValueError(f"{n} must lie in [0, 1)")
        for n in ("s_P", "s_M"):
            if getattr(self, n) < 0.0:
                raise ValueError(f"{n} must be non-negative")

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
             math.log(max(self.s_P, 1e-6)), math.log(max(self.s_M, 1e-6))]])

    @classmethod
    def _from_vec(cls, v: np.ndarray, p: int) -> "Params":
        return cls(alpha=tuple(v[:p]), Q=math.exp(v[p]), s2=math.exp(v[p + 1]),
                   phi_P=_expit(v[p + 2]), phi_M=_expit(v[p + 3]),
                   s_P=math.exp(v[p + 4]), s_M=math.exp(v[p + 5]))


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
    state_mean: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    state_cov: np.ndarray = field(default_factory=lambda: np.empty((0, 0, 0)))
    loglik: float = 0.0

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
    """

    def __init__(self, params: Params | None = None, order: int = 5):
        if order < 3:
            raise ValueError("order must be at least 3")
        self.params = params
        self.order = int(order)
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
        return {"params": self.params.to_dict(), "order": self.order}

    @classmethod
    def from_dict(cls, d: dict) -> "OdeFilter":
        return cls(Params.from_dict(d["params"]), order=int(d.get("order", 5)))

    # ------------------------------------------------------------------ grid
    def _build(self):
        key = (self.params, self.order)
        if self._built is not None and self._built[0] == key:
            return self._built[1]
        pr, n = self.params, self.order
        lamP, wP, TP = _chain(pr.phi_P, pr.s_P, n)
        lamM, wM, TM = _chain(pr.phi_M, pr.s_M, n)
        g = {"LP": np.repeat(lamP, n), "LM": np.tile(lamM, n),
             "T": np.kron(TP, TM), "pi0": np.kron(wP, wM),
             "F": pr.companion}
        g["Qg"] = pr.Q * np.exp(np.clip(g["LP"], -60.0, 60.0))
        g["Rg"] = pr.s2 * np.exp(np.clip(g["LM"], -60.0, 60.0))
        self._built = (key, g)
        return g

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
        F = g["F"]
        if self._pi is None:
            self._pi = g["pi0"].copy()
            y0 = float(y) if np.isfinite(y) else 0.0
            self._m = np.full(p, y0)
            self._P = np.eye(p) * float(g["Rg"].max() + g["Qg"].max()) * p

        pi = self._pi @ g["T"]
        m_pri = F @ self._m
        A = F @ self._P @ F.T                     # shared across grid nodes

        if not np.isfinite(y):                    # missing observation
            self._pi = pi
            A[0, 0] += float(pi @ g["Qg"])
            self._m, self._P = m_pri, A
            lamP, lamM = float(pi @ g["LP"]), float(pi @ g["LM"])
            st = Step(float(m_pri[0]), float(A[0, 0]), math.nan, 0.0,
                      1.0, 0.0, 0.0,
                      lamP - self.params.phi_P * self._prev_lamP,
                      self.params.phi_P * self._prev_lamP,
                      lamM - self.params.phi_M * self._prev_lamM,
                      self.params.phi_M * self._prev_lamM,
                      self._whiteness())
            self._prev_lamP, self._prev_lamM = lamP, lamM
            return st

        a00, a0 = A[0, 0], A[:, 0]
        S = a00 + g["Qg"] + g["Rg"]               # predictive variance per node
        # An unconstrained search reaches explosive alpha, where P overflows and
        # S goes non-finite or non-positive.  Signal it rather than emitting a
        # nan log-likelihood the optimiser would then chase.
        if not np.all(np.isfinite(S)) or np.any(S <= 0.0):
            raise _Numerical("non-positive predictive variance")
        e = float(y) - m_pri[0]
        lg = -0.5 * (np.log(S) + e * e / S)
        mx = float(lg.max())
        w = pi * np.exp(lg - mx)
        Z = float(w.sum())
        ll = math.log(Z) + mx - 0.5 * _LOG2PI
        pi = w / Z

        # K_g = (A[:,0] + Qg e1) / S_g  -- differs across nodes only in slot 0
        K = np.outer(1.0 / S, a0).copy()          # (G, p)
        K[:, 0] += g["Qg"] / S
        Kbar = pi @ K                             # (p,)
        m_new = m_pri + Kbar * e

        # collapse (GPB1): mean conditional covariance + spread of the means,
        # and m_g - m_new = (K_g - Kbar) e
        row = np.tile(a0, (len(S), 1))            # (G, p), row 0 of P_pri^g
        row[:, 0] = a00 + g["Qg"]
        Pbar = A + np.eye(p) * 0.0
        Pbar = A.copy()
        Pbar[0, 0] += float(pi @ g["Qg"])
        Pbar -= np.einsum("g,gi,gj->ij", pi, K, row)
        dK = K - Kbar
        Pbar += (e * e) * np.einsum("g,gi,gj->ij", pi, dK, dK)

        lamP, lamM = float(pi @ g["LP"]), float(pi @ g["LM"])
        st = Step(
            mean=float(m_new[0]), var=float(Pbar[0, 0]), innovation=e, loglik=ll,
            share_prior=float(pi @ (a00 / S)),
            share_process=float(pi @ (g["Qg"] / S)),
            share_measurement=float(pi @ (g["Rg"] / S)),
            process_anomaly=lamP - self.params.phi_P * self._prev_lamP,
            process_regime=self.params.phi_P * self._prev_lamP,
            measurement_anomaly=lamM - self.params.phi_M * self._prev_lamM,
            measurement_regime=self.params.phi_M * self._prev_lamM,
            whiteness=self._accum_whiteness(e, float(pi @ S)),
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
        F = g["F"]
        pi, m, P = self._pi, self._m.copy(), self._P.copy()
        for _ in range(int(horizon)):
            pi = pi @ g["T"]
            m = F @ m
            P = F @ P @ F.T
            P[0, 0] += float(pi @ g["Qg"])
        return float(m[0]), float(P[0, 0])

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
                    "measurement_anomaly", "measurement_regime", "whiteness")
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
            scales: bool = True) -> "OdeFilter":
        """Learn every parameter from a series and return a fitted filter.

        ``p`` is the order of the recurrence: 3 is a second-order ODE plus a
        constant offset, which is the class this filter targets.  ``p`` is a
        modelling commitment, not a tuning parameter -- and because each root of
        the characteristic polynomial is a channel, choosing ``p`` is the same
        act as counting channels.

        ``scales=False`` pins the two log-scale channels off, giving an ordinary
        (non-adaptive) recurrence filter.  Useful as a baseline and much faster.
        """
        f = cls(order=order)
        f.fit_(y, p=p, max_iter=max_iter, scales=scales)
        return f

    def fit_(self, y, p: int = 3, max_iter: int = 400,
             scales: bool = True) -> "OdeFilter":
        """Fit in place.  Returns self.

        Staged, because a nine-dimensional search from one start is not
        reliable:

        stage 0   alpha by instrumental variables, using lags p+1..3p as
                  instruments.  Regressing on observed lags is errors-in-
                  variables and does not merely attenuate the dynamics -- it
                  deletes the oscillation outright (exploration/0002).  Lagging
                  past the order annihilates the measurement noise exactly, for
                  every (Q, S2), without needing either.
        stage 1   Q and S2 in closed form from the residual autocovariances,
                  the analogue of the parent's variogram identity.
        stage 2   alpha, Q, S2 by maximum likelihood with the scales off.
        stage 3   the two scale channels, then everything together.

        All of it is scaffolding: the estimate is the maximum likelihood
        estimate however it is reached.  ``max_iter`` is a compute budget.
        """
        from scipy.optimize import minimize

        y = np.asarray(y, dtype=float)
        good = y[np.isfinite(y)]
        if good.size < 8 * p:
            raise ValueError(f"need at least {8 * p} finite observations")

        a0 = _iv_alpha(good, p)
        Q0, s20 = _moment_noises(good, a0)

        def nll(v, n):
            try:
                self.params = Params._from_vec(v, p)
            except (ValueError, OverflowError):
                return np.inf
            r = self._run(y, want=False)
            return np.inf if not np.isfinite(r) else -r / n

        n = max(y.size, 1)
        off = math.log(1e-6)
        base = np.concatenate([a0, [math.log(Q0), math.log(s20),
                                    _logit(0.5), _logit(0.5), off, off]])

        # stage 1b -- Q is badly conditioned from moments (see _moment_noises),
        # so scan it by likelihood at the moment S2 rather than believing it
        best_q, best_v = Q0, -np.inf
        for Qc in Q0 * np.logspace(-2.0, 1.0, 13):
            v = base.copy()
            v[p] = math.log(Qc)
            val = -nll(v, n)
            if val > best_v:
                best_q, best_v = Qc, val
        base[p] = math.log(best_q)

        # stage 2 -- alpha, Q, S2 with the scales pinned off
        idx = list(range(p + 2))
        full = base.copy()

        def sub(vs):
            v = full.copy()
            v[idx] = vs
            return nll(v, n)

        r2 = minimize(sub, full[idx], method="Nelder-Mead",
                      options=dict(maxiter=int(max_iter), xatol=1e-3, fatol=1e-5))
        full[idx] = r2.x

        if not scales:
            self.params = Params._from_vec(full, p)
            self._built = None
            return self.reset()

        # stage 3 -- the scale channels, then a joint polish
        best, bestf = None, np.inf
        for s0 in (0.05, 0.5):
            v = full.copy()
            v[p + 4] = v[p + 5] = math.log(s0)
            sidx = [p + 2, p + 3, p + 4, p + 5]

            def sub2(vs, v=v, sidx=sidx):
                w = v.copy()
                w[sidx] = vs
                return nll(w, n)

            r3 = minimize(sub2, v[sidx], method="Nelder-Mead",
                          options=dict(maxiter=int(max_iter), xatol=2e-3,
                                       fatol=1e-5))
            v[sidx] = r3.x
            if r3.fun < bestf:
                best, bestf = v, r3.fun

        r4 = minimize(lambda v: nll(v, n), best, method="Nelder-Mead",
                      options=dict(maxiter=int(max_iter * 2), xatol=2e-3,
                                   fatol=1e-6))
        self.params = Params._from_vec(r4.x if r4.fun < bestf else best, p)
        self._built = None
        return self.reset()


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
