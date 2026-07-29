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

Inference is an exact forward recursion over a Gauss-Hermite quadrature grid on
the joint log-scale, with the level posterior collapsed to a single Gaussian per
step (GPB1).  That collapse is the one approximation in the method.

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

import math
from dataclasses import dataclass, asdict, field

import numpy as np

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
    return 1.0 / (1.0 + math.exp(-z))


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
def _gauss_hermite(n: int):
    """Nodes and normalised weights for a standard normal."""
    z, w = np.polynomial.hermite_e.hermegauss(n)
    return z, w / w.sum()


def _chain(phi: float, s: float, n: int):
    """Quadrature grid for a stationary AR(1) log-scale, and its transition matrix.

    Nodes are the Gauss-Hermite abscissae of the stationary law; the transition
    is the exact Gaussian kernel evaluated on those nodes and reweighted by the
    stationary density.  The only choice is n, which is a resolution.
    """
    z, w = _gauss_hermite(n)
    lam = s * z
    if s <= 0.0:
        return np.zeros(n), w, np.tile(w, (n, 1))
    nu = max(s * s * (1.0 - phi * phi), 1e-12)
    ex = (-0.5 * (lam[None, :] - phi * lam[:, None]) ** 2 / nu
          + 0.5 * lam[None, :] ** 2 / (s * s))
    T = w[None, :] * np.exp(np.clip(ex, -700.0, 700.0))
    T /= T.sum(1, keepdims=True)
    return lam, w, T


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

    def _run(self, x: np.ndarray, want: bool):
        if self.params is None:
            raise ValueError("filter is not fitted; call fit() or pass params")
        if x.ndim != 1 or x.size == 0:
            raise ValueError("x must be a non-empty 1-D array")
        saved = (self._pi, self._m, self._P, self._prev_lamP,
                 self._prev_lamM, self._loglik)
        try:
            self.reset()
            if not want:
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
    def fit(cls, x, order: int = 5, max_iter: int = 500) -> "AdaptiveFilter":
        """Learn all six parameters from a series and return a fitted filter."""
        f = cls(order=order)
        f.fit_(x, max_iter=max_iter)
        return f

    def fit_(self, x, max_iter: int = 500) -> "AdaptiveFilter":
        """Fit in place.  Returns self.

        Staged, because a six-dimensional search from one start is not reliable:

        stage 0    scan Q with s2 pinned by the variogram identity
                   gamma_0 = Q + 2 s2, so every candidate is admissible and the
                   range comes from the data's own scale;
        stage 0.5  coarse 5x5 scan over (phi_P, phi_M).  Without it the search
                   sits at its initialisation on impulsive data, which showed up
                   as a factor-1.3 worst case rather than 1.017;
        stage 1    full 6-D maximum likelihood from that start, run from a quiet
                   and a volatile log-scale, better likelihood kept.

        All of it is numerical scaffolding: the estimate is the maximum
        likelihood estimate however it is reached.  ``max_iter`` is a compute
        budget, not a tuning parameter.
        """
        from scipy.optimize import minimize          # only needed for fitting

        x = np.asarray(x, dtype=float)
        good = x[np.isfinite(x)]
        if good.size < 10:
            raise ValueError("need at least 10 finite observations to fit")
        d = np.diff(good)
        g0 = float(np.mean(d * d))
        if not g0 > 0:
            raise ValueError("series is constant; nothing to fit")

        def ll(vec) -> float:
            try:
                self.params = Params._from_vec(vec)
            except ValueError:
                return -np.inf
            return self._run(x, want=False)

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
                v = np.array([math.log(Q0), math.log(s20), _logit(pp), _logit(pm),
                              math.log(0.6), math.log(0.6)])
                val = ll(v)
                if val > best_v:
                    best_ph, best_v = (_logit(pp), _logit(pm)), val

        # stage 1 -- full ML
        n = max(x.size, 1)
        best_vec, best_f = None, np.inf
        for s0 in (0.03, 0.6):
            start = np.array([math.log(Q0), math.log(s20), best_ph[0], best_ph[1],
                              math.log(s0), math.log(s0)])
            r = minimize(lambda v: -ll(v) / n, start, method="Nelder-Mead",
                         options=dict(maxiter=int(max_iter), xatol=2e-3, fatol=1e-5))
            if r.fun < best_f:
                best_vec, best_f = r.x, r.fun

        self.params = Params._from_vec(best_vec)
        self._built = None
        self.reset()
        return self
