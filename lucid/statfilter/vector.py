"""Multivariate adaptive local-level filter with a supplied measurement matrix.

This is the vector-valued generalisation of :class:`statfilter.AdaptiveFilter`.
The model is

    theta_t = theta_{t-1} + w_t,   w_t ~ N(0, Q0 * exp(lamP_t))     Q0  n x n
    y_t     = H theta_t   + v_t,   v_t ~ N(0, R0 * exp(lamM_t))     R0  m x m
    lam^c_t = phi_c * lam^c_{t-1} + sqrt(nu_c) z_t,   c in {P, M}

with an n-vector state, an m-vector observation, and a **supplied** measurement
matrix ``H`` (m x n).  ``H`` is structural -- the observation model the caller
built, exactly like ``OdeFilter``'s ``linearized_dynamics`` -- so ``fit`` learns
only the noise: the full-symmetric base covariances ``Q0, R0`` and the four
scale-channel numbers ``phi_P, phi_M, s_P, s_M``.  Give the filter what you know
(how the sensors read the state), it infers what you don't (the live noise).

At ``n = m = 1`` and ``H = [[1]]`` every formula collapses to the scalar
``AdaptiveFilter``; the test suite pins the agreement to 1e-8.

Two things generalise from the scalar core; nothing else does:

  * **The Kalman node** becomes the standard matrix update, with the mixture over
    grid nodes collapsed to one Gaussian per step (multivariate GPB1).
  * **The amplitude conservation law** becomes a trace decomposition of the
    predictive covariance: with ``S = H P H^T + H Qg H^T + Rg`` (three pieces that
    sum to ``S``), ``share_* = tr(S^-1 * piece) / m``, which sums to 1 and reduces
    to the scalar ``P/S, Qg/S, Rg/S`` at ``m = 1``.  (An innovation-weighted
    Mahalanobis form also reduces, but is 0/0 at ``e = 0`` -- the first step, and
    any exact hit -- so the trace form is the faithful one.)

The **scale channels stay scalar** -- one per matrix, ``Q0 exp(lamP)`` and
``R0 exp(lamM)`` -- because that is what keeps the quadrature grid at ``order**2``
states.  Per-component scale deduction ("which sensor is hot right now") is
genuinely richer but breaks the tensor-product grid; it is a recorded open, not a
parameter here.

The base covariances ``Q0, R0`` are MEDIAN (geometric-mean) covariances, the
matrix analogue of the scalar core's median-variance convention: ``exp(lam)`` has
median 1, so the overall magnitude lives in ``Q0`` and the breathing in the scale
channel, and the two do not confound beyond the scalar ridge the core already
documents.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .core import _chain, _LOG2PI

__all__ = ["VectorFilter", "VecParams", "VecStep", "VecFilterResult"]


# --------------------------------------------------------------------- params
@dataclass
class VecParams:
    """The learned noise numbers of a :class:`VectorFilter`.

    Q0, R0     median (geometric-mean) base covariances of the process and
               measurement noise -- full symmetric positive-definite, n x n and
               m x m.  With s_P = s_M = 0 the model is the ordinary multivariate
               local-level model and these are its Q and R exactly.
    phi_P/M    persistence of each channel's scalar log-scale, in [0, 1).
    s_P/M      log-SD of each channel's scalar scale; 0 means homoscedastic.

    ``H`` is not here -- it is supplied to the filter, not learned.
    """

    Q0: np.ndarray
    R0: np.ndarray
    phi_P: float = 0.0
    phi_M: float = 0.0
    s_P: float = 0.0
    s_M: float = 0.0

    def __post_init__(self):
        self.Q0 = np.atleast_2d(np.asarray(self.Q0, dtype=float))
        self.R0 = np.atleast_2d(np.asarray(self.R0, dtype=float))
        for name, M in (("Q0", self.Q0), ("R0", self.R0)):
            if M.ndim != 2 or M.shape[0] != M.shape[1]:
                raise ValueError(f"{name} must be square")
            if not np.allclose(M, M.T, atol=1e-10):
                raise ValueError(f"{name} must be symmetric")
            w = np.linalg.eigvalsh(M)
            if w[0] <= 0:
                raise ValueError(f"{name} must be positive definite")
        for name in ("phi_P", "phi_M"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must lie in [0, 1)")
        for name in ("s_P", "s_M"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def n(self) -> int:
        return self.Q0.shape[0]

    @property
    def m(self) -> int:
        return self.R0.shape[0]

    def to_dict(self) -> dict:
        return {"Q0": self.Q0.tolist(), "R0": self.R0.tolist(),
                "phi_P": self.phi_P, "phi_M": self.phi_M,
                "s_P": self.s_P, "s_M": self.s_M}

    @classmethod
    def from_dict(cls, d: dict) -> "VecParams":
        return cls(Q0=np.asarray(d["Q0"], float), R0=np.asarray(d["R0"], float),
                   phi_P=d.get("phi_P", 0.0), phi_M=d.get("phi_M", 0.0),
                   s_P=d.get("s_P", 0.0), s_M=d.get("s_M", 0.0))


# --------------------------------------------------------------------- results
@dataclass
class VecStep:
    """Everything the filter knows after one vector observation."""

    mean: np.ndarray            #: posterior mean of the level (n,)
    var: np.ndarray             #: posterior covariance of the level (n, n)
    innovation: np.ndarray      #: y_t - H * prior mean (m,)
    loglik: float               #: log predictive density of y_t

    # amplitude conservation -- these three sum to 1
    share_prior: float          #: attributed to already being wrong about theta
    share_process: float        #: attributed to the level genuinely moving
    share_measurement: float    #: attributed to measurement noise

    # the four signed mode coordinates, in log-scale nats (scalar channels)
    process_anomaly: float
    process_regime: float
    measurement_anomaly: float
    measurement_regime: float

    @property
    def process_scale(self) -> float:
        return self.process_anomaly + self.process_regime

    @property
    def measurement_scale(self) -> float:
        return self.measurement_anomaly + self.measurement_regime


@dataclass
class VecFilterResult:
    """Batch output.  ``mean`` is (n_steps, n); ``var`` is (n_steps, n, n)."""

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

    def __len__(self) -> int:
        return len(self.mean)


# -------------------------------------------------- log-Cholesky (fit helpers)
# A symmetric PD matrix is fitted through its lower Cholesky factor L (M = L L^T)
# with the diagonal logged, so the coordinates are unconstrained and the map is a
# bijection onto the PD cone.  This is the matrix analogue of fitting log(sigma^2).
def _n_chol(n: int) -> int:
    return n * (n + 1) // 2


def _chol_to_vec(L: np.ndarray) -> np.ndarray:
    n = L.shape[0]
    out = np.empty(_n_chol(n))
    k = 0
    for i in range(n):
        for j in range(i + 1):
            out[k] = math.log(L[i, i]) if i == j else L[i, j]
            k += 1
    return out


def _vec_to_chol(v: np.ndarray, n: int) -> np.ndarray:
    L = np.zeros((n, n))
    k = 0
    for i in range(n):
        for j in range(i + 1):
            L[i, j] = math.exp(v[k]) if i == j else v[k]
            k += 1
    return L


def _cov_from_vec(v: np.ndarray, n: int) -> np.ndarray:
    L = _vec_to_chol(v, n)
    return L @ L.T


# ------------------------------------------------------------------ the filter
class VectorFilter:
    """Multivariate adaptive local-level filter with a supplied ``H``.

    Parameters
    ----------
    params : VecParams, optional
        The learned noise numbers.  Omit and call :meth:`fit` to learn them.
    H : array (m, n)
        The measurement matrix -- supplied, not learned.  Required (either here
        or, for :meth:`fit`, as the ``H`` argument).  Defaults to the identity
        when ``params`` is given and ``H`` is omitted, i.e. the state is observed
        directly.
    order : int
        Quadrature nodes per scale channel; the joint grid has ``order**2``
        states.  A numerical resolution, not a model choice.
    """

    def __init__(self, params: VecParams | None = None, H=None, order: int = 5):
        if order < 3:
            raise ValueError("order must be at least 3")
        self.params = params
        self.order = int(order)
        if H is not None:
            H = np.atleast_2d(np.asarray(H, dtype=float))
        elif params is not None:
            H = np.eye(params.n)
        self.H = H
        if params is not None and H is not None:
            if H.shape != (params.m, params.n):
                raise ValueError(f"H must have shape ({params.m}, {params.n}), "
                                 f"got {H.shape}")
        self._built = None
        self.reset()

    # ---------------------------------------------------------------- plumbing
    def __repr__(self) -> str:
        if self.params is None:
            return f"VectorFilter(unfitted, order={self.order})"
        p = self.params
        return (f"VectorFilter(n={p.n}, m={p.m}, "
                f"phi_P={p.phi_P:.3f}, s_P={p.s_P:.3f}, "
                f"phi_M={p.phi_M:.3f}, s_M={p.s_M:.3f}, order={self.order})")

    def to_dict(self) -> dict:
        if self.params is None:
            raise ValueError("filter is not fitted")
        return {"params": self.params.to_dict(), "H": self.H.tolist(),
                "order": self.order}

    @classmethod
    def from_dict(cls, d: dict) -> "VectorFilter":
        return cls(VecParams.from_dict(d["params"]),
                   H=np.asarray(d["H"], float), order=int(d.get("order", 5)))

    def _build(self):
        """Precompute the grid and the per-node covariance matrices."""
        p, n = self.params, self.order
        key = (id(p), p.phi_P, p.phi_M, p.s_P, p.s_M,
               p.Q0.tobytes(), p.R0.tobytes(), n)
        if self._built is not None and self._built[0] == key:
            return self._built[1]
        if not (p.s_P > 0.0 or p.s_M > 0.0):
            # THE s = 0 FACE.  Neither log-scale walks, so `_chain` puts every one of its
            # nodes at lam = 0 and the order x order grid is one state repeated -- an
            # ordinary multivariate local-level model paying order**2 times over for it.
            # This is the whole of a `dynamics=False` fit and the first stage of any other,
            # so it is where that factor was being spent.  The collapsed grid IS the model:
            # one node, weight one, at the base covariances.
            n = 1
            lamP = lamM = np.zeros(1)
            wP = wM = np.ones(1)
            TP = TM = np.ones((1, 1))
        else:
            lamP, wP, TP = _chain(p.phi_P, p.s_P, n)
            lamM, wM, TM = _chain(p.phi_M, p.s_M, n)
        LP = np.repeat(lamP, n)
        LM = np.tile(lamM, n)
        g = {
            "LP": LP, "LM": LM,
            "T": np.kron(TP, TM),
            "pi0": np.kron(wP, wM),
            "Qg": p.Q0[None] * np.exp(np.clip(LP, -60.0, 60.0))[:, None, None],
            "Rg": p.R0[None] * np.exp(np.clip(LM, -60.0, 60.0))[:, None, None],
        }
        # the process share's numerator: fixed by H and the node covariances, so it belongs
        # to the grid rather than to the step that used to rebuild it every time
        g["HQHt"] = np.einsum("ij,gjk,lk->gil", self.H, g["Qg"], self.H)
        self._built = (key, g)
        return g

    # ------------------------------------------------------------- streaming
    def reset(self, mean=None, var=None) -> "VectorFilter":
        """Clear the streaming state.  Returns self, so it chains."""
        self._pi = None
        self._m = None if mean is None else np.asarray(mean, float)
        self._P = None if var is None else np.atleast_2d(np.asarray(var, float))
        self._prev_lamP = 0.0
        self._prev_lamM = 0.0
        self._loglik = 0.0
        return self

    def update(self, y) -> VecStep:
        """Absorb one vector observation and return everything known after it.

        A row that is all-NaN is treated as missing: the state is propagated but
        not corrected.  Partially missing observations are not supported yet.
        """
        return self._advance(y, True)

    def _advance(self, y, want):
        """One step.  ``want`` asks for the full `VecStep`; without it only the
        log-likelihood is returned and the read-outs are not formed at all.

        A fit is a few hundred likelihood evaluations over the whole series and looks at
        nothing else, so the shares, the regime split and the state copies were most of
        what the search spent its time on.  The recursion itself is one code path either
        way -- everything below the branch is the arithmetic the state depends on.
        """
        if self.params is None:
            raise ValueError("filter is not fitted; call fit() or pass params")
        p = self.params
        n, m = p.n, p.m
        H = self.H
        y = np.atleast_1d(np.asarray(y, dtype=float))
        if y.shape != (m,):
            raise ValueError(f"observation must have shape ({m},), got {y.shape}")
        g = self._build()
        Qg, Rg = g["Qg"], g["Rg"]
        if self._pi is None:
            self._pi = g["pi0"].copy()
            if self._m is None:
                self._m = (np.linalg.lstsq(H, y, rcond=None)[0]
                           if np.all(np.isfinite(y)) else np.zeros(n))
            if self._P is None:
                p0 = float(Rg.reshape(Rg.shape[0], -1).max()
                           + Qg.reshape(Qg.shape[0], -1).max())
                self._P = np.eye(n) * p0 * n

        pi = self._pi @ g["T"]
        missing = not np.all(np.isfinite(y))

        if missing:                                  # propagate, do not correct
            self._pi = pi
            EQ = np.einsum("g,gij->ij", pi, Qg)
            self._P = self._P + EQ
            lamP = float(pi @ g["LP"]); lamM = float(pi @ g["LM"])
            step = VecStep(self._m.copy(), self._P.copy(),
                           np.full(m, np.nan), 0.0, 1.0, 0.0, 0.0,
                           lamP - p.phi_P * self._prev_lamP, p.phi_P * self._prev_lamP,
                           lamM - p.phi_M * self._prev_lamM, p.phi_M * self._prev_lamM) \
                if want else 0.0
            self._prev_lamP, self._prev_lamM = lamP, lamM
            return step

        P = self._P
        Ppred = P[None] + Qg                          # (G, n, n)
        e = y - H @ self._m                           # (m,) shared across nodes
        PHt = np.einsum("gij,kj->gik", Ppred, H)      # (G, n, m) = Ppred H^T
        S = np.einsum("ij,gjk->gik", H, PHt) + Rg     # (G, m, m)
        Sinv = np.linalg.inv(S)
        K = np.einsum("gik,gkl->gil", PHt, Sinv)      # (G, n, m)

        sign, logdet = np.linalg.slogdet(S)
        maha = np.einsum("i,gij,j->g", e, Sinv, e)
        lg = -0.5 * (m * _LOG2PI + logdet + maha)
        mx = float(lg.max())
        w = pi * np.exp(lg - mx)
        Z = float(w.sum())
        ll = math.log(Z) + mx
        pi = w / Z

        Ke = np.einsum("gil,l->gi", K, e)             # (G, n)
        Kbar = np.einsum("g,gil->il", pi, K)          # (n, m)
        m_new = self._m + Kbar @ e
        mpost = self._m[None] + Ke                    # (G, n)
        dm = mpost - m_new                            # (G, n)
        KH = np.einsum("gil,lj->gij", K, H)           # (G, n, n)
        Ppost = Ppred - np.einsum("gij,gjk->gik", KH, Ppred)
        P_new = (np.einsum("g,gij->ij", pi, Ppost)
                 + np.einsum("g,gi,gj->ij", pi, dm, dm))
        P_new = 0.5 * (P_new + P_new.T)               # symmetrise against drift

        if want:
            # trace-form shares: S = H P H^T + H Qg H^T + Rg  (pieces sum to S)
            HPHt = H @ P @ H.T                        # (m, m), same for every node
            sh_prior = float(pi @ (np.einsum("gij,ji->g", Sinv, HPHt) / m))
            sh_proc = float(pi @ (np.einsum("gij,gji->g", Sinv, g["HQHt"]) / m))
            sh_meas = float(pi @ (np.einsum("gij,gji->g", Sinv, Rg) / m))
            lamP = float(pi @ g["LP"]); lamM = float(pi @ g["LM"])
            step = VecStep(
                mean=m_new.copy(), var=P_new.copy(), innovation=e.copy(), loglik=ll,
                share_prior=sh_prior, share_process=sh_proc, share_measurement=sh_meas,
                process_anomaly=lamP - p.phi_P * self._prev_lamP,
                process_regime=p.phi_P * self._prev_lamP,
                measurement_anomaly=lamM - p.phi_M * self._prev_lamM,
                measurement_regime=p.phi_M * self._prev_lamM,
            )
        else:
            lamP = float(pi @ g["LP"]); lamM = float(pi @ g["LM"])
            step = ll
        self._pi, self._m, self._P = pi, m_new, P_new
        self._prev_lamP, self._prev_lamM = lamP, lamM
        self._loglik += ll
        return step

    def predict(self, horizon: int = 1):
        """Mean and covariance of theta_{t+h} given everything so far.

        The level is a random walk, so the forecast mean is flat and the
        covariance grows by the expected process covariance per step.
        """
        if self._pi is None:
            raise ValueError("nothing observed yet")
        g = self._build()
        pi, var = self._pi, self._P
        for _ in range(int(horizon)):
            pi = pi @ g["T"]
            var = var + np.einsum("g,gij->ij", pi, g["Qg"])
        return self._m.copy(), var

    # ----------------------------------------------------------------- batch
    def loglik(self, Y) -> float:
        """Exact marginal log-likelihood of a series.  Does not touch state."""
        return self._run(np.asarray(Y, dtype=float), want=False)

    def filter(self, Y) -> VecFilterResult:
        """Run over a whole series from a fresh state.  Does not touch state."""
        return self._run(np.asarray(Y, dtype=float), want=True)

    def _run(self, Y: np.ndarray, want: bool):
        if self.params is None:
            raise ValueError("filter is not fitted; call fit() or pass params")
        Y = np.atleast_2d(Y)
        if Y.ndim != 2 or Y.shape[0] == 0 or Y.shape[1] != self.params.m:
            raise ValueError(f"Y must be (T, {self.params.m})")
        saved = (self._pi, self._m, self._P, self._prev_lamP,
                 self._prev_lamM, self._loglik)
        try:
            self.reset()
            if not want:
                total = 0.0
                for row in Y:
                    total += self._advance(row, False)
                return total
            T, n, m = Y.shape[0], self.params.n, self.params.m
            out = {c: np.empty(T) for c in
                   ("share_prior", "share_process", "share_measurement",
                    "process_anomaly", "process_regime",
                    "measurement_anomaly", "measurement_regime")}
            mean = np.empty((T, n)); var = np.empty((T, n, n)); inn = np.empty((T, m))
            total = 0.0
            for i, row in enumerate(Y):
                st = self.update(row)
                mean[i] = st.mean; var[i] = st.var; inn[i] = st.innovation
                for c in out:
                    out[c][i] = getattr(st, c)
                total += st.loglik
            return VecFilterResult(mean=mean, var=var, innovation=inn,
                                   loglik=total, **out)
        finally:
            (self._pi, self._m, self._P, self._prev_lamP,
             self._prev_lamM, self._loglik) = saved

    # ------------------------------------------------------------------- fit
    @classmethod
    def fit(cls, Y, H, order: int = 5, max_iter: int = 400,
            dynamics: bool = True) -> "VectorFilter":
        """Learn ``Q0, R0`` and the four scale numbers from a series, ``H`` supplied.

        ``Y`` is ``(T, m)``; ``H`` is the ``(m, n)`` measurement matrix.  ``n`` is
        read from ``H``.  ``dynamics=False`` fits the homoscedastic model
        (``s_P = s_M = 0``) only -- the plain multivariate local-level model.
        """
        H = np.atleast_2d(np.asarray(H, dtype=float))
        f = cls(H=H, order=order)
        f.fit_(Y, max_iter=max_iter, dynamics=dynamics)
        return f

    def fit_(self, Y, max_iter: int = 400, dynamics: bool = True) -> "VectorFilter":
        """Fit in place.  Returns self.

        Staged: (1) the homoscedastic face -- ``Q0, R0`` by log-Cholesky L-BFGS,
        the multivariate analogue of the scalar core's ``s = 0`` face; (2) a small
        screen over the scale-channel starts with those covariances held; (3) full
        maximum likelihood over ``[chol(Q0), chol(R0), phi_P, phi_M, s_P, s_M]``.
        Gradients are central differences.  Nothing here is a tuning choice -- the
        estimate is the maximum-likelihood estimate and ``max_iter`` is a budget.
        """
        from scipy.optimize import minimize

        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        H = self.H
        if H is None:
            raise ValueError("H (the measurement matrix) must be supplied to fit")
        m, n = H.shape
        if Y.shape[1] != m:
            raise ValueError(f"Y must be (T, {m}) to match H")
        finite = np.all(np.isfinite(Y), axis=1)
        if int(finite.sum()) < max(10, 2 * (n + m)):
            raise ValueError("not enough finite observations to fit")
        T = int(finite.sum())
        nQ, nR = _n_chol(n), _n_chol(m)

        # crude scale reference: map observation first-differences up to the state
        dY = np.diff(Y[finite], axis=0)
        Hpinv = np.linalg.pinv(H)
        dstate = dY @ Hpinv.T
        cov_state = np.cov(dstate, rowvar=False) + 1e-6 * np.eye(n)
        cov_obs = np.cov(dY, rowvar=False) + 1e-6 * np.eye(m)
        # split the observed diff-covariance ~half process / half measurement
        Q_init = 0.5 * np.atleast_2d(cov_state)
        R_init = 0.5 * np.atleast_2d(cov_obs)
        vQ0 = _chol_to_vec(np.linalg.cholesky(Q_init))
        vR0 = _chol_to_vec(np.linalg.cholesky(R_init))

        def negll(cov_vec, phi_P, phi_M, s_P, s_M):
            try:
                Q0 = _cov_from_vec(cov_vec[:nQ], n)
                R0 = _cov_from_vec(cov_vec[nQ:nQ + nR], m)
                self.params = VecParams(Q0, R0, phi_P, phi_M, s_P, s_M)
                self._built = None
                return -self._run(Y, want=False) / T
            except (ValueError, np.linalg.LinAlgError, OverflowError):
                return 1e9

        # ---- stage 1: homoscedastic Q0, R0 (s = 0) by L-BFGS on log-Cholesky
        def fg_homo(cv):
            base = negll(cv, 0.0, 0.0, 0.0, 0.0)
            h = 1e-4
            grad = np.empty_like(cv)
            for k in range(cv.size):
                cv2 = cv.copy(); cv2[k] += h
                grad[k] = (negll(cv2, 0.0, 0.0, 0.0, 0.0) - base) / h
            return base, grad

        cv0 = np.concatenate([vQ0, vR0])
        r1 = minimize(fg_homo, cv0, jac=True, method="L-BFGS-B",
                      options=dict(maxiter=int(max_iter), ftol=1e-11, gtol=1e-7))
        cv_homo = r1.x
        if not dynamics:
            negll(cv_homo, 0.0, 0.0, 0.0, 0.0)     # set params
            self._built = None
            self.reset()
            return self

        # ---- stage 2: screen scale-channel starts with the covariances held
        _PHI = (0.25, 0.6, 0.9)
        _SS = ((0.0, 0.0), (0.4, 0.0), (0.0, 0.4), (0.4, 0.4))
        best_start, best_v = (0.0, 0.0, 0.0, 0.0), -1e18
        for pp in _PHI:
            for pm in _PHI:
                for sp, sm in _SS:
                    v = -negll(cv_homo, pp, pm, sp, sm)
                    if v > best_v:
                        best_v, best_start = v, (pp, pm, sp, sm)

        # ---- stage 3: full ML over covariances + scale channels
        # unconstrained scale coords: logit(phi), log(s) with a small floor
        def pack(cv, ph_P, ph_M, s_P, s_M):
            lg = lambda p: math.log(min(max(p, 1e-6), 1 - 1e-6) /
                                    (1 - min(max(p, 1e-6), 1 - 1e-6)))
            return np.concatenate([cv, [lg(ph_P), lg(ph_M),
                                        math.log(max(s_P, 1e-6)),
                                        math.log(max(s_M, 1e-6))]])

        def unpack(v):
            cv = v[:nQ + nR]
            ex = lambda z: 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))
            return (cv, ex(v[-4]), ex(v[-3]),
                    math.exp(min(v[-2], 4.0)), math.exp(min(v[-1], 4.0)))

        def fg_full(v):
            cv, pp, pm, sp, sm = unpack(v)
            base = negll(cv, pp, pm, sp, sm)
            h = 1e-4
            grad = np.empty_like(v)
            for k in range(v.size):
                v2 = v.copy(); v2[k] += h
                cv2, pp2, pm2, sp2, sm2 = unpack(v2)
                grad[k] = (negll(cv2, pp2, pm2, sp2, sm2) - base) / h
            return base, grad

        v0 = pack(cv_homo, *best_start)
        r3 = minimize(fg_full, v0, jac=True, method="L-BFGS-B",
                      options=dict(maxiter=int(max_iter), ftol=1e-11, gtol=1e-7))
        cv, pp, pm, sp, sm = unpack(r3.x)
        negll(cv, pp, pm, sp, sm)                  # set params to the optimum
        self._built = None
        self.reset()
        return self
