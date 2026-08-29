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

Everything is vector; a scalar problem is length 1.  `dynamics=None` (learn the dynamics) is the one
open cell -- it belongs to the ODE-learning filter and raises `NotImplementedError` for now.

This is a benchmark toy: the RMSE for a given amount of supplied knowledge is the bound a real
implementation can aim at.  The mechanism (per-component walk, GPB1 grid, derived spectral truncation,
finding-18 loop) is the parameter-free `WalkingVectorFilter` (now moved to research/ as a specimen),
lifted with a supplied `F` and wrapped in the `(phi, s)` bank of `WalkingBank`.
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

    def __len__(self) -> int:
        return len(self.mean)


def _logsumexp(a: np.ndarray) -> float:
    m = float(np.max(a))
    return m + math.log(float(np.sum(np.exp(a - m))))


# ------------------------------------------------------- the per-(phi,s) engine
class _WalkEngine:
    """One bank member: the per-component walking grid filter WITH supplied dynamics F (+ forcing B u).

    Identical to the research `WalkingVectorFilter` except the state prediction is the supplied linear
    dynamics ``mpred = F m + B u``, ``Ppred = F P F^T + Q`` (F defaults to the identity -> random walk).
    Parameter-free: gain ``K* = (1-phi)/4``, drift ``q_mu`` and the spectral-freeze floor are all derived
    from the class ``(phi, s)``; no EMA, no tuned constant.
    """

    def __init__(self, Q0, R0, H, F, B, phi, s):
        n = Q0.shape[0]
        lam, V = np.linalg.eigh(Q0)
        self.n, self.m = n, R0.size
        self.D = n + self.m
        self.V, self.lam, self.rho, self.H = V, lam, R0, H
        self.F = F
        self.B = B
        self.p = 0 if B is None else B.shape[1]
        self.HV = H @ V
        # Each Fisher direction is a scalar times a channel-fixed matrix:
        #   dS_k(scale) = _dsbase[k] * exp(min(scale[k], 60)) * _dsM[k]
        # (process k: lam[k] * outer(HV[:,k], HV[:,k]); sensor i: rho[i] * e_i e_i^T)
        # so the whole grid of dS matrices is one broadcast, not a per-node loop.
        self._dsM = np.zeros((self.D, self.m, self.m))
        self._dsbase = np.empty(self.D)
        for k in range(n):
            hv = self.HV[:, k]
            self._dsM[k] = np.outer(hv, hv)
            self._dsbase[k] = self.lam[k]
        for i in range(self.m):
            self._dsM[n + i, i, i] = 1.0
            self._dsbase[n + i] = self.rho[i]
        self.phi, self.s = float(phi), float(s)
        self.gap = _GAP_FACTOR * self.s
        self._Kstar = (1.0 - self.phi) / 4.0
        self._Ichar = self._steady_fisher()
        self._Ifloor = (1.0 - self.phi) / (4.0 * (_SPAN_S * self.s) ** 2)
        self.active = self._Ichar >= self._Ifloor
        with np.errstate(divide="ignore"):
            self._qmu = self._Kstar ** 2 / (self._Ichar * (1.0 - self._Kstar))
        self._build_window()
        self.reset()

    # -- grid window: an axial stencil, not the tensor product --
    def _build_window(self):
        """Lay one 1-D grid along each identifiable channel's axis through ``mu``.

        The tensor product over the D channels needs ``(2K+1)**A`` nodes for A
        active channels -- 5**25 for a 5-DOF arm in one block, which cannot be
        formed.  The scale posterior is carried instead as a product of
        per-channel marginals, and every expectation the step needs is taken on
        the axial stencil: one 1-D grid per active channel, ``(2K+1) * A`` nodes,
        linear in the number of channels.  With one active channel the stencil
        *is* the tensor grid, so that case is unchanged.
        """
        K = int(math.ceil(_SPAN_S / _GAP_FACTOR))
        off1 = self.gap * np.arange(-K, K + 1)
        w1 = np.exp(-0.5 * (off1 / self.s) ** 2); w1 /= w1.sum()
        nu = max(self.s * self.s * (1.0 - self.phi ** 2), 1e-12)
        T1 = np.exp(np.clip(-0.5 * (off1[None, :] - self.phi * off1[:, None]) ** 2 / nu, -700.0, 700.0))
        T1 /= T1.sum(1, keepdims=True)
        self._off1, self._T1 = off1, T1
        self._axes = np.flatnonzero(self.active)
        A, nk = self._axes.size, off1.size
        if A == 0:                          # nothing identifiable -- a single node at mu
            self._mesh = np.zeros((1, self.D))
            self._pim0, self._G = np.ones((0, nk)), 1
            return
        mesh = np.zeros((A, nk, self.D))
        for a, k in enumerate(self._axes):
            mesh[a, :, k] = off1
        self._mesh = mesh.reshape(A * nk, self.D)
        self._pim0, self._G = np.tile(w1, (A, 1)), A * nk

    def _scale_mean(self, pim):
        """Posterior mean offset per channel, from the marginals."""
        wmean = np.zeros(self.D)
        for a, k in enumerate(self._axes):
            wmean[k] = float(pim[a] @ self._off1)
        return wmean

    def _Q_of(self, xi):
        return self.V @ np.diag(self.lam * np.exp(np.clip(xi, -60, 60))) @ self.V.T

    def _R_of(self, eta):
        return np.diag(self.rho * np.exp(np.clip(eta, -60, 60)))

    def _Q_grid(self, scales):
        """(G, n, n) stack of ``V diag(lam e^xi) V^T`` -- the per-node loop, vectorised."""
        d = self.lam[None, :] * np.exp(np.clip(scales[:, :self.n], -60, 60))
        return np.matmul(self.V * d[:, None, :], self.V.T)

    def _R_grid(self, scales):
        """(G, m, m) stack of ``diag(rho e^eta)``."""
        Rg = np.zeros((scales.shape[0], self.m, self.m))
        i = np.arange(self.m)
        Rg[:, i, i] = self.rho[None, :] * np.exp(np.clip(scales[:, self.n:], -60, 60))
        return Rg

    def _dS_grid(self, scales, k):
        """(G, m, m) stack of the channel-``k`` Fisher direction over the grid."""
        c = self._dsbase[k] * np.exp(np.minimum(scales[:, k], 60.0))
        return c[:, None, None] * self._dsM[k]

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
            Ppred = F @ P @ F.T + Q0
            S = H @ Ppred @ H.T + R0
            K = Ppred @ H.T @ np.linalg.inv(S)
            P = Ppred - K @ H @ Ppred
        Ppred = F @ P @ F.T + Q0
        Si = np.linalg.inv(H @ Ppred @ H.T + R0)
        dS = self._dS_list(np.zeros(self.D))
        return np.array([0.5 * np.trace(Si @ d @ Si @ d) for d in dS]) + _RIDGE

    def reset(self, mean=None, scale=None):
        self._pim = None
        self._m = None if mean is None else np.asarray(mean, float)
        self._P = None
        self.mu = np.zeros(self.D) if scale is None else np.asarray(scale, float).copy()
        self._Pmu = np.full(self.D, self.s * self.s)
        self.loglik = 0.0
        return self

    def update(self, y, u=None):
        n, m, H, F = self.n, self.m, self.H, self.F
        bu = (self.B @ u) if self.B is not None else 0.0
        y = np.atleast_1d(np.asarray(y, dtype=float))
        A, nk = self._pim0.shape
        scales = self.mu[None, :] + self._mesh
        Qg = self._Q_grid(scales)
        Rg = self._R_grid(scales)
        if self._pim is None:
            self._pim = self._pim0.copy()
            if self._m is None:
                self._m = (np.linalg.lstsq(H, y, rcond=None)[0]
                           if np.all(np.isfinite(y)) else np.zeros(n))
            if self._P is None:
                self._P = np.eye(n) * float(Rg.reshape(self._G, -1).max()
                                            + Qg.reshape(self._G, -1).max()) * n
        # walk each channel's marginal, then weight the axial nodes by it.  The
        # node weights are the marginal mixture pim/A: a proper distribution over
        # the stencil, and exactly pim when there is a single axis.
        pim = self._pim @ self._T1 if A else self._pim
        pi = (pim / A).ravel() if A else np.ones(1)
        mpred = F @ self._m + bu
        FPFt = F @ self._P @ F.T
        if not np.all(np.isfinite(y)):
            self._pim = pim
            self._P = FPFt + np.einsum("g,gij->ij", pi, Qg)
            self._m = mpred
            wmean = self._scale_mean(pim)
            return LucidStep(self._m.copy(), self._P.copy(), np.full(m, np.nan), 0.0,
                             self.mu[:n] + wmean[:n], self.mu[n:] + wmean[n:])
        Ppred = FPFt[None] + Qg
        e = y - H @ mpred
        PHt = np.einsum("gij,kj->gik", Ppred, H)
        S = np.einsum("ij,gjk->gik", H, PHt) + Rg
        Si = np.linalg.inv(S)
        sgn, logdet = np.linalg.slogdet(S)
        maha = np.einsum("i,gij,j->g", e, Si, e)
        lg = -0.5 * (m * _LOG2PI + logdet + maha)
        mx = float(lg.max())
        w = pi * np.exp(lg - mx); Z = float(w.sum())
        ll = math.log(Z) + mx
        pi = w / Z
        if A:                     # mean-field coordinate update of each marginal
            lgm = lg.reshape(A, nk)
            wm = pim * np.exp(lgm - lgm.max(1, keepdims=True))
            pim = wm / wm.sum(1, keepdims=True)
        K = np.einsum("gik,gkl->gil", PHt, Si)
        Kbar = np.einsum("g,gil->il", pi, K)
        m_new = mpred + Kbar @ e
        mpost = mpred[None] + np.einsum("gil,l->gi", K, e)
        dm = mpost - m_new
        KH = np.einsum("gil,lj->gij", K, H)
        Ppost = Ppred - np.einsum("gij,gjk->gik", KH, Ppred)
        P_new = np.einsum("g,gij->ij", pi, Ppost) + np.einsum("g,gi,gj->ij", pi, dm, dm)
        P_new = 0.5 * (P_new + P_new.T)
        Sie = np.einsum("gij,j->gi", Si, e)
        # each channel's score and information are taken on its own 1-D grid,
        # against its own marginal -- the mean-field expectation for that channel
        for a, k in enumerate(self._axes):
            sl = slice(a * nk, (a + 1) * nk)
            Si_k, Sie_k, pk = Si[sl], Sie[sl], pim[a]
            dpk = self._dS_grid(scales[sl], k)
            score_g = 0.5 * (np.einsum("gi,gij,gj->g", Sie_k, dpk, Sie_k)
                             - np.einsum("gij,gji->g", Si_k, dpk))
            SidS = np.einsum("gij,gjk->gik", Si_k, dpk)
            info_g = 0.5 * np.einsum("gij,gji->g", SidS, SidS)
            info = float(pk @ info_g) + _RIDGE
            grad = float(pk @ score_g)
            K_mu = self._Pmu[k] / (self._Pmu[k] + 1.0 / info)
            self.mu[k] += float(np.clip(K_mu * (grad / info), -self.gap, self.gap))
            self._Pmu[k] = (1.0 - K_mu) * self._Pmu[k] + self._qmu[k]
        self._pim, self._m, self._P = pim, m_new, P_new
        self.loglik += ll
        wmean = self._scale_mean(pim)
        return LucidStep(m_new.copy(), P_new.copy(), e.copy(), ll,
                         self.mu[:n] + wmean[:n], self.mu[n:] + wmean[n:])


# ------------------------------------------------------------- the public filter
class LucidFilter:
    """The single public filter: supply dynamics + observation, it infers all the noise online.

    Parameters (each: explicit value, or a null/default for "none", or the give-what-you-know rule)
    ----------
    dynamics : (n, n) array, or 0
        The linear state dynamics ``F``.  ``0`` (default) means no dynamics -> a random-walk level.
        ``None`` (learn the dynamics) is not implemented -- that is the ODE-learning filter's cell.
    control : (n, p) array, optional
        Forcing/bias map ``B`` for a known input ``u`` (the ODE bias).  If given, ``update``/``filter``
        require ``u``/``U``.
    H : (m, n) array, optional
        Measurement matrix.  Defaults to the identity.
    process, measurement : arrays, optional
        Base noise magnitudes ``Q0`` (n, n, PD) and ``R0`` (m, diagonal).  Default to identity/unit --
        the walk breathes around them with unbounded reach.  The base is not merely cosmetic,
        though: the per-channel identifiability gate (``active``) is evaluated once at the base,
        and a base wrong by ~10x in the Q/R *ratio* can freeze exactly the channels that would
        walk back to the truth.  Give the right order of magnitude.
    n : int, optional
        State dimension, when it cannot be inferred from ``dynamics``/``process``/``H`` (default 1).
    phis, ss : sequences, optional
        The ``(phi, s)`` box the bank averages over.  Defaults to a broad dead-zone-free range;
        not a fitted value.  ``forget`` (default 0.999) is the weight memory -- the one residual.
    """

    def __init__(self, dynamics=0, control=None, H=None, process=None, measurement=None,
                 n=None, phis=_PHIS, ss=_SS, forget=0.999):
        if dynamics is None:
            raise NotImplementedError(
                "dynamics=None (learn the dynamics) is the open ODE-learning cell; supply F or 0")
        H = None if H is None else np.atleast_2d(np.asarray(H, float))
        B = None if control is None else np.atleast_2d(np.asarray(control, float))
        Fm = None if np.ndim(dynamics) == 0 else np.atleast_2d(np.asarray(dynamics, float))
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
        F = np.eye(n) if Fm is None else Fm
        if F.shape != (n, n):
            raise ValueError(f"dynamics must be ({n}, {n})")
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

        self.n, self.m, self.D = n, m, n + m
        self.p = 0 if B is None else B.shape[1]
        self.B = B
        self.forget = float(forget)
        self.phi_arr = np.array([p for p in phis for _ in ss], float)
        self.s_arr = np.array([sv for _ in phis for sv in ss], float)
        self._members = [_WalkEngine(Q0, R0, Hm, F, B, p, sv) for p in phis for sv in ss]
        self.reset()

    def reset(self):
        for f in self._members:
            f.reset()
        self._logw = np.zeros(len(self._members))
        self.loglik = 0.0
        return self

    def update(self, y, u=None) -> LucidStep:
        if self.B is not None and u is None:
            raise ValueError(f"this filter has a control input; pass u (length {self.p})")
        if self.B is None and u is not None:
            raise ValueError("filter has no control map; do not pass u")
        M = len(self._members)
        prior = self._logw - _logsumexp(self._logw)
        steps = [f.update(y, u=u) for f in self._members]
        ll = np.array([st.loglik for st in steps])
        yv = np.atleast_1d(np.asarray(y, float))
        if np.all(np.isfinite(yv)):
            bank_ll = _logsumexp(prior + ll)
            self._logw = self.forget * prior + ll
        else:
            bank_ll = 0.0
            self._logw = prior
        post = np.exp(self._logw - _logsumexp(self._logw))
        mean = sum(post[i] * steps[i].mean for i in range(M))
        var = sum(post[i] * (steps[i].var + np.outer(steps[i].mean - mean, steps[i].mean - mean))
                  for i in range(M))
        ps = sum(post[i] * steps[i].process_scale for i in range(M))
        ms = sum(post[i] * steps[i].measurement_scale for i in range(M))
        innov = sum(post[i] * steps[i].innovation for i in range(M))
        self.loglik += bank_ll
        return LucidStep(mean, var, innov, bank_ll, ps, ms)

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
        total = 0.0
        for i, row in enumerate(Y):
            st = self.update(row, None if U is None else U[i])
            mean[i] = st.mean; var[i] = st.var; inn[i] = st.innovation
            ps[i] = st.process_scale; ms[i] = st.measurement_scale; total += st.loglik
        return LucidResult(mean=mean, var=var, innovation=inn,
                           process_scale=ps, measurement_scale=ms, loglik=total)

    def loglik_of(self, Y, U=None) -> float:
        return self.filter(Y, U).loglik
