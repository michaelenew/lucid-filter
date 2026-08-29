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
        self.phi, self.s = float(phi), float(s)
        self.gap = _GAP_FACTOR * self.s
        self._Kstar = (1.0 - self.phi) / 4.0
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
            w[k] = float(pi_ax[i] @ self._off1)
        return w

    def _dS_axis(self, k):
        """dS/dxi_k at each of axis k's window nodes (dS_k depends only on the k-coordinate)."""
        a = np.exp(np.minimum(self.mu[k] + self._off1, 60.0))
        if k < self.n:
            hv = self.HV[:, k]
            return (self.lam[k] * a)[:, None, None] * np.outer(hv, hv)[None]
        out = np.zeros((self._nn, self.m, self.m))
        i = k - self.n
        out[:, i, i] = self.rho[i] * a
        return out

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
            Ppred = F @ P @ F.T + Q0
            S = H @ Ppred @ H.T + R0
            K = Ppred @ H.T @ np.linalg.inv(S)
            P = Ppred - K @ H @ Ppred
        Ppred = F @ P @ F.T + Q0
        Si = np.linalg.inv(H @ Ppred @ H.T + R0)
        dS = self._dS_list(np.zeros(self.D))
        return np.array([0.5 * np.trace(Si @ d @ Si @ d) for d in dS]) + _RIDGE

    def reset(self, mean=None, scale=None):
        self._pi_ax = None
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
        r = len(self._act)
        Qg, Rg = self._star_QR()
        if self._pi_ax is None:
            self._pi_ax = np.tile(self._w1, (r, 1))
            if self._m is None:
                self._m = (np.linalg.lstsq(H, y, rcond=None)[0]
                           if np.all(np.isfinite(y)) else np.zeros(n))
            if self._P is None:
                self._P = np.eye(n) * float(Rg.reshape(self._G, -1).max()
                                            + Qg.reshape(self._G, -1).max()) * n
        pi_ax = self._pi_ax @ self._T1
        mpred = F @ self._m + bu
        FPFt = F @ self._P @ F.T
        if not np.all(np.isfinite(y)):
            self._pi_ax = pi_ax
            w = self._star_weights(pi_ax)
            self._P = FPFt + np.einsum("g,gij->ij", w, Qg)
            self._m = mpred
            wmean = self._wmean(pi_ax)
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
        KH = np.einsum("gil,lj->gij", K, H)
        Ppost = Ppred - np.einsum("gij,gjk->gik", KH, Ppred)
        P_new = np.einsum("g,gij->ij", pi, Ppost) + np.einsum("g,gi,gj->ij", pi, dm, dm)
        P_new = 0.5 * (P_new + P_new.T)
        # finding-18 walk per active axis, score/Fisher averaged over that axis's window
        # posterior only (the caltrop: dS_k depends only on the k-coordinate, so the axial
        # profile carries the axis's evidence at linear cost).
        for i, k in enumerate(self._act):
            idx = self._axwin[k]
            dpk = self._dS_axis(k)
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
            self._Pmu[k] = min((1.0 - K_mu) * self._Pmu[k] + self._qmu[k], self._Pmu_cap)
        self._pi_ax, self._m, self._P = pi_ax, m_new, P_new
        self.loglik += ll
        wmean = self._wmean(pi_ax)
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
        the walk breathes around them with unbounded reach, so a rough base is fine.
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
