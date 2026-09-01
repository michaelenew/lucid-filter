"""Per-component multivariate walking filter -- noise scales tracked online.

This is the multivariate, per-component generalisation of
:class:`~statfilter.walking.WalkingFilter`.  Where ``WalkingFilter`` walks a single
scalar process log-scale, this filter walks a **vector of log-scales** -- one per
*significant process eigenmode* and one per *sensor* -- so it deduces, online and
per step, **which component's noise is elevated**: which mechanical mode is
drifting harder, which sensor is glitching.

Model (n-vector state, m-vector observation, supplied ``H`` (m x n))::

    theta_t = theta_{t-1} + w_t,   w_t ~ N(0, Q(t))
    y_t     = H theta_t   + v_t,   v_t ~ N(0, R(t))
    Q(t) = V diag(lam_k e^{xi_k(t)}) V^T,   R(t) = diag(rho_i e^{eta_i(t)})
    each log-scale is a stationary AR(1) with class pair (phi, s)

``Q0 = V diag(lam) V^T`` is the base process covariance (symmetric PD); its
**eigenvectors ``V`` are fixed** (the noise *directions* are structural; only the
*magnitudes* breathe -- see the open on learning ``V``).  ``R0 = diag(rho)`` is
diagonal: a sensor's inherent noise has no reason to correlate with another's
(the correlated-sensor case is an open).  ``H`` defaults to the identity
(``[[1]]`` at ``n = 1``).

**How it works, in one line per piece:**

  * **the grid is dense and walks.**  Each active log-scale axis carries a dense
    window of nodes at the resolution spacing ``gap = 1.5 s`` (the Sparrow-limit
    spacing, finding 11 -- *no* ``order``/``nodes`` knob; the node count is derived
    from a fixed support span).  The joint window is the tensor product over active
    axes; its centre **walks** to follow the scale, with unbounded reach.
  * **the state is a Kalman filter at every node, collapsed by GPB1** -- exactly the
    ``VectorFilter`` recursion, but over the per-component (not scalar) scale grid.
  * **each axis walks by the finding-18 loop.**  The window centre ``mu_k``
    integrates the per-axis grid-shift score with the critically-damped gain
    ``K* = (1-phi)/4`` and derived drift variance ``q_mu`` -- the ``WalkingFilter``
    loop, one copy per axis.
  * **spectral truncation, derived (no free parameter).**  An axis the data cannot
    resolve is *frozen*: an unbounded walk on a near-zero-Fisher axis integrates
    noise into a drift.  The freeze threshold is derived (0010) -- the walk
    delocalises when its steady spread ``(1-phi)/(4 I_char)`` (finding-18 Th. 2)
    exceeds its window ``(SPAN_S s)^2``, so ``freeze <=> I_char < (1-phi)/(4 (SPAN_S s)^2)``
    -- a function of the class ``(phi, s)`` and the coverage budget only.  This also
    keeps the joint grid ``nodes**(#active axes)`` small.

**Scope / limits (a testbed filter).** ``statfilter`` is a theory testbed; this
filter's value is isolating the per-component noise-deduction mechanism, not
production scale (the joint grid is exponential in the *active*-axis count -- a
handful of significant modes + sensors).  Measured against the exact grid over 6
seeds, the walk is faithful: when a strong process mode is hot the sensor reads ~0.26
and the exact grid *also* reads ~0.31 there -- with a mixing ``H`` process and
measurement noise are genuinely partly confounded, so that is the *true* posterior
coupling, not a walk defect (a clean sensor stays clean when another is hot).  The
only walk artifact is a ~0.1-nat static drift on the strong axis (bounded).  See
``research/multivariate-statfilter/SUMMARY.md``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["WalkingVectorFilter", "WalkVecStep", "WalkVecResult"]

_LOG2PI = math.log(2.0 * math.pi)
_GAP_FACTOR = 1.5           # gap = 1.5 s: resolution (Sparrow) spacing (finding 11)
_SPAN_S = 3.0               # window half-span in units of s (support budget; derives node count)
_RIDGE = 1e-4               # Fisher stabiliser before dividing


# --------------------------------------------------------------------- results
@dataclass
class WalkVecStep:
    """What the filter knows after one vector observation."""

    mean: np.ndarray            #: posterior state mean (n,)
    var: np.ndarray             #: posterior state covariance (n, n)
    innovation: np.ndarray      #: y_t - H * prior mean (m,)
    loglik: float               #: log predictive density of y_t
    process_scale: np.ndarray   #: tracked process-eigenmode log-scales (n,)
    measurement_scale: np.ndarray  #: tracked per-sensor log-scales (m,)

    @property
    def scale(self) -> np.ndarray:
        """The full log-scale vector (process eigenmodes ++ sensors), length n+m."""
        return np.concatenate([self.process_scale, self.measurement_scale])


@dataclass
class WalkVecResult:
    """Batch output.  ``mean`` is (T, n); ``var`` is (T, n, n)."""

    mean: np.ndarray
    var: np.ndarray
    innovation: np.ndarray
    process_scale: np.ndarray       #: (T, n)
    measurement_scale: np.ndarray   #: (T, m)
    loglik: float = 0.0

    def __len__(self) -> int:
        return len(self.mean)


# ---------------------------------------------------------------- the filter
class WalkingVectorFilter:
    """Track per-component noise scales online for a multivariate local-level model.

    Parameters
    ----------
    Q0 : array (n, n)
        Base process covariance (symmetric positive-definite).  Its eigenvectors are
        the fixed noise directions; its eigenvalues the base magnitudes that breathe.
    R0 : array (m,) or (m, m), optional
        Base per-sensor variances (diagonal).  A vector, or a diagonal matrix.
        Defaults to ones of length ``m``.
    H : array (m, n), optional
        Measurement matrix.  Defaults to the identity (``[[1]]`` at ``n = 1``).
    phi, s : float
        The AR(1) class pair of every log-scale -- persistence and swing.  ``s`` sets
        the grid spacing (``gap = 1.5 s``).  Both shared across axes (the class
        commitment); per-axis pairs are an easy extension.
    """

    def __init__(self, Q0, R0=None, H=None, phi: float = 0.9, s: float = 0.3):
        Q0 = np.atleast_2d(np.asarray(Q0, dtype=float))
        n = Q0.shape[0]
        if Q0.shape != (n, n) or not np.allclose(Q0, Q0.T, atol=1e-10):
            raise ValueError("Q0 must be a square symmetric matrix")
        lam, V = np.linalg.eigh(Q0)
        if lam[0] <= 0:
            raise ValueError("Q0 must be positive definite")
        if R0 is None:
            R0 = np.ones(1 if H is None else np.atleast_2d(H).shape[0])
        R0 = np.asarray(R0, dtype=float)
        rho = np.diag(R0) if R0.ndim == 2 else R0
        m = rho.size
        if np.any(rho <= 0):
            raise ValueError("R0 must have positive diagonal")
        H = np.eye(n) if H is None else np.atleast_2d(np.asarray(H, dtype=float))
        if H.shape != (m, n):
            raise ValueError(f"H must have shape ({m}, {n}), got {H.shape}")
        if not 0.0 <= phi < 1.0:
            raise ValueError("phi must lie in [0, 1)")
        if not s > 0.0:
            raise ValueError("s must be positive (it sets the grid spacing)")

        self.n, self.m, self.D = n, m, n + m
        self.V, self.lam, self.rho, self.H = V, lam, rho, H
        self.HV = H @ V                                  # (m, n): H applied to each eigenvector
        self.phi, self.s = float(phi), float(s)
        self.gap = _GAP_FACTOR * self.s

        # finding-18 walk loop, one copy per axis (parameter-free)
        self._Kstar = (1.0 - self.phi) / 4.0
        self._dSs = self._dS_shapes()                   # (D, m, m), constant
        self._Ichar = self._steady_fisher()             # (D,) per-axis steady Fisher
        # SPECTRAL TRUNCATION -- derived, not tuned.  An axis is FROZEN when its walk
        # cannot stay localised in its own window: an unbounded walk on a near-zero-
        # Fisher axis delocalises and drifts (research 0006/0009).  Finding 18's
        # Theorem 2 gives the walk's steady posterior variance exactly as
        # (1-phi)/(4 I_char); the window half-span is _SPAN_S*s.  The walk drifts
        # precisely when the former exceeds the latter squared, so
        #     freeze  <=>  I_char < (1 - phi) / (4 (_SPAN_S s)^2).
        # A pure function of the class (phi, s) and the coverage budget _SPAN_S -- no
        # free parameter.  Derivation + delocalisation onset: research 0010.
        self._Ifloor = (1.0 - self.phi) / (4.0 * (_SPAN_S * self.s) ** 2)
        self.active = self._Ichar >= self._Ifloor
        with np.errstate(divide="ignore"):
            self._qmu = self._Kstar ** 2 / (self._Ichar * (1.0 - self._Kstar))
        self._build_window()
        self.reset()

    def __repr__(self) -> str:
        return (f"WalkingVectorFilter(n={self.n}, m={self.m}, "
                f"active={int(self.active.sum())}/{self.D}, "
                f"phi={self.phi:.3f}, s={self.s:.3f})")

    # ------------------------------------------------------------- the window
    def _build_window(self):
        """The dense joint window over ACTIVE axes: offsets, stationary weights, T.

        Node count per axis is derived from the support span (+/- _SPAN_S * s) at the
        resolution spacing (1.5 s) -- no ``order``/``nodes`` parameter.  Frozen axes
        contribute a single node at offset 0.
        """
        K = int(math.ceil(_SPAN_S / _GAP_FACTOR))       # nodes each side to cover the span
        off1 = self.gap * np.arange(-K, K + 1)          # 1-D window offsets (same for every axis)
        w1 = np.exp(-0.5 * (off1 / self.s) ** 2); w1 /= w1.sum()
        nu = max(self.s * self.s * (1.0 - self.phi ** 2), 1e-12)
        T1 = np.exp(np.clip(-0.5 * (off1[None, :] - self.phi * off1[:, None]) ** 2 / nu,
                            -700.0, 700.0))
        T1 /= T1.sum(1, keepdims=True)

        offsets = [off1 if a else np.array([0.0]) for a in self.active]
        weights = [w1 if a else np.array([1.0]) for a in self.active]
        trans = [T1 if a else np.array([[1.0]]) for a in self.active]
        # joint tensor grid over all D axes (frozen axes are singletons)
        self._mesh = np.array(np.meshgrid(*offsets, indexing="ij")).reshape(self.D, -1).T
        pi0 = weights[0]; Tj = trans[0]
        for k in range(1, self.D):
            pi0 = np.kron(pi0, weights[k]); Tj = np.kron(Tj, trans[k])
        self._pi0, self._T = pi0, Tj
        self._G = pi0.size

    def _Q_of(self, xi):
        return self.V @ np.diag(self.lam * np.exp(np.clip(xi, -60, 60))) @ self.V.T

    def _R_of(self, eta):
        return np.diag(self.rho * np.exp(np.clip(eta, -60, 60)))

    def _dS_shapes(self):
        """The constant matrix each axis's ``dS/dpsi_k`` is a positive multiple of.

        Only the multiple moves with the scale: a process mode contributes
        ``outer(HV[:, k], HV[:, k])`` and a sensor contributes its own unit diagonal,
        neither of which depends on where the walk is.  Built once.
        """
        out = [np.outer(self.HV[:, k], self.HV[:, k]) for k in range(self.n)]
        for i in range(self.m):
            E = np.zeros((self.m, self.m))
            E[i, i] = 1.0
            out.append(E)
        return np.stack(out)                            # (D, m, m)

    def _dS_coef(self, scale, k):
        """The multiple in front of axis ``k``'s shape, at one scale vector."""
        return (self.lam[k] if k < self.n else self.rho[k - self.n]) * math.exp(
            min(scale[k], 60))

    def _dS_list(self, scale, Ppred):
        """dS/dpsi_k at a scale vector, given the current predictive S depends on it."""
        return [self._dS_coef(scale, k) * self._dSs[k] for k in range(self.D)]

    def _dS_axis(self, k, scales):
        """``dS/dpsi_k`` at every node of the window, stacked (G, m, m).

        The looped form built all D matrices for each node and kept one, which is D
        times the work at every axis and D**2 times it per step.
        """
        c = np.array([self._dS_coef(sc, k) for sc in scales])
        return c[:, None, None] * self._dSs[k]

    def _steady_fisher(self) -> np.ndarray:
        """Per-axis steady expected Fisher at the base regime (DARE fixed point)."""
        H, n, m = self.H, self.n, self.m
        P = np.eye(n) * (self.lam.max() + self.rho.max())
        Q0, R0 = self._Q_of(np.zeros(n)), self._R_of(np.zeros(m))
        for _ in range(400):
            Ppred = P + Q0
            S = H @ Ppred @ H.T + R0
            K = Ppred @ H.T @ np.linalg.inv(S)
            P = Ppred - K @ H @ Ppred
        Ppred = P + Q0
        S = H @ Ppred @ H.T + R0
        Si = np.linalg.inv(S)
        dS = self._dS_list(np.zeros(self.D), Ppred)
        info = np.array([0.5 * np.trace(Si @ d @ Si @ d) for d in dS]) + _RIDGE
        return info

    # ------------------------------------------------------------- streaming
    def reset(self, mean=None, scale=None) -> "WalkingVectorFilter":
        """Clear streaming state.  ``scale`` seeds the walk centres (D,).  Chains."""
        self._pi = None
        self._m = None if mean is None else np.asarray(mean, float)
        self._P = None
        self.mu = np.zeros(self.D) if scale is None else np.asarray(scale, float).copy()
        self._Pmu = np.full(self.D, self.s * self.s)
        self.loglik = 0.0
        return self

    def update(self, y) -> WalkVecStep:
        """Absorb one observation, collapse the state, walk each active scale."""
        n, m, H = self.n, self.m, self.H
        y = np.atleast_1d(np.asarray(y, dtype=float))
        if y.shape != (m,):
            raise ValueError(f"observation must have shape ({m},), got {y.shape}")
        scales = self.mu[None, :] + self._mesh                 # (G, D) absolute log-scales
        Qg = np.stack([self._Q_of(sc[:n]) for sc in scales])   # (G, n, n)
        Rg = np.stack([self._R_of(sc[n:]) for sc in scales])   # (G, m, m)
        if self._pi is None:
            self._pi = self._pi0.copy()
            if self._m is None:
                self._m = (np.linalg.lstsq(H, y, rcond=None)[0]
                           if np.all(np.isfinite(y)) else np.zeros(n))
            if self._P is None:
                self._P = np.eye(n) * float(Rg.reshape(self._G, -1).max()
                                            + Qg.reshape(self._G, -1).max()) * n

        pi = self._pi @ self._T
        if not np.all(np.isfinite(y)):                          # missing: propagate only
            self._pi = pi
            self._P = self._P + np.einsum("g,gij->ij", pi, Qg)
            ps = self.mu[:n] + (pi @ self._mesh)[:n]
            ms = self.mu[n:] + (pi @ self._mesh)[n:]
            return WalkVecStep(self._m.copy(), self._P.copy(), np.full(m, np.nan),
                               0.0, ps, ms)

        P = self._P
        Ppred = P[None] + Qg
        e = y - H @ self._m
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

        K = np.einsum("gik,gkl->gil", PHt, Si)
        Kbar = np.einsum("g,gil->il", pi, K)
        m_new = self._m + Kbar @ e
        mpost = self._m[None] + np.einsum("gil,l->gi", K, e)
        dm = mpost - m_new
        KH = np.einsum("gil,lj->gij", K, H)
        Ppost = Ppred - np.einsum("gij,gjk->gik", KH, Ppred)
        P_new = (np.einsum("g,gij->ij", pi, Ppost)
                 + np.einsum("g,gi,gj->ij", pi, dm, dm))
        P_new = 0.5 * (P_new + P_new.T)

        # walk each active axis by the finding-18 loop (analytic residual score/Fisher)
        Sie = np.einsum("gij,j->gi", Si, e)
        for k in range(self.D):
            if not self.active[k]:
                continue
            dpk = self._dS_axis(k, scales)                              # (G, m, m)
            score_g = 0.5 * (np.einsum("gi,gij,gj->g", Sie, dpk, Sie)
                             - np.einsum("gij,gji->g", Si, dpk))
            SidS = np.einsum("gij,gjk->gik", Si, dpk)
            info_g = 0.5 * np.einsum("gij,gji->g", SidS, SidS)
            grad = float(pi @ score_g)
            info = float(pi @ info_g) + _RIDGE
            R_mu = 1.0 / info
            K_mu = self._Pmu[k] / (self._Pmu[k] + R_mu)
            self.mu[k] += float(np.clip(K_mu * (grad / info), -self.gap, self.gap))
            self._Pmu[k] = (1.0 - K_mu) * self._Pmu[k] + self._qmu[k]

        self._pi, self._m, self._P = pi, m_new, P_new
        self.loglik += ll
        wmean = pi @ self._mesh
        ps = self.mu[:n] + wmean[:n]
        ms = self.mu[n:] + wmean[n:]
        return WalkVecStep(m_new.copy(), P_new.copy(), e.copy(), ll, ps, ms)

    # ----------------------------------------------------------------- batch
    def loglik_of(self, Y) -> float:
        return self._run(np.asarray(Y, dtype=float), want=False)

    def filter(self, Y) -> WalkVecResult:
        return self._run(np.asarray(Y, dtype=float), want=True)

    def _run(self, Y, want):
        Y = np.atleast_2d(Y)
        if Y.ndim != 2 or Y.shape[0] == 0 or Y.shape[1] != self.m:
            raise ValueError(f"Y must be (T, {self.m})")
        saved = (self._pi, self._m, self._P, self.mu.copy(), self._Pmu.copy(), self.loglik)
        try:
            self.reset()
            if not want:
                total = 0.0
                for row in Y:
                    total += self.update(row).loglik
                return total
            T, n, m = Y.shape[0], self.n, self.m
            mean = np.empty((T, n)); var = np.empty((T, n, n)); inn = np.empty((T, m))
            ps = np.empty((T, n)); ms = np.empty((T, m))
            total = 0.0
            for i, row in enumerate(Y):
                st = self.update(row)
                mean[i] = st.mean; var[i] = st.var; inn[i] = st.innovation
                ps[i] = st.process_scale; ms[i] = st.measurement_scale
                total += st.loglik
            return WalkVecResult(mean=mean, var=var, innovation=inn,
                                 process_scale=ps, measurement_scale=ms, loglik=total)
        finally:
            (self._pi, self._m, self._P, self.mu, self._Pmu, self.loglik) = saved
