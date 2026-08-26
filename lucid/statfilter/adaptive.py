"""Adaptive Kalman filter: supplied dynamics + measurement, noise learned online.

This is the **production** member of ``statfilter`` -- the one built for robotics, not the
theory testbed.  You supply the (linearised) dynamics ``F`` and the measurement matrix ``H``;
the filter learns everything else -- the per-component process and sensor noise magnitudes --
online, per step, at **polynomial** cost in the number of active noise axes (no exponential
grid).

Model (n-vector state, m-vector observation)::

    theta_t = F theta_{t-1} + w_t,   w_t ~ N(0, Q(t))
    y_t     = H theta_t     + v_t,   v_t ~ N(0, R(t))
    Q(t) = V diag(lam_k e^{xi_k(t)}) V^T      (V = eigvecs(Q0), FIXED; magnitudes breathe)
    R(t) = diag(rho_i e^{eta_i(t)})           (per-sensor variances breathe)
    each log-scale xi_k, eta_i is a stationary AR(1) with class pair (phi, s).

Two things distinguish it from :class:`WalkingVectorFilter` (the exponential testbed):

* **General dynamics ``F`` + the derivative mode.**  A robotic arm has momentum -- position,
  velocity and acceleration are *coupled* by the integrator (``x' = v``), not free axes.  With
  the local-level default ``F = I`` there is nothing to *coast on* when a sensor-noise burst
  hits; a kinematic ``F`` rides through it on velocity, and 10-40x better on position RMSE than
  the random walk in the crusher regime (research 0024).  :meth:`kinematic` builds the
  position/velocity(/acceleration) model; it fuses whatever you measure -- encoder (position),
  gyro/tacho (velocity), accelerometer/IMU (acceleration), or several at once -- into all the
  derivatives, and :meth:`derivatives` reads them back per DOF.

* **Known forcing / control input ``B u``.**  Without it the estimate *lags* while the state is
  being driven (a commanded arm trajectory): a constant-velocity model can't anticipate the
  commanded acceleration.  Supply ``B`` and pass ``u`` each step -- the prediction becomes
  ``F theta + B u`` -- and the lag collapses (velocity RMSE ~3x+ smaller; the state is tracked
  to the sensor floor even mid-swing).  ``kinematic(..., control=True)`` wires ``u`` to the
  per-DOF commanded top derivative.

* **Whiteness-gated noise adaptation (breaks the Q-vs-R confound).**  A single innovation is
  explained equally by more process noise *or* more sensor noise, so a per-step likelihood walk
  cannot separate them -- it picks the stiffer axis, down-weights a good sensor to zero and
  diverges (research 0024).  The innovation *sequence* separates them (Mehra 1970): process
  noise makes the filter LAG -> positive lag-1 innovation autocorrelation; sensor noise inflates
  the innovation variance but stays WHITE.  So each active scale walks by the finding-18 loop on
  its analytic residual score, **gated**: a process axis may only *raise* when the innovations
  are lag-correlated, a sensor axis only when they are white (either may always *lower*).  The
  gate turns on above ``2 sqrt(beta)`` -- the 2-sigma significance of the innovation-correlation
  EMA -- so it is tied to the adaptation timescale, not a separate knob.  Cost per step is
  ``O(r m^2)`` -- polynomial, no grid.

**No theoretically relevant free parameters.**  The spectral floor is derived
(``(1-phi)/(4 (SPAN_S s)^2)``); the walk gain ``K* = (1-phi)/4`` and drift ``q_mu`` are the
finding-18 loop; the grid spacing is the Sparrow limit ``1.5 s``; the gate threshold is the
2-sigma EMA significance.  ``SPAN_S`` and ``_BETA`` (the adaptation timescale) are *labeled
budgets* -- they trade responsiveness for smoothness, they do not move the fixed point.

Open items / known warts (see ``research/multivariate-statfilter/SUMMARY.md``):

* **Simultaneous process+sensor burst.**  When process AND sensor noise spike together (the
  crusher "both" regime) the innovations are only partly correlated, so the gate splits the
  excess imperfectly; state RMSE is ~parity with (a few % worse than) a well-tuned fixed filter,
  where the oracle is ~10% better.  The pure process-hot and sensor-hot bursts are handled well
  (research 0025).  A full multivariate Mehra solve (joint ``Q,R`` from the autocorrelation
  sequence) is the planned upgrade.
* **Per-component diagnostic de-mix.**  The Fisher-eigenbasis walk (research 0018-0023,
  ``WalkingVectorFilter`` lineage) gives a faithful *which-sensor / which-mode* attribution
  under a mixing ``H`` at a dimension-stable false-alarm floor; it is not yet unified into this
  production state filter (which prioritises robust state estimation).  Unifying them -- the
  eigenbasis de-mix *within* each whiteness-gated block -- is an open.
* **Correlated sensors / learned ``V``.**  ``R0`` is diagonal and ``V`` is fixed; a genuinely
  correlated sensor bank, or a drifting noise *direction*, is not yet learned.
* **Adaptation lag.**  A burst is caught over ``~1/beta`` steps (the EMA memory); very brief
  bursts are under-corrected.  ``beta`` is the labeled responsiveness/smoothness budget.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["AdaptiveKalmanFilter", "AdaptiveStep", "AdaptiveResult"]

_LOG2PI = math.log(2.0 * math.pi)
_GAP_FACTOR = 1.5           # gap = 1.5 s: resolution (Sparrow) spacing (finding 11)
_SPAN_S = 3.0               # spectral-truncation coverage budget (half-span in units of s)
_BETA = 0.02               # innovation-statistics EMA rate (labeled adaptation-timescale budget)
_Q_DRIVE = 0.2             # process-scale whitening rate (labeled adaptation-rate budget)
_RIDGE = 1e-9


@dataclass
class AdaptiveStep:
    """What the filter knows after one vector observation."""

    mean: np.ndarray               #: posterior state mean (n,)
    var: np.ndarray                #: posterior state covariance (n, n)
    innovation: np.ndarray         #: y_t - H F * (prev mean)  (m,)
    loglik: float                  #: log predictive density of y_t
    process_scale: np.ndarray      #: tracked process-eigenmode log-scales (n,)
    measurement_scale: np.ndarray  #: tracked per-sensor log-scales (m,)

    @property
    def scale(self) -> np.ndarray:
        """Full log-scale vector (process eigenmodes ++ sensors), length n+m."""
        return np.concatenate([self.process_scale, self.measurement_scale])


@dataclass
class AdaptiveResult:
    """Batch output.  ``mean`` is (T, n); ``var`` is (T, n, n)."""

    mean: np.ndarray
    var: np.ndarray
    innovation: np.ndarray
    process_scale: np.ndarray       #: (T, n)
    measurement_scale: np.ndarray   #: (T, m)
    loglik: float = 0.0

    def __len__(self) -> int:
        return len(self.mean)


class AdaptiveKalmanFilter:
    """Supplied dynamics ``F`` and measurement ``H``; per-component noise learned online.

    Parameters
    ----------
    Q0 : array (n, n)
        Base process covariance (symmetric positive-definite).  Its eigenvectors are the fixed
        noise directions; its eigenvalues the base magnitudes that breathe.
    R0 : array (m,) or (m, m), optional
        Base per-sensor variances (diagonal).  Defaults to ones of length ``m``.
    H : array (m, n), optional
        Measurement matrix.  Defaults to the identity.
    F : array (n, n), optional
        State-transition matrix (the supplied, possibly linearised, dynamics).  Defaults to the
        identity (local-level random walk).  Use :meth:`kinematic` for a position/velocity model.
    phi, s : float
        AR(1) class pair of every log-scale -- persistence and swing.  ``s`` sets the grid
        spacing (``gap = 1.5 s``).
    """

    def __init__(self, Q0, R0=None, H=None, F=None, B=None, phi: float = 0.9, s: float = 0.3):
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
        F = np.eye(n) if F is None else np.atleast_2d(np.asarray(F, dtype=float))
        if F.shape != (n, n):
            raise ValueError(f"F must have shape ({n}, {n}), got {F.shape}")
        if B is not None:
            B = np.atleast_2d(np.asarray(B, dtype=float))
            if B.shape[0] != n:
                raise ValueError(f"B must have {n} rows (state dim), got {B.shape}")
        if not 0.0 <= phi < 1.0:
            raise ValueError("phi must lie in [0, 1)")
        if not s > 0.0:
            raise ValueError("s must be positive (it sets the grid spacing)")

        self.n, self.m, self.D = n, m, n + m
        self.V, self.lam, self.rho, self.H, self.F, self.B = V, lam, rho, H, F, B
        self.p = 0 if B is None else B.shape[1]           # control-input dimension
        self.n_dof = self.order = None                    # set by kinematic()
        self.HV = H @ V                                   # (m, n)
        self.phi, self.s = float(phi), float(s)
        self.gap = _GAP_FACTOR * self.s
        self._Kstar = (1.0 - self.phi) / 4.0
        self._floor = (1.0 - self.phi) / (4.0 * (_SPAN_S * self.s) ** 2)
        self._Ichar = self._steady_fisher()
        # ACTIVATE structurally-observable axes -- NOT by the delocalisation floor, and NOT by a
        # threshold relative to the loudest axis (a quiet process mode next to a loud sensor would
        # be frozen and then coast rigidly on a wrong velocity when the sensor is down-weighted --
        # research 0024).  A process eigenmode is live iff it carries base variance AND is seen by
        # H (lam_k > 0 and H v_k != 0); a sensor is always live.  The whiteness gate + bounded
        # drift keep a genuinely quiet axis from wandering.
        obs = np.zeros(self.D, dtype=bool)
        hv_norm = np.linalg.norm(self.HV, axis=0)                # ||H v_k|| per process mode
        for k in range(self.n):
            obs[k] = self.lam[k] > 1e-12 * self.lam.max() and hv_norm[k] > 1e-8
        obs[self.n:] = True                                     # sensors always observable
        self.active = np.where(obs)[0]
        self.r = self.active.size
        self._qmu = np.array([self._Kstar ** 2
                              / (max(self._Ichar[int(k)], self._floor) * (1.0 - self._Kstar))
                              for k in self.active])
        self._Pmu_cap = (_SPAN_S * self.s) ** 2      # walk covariance ceiling (bounds wander)
        self.reset()

    # -------------------------------------------------------------- constructors
    @classmethod
    def kinematic(cls, n_dof: int, order: int = 2, dt: float = 1.0,
                  process_var=1e-3, meas_var=1.0, phi: float = 0.9, s: float = 0.3,
                  measured=("pos",), control: bool = False):
        """Build a per-DOF kinematic filter (the derivative mode).

        ``order`` = number of derivatives tracked per DOF: 1 = position (random walk),
        2 = (position, velocity), 3 = (position, velocity, acceleration).  ``F`` is the
        constant-``order`` motion model (a Taylor integrator, ``dt`` step); process noise enters
        through the top derivative and propagates down (the standard continuous-white-noise
        model).  Because position, velocity and acceleration are *coupled* by the integrator
        (``x' = v``, not free axes), the filter fuses whatever you measure into all of them --
        so an encoder pins position *and* sharpens the velocity/acceleration estimates.

        ``measured`` names which derivative each sensor reads, per DOF: ``("pos",)`` for encoders,
        ``("acc",)`` for accelerometers/IMUs, ``("pos", "acc")`` to fuse both, ``("vel",)`` for a
        tachometer/gyro.  Read the estimated derivatives back with :meth:`derivatives`.

        ``control=True`` adds a known-forcing input ``B`` (see :meth:`update`): the per-DOF
        commanded *top derivative* (acceleration for ``order=2``, jerk for ``order=3``), which
        removes the lag while the arm is driven along a commanded trajectory.

        State layout is DOF-major: ``[x0, x0', x0'', ..., x1, x1', ...]``.
        """
        if order < 1:
            raise ValueError("order must be >= 1")
        Fb = np.eye(order)
        for i in range(order):
            for j in range(i + 1, order):
                Fb[i, j] = dt ** (j - i) / math.factorial(j - i)
        g = np.array([dt ** (order - i) / math.factorial(order - i) for i in range(order)])  # integrator gain
        Qb = process_var * np.outer(g, g) + 1e-12 * np.eye(order)
        F = np.kron(np.eye(n_dof), Fb)
        Q0 = np.kron(np.eye(n_dof), Qb)
        B = np.kron(np.eye(n_dof), g[:, None]) if control else None   # commanded top-derivative
        idx = {"pos": 0, "vel": 1, "acc": 2}
        rows = []
        for d in range(n_dof):
            for name in measured:
                if idx[name] >= order:
                    raise ValueError(f"measured '{name}' needs order > {idx[name]} (order={order})")
                e = np.zeros(order * n_dof); e[d * order + idx[name]] = 1.0; rows.append(e)
        H = np.array(rows)
        R0 = np.full(len(rows), float(meas_var))
        f = cls(Q0, R0=R0, H=H, F=F, B=B, phi=phi, s=s)
        f.n_dof, f.order = n_dof, order
        return f

    def __repr__(self) -> str:
        return (f"AdaptiveKalmanFilter(n={self.n}, m={self.m}, active={self.r}/{self.D}, "
                f"phi={self.phi:.3f}, s={self.s:.3f})")

    # ------------------------------------------------------------------ noise maps
    def _Q_of(self, xi):
        return self.V @ np.diag(self.lam * np.exp(np.clip(xi, -60, 60))) @ self.V.T

    def _R_of(self, eta):
        return np.diag(self.rho * np.exp(np.clip(eta, -60, 60)))

    def _dS(self, scale, k):
        """dS/dpsi_k (M x M): how the innovation covariance moves with log-scale k."""
        if k < self.n:
            hv = self.HV[:, k]
            return self.lam[k] * math.exp(min(scale[k], 60)) * np.outer(hv, hv)
        E = np.zeros((self.m, self.m)); i = k - self.n
        E[i, i] = self.rho[i] * math.exp(min(scale[k], 60)); return E

    def _steady_fisher(self) -> np.ndarray:
        H, F, n, m = self.H, self.F, self.n, self.m
        P = np.eye(n) * (self.lam.max() + self.rho.max())
        Q0, R0 = self._Q_of(np.zeros(n)), self._R_of(np.zeros(m))
        for _ in range(500):
            Pp = F @ P @ F.T + Q0
            S = H @ Pp @ H.T + R0
            K = Pp @ H.T @ np.linalg.inv(S)
            P = Pp - K @ H @ Pp
        Pp = F @ P @ F.T + Q0
        Si = np.linalg.inv(H @ Pp @ H.T + R0)
        z = np.zeros(self.D)
        return np.array([0.5 * np.trace(Si @ self._dS(z, k) @ Si @ self._dS(z, k))
                         for k in range(self.D)]) + 1e-12

    # ------------------------------------------------------------------- streaming
    def reset(self, mean=None, scale=None) -> "AdaptiveKalmanFilter":
        self._m = None if mean is None else np.asarray(mean, float)
        self._P = None
        self.mu = np.zeros(self.D) if scale is None else np.asarray(scale, float).copy()
        self._Pmu = np.full(self.r, self.s * self.s)   # per-axis walk variance
        self._C0 = np.diag(self.rho).astype(float)     # EMA innovation covariance
        self._C1 = np.zeros((self.m, self.m))          # EMA lag-1 innovation covariance
        self._e_prev = None
        self.loglik = 0.0
        return self

    def update(self, y, u=None) -> AdaptiveStep:
        """Absorb one observation: predict with F (+ known forcing B u), walk the noise scales,
        correct the state.  ``u`` is the control / known-forcing input this step (length ``p``);
        supplying it removes the lag while the state is being driven (e.g. a commanded arm
        trajectory).  Required iff the filter was built with ``B``."""
        n, m, H, F = self.n, self.m, self.H, self.F
        y = np.atleast_1d(np.asarray(y, dtype=float))
        if y.shape != (m,):
            raise ValueError(f"observation must have shape ({m},), got {y.shape}")
        if self.B is not None:
            if u is None:
                raise ValueError(f"this filter has control input; pass u (length {self.p})")
            u = np.atleast_1d(np.asarray(u, dtype=float))
            if u.shape != (self.p,):
                raise ValueError(f"u must have shape ({self.p},), got {u.shape}")
        elif u is not None:
            raise ValueError("filter has no B; do not pass u")
        bu = (self.B @ u) if self.B is not None else 0.0
        if self._m is None:
            self._m = (np.linalg.lstsq(H, y, rcond=None)[0]
                       if np.all(np.isfinite(y)) else np.zeros(n))
        if self._P is None:
            self._P = np.eye(n) * (self.lam.max() + self.rho.max()) * n

        Ppost = self._P                                    # posterior covariance from last step
        mpred = F @ self._m + bu                           # known forcing enters the prediction
        Pp = F @ Ppost @ F.T + self._Q_of(self.mu[:n])
        e = y - H @ mpred

        if not np.all(np.isfinite(y)):                     # missing: propagate only
            self._m, self._P = mpred, Pp
            return AdaptiveStep(mpred.copy(), Pp.copy(), np.full(m, np.nan), 0.0,
                                self.mu[:n].copy(), self.mu[n:].copy())

        # ---- innovation statistics: separate Q from R by the time-correlation (Mehra; 0025) ----
        # A single innovation is explained equally by more process OR more sensor noise; only the
        # innovation SEQUENCE separates them.  Process noise makes the filter LAG -> the innovation
        # picks up lag-1 autocorrelation, aligned with the process mode's direction (H v_k).  Sensor
        # noise inflates the innovation variance but stays WHITE.  So drive each PROCESS scale to
        # whiten its own lag-1 correlation, and each SENSOR scale to match the WHITE residual
        # variance -- gated so a sensor absorbs only the uncorrelated part.  This is what stops
        # process noise masquerading as sensor noise (the runaway that diverges; 0024).
        self._C0 = (1 - _BETA) * self._C0 + _BETA * np.outer(e, e)
        if self._e_prev is not None:
            self._C1 = (1 - _BETA) * self._C1 + _BETA * np.outer(e, self._e_prev)
        self._e_prev = e.copy()
        HPHt = H @ Pp @ H.T
        C1s = 0.5 * (self._C1 + self._C1.T)
        thr = 2.0 * math.sqrt(_BETA)                                    # 2-sigma significance of the EMA
        rho1 = float(np.trace(C1s)) / (float(np.trace(self._C0)) + 1e-12)
        white_gate = float(np.clip(1.0 - (rho1 - thr) / thr, 0.0, 1.0))  # 1 white, 0 correlated

        for a, k in enumerate(self.active):
            if k < n:                                                  # process eigenmode: whiten its lag-1 corr
                hv = self.HV[:, k]
                c0k = float(hv @ self._C0 @ hv) + 1e-12
                sig = float(hv @ C1s @ hv) / c0k                       # lag-1 autocorr in mode k's direction
                drive = np.sign(sig) * max(abs(sig) - thr, 0.0)        # act only when significant
                self.mu[k] += float(np.clip(_Q_DRIVE * drive, -self.gap, self.gap))
            else:                                                      # sensor: match the WHITE residual variance
                i = k - n
                resid = float(self._C0[i, i] - HPHt[i, i])
                target = math.log(max(resid, 1e-8) / self.rho[i])
                self.mu[k] += float(np.clip(self._Kstar * white_gate * (target - self.mu[k]),
                                            -self.gap, self.gap))
        self.mu[:n] = np.clip(self.mu[:n], -8.0, 20.0)
        self.mu[n:] = np.clip(self.mu[n:], -8.0, 20.0)
        # rebuild the state prior at the walked scale
        Pp = F @ Ppost @ F.T + self._Q_of(self.mu[:n])
        S = H @ Pp @ H.T + self._R_of(self.mu[n:]) + _RIDGE * np.eye(m)
        Si = np.linalg.inv(S)

        # ---- state correction (general-F Kalman) ----
        sgn, logdet = np.linalg.slogdet(S)
        ll = -0.5 * (m * _LOG2PI + logdet + float(e @ Si @ e))
        K = Pp @ H.T @ Si
        m_new = mpred + K @ e
        P_new = Pp - K @ H @ Pp; P_new = 0.5 * (P_new + P_new.T)
        self._m, self._P = m_new, P_new
        self.loglik += ll
        return AdaptiveStep(m_new.copy(), P_new.copy(), e.copy(), ll,
                            self.mu[:n].copy(), self.mu[n:].copy())

    def derivatives(self, mean):
        """Reshape a kinematic-model state (or state trajectory) into per-DOF derivatives.

        Returns an array whose last two axes are ``(n_dof, order)`` -- e.g. ``[..., d, 0]`` is
        position of DOF ``d``, ``[..., d, 1]`` its velocity, ``[..., d, 2]`` its acceleration.
        Only defined for filters built with :meth:`kinematic`.
        """
        if self.n_dof is None:
            raise ValueError("derivatives() is only defined for a kinematic() filter")
        mean = np.asarray(mean, dtype=float)
        return mean.reshape(mean.shape[:-1] + (self.n_dof, self.order))

    # ---------------------------------------------------------------------- batch
    def loglik_of(self, Y, U=None) -> float:
        return self._run(np.asarray(Y, dtype=float), U, want=False)

    def filter(self, Y, U=None) -> AdaptiveResult:
        """Filter a batch.  ``U`` (T, p) is the control / known-forcing sequence, required iff
        the filter was built with ``B``."""
        return self._run(np.asarray(Y, dtype=float), U, want=True)

    def _run(self, Y, U, want):
        Y = np.atleast_2d(Y)
        if Y.ndim != 2 or Y.shape[0] == 0 or Y.shape[1] != self.m:
            raise ValueError(f"Y must be (T, {self.m})")
        if self.B is not None:
            if U is None:
                raise ValueError(f"this filter has control input; pass U of shape ({Y.shape[0]}, {self.p})")
            U = np.atleast_2d(np.asarray(U, dtype=float))
            if U.shape != (Y.shape[0], self.p):
                raise ValueError(f"U must be ({Y.shape[0]}, {self.p}), got {U.shape}")
        saved = (self._m, self._P, self.mu.copy(), self._Pmu.copy(), self.loglik)
        try:
            self.reset()
            us = (None for _ in range(Y.shape[0])) if U is None else iter(U)
            if not want:
                return sum(self.update(row, next(us)).loglik for row in Y)
            T, n, m = Y.shape[0], self.n, self.m
            mean = np.empty((T, n)); var = np.empty((T, n, n)); inn = np.empty((T, m))
            ps = np.empty((T, n)); ms = np.empty((T, m)); total = 0.0
            for i, row in enumerate(Y):
                st = self.update(row, next(us))
                mean[i] = st.mean; var[i] = st.var; inn[i] = st.innovation
                ps[i] = st.process_scale; ms[i] = st.measurement_scale; total += st.loglik
            return AdaptiveResult(mean=mean, var=var, innovation=inn,
                                  process_scale=ps, measurement_scale=ms, loglik=total)
        finally:
            (self._m, self._P, self.mu, self._Pmu, self.loglik) = saved
