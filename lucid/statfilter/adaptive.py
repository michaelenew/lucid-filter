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
  its analytic residual score, with the sensor axis absorbing only its **derived share** of the
  residual.  The share is not a tuned gate: chasing the oracle across the sensor<->process
  continuum (research 0032) gives the process fraction of a channel's innovation variance as
  ``c/S + rho1`` (``c/S = (H Pp H^T)/S`` the nominal state share, ``rho1`` the lag-1 autocorr), so
  the sensor's fraction of the residual ``C0 - c`` is ``1 - rho1 (S/R)`` -- one smooth line whose
  two arms are its limits (``rho1 = R/S`` all-process, ``rho1 <= 0`` all-sensor), the transition
  WIDTH the channel's own ``R/S``, no threshold and no ramp.  The gate is **per-sensor** (research
  0027): process noise on a state a sensor reads directly -- an accelerometer on the jerk-driven
  acceleration -- shows as lag-correlation in that channel, and a per-sensor share stops it
  masquerading as that sensor's noise where a pooled gate would average it away.  ``rho1`` is a
  noisy EMA, so it is **denoised** by a non-negative garrote at its own ``2 sqrt(beta)`` (the
  2-sigma EMA noise floor, a labeled budget tied to the adaptation timescale) -- continuous and
  unbiased for a significant correlation, zero below the floor so a failing sensor still sheds
  fully.  Cost per step is ``O(r m^2)`` -- polynomial, no grid.

* **Robust measurement update, derived (the hot regimes).**  The expensive extreme is a failing
  *absolute* reference: when a bad potentiometer degrades further, position observability
  collapses (an accelerometer only integrates to a *drifting* position), and the walked scale
  trusts the failing sensor for its first steps -- a spike the drifting state then carries
  (research 0029/0030, adaptive-vs-oracle 1.98x).  The measurement noise scale is *uncertain*
  (it breathes), so marginalising the Gaussian correction over that uncertainty is a
  **heavy-tail**: the state correction MAPs each sensor's log-scale for THIS innovation (prior =
  its walked scale, spread the class swing ``s^2``), inflating ``R`` smoothly on an outlier with
  **no threshold and no branch** -- the 4-sigma cutoff of the earlier gate is now the derived
  heavy-tail (research 0031, ``_robust_eta``).  This is what reacts to a failing sensor at the first
  corrupted sample; the walked scale then follows over the adaptation window.  Result: the
  failing-absolute-sensor regimes track near the online floor (pot-hot ~**1.13x**, process+pot
  ~**1.18x** oracle, from 1.98x) with **no shed** (research 0037: the empirical fast shed
  ``_SHED``/``_WHITE_MIN`` bought only ~0.1x more and is removed).

**No tuning parameters.**  The spectral floor is derived (``(1-phi)/(4 (SPAN_S s)^2)``); the sensor
walk gain ``K* = (1-phi)/4`` and drift ``q_mu`` are the finding-18 loop; the grid spacing is the
Sparrow limit ``1.5 s``; the sensor share is the derived ``1 - rho1 (S/R)`` (research 0032) with
``rho1`` garrote-denoised at its 2-sigma EMA noise floor; the process-walk gain is the derived
Newton whitening rate ``K*/b_k`` (research 0035).  The robust measurement update is fully derived
(the heavy-tail from the scale swing ``s``, no cutoff).  The only remaining inputs are the model
**class** -- the pair ``(phi, s)``, which is the *definition of the class, not a parameter within
it* (optimality-proof Prop 1: with free scale motion, "the level jumped" and "the sensor glitched"
are identically distributed, so the class must fix how fast scales move) -- and the *labeled
budgets* ``_BETA``, ``_SPAN_S``, ``_Q_REVERT`` (adaptation timescales that trade responsiveness for
smoothness, they do not move the fixed point).

Even the ``(phi, s)`` POINT need not be committed: it lives on a sloppy identification ridge that is
flat in what matters, so it can be integrated over by a model-averaged bank (adaptive-grid findings
13-16; shipped scalar as ``WalkingBank``).  :class:`AdaptiveBank` is the multivariate analogue --
**work in progress**: it retires the ``(phi, s)`` commitment when the class varies slowly, but the
model average concentrates on the calm-optimal member and under-serves a short burst, so it does not
yet match a single member on the hot regimes (research 0037).  The right within-member cure is a
scale POSTERIOR (the windowed GPB1, 0008) whose reach is the finding-18 analogue ``q`` -- an open
derivation.

Open items / known warts (see ``research/multivariate-statfilter/SUMMARY.md``):

* **BOTH is observability-limited, not a fixable leak (research 0033).**  When a *dynamic* sensor
  is noisy AND a process disturbance is active at once, the process is *masked*: the only sensor
  that reads it directly is drowned by its own noise (its lag-1 correlation dilutes to ~0), so the
  jerk is nearly unobservable.  Freezing ``Q`` at oracle closes BOTH (~1.1x); the achievable floor
  for *inferring* the masked ``Q`` (oracle-R) is ~3x oracle, and the adaptive already sits below it
  -- the distance to the *full* oracle is the oracle discounting an unobservable jerk, so the
  full-oracle ratio overstates BOTH.  No smooth-transition arm is missing here; the binding
  constraint is observability.  The full-oracle ratio overstates BOTH by the unobservable-Q term.

* **Collinear process/sensor modes (measured, research 0027).**  When a sensor reads the very
  state the process noise enters (an accelerometer on the jerk-driven acceleration), the two
  scale axes are *collinear* in innovation space -- the exact scale-Fisher correlation is
  ``|C| = 1.0`` -- so at steady state, after the process scale adapts and whitens the channel,
  they are single-step indistinguishable and the sensor scale drifts partway up on the whitened
  residual.  The per-sensor gate removes most of the onset misattribution (leak 1.29 -> 0.76
  nats); the steady-state residual needs a joint Mehra solve (``Q,R`` from the full
  autocorrelation sequence) and is the planned upgrade.  Note the *state* cost of this
  collinearity is small (collinear modes only need their total right, which the filter gets);
  the genuinely expensive regime is **observability loss** -- process disturbance while the
  *absolute* sensor degrades -- which is a different problem, not this confound.
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
_Q_REVERT = 0.008          # process-scale reversion to baseline (an elapsed disturbance decays out)
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
        self._qgain = self._whiten_gain()                 # derived per-mode whitening gain (0035)
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
        ``meas_var`` is the base variance of each sensor -- a scalar for all, or a dict keyed by
        the ``measured`` name (e.g. ``{"pos": 0.08**2, "acc": 0.01**2}`` for a bad potentiometer
        fused with a good accelerometer).  The filter learns the live noise from here.

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
        rows = []; rvar = []
        for d in range(n_dof):
            for name in measured:
                if idx[name] >= order:
                    raise ValueError(f"measured '{name}' needs order > {idx[name]} (order={order})")
                e = np.zeros(order * n_dof); e[d * order + idx[name]] = 1.0; rows.append(e)
                rvar.append(meas_var[name] if isinstance(meas_var, dict) else meas_var)
        H = np.array(rows)
        R0 = np.array(rvar, dtype=float)
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

    @staticmethod
    def _robust_eta(ei2, c, rho, mu, sig2):
        """MAP of a sensor log-scale eta given this innovation, prior N(mu, sig2).

        The measurement variance R = rho e^eta is uncertain (its scale breathes); marginalising
        the Gaussian likelihood over that uncertainty is heavy-tailed, so a large innovation is
        partly attributed to a larger scale -- inflating R and down-weighting the sample SMOOTHLY,
        with no threshold.  With innovation variance S(eta) = c + rho e^eta (c = the state's share
        (H Pp H^T)_ii), the MAP condition L'(eta)=0 is
            eta - mu = 1/2 sig2 (1 - c/S)(ei2/S - 1),
        signed (raises on ei2>S, lowers on ei2<S -- no branch), attributing the excess by the
        sensor's own share (1 - c/S) of the innovation variance.  sig2 is the class scale-swing
        s^2 (no new knob).  Solved by a few damped Newton steps.  This REPLACES the 4-sigma robust
        gate (research 0031): the cutoff and its hinge become the derived heavy-tail.
        """
        if sig2 < 1e-12:
            return mu
        eta = mu
        for _ in range(6):
            S = c + rho * math.exp(min(max(eta, -60.0), 60.0))
            A = 1.0 - c / S; B = ei2 / S - 1.0
            Lp = 0.5 * A * B - (eta - mu) / sig2
            Lpp = 0.5 * ((S - c) / S ** 2) * (c * B - A * ei2) - 1.0 / sig2
            eta -= max(-2.0, min(2.0, Lp / Lpp))
        return eta

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

    def _gain_for(self, Q, R0):
        """Steady-state Kalman gain the filter uses when it ASSUMES process cov Q."""
        H, F, n = self.H, self.F, self.n
        P = np.eye(n) * (self.lam.max() + self.rho.max())
        for _ in range(500):
            Pp = F @ P @ F.T + Q
            K = Pp @ H.T @ np.linalg.inv(H @ Pp @ H.T + R0)
            P = Pp - K @ H @ Pp
        Pp = F @ P @ F.T + Q
        return Pp @ H.T @ np.linalg.inv(H @ Pp @ H.T + R0)

    def _whiten_gain(self) -> np.ndarray:
        """Per process eigenmode, the DERIVED whitening gain K*/b_k (research 0035) that replaces
        the empirical _Q_DRIVE.  b_k = -d sig_k/d mu_k is the steady-state sensitivity of the mode's
        lag-1 innovation autocorrelation to its OWN assumed log-scale: with the gain from the assumed
        scale but the true process cov Q0, the actual a-priori error cov M solves the closed-loop
        Lyapunov M = A M A^T + F K R K^T F^T + Q0 (A = F(I-KH)), the lag-1 innovation autocovariance
        is the Mehra C1 = H A M H^T - H F K R (zero at the optimum), and sig_k its mode-direction
        autocorrelation.  Then the process walk whitens at the SAME Newton rate K* as the sensor
        walk: mu += (K*/b_k) sign(sig)(|sig| - thr).  A finite difference in mu_k about the optimum."""
        H, F, n, m = self.H, self.F, self.n, self.m
        R0 = self._R_of(np.zeros(m)); Q0 = self._Q_of(np.zeros(n)); I = np.eye(n)
        d = 0.5
        B_MIN = 4.0 * self._Kstar          # cap the Newton gain at K*/B_MIN = 1/4: a mode that barely
        #                                    whitens (small b) is not over-relaxed past the SOR limit
        g = np.zeros(n)                     # default: structurally unobservable modes stay frozen

        def sig_k(mu_k, k):
            mu = np.zeros(n); mu[k] = mu_k
            K = self._gain_for(self._Q_of(mu), R0)
            A = F @ (I - K @ H); W = F @ K @ R0 @ K.T @ F.T + Q0
            M = I.copy()
            for _ in range(2000):
                M = A @ M @ A.T + W
            C1 = H @ A @ M @ H.T - H @ F @ K @ R0
            hv = self.HV[:, k]
            num = float(hv @ (0.5 * (C1 + C1.T)) @ hv)
            return num / (float(hv @ (H @ M @ H.T + R0) @ hv) + 1e-12)

        for k in range(n):
            if self.lam[k] <= 1e-9 * self.lam.max() or np.linalg.norm(self.HV[:, k]) < 1e-8:
                continue                                   # structurally unobservable -> frozen
            b = -(sig_k(d, k) - sig_k(-d, k)) / (2.0 * d)
            g[k] = self._Kstar / max(b, B_MIN)             # Newton whitening gain, SOR-capped
        return g

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
        dC0 = np.diag(self._C0); dC1 = np.diag(C1s)
        Sdiag = np.diag(HPHt) + self.rho * np.exp(np.clip(self.mu[n:], -60, 60))   # predicted innov var

        for a, k in enumerate(self.active):
            if k < n:                                                  # process eigenmode: whiten its lag-1 corr
                hv = self.HV[:, k]
                c0k = float(hv @ self._C0 @ hv) + 1e-12
                sig = float(hv @ C1s @ hv) / c0k                       # lag-1 autocorr in mode k's direction
                excess = max(abs(sig) - thr, 0.0)
                # DERIVED whitening gain K*/b_k (research 0035): the process mode whitens its lag-1
                # correlation at the SAME Newton rate K* as the sensor walk, scaled by the mode's
                # steady-state sensitivity b_k = -d sig/d mu.  Replaces the empirical _Q_DRIVE.
                self.mu[k] += float(np.clip(self._qgain[k] * np.sign(sig) * excess,
                                            -self.gap, self.gap))
                self.mu[k] *= (1.0 - _Q_REVERT)                        # mild reversion to baseline so an
                #                                                        elapsed disturbance decays out
            else:                                                      # sensor: match the WHITE residual variance
                i = k - n
                # PER-SENSOR whiteness gate (research 0027/0032): sensor i absorbs the residual only
                # to the extent its OWN channel is white.  DERIVED width (0032): the sensor's fraction
                # of the residual is 1 - rho1*(S/R) -- the process share of the innovation variance is
                # c/S + rho1, and the residual C0 - c is net of c, so its process fraction is
                # rho1/(R/S).  wg reaches 0 exactly when the residual is fully process-explained
                # (rho1 = R/S), a per-channel width, replacing the global 2*thr ramp.  thr = 2 sqrt
                # (beta) is the 2-sigma lag-1 EMA noise floor (a labeled budget): below it the
                # correlation is insignificant, so a failing sensor still sheds fully.
                rho1_i = dC1[i] / (dC0[i] + 1e-12)
                SRi = Sdiag[i] / (self.rho[i] * math.exp(min(max(self.mu[k], -60.0), 60.0)) + 1e-12)
                rho1_hat = (rho1_i - thr * thr / rho1_i) if rho1_i > thr else 0.0   # garrote denoise
                wg = float(np.clip(1.0 - rho1_hat * SRi, 0.0, 1.0))
                resid = float(self._C0[i, i] - HPHt[i, i])
                target = math.log(max(resid, 1e-8) / self.rho[i])
                step = target - self.mu[k]
                # Clean derived walk at the class rate K* -- NO shed.  The fast jump-reaction that a
                # single (phi, s) lacks (its K* = (1-phi)/4 is capped by the persistent prior) is
                # supplied instead by the IMPULSIVE members of a (phi, s) bank: a small-phi member
                # has a large K* and sheds fast, and a burst simply shifts model-average weight onto
                # it, by likelihood, no threshold (research 0037; adaptive-grid findings 13-16, the
                # ridge is integrated over rather than a point (phi,s) chosen).  Within a member the
                # sensor still absorbs only its derived whiteness share wg (0032).
                self.mu[k] += float(np.clip(self._Kstar * wg * step, -self.gap, self.gap))
        self.mu[:n] = np.clip(self.mu[:n], -8.0, 20.0)
        self.mu[n:] = np.clip(self.mu[n:], -8.0, 20.0)
        # rebuild the state prior at the walked scale
        Pp = F @ Ppost @ F.T + self._Q_of(self.mu[:n])
        Rw = self._R_of(self.mu[n:])
        # ---- robust measurement update (research 0031): protect the STATE from an outlier at the
        # FIRST corrupted sample, before the walked scale has ramped.  Instead of a 4-sigma cutoff,
        # MAP each sensor's scale for THIS correction given its innovation (prior = the walked scale
        # mu_i with spread s^2, tightened by the whiteness wg so a correlated process disturbance
        # can't trigger it).  A smooth, derived heavy-tail -- no threshold, no branch.
        Hpp = H @ Pp @ H.T; Hd = np.diag(Hpp)
        for a, k in enumerate(self.active):
            if k >= n:
                i = k - n
                rho1_i = dC1[i] / (dC0[i] + 1e-12)
                wgi = float(np.clip(1.0 - (rho1_i - thr) / thr, 0.0, 1.0))
                eta_r = self._robust_eta(float(e[i] ** 2), float(Hd[i]), float(self.rho[i]),
                                         float(self.mu[n + i]), self.s ** 2 * wgi)
                Rw[i, i] = self.rho[i] * math.exp(min(max(eta_r, -60.0), 60.0))
        S = Hpp + Rw + _RIDGE * np.eye(m)
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


def _logsumexp(a: np.ndarray) -> float:
    m = float(np.max(a))
    return m + math.log(float(np.sum(np.exp(a - m))))


@dataclass
class AdaptiveBankStep:
    """Model-averaged state after one observation (the bank's output)."""

    mean: np.ndarray               #: posterior state mean (n,), weight-averaged over the bank
    var: np.ndarray                #: posterior state covariance (n, n), mixture-collapsed
    innovation: np.ndarray         #: weight-averaged innovation (m,)
    loglik: float                  #: mixture predictive log-density of this observation
    process_scale: np.ndarray      #: weight-averaged process-eigenmode log-scales (n,)
    measurement_scale: np.ndarray  #: weight-averaged per-sensor log-scales (m,)
    n_eff: float                   #: effective number of live members (1 / sum w_i^2)
    phi_hat: float                 #: posterior-mean class persistence
    s_hat: float                   #: posterior-mean class swing


class AdaptiveBank:
    """A ``(phi, s)`` bank of :class:`AdaptiveKalmanFilter`s, online Bayesian model-averaged.

    A single filter needs the class pair ``(phi, s)``.  Those live on a *sloppy ridge* the data
    identifies only weakly, and tracking is nearly flat along it (adaptive-grid findings 13-16), so
    the right move is not to pick a point but to run a bank across a ``(phi, s)`` grid and combine
    by online Bayesian model averaging ``w_i propto w_i^forget * p_i(y)`` -- the data pours weight
    onto the ridge, the flat sloppy direction averages out, and the caller commits only to the model
    *class* and a broad grid range, no fitted ``(phi, s)`` number.

    The grid also spans ``phi`` down to the **impulsive** end.  A small-``phi`` member has a large
    walk gain ``K* = (1-phi)/4`` and so reacts fast to a jump; a burst simply shifts model-average
    weight onto it, by likelihood.  That is what supplies the fast reaction a single persistent
    member lacks -- and why the member filter needs **no shed** (research 0037).

    ``forget`` is the one residual, and it is a not-a-parameter: it governs the drift rate of
    ``(phi, s)``, the slowest-varying quantities and the ones on the flat ridge, so its value barely
    reaches the estimate (identical tracking across ``[0.99, 1.0]``).  ``1.0`` is clean pure-Bayes
    (weights concentrate and freeze); the default ``0.999`` keeps the bank re-selectable if the
    class drifts.
    """

    def __init__(self, members, phis, ss, forget: float = 0.999):
        if not 0.0 < forget <= 1.0:
            raise ValueError("forget must lie in (0, 1]")
        if not members:
            raise ValueError("the bank needs at least one member")
        self.filters = list(members)
        self.phi_arr = np.asarray(phis, dtype=float)
        self.s_arr = np.asarray(ss, dtype=float)
        self.forget = float(forget)
        self.n = self.filters[0].n
        self.m = self.filters[0].m
        self.p = self.filters[0].p
        self.B = self.filters[0].B
        self.reset()

    @classmethod
    def kinematic(cls, n_dof, order=2, dt=1.0, process_var=1e-3, meas_var=1.0, measured=("pos",),
                  control=False, phis=(0.3, 0.6, 0.85, 0.95), ss=(0.3, 0.5, 0.8), forget=0.999):
        """Build a bank of :meth:`AdaptiveKalmanFilter.kinematic` members over the ``(phi, s)`` grid.
        The grid spans the impulsive (small ``phi``) to persistent (``phi`` -> 1) ends; widen it
        freely, the data down-weights the unsupported corners."""
        members, pa, sa = [], [], []
        for phi in phis:
            for s in ss:
                members.append(AdaptiveKalmanFilter.kinematic(
                    n_dof, order, dt, process_var=process_var, meas_var=meas_var,
                    measured=measured, control=control, phi=phi, s=s))
                pa.append(phi); sa.append(s)
        return cls(members, pa, sa, forget=forget)

    def reset(self) -> "AdaptiveBank":
        for f in self.filters:
            f.reset()
        self._logw = np.zeros(len(self.filters))          # uniform prior (unnormalised)
        self.loglik = 0.0
        return self

    def update(self, y, u=None) -> AdaptiveBankStep:
        M = len(self.filters)
        prior = self._logw - _logsumexp(self._logw)
        ll = np.empty(M)
        means, vars_, ps, ms, inn = [], [], [], [], []
        for i, f in enumerate(self.filters):
            st = f.update(y, u=u)
            ll[i] = st.loglik
            means.append(st.mean); vars_.append(st.var); inn.append(st.innovation)
            ps.append(st.process_scale); ms.append(st.measurement_scale)
        y = np.atleast_1d(np.asarray(y, dtype=float))
        if np.all(np.isfinite(y)):
            bank_ll = _logsumexp(prior + ll)
            self._logw = self.forget * prior + ll         # Bayes update with forgetting
        else:
            bank_ll = 0.0
            self._logw = prior
        post = np.exp(self._logw - _logsumexp(self._logw))
        mean = sum(post[i] * means[i] for i in range(M))
        var = sum(post[i] * (vars_[i] + np.outer(means[i] - mean, means[i] - mean))
                  for i in range(M))
        pscale = sum(post[i] * ps[i] for i in range(M))
        mscale = sum(post[i] * ms[i] for i in range(M))
        innov = sum(post[i] * inn[i] for i in range(M))
        self.loglik += bank_ll
        return AdaptiveBankStep(mean, var, innov, bank_ll, pscale, mscale,
                                float(1.0 / (post @ post)),
                                float(post @ self.phi_arr), float(post @ self.s_arr))

    def filter(self, Y, U=None) -> AdaptiveResult:
        """Filter a batch; returns the model-averaged trajectory (same shape as a single filter)."""
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        if U is not None:
            U = np.atleast_2d(np.asarray(U, dtype=float))
        self.reset()
        T = Y.shape[0]
        mean = np.empty((T, self.n)); var = np.empty((T, self.n, self.n))
        inn = np.empty((T, self.m)); ps = np.empty((T, self.n)); ms = np.empty((T, self.m))
        total = 0.0
        for i, row in enumerate(Y):
            st = self.update(row, None if U is None else U[i])
            mean[i] = st.mean; var[i] = st.var; inn[i] = st.innovation
            ps[i] = st.process_scale; ms[i] = st.measurement_scale; total += st.loglik
        return AdaptiveResult(mean=mean, var=var, innovation=inn,
                              process_scale=ps, measurement_scale=ms, loglik=total)
