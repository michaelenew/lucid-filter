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

Everything is vector; a scalar problem is length 1.

**The dynamics channel.**  `dynamics=None` learns `F` (and `B`) online from the random-walk prior;
`dynamics=F0, faults=rho` says the supplied dynamics may CHANGE -- a payload attached to a drone, a
tire blown out -- and the filter detects the change and recovers the new dynamics without a refit, a
threshold, or a fitted constant.  It is realised as a state augmentation `(x, g)` with
`F = F0 + sum_j g_j A_j`, so the noise machinery above runs on top of it unchanged, which is what
separating a wrong `F` from elevated `Q` requires: the two compete as hypotheses under a live noise
walk rather than through a bolted-on whiteness statistic.  The bank carries the nominal member
forever (a false detection then costs ~nothing), optional named fault `anchors` (the fastest
detector when the failure modes can be named), and a departure walker whose variance is bounded at
the class cap and re-priced to it when a fault is confirmed -- bounded, never frozen.  The one
labeled prior is the fault hazard `rho`; every gain, drift, cap and restart width follows from it
and the class size, and the detection delay it buys is derived (`log(1/rho) / KL-rate`), not tuned.
The derivations and the measured acceptance results are `research/dynamics-learning/SUMMARY.md`.

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
_HAZARD = 1e-4              # default fault rate: ~1 dynamics fault per 10,000 steps.  A LABELED
                            # prior of the same standing as `forget`, not a tuning constant: it is
                            # the operating point on the false-alarm/delay frontier, and the delay
                            # it buys is derived, log(1/rho) / KL-rate (research 0001).


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
    dynamics: np.ndarray = None    #: posterior-mean ``F`` (n, n) -- ``None`` when supplied fixed
    control: np.ndarray = None     #: posterior-mean ``B`` (n, p) -- ``None`` when fixed or absent
    fault: float = 0.0             #: posterior probability the dynamics have left the nominal

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
    dynamics: np.ndarray = None    #: (T, n, n) learned ``F``, or ``None`` when supplied fixed
    control: np.ndarray = None     #: (T, n, p) learned ``B``, or ``None``
    fault: np.ndarray = None       #: (T,) posterior probability of a dynamics fault

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

    def __init__(self, Q0, R0, H, F, B, phi, s, walk_axes=None, cap=None):
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
        self._cap = None if cap is None else np.asarray(cap, float)
        self._dyn = None            # optional (mean, u) -> (Jacobian, predicted mean) callable
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
        if walk_axes is not None:
            # Axes whose scale is a CLASS COMMITMENT, not a live noise scale (the dynamics
            # channel's theta block: its drift is q_theta = cap * rho from the fault class).
            # This bounds nothing about theta itself -- theta is a state with a live gain and a
            # cap, never frozen (research 0003); only the log-scale WALK of its drift is off.
            self.active &= np.asarray(walk_axes, bool)
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

    def _cap_P(self, P):
        """Bound a state's variance at its class cap by symmetric row/column scaling.

        A BOUND, never a freeze: the gain stays live, so later evidence still moves the
        component (research 0003 measured the latched-freeze alternative at 20x worse)."""
        if self._cap is None:
            return P
        d = np.diag(P)
        over = d > self._cap
        if not over.any():
            return P
        sc = np.ones_like(d)
        sc[over] = np.sqrt(self._cap[over] / np.maximum(d[over], 1e-300))
        return P * np.outer(sc, sc)

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
            Ppred = self._cap_P(F @ P @ F.T + Q0)
            S = H @ Ppred @ H.T + R0
            K = Ppred @ H.T @ np.linalg.inv(S)
            P = Ppred - K @ H @ Ppred
        Ppred = self._cap_P(F @ P @ F.T + Q0)
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

    def reprice(self, idx):
        """A detected jump re-prices ignorance: variance back to the class cap on ``idx``,
        cross-covariances cleared (research 0003).  The estimate itself is kept -- the data
        moves it at cap gain.  A fault is ONE event, so this fires on EVERY departure axis,
        excited or not: an unexcited axis then stands at honest cap width instead of
        reporting a stale coefficient it has no evidence for."""
        if self._P is None or self._cap is None or len(idx) == 0:
            return
        self._P[idx, :] = 0.0
        self._P[:, idx] = 0.0
        self._P[idx, idx] = self._cap[idx]

    def update(self, y, u=None):
        n, m, H = self.n, self.m, self.H
        y = np.atleast_1d(np.asarray(y, dtype=float))
        r = len(self._act)
        Qg, Rg = self._star_QR()
        if self._pi_ax is None:
            self._pi_ax = np.tile(self._w1, (r, 1))
            if self._m is None:
                self._m = (np.linalg.lstsq(H, y, rcond=None)[0]
                           if np.all(np.isfinite(y)) else np.zeros(n))
            if self._P is None:
                self._P = self._cap_P(
                    np.eye(n) * float(Rg.reshape(self._G, -1).max()
                                      + Qg.reshape(self._G, -1).max()) * n)
        pi_ax = self._pi_ax @ self._T1
        # The dynamics are a CALLABLE when they are learned or nonlinear: it returns the
        # linearisation (for the covariance) and the predicted mean (from f, not the
        # Jacobian -- they differ once the transition depends on the state).  Real F/B
        # arrive linearised per operating point, so this is the general path.
        if self._dyn is None:
            F = self.F
            mpred = F @ self._m + ((self.B @ u) if self.B is not None else 0.0)
        else:
            F, mpred = self._dyn(self._m, u)
        FPFt = F @ self._P @ F.T
        if not np.all(np.isfinite(y)):
            self._pi_ax = pi_ax
            w = self._star_weights(pi_ax)
            self._P = self._cap_P(FPFt + np.einsum("g,gij->ij", w, Qg))
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
        P_new = self._cap_P(0.5 * (P_new + P_new.T))
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


# ------------------------------------------------------- the dynamics channel
class _Departure:
    """One learned-dynamics hypothesis: ``F(g) = F0 + sum_j g_j A_j``, ``B(g) = B0 + sum_j g_j C_j``.

    Realised as a STATE AUGMENTATION.  With the augmented state ``(x, g)``, the augmented
    observation ``[H | 0]`` and the augmented transition

        [[ F(g),  d(F(g)x + B(g)u)/dg ],
         [ 0,     I                   ]]

    the departure channel is just more state, so the whole noise machinery above (the caltrop
    scale walk, the structural activation, the bank) runs on top of it unchanged -- which is what
    the record demands: the Q<->F confound is split by per-hypothesis MEANS competing under a
    LIVE noise walk (research 0002), not by a whiteness statistic bolted on the side.

    Three commitments, all from the fault class, none tuned:

    * ``g``'s drift is ``q_g = sigma^2 rho`` (a jump of class size ``sigma`` at hazard ``rho``
      has that mean square per step) and its variance is capped at ``sigma^2`` -- **bounded,
      never frozen**: the gain stays live so an axis the data cannot see today still moves when
      excitation arrives (research 0003 measured the latched-freeze alternative at 20x worse).
    * ``g``'s scale axes are excluded from the noise walk: their drift is a class commitment,
      not a live noise scale.  (Structural, per 0024 -- ``[H | 0]`` cannot see ``g`` directly,
      so the engine's own activation rule already reaches this conclusion; the mask makes it
      exact under eigenvalue degeneracy.)
    * ``sigma = 1`` per unit-Frobenius direction: the entries of a stable discrete transition
      are O(1), so "F moves by one in Frobenius norm" is the class-size statement.  A supplied
      direction is normalised, so its coefficient is in class units by construction.
    """

    def __init__(self, F0, B0, dirs, rho, n, p):
        self.F0, self.B0, self.n, self.p = F0, B0, n, p
        self.A = np.stack([a for a, _ in dirs]) if dirs else np.zeros((0, n, n))
        self.C = (np.stack([c for _, c in dirs]) if dirs and p
                  else np.zeros((0, n, max(p, 1))))
        self.k = self.A.shape[0]
        self.na = n + self.k
        sig2 = np.ones(self.k)                      # unit-Frobenius directions -> class size 1
        self.cap = np.concatenate([np.full(n, np.inf), sig2])
        self.q_g = sig2 * rho
        self.gidx = np.arange(n, self.na)

    def augment(self, Q0, R0, H):
        """(Q0_aug, R0, H_aug, walk_axes) -- walk_axes in EIGENMODE space, so it is exact even
        if a departure's drift coincides with a process eigenvalue."""
        Qa = np.zeros((self.na, self.na))
        Qa[:self.n, :self.n] = Q0
        Qa[self.gidx, self.gidx] = self.q_g
        Ha = np.zeros((H.shape[0], self.na))
        Ha[:, :self.n] = H
        _, V = np.linalg.eigh(Qa)                   # the engine repeats this deterministically
        walk = np.ones(self.na + H.shape[0], bool)
        walk[:self.na] = np.linalg.norm(V[self.n:], axis=0) < 0.5
        return Qa, R0, Ha, walk

    def dynamics_of(self, g):
        F = self.F0 + np.einsum("j,jab->ab", g, self.A) if self.k else self.F0
        if self.B0 is None:
            return F, None
        B = self.B0 + np.einsum("j,jab->ab", g, self.C) if self.k else self.B0
        return F, B

    def callable_for(self, engine):
        """The (mean, u) -> (Jacobian, predicted mean) hook the engine calls each step."""
        n, k = self.n, self.k

        def _dyn(m, u):
            x, g = m[:n], m[n:]
            F, B = self.dynamics_of(g)
            mp = np.empty_like(m)
            mp[:n] = F @ x + (B @ u if B is not None else 0.0)
            mp[n:] = g
            J = np.zeros((self.na, self.na))
            J[:n, :n] = F
            J[n:, n:] = np.eye(k)
            if k:
                J[:n, n:] = np.einsum("jab,b->aj", self.A, x)
                if B is not None:
                    J[:n, n:] += np.einsum("jab,b->aj", self.C, np.atleast_1d(u))
            return J, mp

        return _dyn


def _unit(M):
    nrm = float(np.linalg.norm(M))
    return M / nrm if nrm > 0 else M


def _basis(n, p, departures):
    """Departure directions as (A, C) pairs, unit-Frobenius.

    Default (research mechanism (c), the full walk): every elementary entry of F, and of B when
    a control map is supplied.  Supplying ``departures`` is mechanism (b) -- the low-rank
    physical channel (a drone's added mass moves F AND B through one coefficient), which is the
    production form and is `n**2 / len(departures)` times cheaper.
    """
    if departures is None:
        out = []
        for i in range(n):
            for j in range(n):
                A = np.zeros((n, n)); A[i, j] = 1.0
                out.append((A, np.zeros((n, max(p, 1)))))
        for i in range(n):
            for j in range(p):
                C = np.zeros((n, p)); C[i, j] = 1.0
                out.append((np.zeros((n, n)), C))
        return out
    out = []
    for d in departures:
        A, C = d if isinstance(d, tuple) else (d, None)
        A = np.atleast_2d(np.asarray(A, float))
        if A.shape != (n, n):
            raise ValueError(f"each departure direction must be ({n}, {n})")
        C = (np.zeros((n, max(p, 1))) if C is None
             else np.atleast_2d(np.asarray(C, float)))
        if p and C.shape != (n, p):
            raise ValueError(f"each departure control direction must be ({n}, {p})")
        nrm = math.sqrt(float(np.sum(A * A) + np.sum(C * C)))
        if nrm == 0:
            raise ValueError("a departure direction must be non-zero")
        out.append((A / nrm, C / nrm))
    return out


# ------------------------------------------------------------- the public filter
class LucidFilter:
    """The single public filter: supply dynamics + observation, it infers all the noise online.

    Parameters (each: explicit value, or a null/default for "none", or the give-what-you-know rule)
    ----------
    dynamics : (n, n) array, 0, or None
        The linear state dynamics ``F``.  ``0`` (default) means no dynamics -> a random-walk level.
        ``None`` means **learn them**: the prior is the random walk (``F = I``) and the filter
        recovers ``F`` (and ``B``) online.  See *The dynamics channel* below.
    faults : float or True, optional
        Turn on the dynamics channel around a SUPPLIED ``F``: the value is the hazard ``rho``,
        the per-step probability that the dynamics change (``True`` -> ``1e-4``, about one fault
        per 10,000 steps).  Implied by ``dynamics=None``.  This is a labeled prior of the same
        standing as ``forget``, not a tuning constant -- it is the operating point on the
        false-alarm/delay frontier, and the detection delay it buys is derived,
        ``log(1/rho) / KL-rate``.
    departures : sequence, optional
        The directions the dynamics may move along: each an ``(n, n)`` matrix, or an
        ``(A, C)`` pair when the same physical parameter moves ``F`` and ``B`` together (a
        drone's added mass does).  Default: every entry of ``F`` (and of ``B``) -- fully general
        and ``n**2`` coefficients.  Supplying the physical directions is the production form and
        is far cheaper.
    anchors : sequence, optional
        Named fault hypotheses -- each an ``(n, n)`` ``F`` or an ``(F, B)`` pair -- carried as
        their own full filters.  When the failure modes are nameable (blown-left, blown-right,
        payload-attached) this is the fastest detector there is; the walker still refines
        whatever the anchors only bracket.
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
                 n=None, faults=None, departures=None, anchors=None,
                 phis=_PHIS, ss=_SS, forget=0.999):
        learn = dynamics is None or faults is not None
        if faults is None or faults is True:
            rho = _HAZARD
        else:
            rho = float(faults)
            if not 0.0 < rho < 1.0:
                raise ValueError("faults (the hazard rho) must lie in (0, 1)")
        if anchors is not None and not learn:
            learn = True                        # named fault hypotheses imply a fault class
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
        self.phi_arr = np.array([ph for ph in phis for _ in ss], float)
        self.s_arr = np.array([sv for _ in phis for sv in ss], float)
        cells = [(ph, sv) for ph in phis for sv in ss]

        # -------- the dynamics hypotheses (the ladder of research/dynamics-learning) --------
        # The NOMINAL member is always present and never leaves: it is the hedge that makes a
        # false detection cost ~nothing, which is in turn what makes the fast end of the
        # detection frontier affordable (research 0001).  Named ANCHORS come next -- when the
        # failure modes can be named, they are the fastest detector there is (0005).  The
        # WALKER refines whatever the anchors only bracket: detection degrades gracefully under
        # a mis-specified anchor set, recovery does not (0001 s4).
        specs = [(F, B, None)]
        for a in (anchors or []):
            Fa, Ba = a if isinstance(a, tuple) else (a, B)
            Fa = np.atleast_2d(np.asarray(Fa, float))
            if Fa.shape != (n, n):
                raise ValueError(f"each anchor must be ({n}, {n})")
            specs.append((Fa, None if Ba is None else np.atleast_2d(np.asarray(Ba, float)),
                          None))
        if learn:
            specs.append((F, B, _Departure(F, B, _basis(n, self.p, departures),
                                           rho, n, self.p)))

        self._members, self._pidx, self._specs = [], [], specs
        for Fs, Bs, dep in specs:
            if dep is None:
                self._members += [_WalkEngine(Q0, R0, Hm, Fs, Bs, ph, sv) for ph, sv in cells]
                self._pidx += [np.arange(n)] * len(cells)
                continue
            Qa, Ra, Ha, walk = dep.augment(Q0, R0, Hm)
            xmode = np.flatnonzero(walk[:dep.na])
            if xmode.size != n:
                raise RuntimeError("augmented process eigenbasis did not separate")
            Fa = np.eye(dep.na)
            Fa[:n, :n] = Fs                     # the Jacobian at g = 0, x = 0: the
            Ba = (None if Bs is None                     # characteristic linearisation the
                  else np.vstack([Bs, np.zeros((dep.k, self.p))]))   # steady Fisher wants
            for ph, sv in cells:
                e = _WalkEngine(Qa, Ra, Ha, Fa, Ba, ph, sv, walk_axes=walk, cap=dep.cap)
                e._dyn = dep.callable_for(e)
                self._members.append(e)
                self._pidx.append(xmode)
        self._nd, self._nc = len(specs), len(cells)
        k = self._nd
        # The fault class's kernel: a uniform-leak chain with probability rho per step of
        # leaving the current dynamics hypothesis (Shiryaev's rule for a jump process).  It
        # mixes WITHIN each (phi, s) cell -- a dynamics fault does not change the noise class,
        # so the joint kernel over the two nuisances is their product.
        self._Md = (np.eye(k) * (1.0 - rho * k / (k - 1)) + np.ones((k, k)) * (rho / (k - 1))
                    if k > 1 else np.ones((1, 1)))
        self._learn, self.hazard = learn, rho
        self.reset()

    def reset(self):
        for f in self._members:
            f.reset()
        self._logw = np.zeros(len(self._members))
        self.loglik = 0.0
        self._alarm = False
        return self

    def _hazard_mix(self, logw):
        """Propagate the bank prior through the fault class's kernel."""
        W = np.exp(logw - float(logw.max())).reshape(self._nd, self._nc)
        out = np.log(np.maximum(self._Md.T @ W, 1e-300)).ravel()
        return out - _logsumexp(out)

    def _dynamics_mean(self, post):
        """Posterior-mean (F, B) across the bank -- fixed members contribute their own."""
        W = post.reshape(self._nd, self._nc)
        Fh = np.zeros((self.n, self.n))
        Bh = None if self.B is None else np.zeros((self.n, self.p))
        for d, (Fs, Bs, dep) in enumerate(self._specs):
            if dep is None:
                w = float(W[d].sum())
                Fh += w * Fs
                if Bh is not None and Bs is not None:
                    Bh += w * Bs
                continue
            for c in range(self._nc):
                e = self._members[d * self._nc + c]
                g = np.zeros(dep.k) if e._m is None else e._m[self.n:]
                Fg, Bg = dep.dynamics_of(g)
                Fh += W[d, c] * Fg
                if Bh is not None and Bg is not None:
                    Bh += W[d, c] * Bg
        return Fh, Bh

    def _reprice(self):
        """Fire the shared-event restart on every walker (research 0003)."""
        for d, (_, _, dep) in enumerate(self._specs):
            if dep is None:
                continue
            for c in range(self._nc):
                self._members[d * self._nc + c].reprice(dep.gidx)

    def update(self, y, u=None) -> LucidStep:
        if self.B is not None and u is None:
            raise ValueError(f"this filter has a control input; pass u (length {self.p})")
        if self.B is None and u is not None:
            raise ValueError("filter has no control map; do not pass u")
        M = len(self._members)
        prior = self._logw - _logsumexp(self._logw)
        if self._nd > 1:
            prior = self._hazard_mix(prior)
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
        n = self.n
        mn = [st.mean[:n] for st in steps]
        mean = sum(post[i] * mn[i] for i in range(M))
        var = sum(post[i] * (steps[i].var[:n, :n] + np.outer(mn[i] - mean, mn[i] - mean))
                  for i in range(M))
        ps = sum(post[i] * steps[i].process_scale[self._pidx[i]] for i in range(M))
        ms = sum(post[i] * steps[i].measurement_scale for i in range(M))
        innov = sum(post[i] * steps[i].innovation for i in range(M))
        self.loglik += bank_ll
        if self._nd == 1:
            return LucidStep(mean, var, innov, bank_ll, ps, ms)
        Fh, Bh = self._dynamics_mean(post)
        # The fault readout is a marginal of the posterior -- the filter itself never
        # thresholds, it mixes.  Its RISING EDGE re-prices the walkers' ignorance: a jump has
        # just been confirmed, so what the walkers think they know about the departure is
        # priced back to the class prior (research 0003).
        fault = float(1.0 - post.reshape(self._nd, self._nc)[0].sum())
        alarm = fault > 0.5
        if alarm and not self._alarm:
            self._reprice()
        self._alarm = alarm
        return LucidStep(mean, var, innov, bank_ll, ps, ms, Fh, Bh, fault)

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
        live = self._nd > 1
        dyn = np.empty((T, self.n, self.n)) if live else None
        ctl = np.empty((T, self.n, self.p)) if live and self.B is not None else None
        flt = np.empty(T) if live else None
        total = 0.0
        for i, row in enumerate(Y):
            st = self.update(row, None if U is None else U[i])
            mean[i] = st.mean; var[i] = st.var; inn[i] = st.innovation
            ps[i] = st.process_scale; ms[i] = st.measurement_scale; total += st.loglik
            if live:
                dyn[i] = st.dynamics; flt[i] = st.fault
                if ctl is not None:
                    ctl[i] = st.control
        return LucidResult(mean=mean, var=var, innovation=inn,
                           process_scale=ps, measurement_scale=ms, loglik=total,
                           dynamics=dyn, control=ctl, fault=flt)

    def loglik_of(self, Y, U=None) -> float:
        return self.filter(Y, U).loglik
