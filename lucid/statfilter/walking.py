"""A local-level filter whose noise scale is *tracked online*, not fitted.

Where :class:`~statfilter.core.AdaptiveFilter` learns its six numbers once, from a
whole series, by maximum likelihood (``fit()``), this filter learns the changing
part of the model *as it streams*.  It carries a small, fine quadrature window
over the process log-scale and lets that window **walk** to wherever the scale
actually is -- so the process volatility is followed step by step rather than
summarised by a fixed fit.

The model is the ordinary local-level model with a drifting process scale::

    theta_t = theta_{t-1} + w_t,     w_t ~ N(0, Q * exp(lam_t))
    x_t     = theta_t     + v_t,     v_t ~ N(0, s2)
    lam_t   = phi * lam_{t-1} + sqrt(nu) z_t,     nu = s^2 (1 - phi^2)

Only the AR(1) pair ``(phi, s)`` -- how sticky the volatility is, and how far it
swings -- plus the base scales ``(Q, s2)`` are supplied.  *Everything else the
filter needs is derived or learned online*, which is what sets it apart from
every other filter in this package:

  * **where the scale is** -- the window centre ``mu`` integrates the exact
    grid-shift score, walking (with unbounded reach) to the truth;
  * **how fast to correct it** -- the step gain is a scalar Kalman gain built from
    the per-step Fisher information ``I`` read straight off the grid, with the
    drift variance set online to the critical-damping point ``q_mu = r*/I``
    (``r* = 3.5e-4``: the fastest tracking with no overshoot);
  * **the grid spacing** -- ``gap = 1.5 s`` (nodes at ~2/3 of a posterior width,
    the resolution limit: fine enough for no dead zone, no finer);
  * **the step cap** -- ``mu_cap = gap`` keeps successive windows overlapping, so
    the walk stays dense.

There is no ``fit()``.  The only free numbers are ``(phi, s)``, which are not a
tuning knob but the model of the process itself -- the irreducible class
commitment (a filter that assumes nothing about how fast volatility moves cannot
separate a real regime change from a run of noise).  See ``theory/adaptive-grid``
for the derivations behind each "derived" line above.

Basic use::

    from statfilter import WalkingFilter

    f = WalkingFilter(Q=1.0, s2=1.0, phi=0.9, s=0.30)
    r = f.filter(x)
    r.mean                 # tracked level
    r.process_scale        # tracked process log-scale, online -- no fit

    f.reset()              # or stream
    for v in stream:
        step = f.update(v)
        print(step.mean, step.process_scale)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["WalkingFilter", "WalkStep", "WalkResult"]

_LOG2PI = math.log(2.0 * math.pi)

# Derived constants (see theory/adaptive-grid/SUMMARY.md findings 10-11):
_R_STAR = 3.5e-4        # critical-damping tracking index r = q_mu * I (finding 10)
_GAP_FACTOR = 1.5       # gap = 1.5 * s: the resolution/Sparrow spacing (finding 11)
_RIDGE = 1e-4           # stabiliser on the Fisher information before dividing


# --------------------------------------------------------------------- results
@dataclass
class WalkStep:
    """Everything the walking filter knows after one observation."""

    mean: float            #: posterior mean of the level
    var: float             #: posterior variance of the level
    innovation: float      #: x_t - prior mean
    loglik: float          #: log predictive density of x_t
    process_scale: float   #: tracked process log-scale E[lam_t | data], = mu + window mean
    scale_step: float      #: how far the window moved this step (nats), clamped to the gap
    info: float            #: per-step Fisher information about the scale (the observability)


@dataclass
class WalkResult:
    """Batch output.  Every field is an array of length n except ``loglik``."""

    mean: np.ndarray
    var: np.ndarray
    innovation: np.ndarray
    process_scale: np.ndarray
    scale_step: np.ndarray
    info: np.ndarray
    loglik: float = 0.0

    def __len__(self) -> int:
        return len(self.mean)


# ---------------------------------------------------------------- the filter
class WalkingFilter:
    """Local-level filter that tracks the process noise scale online by walking a grid.

    Parameters
    ----------
    Q, s2 : float
        Median (geometric-mean) process and measurement variance.  ``Q`` is only
        a reference point: a wrong ``Q`` is absorbed by the walking window (the
        scale it reports shifts to compensate), so the estimate is unchanged.
    phi : float
        Persistence of the process log-scale AR(1), in ``[0, 1)`` -- how sticky
        the volatility is.
    s : float
        Stationary log-SD of the process scale -- how far the volatility swings,
        in nats.  Sets the grid spacing (``gap = 1.5 s``); must be positive.
    nodes : int
        Number of grid nodes (odd).  A numerical resolution, not a model choice:
        the window spans ``+-(nodes//2) * gap`` and *walks* for anything beyond,
        so this trades instant reach for per-step cost, nothing else.  Default 7.

    Notes
    -----
    Stateful when streaming: :meth:`update` advances the state, :meth:`filter`
    and :meth:`loglik` run from a fresh state and leave the object untouched.
    """

    def __init__(self, Q: float, s2: float, phi: float, s: float, nodes: int = 7):
        if not (Q > 0.0 and s2 > 0.0):
            raise ValueError("Q and s2 must be positive")
        if not 0.0 <= phi < 1.0:
            raise ValueError("phi must lie in [0, 1)")
        if not s > 0.0:
            raise ValueError("s must be positive (it sets the grid spacing)")
        if nodes < 3 or nodes % 2 == 0:
            raise ValueError("nodes must be an odd integer >= 3")
        self.Q, self.s2 = float(Q), float(s2)
        self.phi, self.s = float(phi), float(s)
        self.nodes = int(nodes)
        self.gap = _GAP_FACTOR * self.s                 # resolution spacing (finding 11)
        self.cap = self.gap                             # overlap: <= one node / step
        self._build()
        self.reset()

    def __repr__(self) -> str:
        return (f"WalkingFilter(Q={self.Q:.4g}, s2={self.s2:.4g}, "
                f"phi={self.phi:.3f}, s={self.s:.3f}, nodes={self.nodes})")

    def _build(self):
        """The fixed uniform window: nodes, stationary weights, AR(1) transition."""
        K = self.nodes // 2
        lam = self.gap * np.arange(-K, K + 1, dtype=float)
        w0 = np.exp(-0.5 * (lam / self.s) ** 2)
        w0 /= w0.sum()
        nu = max(self.s * self.s * (1.0 - self.phi * self.phi), 1e-12)
        T = np.exp(np.clip(-0.5 * (lam[None, :] - self.phi * lam[:, None]) ** 2 / nu,
                           -700.0, 700.0))
        T /= T.sum(1, keepdims=True)
        self.lam, self.w0, self.T = lam, w0, T

    # ------------------------------------------------------------- streaming
    def reset(self, level: float | None = None, scale: float = 0.0) -> "WalkingFilter":
        """Clear the streaming state.  ``scale`` seeds the window centre.  Chains."""
        self._pi = None
        self._m = level
        self._P = None
        self.mu = float(scale)
        self._Pmu = 25.0            # diffuse start on the scale: washes out in a few steps
        self.loglik = 0.0
        return self

    def update(self, x: float) -> WalkStep:
        """Absorb one observation; walk the window; return everything known.

        A non-finite observation is treated as missing: the state is propagated
        but not corrected.
        """
        lam, T, Q, s2 = self.lam, self.T, self.Q, self.s2
        Qg = Q * np.exp(np.clip(lam + self.mu, -60.0, 60.0))
        if self._pi is None:
            self._pi = self.w0.copy()
            if self._m is None:
                self._m = float(x) if np.isfinite(x) else 0.0
            if self._P is None:
                self._P = float(Qg.max() + s2)

        pi = self._pi @ T
        if not np.isfinite(x):                          # missing: propagate only
            self._pi = pi
            self._P = float(self._P + pi @ Qg)
            return WalkStep(self._m, self._P, math.nan, 0.0,
                            self.mu + float(pi @ lam), 0.0, 0.0)

        P = self._P
        S = P + Qg + s2
        e = float(x) - self._m
        e2 = e * e
        lg = -0.5 * (np.log(S) + e2 / S)
        mx = float(lg.max())
        w = pi * np.exp(lg - mx)
        Z = float(w.sum())
        ll = float(np.log(Z)) + mx - 0.5 * _LOG2PI
        pi = w / Z

        K = (P + Qg) / S
        Kbar = float(pi @ K)
        m_new = self._m + Kbar * e
        P_new = float(pi @ ((1.0 - K) * (P + Qg)) + e2 * (pi @ (K - Kbar) ** 2))

        # walk the window: natural-gradient step on the grid-shift score, made a
        # scalar Kalman update by the grid-read Fisher information I, capped to
        # one node for overlap, with the drift variance set to critical damping.
        gS = Qg / S
        grad = float(pi @ (0.5 * gS * (e2 / S - 1.0)))
        info = float(pi @ (0.5 * gS * gS)) + _RIDGE
        R_mu = 1.0 / info
        K_mu = self._Pmu / (self._Pmu + R_mu)
        dmu = float(np.clip(K_mu * (grad / info), -self.cap, self.cap))
        self.mu += dmu
        self._Pmu = (1.0 - K_mu) * self._Pmu + _R_STAR * R_mu     # q_mu = r*/I

        self._pi, self._m, self._P = pi, m_new, P_new
        self.loglik += ll
        return WalkStep(m_new, P_new, e, ll,
                        self.mu + float(pi @ lam), dmu, info)

    # ----------------------------------------------------------------- batch
    def loglik_of(self, x) -> float:
        """Marginal log-likelihood of a series.  Does not touch streaming state."""
        return self._run(np.asarray(x, dtype=float), want=False)

    def filter(self, x) -> WalkResult:
        """Run over a whole series from a fresh state.  Does not touch state."""
        return self._run(np.asarray(x, dtype=float), want=True)

    def _run(self, x: np.ndarray, want: bool):
        if x.ndim != 1 or x.size == 0:
            raise ValueError("x must be a non-empty 1-D array")
        saved = (self._pi, self._m, self._P, self.mu, self._Pmu, self.loglik)
        try:
            self.reset()
            if not want:
                total = 0.0
                for v in x:
                    total += self.update(v).loglik
                return total
            n = x.size
            cols = ("mean", "var", "innovation", "process_scale", "scale_step", "info")
            out = {c: np.empty(n) for c in cols}
            total = 0.0
            for i, v in enumerate(x):
                st = self.update(v)
                out["mean"][i] = st.mean
                out["var"][i] = st.var
                out["innovation"][i] = st.innovation
                out["process_scale"][i] = st.process_scale
                out["scale_step"][i] = st.scale_step
                out["info"][i] = st.info
                total += st.loglik
            return WalkResult(loglik=total, **out)
        finally:
            (self._pi, self._m, self._P, self.mu, self._Pmu, self.loglik) = saved
