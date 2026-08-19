"""The move: a fine noise-channel grid whose centre tracks the truth.

The probes settle the design:

  * a node's effective zone is a shelf with a cliff (0003); a dead zone opens
    where the node gap exceeds the cliff reach (~0.8 nats), and the dead zone is
    in the exact likelihood, not just the cheap score (0004) -- so the only cure
    is resolution;
  * the grid-shift score points at the truth and its log recovers the distance
    even far off-grid (0001), and it is monotone precisely while the grid is
    fine enough to have no dead zone (0002, 0003).

Both point to the same architecture: keep the grid FINE (small spread s, so the
max node gap stays under the cliff reach and there is never a dead zone) and
give up on covering the plane with it.  Coverage instead comes from MOVING the
fine window.  Two timescales:

  inner (fast)   a fixed fine grid resolves fluctuations of the log-scale about
                 the window centre; its posterior pi rides the co-moving frame.
  outer (slow)   the window centre mu integrates the grid-shift score, so it
                 slides toward wherever the truth is -- with unbounded reach,
                 because integration has no ceiling the way a fixed grid does.

Because the frame co-moves with the truth, the posterior in the frame stays
centred and needs no re-projection when mu slides; the per-step step is clamped
so consecutive windows overlap (the user's "new grid must overlap the old").

This is a single channel (process).  The plane separates while every channel
stays covered (0004), so channels move independently; a loud UNCOVERED channel
leaks into the others through the shared innovation, which is the reason to keep
every channel's window on its truth.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gridlab import grid, uniform_grid, _LOG2PI  # noqa: E402


class MovingChannel:
    """Fixed fine grid + a centre that integrates the grid-shift score.

    Parameters
    ----------
    Q, s2       median process and measurement variance.
    phi, s      persistence and spread of the (frame-relative) log-scale grid.
                Keep s small: the no-dead-zone rule is max_gap = maxgap(order)*s
                <~ 0.6 nats.  Reach comes from the move, not from s.
    order       quadrature nodes.
    step        which signal drives the window centre (see 0007):
                "recenter" (default) slides mu to zero the within-window
                  posterior mean, dmu = rate * (pi @ lam).  Within coverage this
                  is the unbiased offset (a servo with a restoring force -> no
                  wander), off-window it saturates at +-edge (constant-rate
                  travel, right direction), and it reads relative fit across
                  nodes so it is SNR-robust.  Fast, symmetric, unbiased.
                "fisher" integrates the natural-gradient step (score / Fisher
                  ~0.5(Qg/S)^2): fast but a pure integrator on a noisy gradient,
                  so it wanders and biases.  "score" is the raw gradient
                  (also asymmetric: suppressed by Qg/S below the noise floor).
                  Both kept for the comparison in 0007.
    eta         gain: `rate` for "recenter", integration gain for the others.
    cap         per-step clamp on |d mu|, in nats -- the overlap constraint.
    beta        EMA smoothing of the step signal.
    ridge       stabiliser added to the Fisher information before dividing.
    """

    def __init__(self, Q, s2, phi=0.9, s=0.3, order=5, step="servo",
                 eta=0.4, cap=0.12, beta=0.6, ridge=1e-4, w_score=2.0,
                 tau=60.0, eta_floor=0.05,
                 a_slope=-0.138, b_int=-0.016, R_meas=15.0, q_mu=0.0, P0=25.0,
                 uniform=None, mu_cap=None,
                 hop_thresh=None, hop_patience=3, hop0=1.0, hop_grow=1.8):
        self.Q, self.s2 = float(Q), float(s2)
        self.order = int(order)
        if uniform is not None:
            # uniform=(half_width, gap): equispaced nodes at spacing `gap` over
            # +-half_width.  The optimal moving-grid discretisation -- constant
            # spacing means no over-clustered centre wasting compute and no
            # widening outer gap opening a dead zone.  `order` is then ignored
            # (the node count is fixed by half_width/gap).
            half_width, gap = uniform
            self.lam, self.w0, self.T = uniform_grid(phi, s, half_width, gap)
        else:
            self.lam, self.w0, self.T = grid(phi, s, order)
        self.max_gap = float(np.diff(self.lam).max())
        self.step, self.eta, self.cap = step, eta, cap
        self.beta, self.ridge, self.w_score = beta, ridge, w_score
        self.tau = tau                   # Robbins-Monro decay time
        self.eta_floor = eta_floor       # residual gain for tracking a drift
        # kalman-mode inputs (measured, not tuned): signal slope, measurement
        # variance, drift variance, initial truth-uncertainty
        self.a_slope, self.b_int = a_slope, b_int
        self.R_meas, self.q_mu, self.P0 = R_meas, q_mu, P0
        # unbounded-reach hunter (kalman_auto only; all default to no-op).  The
        # natural-gradient step grad/I ~ (e^2 - S)/Qg is unbounded in the
        # innovation, so a single huge e^2 (a big up-jump lands the truth many
        # nats out) makes mu LEAP past the truth; the collapsed steady gain then
        # cannot walk it back -> the recovery "blow-up" (0019/0022) is an
        # OVERSHOOT, not a lack of reach.  mu_cap clamps |dmu| to a bounded
        # stride (the overlap constraint), so the window walks out steadily and
        # captures any distance.  hop_thresh arms a rail-triggered geometric
        # expansion (Nelder-Mead): when the edge node's responsibility stays
        # above hop_thresh the truth is beyond the window, so jump mu by a
        # stride that grows *hop_grow each rail step -- bracketing the truth in
        # O(log distance) big steps, then the fine grid locks locally.  Compute
        # stays fixed: the fine window MOVES, it never grids the whole span.
        self.mu_cap = mu_cap
        self.hop_thresh = hop_thresh
        self.hop_patience, self.hop0, self.hop_grow = hop_patience, hop0, hop_grow
        self.reset()

    def reset(self, mu=0.0):
        self.mu = float(mu)
        self._pi = None
        self._m = None
        self._P = None
        self._score_ema = 0.0
        self._t = 0
        self._Pmu = self.P0              # kalman truth-uncertainty
        self._upc = 0                    # consecutive top-edge rail steps
        self._dnc = 0                    # consecutive bottom-edge rail steps
        self._hop = self.hop0            # current expansion stride
        self.loglik = 0.0
        return self

    def update(self, x):
        """Absorb one observation; slide the window; return a small dict."""
        lam, w0, T, Q, s2 = self.lam, self.w0, self.T, self.Q, self.s2
        Qg = Q * np.exp(np.clip(lam + self.mu, -60.0, 60.0))
        if self._pi is None:
            self._pi = w0.copy()
            self._m = float(x) if np.isfinite(x) else 0.0
            self._P = float(Qg.max() + s2)

        pi = self._pi @ T
        if not np.isfinite(x):                       # missing: propagate only
            self._pi = pi
            self._P = float(self._P + pi @ Qg)
            return dict(mean=self._m, var=self._P, mu=self.mu,
                        logscale=self.mu + float(pi @ lam), score=0.0, loglik=0.0)

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
        self._m = self._m + Kbar * e
        self._P = float(pi @ ((1.0 - K) * (P + Qg)) + e2 * (pi @ (K - Kbar) ** 2))

        # the move: integrate the (smoothed) grid-shift step, clamp for overlap.
        # grad is the raw score; fisher ~ 0.5 (Qg/S)^2 is the shift information.
        # The natural-gradient step grad/fisher cancels the Qg/S prefactor that
        # suppresses grad far below the measurement floor (see 0007).
        gS = Qg / S
        grad = float(pi @ (0.5 * gS * (e2 / S - 1.0)))
        if self.step == "kalman_auto":
            # Self-calibrating optimal tracker (0013): no a, b, R constants.
            # The per-step Fisher information about the shift is read straight
            # off the current grid, I_t = sum_i pi_i * 0.5 (Qg_i/S_i)^2, so the
            # natural-gradient step g/I is the offset estimate (Qg/S cancels ->
            # no from-below suppression, b=0) and R_t = 1/I_t (Cramer-Rao).  The
            # Kalman gain then down-weights low-information steps automatically.
            # Everything recomputes each step, so it follows a shifting regime.
            I = float(pi @ (0.5 * gS * gS)) + self.ridge
            R = 1.0 / I
            K = self._Pmu / (self._Pmu + R)
            dmu = K * (grad / I)                          # ascend, Kalman-averaged
            if self.mu_cap is not None:                  # bounded stride: no leap
                dmu = float(np.clip(dmu, -self.mu_cap, self.mu_cap))
            self.mu = self.mu + dmu
            self._Pmu = (1.0 - K) * self._Pmu + self.q_mu
            top = float(pi[-1]); bot = float(pi[0])
            if self.hop_thresh is not None:              # rail-triggered expansion
                self._upc = self._upc + 1 if top > self.hop_thresh else 0
                self._dnc = self._dnc + 1 if bot > self.hop_thresh else 0
                if top <= self.hop_thresh and bot <= self.hop_thresh:
                    self._hop = self.hop0                # bracketed: reset stride
                if self._upc >= self.hop_patience:
                    self.mu += self._hop; self._Pmu = self.P0
                    self._hop *= self.hop_grow; self._upc = 0
                elif self._dnc >= self.hop_patience:
                    self.mu -= self._hop; self._Pmu = self.P0
                    self._hop *= self.hop_grow; self._dnc = 0
            self._pi = pi
            self.loglik += ll
            return dict(mean=self._m, var=self._P, mu=self.mu,
                        logscale=self.mu + float(pi @ lam), score=grad,
                        signal=grad, fisher=I, gain=K, loglik=ll,
                        top=top, bot=bot, hop=self._hop)
        if self.step == "kalman":
            # Optimal linearised tracker (0012).  The signal is a noisy linear
            # measurement of the offset: signal ~ a*(mu - truth) + noise.  So
            # z = signal/a estimates the offset and (mu - z) measures the truth;
            # a scalar Kalman filter fuses those measurements.  Gains come only
            # from a (measured signal slope), R (measured measurement variance)
            # and q_mu (drift variance) -- no hand-set eta/beta/tau/cap.
            signal = float(pi @ lam) + self.w_score * grad
            z = (signal - self.b_int) / self.a_slope     # offset estimate (debiased)
            K = self._Pmu / (self._Pmu + self.R_meas)
            self.mu = self.mu - K * z                     # = truth-estimate update
            self._Pmu = (1.0 - K) * self._Pmu + self.q_mu
            self._pi = pi
            self.loglik += ll
            return dict(mean=self._m, var=self._P, mu=self.mu,
                        logscale=self.mu + float(pi @ lam), score=grad,
                        signal=signal, gain=K, loglik=ll)
        if self.step == "servo":
            # posterior mean (drives from below, restoring within coverage) plus
            # the raw score (drives from above, where the shelf is flat and the
            # posterior mean stalls).  Complementary; neither amplifies noise.
            signal = float(pi @ lam) + self.w_score * grad
        elif self.step == "recenter":
            signal = float(pi @ lam)                 # within-window posterior mean
        elif self.step == "fisher":
            fisher = float(pi @ (0.5 * gS * gS))
            signal = grad / (fisher + self.ridge)
        else:
            signal = grad
        score = grad
        self._score_ema = self.beta * self._score_ema + (1.0 - self.beta) * signal
        # Robbins-Monro step size: eta_t = max(eta_floor, eta / (1 + t/tau)).
        # A decaying step is stochastic approximation -- it converges to the true
        # fixed point (mu = truth) with vanishing wander, so the estimate reaches
        # the oracle floor for a static log-scale.  eta_floor > 0 keeps a constant
        # residual gain so a drifting log-scale is still tracked (bandwidth is a
        # tracking budget, not a free parameter).
        self._t += 1
        eta_t = max(self.eta_floor, self.eta / (1.0 + self._t / self.tau))
        dmu = float(np.clip(eta_t * self._score_ema, -self.cap, self.cap))
        self.mu += dmu

        self._pi = pi
        self.loglik += ll
        return dict(mean=self._m, var=self._P, mu=self.mu,
                    logscale=self.mu + float(pi @ lam), score=score,
                    signal=signal, loglik=ll)
