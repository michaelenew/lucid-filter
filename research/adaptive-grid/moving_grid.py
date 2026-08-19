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
from gridlab import grid, _LOG2PI  # noqa: E402


class MovingChannel:
    """Fixed fine grid + a centre that integrates the grid-shift score.

    Parameters
    ----------
    Q, s2       median process and measurement variance.
    phi, s      persistence and spread of the (frame-relative) log-scale grid.
                Keep s small: the no-dead-zone rule is max_gap = maxgap(order)*s
                <~ 0.6 nats.  Reach comes from the move, not from s.
    order       quadrature nodes.
    eta         integration gain on the score.
    cap         per-step clamp on |d mu|, in nats -- the overlap constraint.
    beta        EMA smoothing of the score before the step.
    """

    def __init__(self, Q, s2, phi=0.9, s=0.3, order=5,
                 eta=0.5, cap=0.12, beta=0.6):
        self.Q, self.s2 = float(Q), float(s2)
        self.order = int(order)
        self.lam, self.w0, self.T = grid(phi, s, order)
        self.max_gap = float(np.diff(self.lam).max())
        self.eta, self.cap, self.beta = eta, cap, beta
        self.reset()

    def reset(self, mu=0.0):
        self.mu = float(mu)
        self._pi = None
        self._m = None
        self._P = None
        self._score_ema = 0.0
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

        # the move: integrate the (smoothed) grid-shift score, clamp for overlap
        score = float(pi @ (0.5 * (Qg / S) * (e2 / S - 1.0)))
        self._score_ema = self.beta * self._score_ema + (1.0 - self.beta) * score
        dmu = float(np.clip(self.eta * self._score_ema, -self.cap, self.cap))
        self.mu += dmu

        self._pi = pi
        self.loglik += ll
        return dict(mean=self._m, var=self._P, mu=self.mu,
                    logscale=self.mu + float(pi @ lam), score=score, loglik=ll)
