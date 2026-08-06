"""Two series, one clock: the offset channel.

A second series reads the same latent process at a time offset:

    y1_t = x(t) + v1_t,            v1_t ~ N(0, s2)
    y2_t = c * x(t - tau) + v2_t,  v2_t ~ N(0, s2_2)

tau is time-valued, fractional, signed, and possibly moving.  tau > 0 means y2
LAGS (y1 leads); tau < 0 means y2 leads.  This module detects and tracks the
lead/lag online as a posterior over a tau grid -- a trusted distribution with
the same semantics as every other channel -- and reports trust that the two
series are related at all.

The construction, each piece pinned by a probe in exploration/0042-0057:

DELAY ROW (0042/0043).  The whole extension is one observation row.  In the
lag basis the delayed reading is a **fractional power of the shift**: with F
the companion matrix and V its eigenbasis, x_{t-s} = e1' V diag(z_i^-sigma)
V^-1 applied to p consecutive stored lags -- exact on the recurrence's
solution space, no interpolation stencil ever chosen.  Principal branch;
see `delay_row` for the two guards this needs.

LEADS ARE LAGS IN PROCESSING TIME (0042/0054/0055).  A node with tau < 0 says
y2 reads the latent's future; its bracketing lags exist ceil(-tau) steps
later.  Every node defers UNIFORMLY by the window's max deferral, because
per-node deferral is measurably biased: a node processing later conditions on
more y1, a likelihood subsidy unrelated to tau (0055 -- a spurious posterior
band at the deferral-class boundary, 2.7x lead-side RMS).  Under uniform
deferral the sign of tau is decided at 99:1 in ~20 points either way, and
deferral itself HELPS: y2 is read against partially smoothed lags.

NOTHING HAND-SET (0046).  The gain c is a log-spaced grid crossed with tau; the
restart mass eps is a three-member hyper-grid Bayes-mixed online (a Bayes
mixture trails the best member in hindsight by at most log 3 nats, ever); node
weights evolve by fixed-share, so "the offset does not move" (eps = 0) is an
explicit member.  Grid extents and node counts are compute budgets.

TRUST IS DIRECTED INFORMATION (0046).  Against a MATCHED null -- the caller's
own fitted `OdeFilter` for y2 alone, passed as `null` -- the accumulated
log-odds Lambda is a prequential estimate of the information y1's history adds
about y2 beyond y2's own, and trust = sigma(Lambda).  A weaker null inflates
trust ~4.5x (0046 measured it), so when no null is supplied trust is reported
as nan rather than against a strawman.

THE CLASS GAP, RESOLVED BY ABSORPTION (0045 s5).  The discrete class defines x
between samples as the modal interpolant, so fractional reads carry no
in-model bridge variance.  On data from a continuous world the true fractional
read has a bridge residual (~ Q * frac(tau)-shaped, 0043); it is absorbed into
the second channel's measurement variance s2_2, which the caller estimates for
the second series anyway.  That is the honest resolution until the class
itself is continuous -- the fractional-order program in the repository README.

DELIBERATELY LEFT OUT, with pointers: diffusion and kinetic (tau, taudot)
kernels (0050 -- they buy tau-band calibration during persistent drift, which
0056 priced at ~5 millinats/point for every cross-forecast consumer); the tube
grid's "is it a pure delay?" verdict (0048); the (mu, tau) derivative axis
(0043).  The filter here runs on the fitted model's homoscedastic face (the
volatility channels stay in `OdeFilter`, which keeps owning y1).

Guarantees measured in exploration, not asserted here: dynamics errors cannot
bias tau (0052/0053 -- tau is the symmetry center of the cross-covariance,
first-order immune to model misspecification), and the lead time is the
horizon out to which y1 forecasts y2 at tracking grade (0056's knee law).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .core import OdeFilter, Params, _companion

__all__ = ["OffsetFilter", "OffsetStep", "delay_row", "cross_anchor"]

_LOG2PI = math.log(2.0 * math.pi)


# ----------------------------------------------------------------- delay row
def _modal(alpha):
    """Eigen-structure of the companion matrix, with the two guards the
    principal branch needs: distinct roots (a Vandermonde inverse must exist)
    and no root on the closed negative real axis (z^-sigma would be branch-
    ambiguous there -- an undersampled or alternating mode has no well-defined
    fractional delay; see 0042 s2)."""
    F = _companion(alpha)
    w, V = np.linalg.eig(F)
    scale = float(np.max(np.abs(w))) or 1.0
    d = np.abs(w[:, None] - w[None, :])
    np.fill_diagonal(d, np.inf)
    if d.min() < 1e-6 * scale:
        raise ValueError(
            "offset channel needs distinct roots; alpha has a (near-)repeated "
            "root, whose fractional delay is not defined by the modal form")
    if np.any((w.real < 0.0) & (np.abs(w.imag) < 1e-9 * scale)):
        raise ValueError(
            "a negative real root has no principal fractional power: the mode "
            "alternates sign every step and 'between samples' is ambiguous")
    if np.min(np.abs(w)) < 1e-8:
        raise ValueError("a root at z = 0 cannot be read at negative powers")
    return w, V


def delay_row(alpha, s: float, length: int) -> np.ndarray:
    """The row r with  r @ (x_t, ..., x_{t-length+1}) = x_{t-s},  exactly on
    the recurrence's solution space, for real s in [0, length - 1].

    This is the fractional shift z^-sigma evaluated on the modes (principal
    branch), applied to the p consecutive stored lags nearest the reading
    point.  At integer s it reduces to the exact pick-out row, bit for bit up
    to the eigendecomposition's round-off.
    """
    alpha = np.asarray(alpha, dtype=float)
    p = alpha.size
    if length < p:
        raise ValueError("stored window shorter than the recurrence order")
    if not 0.0 <= s <= length - 1:
        raise ValueError("reading point outside the stored window")
    k = min(int(math.floor(s)), length - p)
    sigma = s - k
    w, V = _modal(alpha)
    local = (V[0] * w.astype(complex) ** (-sigma)) @ np.linalg.inv(V)
    if np.max(np.abs(local.imag)) > 1e-8 * max(np.max(np.abs(local.real)), 1.0):
        raise ValueError("fractional delay row failed to come out real")
    row = np.zeros(length)
    row[k:k + p] = local.real
    return row


# -------------------------------------------------------------------- anchor
def _gamma_modal(alpha, Q):
    """The stationary autocovariance, extended to REAL lags by the modal form
    gamma(s) = Re sum_i b_i z_i^s (principal powers).  Requires a stable
    alpha; the caller handles unit roots by differencing first."""
    alpha = np.asarray(alpha, dtype=float)
    p = alpha.size
    F = _companion(alpha)
    if np.max(np.abs(np.linalg.eigvals(F))) >= 1.0 - 1e-9:
        raise ValueError("gamma extension needs a stable alpha")
    Qm = np.zeros((p, p)); Qm[0, 0] = Q
    n2 = p * p
    G = np.linalg.solve(np.eye(n2) - np.kron(F, F), Qm.reshape(-1)).reshape(p, p)
    gam = G[0]                                   # gamma(0..p-1)
    w, V = _modal(alpha)
    b = np.linalg.solve(np.vander(w, p, increasing=True).T, gam.astype(complex))

    def gamma(s):
        s = np.abs(np.asarray(s, dtype=float))
        return np.real(b @ w[:, None].astype(complex) ** s[None, :]).reshape(
            np.shape(s))
    return gamma


def cross_anchor(y1, y2, params: Params, window=(-6.0, 6.0)) -> float:
    """Closed-form tau start: the sliding cross-covariance, interpolated with
    the model's own autocovariance shape (0049).

    Independent measurement noises leave the cross-covariance unbiased at
    every lag -- the analogue of the parent's variogram identity -- and at
    fractional tau the interpolant between integer lags is gamma(.) itself,
    not a parabola (measured 5x better, within 25% of full ML).  When the
    fitted alpha carries roots at or near the unit circle both series are
    differenced once and the quotient dynamics supply the shape; that is an
    approximation, and this is a START for the tracked filter, not the
    estimate.
    """
    y1 = np.asarray(y1, dtype=float); y2 = np.asarray(y2, dtype=float)
    alpha = np.asarray(params.alpha, dtype=float)
    roots = params.roots
    near_unit = np.abs(roots) >= 1.0 - 1e-6
    if near_unit.any():
        keep = roots[~near_unit]
        if keep.size == 0:
            raise ValueError("no stable modes to shape the anchor with")
        alpha = np.real(np.poly(keep))
        alpha = -alpha[1:]                        # back to recurrence form
        y1 = np.diff(y1); y2 = np.diff(y2)
    gamma = _gamma_modal(alpha, params.Q)
    lo, hi = int(math.floor(window[0])), int(math.ceil(window[1]))
    ks, chat = [], []
    n = min(len(y1), len(y2))
    for k in range(lo, hi + 1):
        a = y2[max(k, 0):n + min(k, 0)]
        b = y1[max(-k, 0):n - max(k, 0)]
        m = min(len(a), len(b))
        if m < 30:
            continue
        ks.append(k); chat.append(float(np.mean(a[:m] * b[:m])))
    ks = np.asarray(ks, dtype=float); chat = np.asarray(chat)
    taus = np.arange(window[0], window[1] + 1e-9, 0.01)
    best, tau_hat = np.inf, float(taus[0])
    for tau in taus:
        g = gamma(ks - tau)
        gg = float(g @ g)
        if gg <= 0.0:
            continue
        r = float(chat @ chat - (g @ chat) ** 2 / gg)
        if r < best:
            best, tau_hat = r, float(tau)
    return tau_hat


# -------------------------------------------------------------------- filter
@dataclass
class OffsetStep:
    """Everything known about the offset after one paired observation."""

    tau_mean: float              #: posterior mean of tau
    tau_sd: float                #: posterior SD of tau
    tau_map: float               #: highest-posterior tau node
    p_lead: float                #: P(tau < 0): the probability y2 LEADS
    c_mean: float                #: posterior mean of the gain
    lam: float                   #: accumulated coupled-vs-null log-odds, nats
    trust: float                 #: sigma(lam); nan when no null was supplied
    loglik2: float               #: log predictive density of the processed y2
    pending: bool                #: True while deferral is still warming up


class OffsetFilter:
    """Online lead/lag tracker for two series sharing one latent process.

    Parameters
    ----------
    params : Params
        The fitted model of the latent as seen through y1 (`OdeFilter.fit` on
        y1's history).  The offset channel runs on its homoscedastic face:
        alpha, Q, s2.  The volatility and dynamics channels stay in
        `OdeFilter`.
    s2_2 : float
        Measurement variance of the second series.  Any bridge residual of
        fractional reads is absorbed here (module docstring, THE CLASS GAP).
    window : (float, float)
        The tau extent tracked.  A compute budget plus one commitment: offsets
        beyond the stored window cannot be represented (a compressed-history
        anchor for large offsets is recorded in 0049 s3, unbuilt).
    taus, c_grid : arrays, optional
        Node budgets.  Defaults: 41 tau nodes over the window; gains at
        +/- geomspace(1/4, 4, 7) -- signed, because anti-correlated coupling
        is a coupling.
    restarts : tuple of float
        The restart-mass hyper-grid, Bayes-mixed online; includes 0 ("the
        offset does not move") as an explicit member.
    null : OdeFilter, optional
        A fitted filter for y2 ALONE -- the matched null that makes `trust` a
        directed-information reading.  It is reset here; pass a fresh or
        reusable instance.
    """

    def __init__(self, params: Params, s2_2: float, window=(-2.0, 3.0),
                 taus=None, c_grid=None, restarts=(0.0, 1e-3, 1e-2),
                 null: OdeFilter | None = None):
        if not s2_2 > 0.0:
            raise ValueError("s2_2 must be positive")
        if not window[0] < window[1]:
            raise ValueError("window must be increasing")
        self.params = params
        p = params.p
        self.taus = (np.linspace(window[0], window[1], 41)
                     if taus is None else np.asarray(taus, dtype=float))
        if c_grid is None:
            g = np.geomspace(0.25, 4.0, 7)
            c_grid = np.concatenate([-g[::-1], g])
        self.c_grid = np.asarray(c_grid, dtype=float)
        self.defer = int(math.ceil(max(0.0, -float(self.taus.min()))))
        length = int(math.ceil(self.defer + float(self.taus.max()))) + p
        self.P = length
        F = np.zeros((length, length))
        F[0, :p] = params.alpha
        F[1:, :-1] = np.eye(length - 1)
        self._F = F
        self._Q = np.zeros((length, length)); self._Q[0, 0] = params.Q
        self._h1 = np.zeros(length); self._h1[0] = 1.0
        self._r1 = params.s2

        rows = np.stack([delay_row(params.alpha, self.defer + t, length)
                         for t in self.taus])
        B = len(self.taus) * len(self.c_grid)
        self._H2 = (self.c_grid[None, :, None]
                    * rows[:, None, :]).reshape(B, length)
        self._R2 = np.full(B, float(s2_2))
        self._B = B
        self._restarts = tuple(float(e) for e in restarts)
        self.null = null
        self.reset()

    def reset(self) -> "OffsetFilter":
        B, L = self._B, self.P
        self._m = np.zeros((B, L))
        self._Pc = np.broadcast_to(1e6 * np.eye(L), (B, L, L)).copy()
        # node weights are carried in LOG space: repeated fixed-share products
        # underflow linear weights to exact zeros on far-from-model data, and
        # a dead eps = 0 node must stay -inf rather than poison the sum
        self._lw = np.tile(np.full(B, -math.log(B)), (len(self._restarts), 1))
        self._hyper = np.zeros(len(self._restarts))
        self._lam = 0.0
        self._buf: deque = deque()
        self._t = 0
        if self.null is not None:
            self.null.reset()
        return self

    # -- internals ---------------------------------------------------------
    def _posterior_nodes(self) -> np.ndarray:
        hw = np.exp(self._hyper - self._hyper.max())
        hw /= hw.sum()
        w = np.exp(self._lw - self._lw.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        return hw @ w

    def update(self, y1: float, y2: float = math.nan) -> OffsetStep:
        """One paired observation.  y2 may be nan (missing); it is processed
        with the window's uniform deferral (module docstring)."""
        F, Q = self._F, self._Q
        self._m = self._m @ F.T
        self._Pc = np.matmul(np.matmul(F, self._Pc), F.T) + Q
        h1, r1 = self._h1, self._r1
        Ph = self._Pc @ h1
        S = Ph @ h1 + r1
        e = float(y1) - self._m @ h1
        K = Ph / S[:, None]
        self._m = self._m + K * e[:, None]
        self._Pc = self._Pc - K[:, :, None] * Ph[:, None, :]

        self._buf.append(float(y2))
        ll2 = math.nan
        pending = len(self._buf) <= self.defer
        if not pending:
            yk = self._buf.popleft()
            if math.isfinite(yk):
                Ph = np.einsum('bij,bj->bi', self._Pc, self._H2)
                S = np.einsum('bi,bi->b', self._H2, Ph) + self._R2
                e = yk - np.einsum('bi,bi->b', self._H2, self._m)
                lp = -0.5 * (np.log(S) + _LOG2PI + e * e / S)
                lls = np.empty(len(self._restarts))
                for i, eps in enumerate(self._restarts):
                    lw = self._lw[i]
                    w = np.exp(lw - lw.max())
                    w /= w.sum()
                    with np.errstate(divide="ignore"):     # fixed-share
                        lw = np.log((1.0 - eps) * w + eps / self._B) + lp
                    mx = float(lw.max())
                    z = mx + math.log(float(np.exp(lw - mx).sum()))
                    lls[i] = z
                    self._lw[i] = lw - z
                a = self._hyper + lls
                m0 = float(a.max())
                ll2 = m0 + math.log(float(np.exp(a - m0).sum())) \
                    - (float(self._hyper.max())
                       + math.log(float(np.exp(self._hyper
                                               - self._hyper.max()).sum())))
                self._hyper = a - m0
                K = Ph / S[:, None]
                self._m = self._m + K * e[:, None]
                self._Pc = self._Pc - K[:, :, None] * Ph[:, None, :]
                self._Pc = 0.5 * (self._Pc + np.swapaxes(self._Pc, 1, 2))
                if self.null is not None:
                    self._lam += ll2 - self.null.update(yk).loglik

        w = self._posterior_nodes().reshape(len(self.taus), len(self.c_grid))
        pt = w.sum(axis=1)
        pc = w.sum(axis=0)
        tau_mean = float(pt @ self.taus)
        tau_sd = float(math.sqrt(max(pt @ self.taus ** 2 - tau_mean ** 2, 0.0)))
        self._t += 1
        return OffsetStep(
            tau_mean=tau_mean, tau_sd=tau_sd,
            tau_map=float(self.taus[int(np.argmax(pt))]),
            p_lead=float(pt[self.taus < 0.0].sum()),
            c_mean=float(pc @ self.c_grid),
            lam=self._lam,
            trust=(1.0 / (1.0 + math.exp(-min(max(self._lam, -500), 500)))
                   if self.null is not None else math.nan),
            loglik2=ll2, pending=pending)

    def filter(self, y1, y2) -> list[OffsetStep]:
        """Batch convenience: one `update` per pair, in order."""
        y1 = np.asarray(y1, dtype=float)
        y2 = np.asarray(y2, dtype=float)
        if len(y1) != len(y2):
            raise ValueError("y1 and y2 must be the same length")
        return [self.update(a, b) for a, b in zip(y1, y2)]

    @property
    def posterior(self):
        """(taus, probs): the trusted distribution of the offset, now."""
        w = self._posterior_nodes().reshape(len(self.taus), len(self.c_grid))
        return self.taus.copy(), w.sum(axis=1)
