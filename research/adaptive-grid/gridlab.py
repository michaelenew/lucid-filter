"""Shared machinery for the adaptive-grid probes.

One process channel of ``statfilter`` in isolation (measurement grid collapsed
to its s_M = 0 face, so R is constant).  :func:`verify` checks the recursion
here against the shipped filter to 1e-7, so the probes run the real code path.

The grid is Gauss-Hermite nodes ``lam_i = s*z_i`` on a stationary AR(1)
log-scale; a *rigid shift* by mu moves the node cloud (and the stationary mean)
to ``lam_i + mu`` while leaving the transition kernel's shape unchanged -- that
is the move the adaptive grid makes, and the quantity every score here
differentiates.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "lucid"))

from statfilter.core import _chain, AdaptiveFilter, Params  # noqa: E402

_LOG2PI = np.log(2.0 * np.pi)


def grid(phi, s, order):
    """(lam, w0, T) for one channel: nodes, stationary weights, transition."""
    return _chain(phi, s, order)


def simulate(rng, lam_star, Q, s2, nt):
    """Random walk with constant excess log-scale lam*, observed with N(0, s2)."""
    qstep = Q * np.exp(lam_star)
    theta = np.cumsum(rng.normal(0.0, np.sqrt(qstep), size=nt))
    return theta + rng.normal(0.0, np.sqrt(s2), size=nt)


def run_channel(x, lam, w0, T, Q, s2, mu=0.0):
    """Batched single-process-channel filter over rows of ``x`` (B, n_t).

    ``mu`` rigidly shifts the node cloud: ``Qg_i = Q*exp(lam_i + mu)``.  Returns
    per-series time-averaged reads, including the local grid-shift score
    (the derivative of the per-step marginal loglik under d mu, holding the
    carried covariance and prior mixture fixed).
    """
    x = np.atleast_2d(x)
    B, nt = x.shape
    Qg = Q * np.exp(lam + mu)                           # (n,)
    R = s2
    m = x[:, 0].astype(float).copy()
    P = np.full(B, float(Qg.max() + R))
    pi = np.tile(w0, (B, 1)).astype(float)

    ll = np.zeros(B)
    score_acc = np.zeros(B)
    mean_acc = np.zeros(B)
    top_acc = np.zeros(B)
    bot_acc = np.zeros(B)
    for t in range(nt):
        pi = pi @ T
        Pp = P[:, None] + Qg[None, :]                  # (B, n)
        S = Pp + R
        e = x[:, t] - m
        e2 = e * e
        lg = -0.5 * (np.log(S) + e2[:, None] / S)
        mx = lg.max(1)
        w = pi * np.exp(lg - mx[:, None])
        Z = w.sum(1)
        ll += np.log(Z) + mx - 0.5 * _LOG2PI
        pi = w / Z[:, None]
        K = Pp / S
        Kbar = (pi * K).sum(1)
        m = m + Kbar * e
        P = (pi * ((1.0 - K) * Pp)).sum(1) + e2 * (pi * (K - Kbar[:, None]) ** 2).sum(1)
        score_acc += (pi * (0.5 * (Qg[None, :] / S) * (e2[:, None] / S - 1.0))).sum(1)
        mean_acc += (pi * (lam + mu)[None, :]).sum(1)
        top_acc += pi[:, -1]
        bot_acc += pi[:, 0]
    return dict(loglik=ll, score=score_acc / nt, postmean=mean_acc / nt,
                top=top_acc / nt, bot=bot_acc / nt)


def responsibilities(x, lam, w0, T, Q, s2, mu=0.0):
    """Time-averaged posterior weight on each node: the (B, n) responsibility."""
    x = np.atleast_2d(x)
    B, nt = x.shape
    Qg = Q * np.exp(lam + mu)
    R = s2
    m = x[:, 0].astype(float).copy()
    P = np.full(B, float(Qg.max() + R))
    pi = np.tile(w0, (B, 1)).astype(float)
    acc = np.zeros((B, lam.size))
    for t in range(nt):
        pi = pi @ T
        Pp = P[:, None] + Qg[None, :]
        S = Pp + R
        e = x[:, t] - m
        e2 = e * e
        lg = -0.5 * (np.log(S) + e2[:, None] / S)
        mx = lg.max(1)
        w = pi * np.exp(lg - mx[:, None])
        Z = w.sum(1)
        pi = w / Z[:, None]
        K = Pp / S
        Kbar = (pi * K).sum(1)
        m = m + Kbar * e
        P = (pi * ((1.0 - K) * Pp)).sum(1) + e2 * (pi * (K - Kbar[:, None]) ** 2).sum(1)
        acc += pi
    return acc / nt


def single_node_loglik(x, lam_i, Q, s2):
    """Per-step loglik of a fixed single-variance filter (one node, mult exp(lam_i)).

    A node's intrinsic 'effectiveness' at representing a truth: run the ordinary
    local-level filter with process variance Q*exp(lam_i) and read its predictive
    log-density.  As a function of the true log-scale this is a bell centred near
    lam_i whose width is set by the filter's sensitivity to variance mismatch --
    a width that does not depend on how the nodes are spaced.
    """
    x = np.atleast_2d(x)
    B, nt = x.shape
    q = Q * np.exp(lam_i)
    R = s2
    m = x[:, 0].astype(float).copy()
    P = np.full(B, float(q + R))
    ll = np.zeros(B)
    for t in range(nt):
        S = P + q + R
        e = x[:, t] - m
        ll += -0.5 * (np.log(S) + e * e / S + _LOG2PI)
        K = (P + q) / S
        m = m + K * e
        P = (1.0 - K) * (P + q)
    return ll / nt


def exact_shift_gradient(x, lam, w0, T, Q, s2, mu=0.0, h=1e-3):
    """Central-difference gradient of the marginal loglik under a rigid mu-shift.

    The full derivative -- the recursive dependence of the carried covariance and
    the mixture on mu included -- against which the cheap local score in
    :func:`run_channel` is the approximation.
    """
    lp = run_channel(x, lam, w0, T, Q, s2, mu + h)["loglik"]
    lm = run_channel(x, lam, w0, T, Q, s2, mu - h)["loglik"]
    return (lp - lm) / (2.0 * h)


def verify(order=5, phi=0.98, s=0.8, Q=1.0, s2=1.0, nt=400):
    """The single-channel recursion equals the shipped filter at s_M = 0."""
    rng = np.random.default_rng(0)
    x = simulate(rng, 0.7, Q, s2, nt)
    lam, w0, T = grid(phi, s, order)
    mine = run_channel(x, lam, w0, T, Q, s2)["loglik"][0]
    f = AdaptiveFilter(Params(Q=Q, s2=s2, phi_P=phi, s_P=s, phi_M=0.0, s_M=0.0),
                       order=order)
    theirs = f.loglik(x)
    assert abs(mine - theirs) < 1e-7, (mine, theirs)
    return mine, theirs


# ----------------------------------------------------- both channels at once
def simulate_joint(rng, lamP_star, lamM_star, Q, s2, nt):
    """Random walk with constant excess log-scales on both channels."""
    theta = np.cumsum(rng.normal(0.0, np.sqrt(Q * np.exp(lamP_star)), size=nt))
    return theta + rng.normal(0.0, np.sqrt(s2 * np.exp(lamM_star)), size=nt)


def run_joint(x, phi_P, s_P, phi_M, s_M, Q, s2, order, muP=0.0, muM=0.0):
    """Two-channel filter (the full plane), returning per-axis shift scores.

    The joint grid is the tensor product used by ``statfilter._build``:
    LP = repeat(lamP), LM = tile(lamM).  Rigid shifts muP, muM move the two node
    clouds independently.  ``verify_joint`` checks the loglik against the shipped
    filter to 1e-7.
    """
    x = np.atleast_2d(x)
    B, nt = x.shape
    lamP, wP, TP = grid(phi_P, s_P, order)
    lamM, wM, TM = grid(phi_M, s_M, order)
    LP = np.repeat(lamP, order) + muP
    LM = np.tile(lamM, order) + muM
    T = np.kron(TP, TM)
    w0 = np.kron(wP, wM)
    Qg = Q * np.exp(np.clip(LP, -60.0, 60.0))
    Rg = s2 * np.exp(np.clip(LM, -60.0, 60.0))

    m = x[:, 0].astype(float).copy()
    P = np.full(B, float(Qg.max() + Rg.max()))
    pi = np.tile(w0, (B, 1)).astype(float)
    ll = np.zeros(B)
    scP = np.zeros(B)
    scM = np.zeros(B)
    for t in range(nt):
        pi = pi @ T
        Pp = P[:, None] + Qg[None, :]
        S = Pp + Rg[None, :]
        e = x[:, t] - m
        e2 = e * e
        lg = -0.5 * (np.log(S) + e2[:, None] / S)
        mx = lg.max(1)
        w = pi * np.exp(lg - mx[:, None])
        Z = w.sum(1)
        ll += np.log(Z) + mx - 0.5 * _LOG2PI
        pi = w / Z[:, None]
        K = Pp / S
        Kbar = (pi * K).sum(1)
        m = m + Kbar * e
        P = (pi * ((1.0 - K) * Pp)).sum(1) + e2 * (pi * (K - Kbar[:, None]) ** 2).sum(1)
        tilt = 0.5 * (e2[:, None] / S - 1.0)
        scP += (pi * (Qg[None, :] / S) * tilt).sum(1)
        scM += (pi * (Rg[None, :] / S) * tilt).sum(1)
    return dict(loglik=ll, scoreP=scP / nt, scoreM=scM / nt)


def verify_joint(order=5, phi_P=0.9, s_P=0.7, phi_M=0.8, s_M=0.5,
                 Q=1.0, s2=1.0, nt=400):
    """The two-channel recursion equals the shipped filter with both channels on."""
    rng = np.random.default_rng(1)
    x = simulate_joint(rng, 0.5, -0.4, Q, s2, nt)
    mine = run_joint(x, phi_P, s_P, phi_M, s_M, Q, s2, order)["loglik"][0]
    f = AdaptiveFilter(Params(Q=Q, s2=s2, phi_P=phi_P, s_P=s_P, phi_M=phi_M, s_M=s_M),
                       order=order)
    theirs = f.loglik(x)
    assert abs(mine - theirs) < 1e-7, (mine, theirs)
    return mine, theirs
