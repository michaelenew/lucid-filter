"""Probe 0051 -- the RECORDED OPEN, built: per-component scales on VectorFilter's grid, clustered.

vector.py line 33-35 flags per-component scale deduction ("which sensor is hot right now") as the open:
it is what the failing-sensor (H-observability) case needs, and it breaks the tensor-product grid --
order^(#channels), exponential. The fix: the confound is LOCAL, so for a block-diagonal H (each joint's
sensors read only that joint's state -- the robotics case) the per-channel grid FACTORISES into
independent per-cluster sub-filters. Each joint is a small grid over its own channels
{process-mode, pot, accel} = order^3 = 125 nodes, run with VectorFilter's exact GPB1 update (reused
verbatim). No EMA, no beta, no walk, no reach -- the scale posterior does the failing-sensor reallocation.

This validates the design: one ClusterGrid == 0050's single joint (sanity), and 5 of them run
independently on the 5-DOF rig, measured against the shipped floor and the oracle across the 0034 regimes.
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
from statfilter.core import _chain, _LOG2PI  # noqa: E402  -- the grid machinery, reused

import importlib.util  # noqa: E402
_s = importlib.util.spec_from_file_location("p34", os.path.join(os.path.dirname(__file__), "0034_profile.py"))
p34 = importlib.util.module_from_spec(_s); _s.loader.exec_module(p34)


def _wide_chain(phi, s, order, span):
    """AR(1) log-scale grid with span DECOUPLED from s: nodes uniform on [-span, span] (wide enough to
    cover a sensor FAILURE, not just the class swing), stationary weights = N(0, s^2) on those nodes,
    transition = the AR(1) Gaussian kernel. span is a resolution choice (cover the failures you care
    about); s and phi are the class. Reduces to _chain's placement when span = (order-1)/2*1.5*s."""
    lam = np.linspace(-span, span, order)
    w = np.exp(-0.5 * (lam / s) ** 2); w /= w.sum()
    nu = max(s * s * (1.0 - phi * phi), 1e-12)
    T = np.exp(np.clip(-0.5 * (lam[None, :] - phi * lam[:, None]) ** 2 / nu, -700, 700))
    T /= T.sum(1, keepdims=True)
    return lam, w, T


class ClusterGrid:
    """Per-channel-scale grid over ONE confound cluster (n-state block, m sensors), reusing
    VectorFilter's GPB1 update. Channels = active process eigenmodes + sensors; joint grid over them."""

    def __init__(self, F, B, H, Q0, rho, phi=0.9, s=0.5, order=9, span=6.0):
        self.F, self.B, self.H = F, B, np.atleast_2d(H)
        self.n, self.m = F.shape[0], self.H.shape[0]
        lam, V = np.linalg.eigh(Q0)
        keep = lam > 1e-6 * lam.max()                  # genuine process modes only (drop numerical dust)
        self.lam, self.V = lam[keep], V[:, keep]
        self.np_ = self.V.shape[1]
        self.rho = np.asarray(rho, float)
        self.phi, self.s, self.order = phi, s, order
        # PER-CHANNEL span from structural confoundability (0043): a sensor's direct process footprint
        # g_i = diag(H Q0 H^T)_i / (that + rho_i). Decoupled sensors (g small) get a WIDE span so they
        # reach a genuine failure; a direct process-readout sensor (g near the cluster max) gets a
        # NARROW span so a process disturbance cannot push its scale up and shed a good sensor (the
        # accel<->process confound). Derived, no beta. Process modes get the wide span (they SHOULD reach
        # a jerk burst). span_hi/span_lo are resolution bounds (cover a failure / the class swing).
        dproc = np.diag(self.H @ Q0 @ self.H.T)
        g = dproc / (dproc + self.rho + 1e-300)
        maxg = max(float(g.max()), 1e-300)
        span_hi, span_lo = span, 1.5 * self.s        # confounded channel: ~one class swing (no reach)
        sensor_spans = span_lo + (span_hi - span_lo) * (1.0 - g / maxg)
        self.spans = np.concatenate([np.full(self.np_, span_hi), sensor_spans])
        self._build()
        self.reset()

    def _build(self):
        o, npn, m = self.order, self.np_, self.m
        chans = npn + m
        chains = [_wide_chain(self.phi, self.s, o, sp) for sp in self.spans]   # per-channel wide chain
        idxg = np.meshgrid(*([np.arange(o)] * chans), indexing="ij")
        idx = np.stack([gg.ravel() for gg in idxg], axis=1)     # (G, chans) node index per channel
        cfg = np.stack([chains[c][0][idx[:, c]] for c in range(chans)], axis=1)   # (G, chans) log-scale
        G = cfg.shape[0]
        self.pi0 = np.prod([chains[c][1][idx[:, c]] for c in range(chans)], axis=0)
        self.pi0 /= self.pi0.sum()
        Tj = np.array([[1.0]])                                  # transition = kron of per-channel chains
        for c in range(chans):
            Tj = np.kron(Tj, chains[c][2])
        self.T = Tj
        xi = cfg[:, :npn]; eta = cfg[:, npn:]
        VVt = np.einsum("k,ik,jk->kij", self.lam, self.V, self.V)          # (np, n, n)
        self.Qg = np.einsum("gk,kij->gij", np.exp(np.clip(xi, -60, 60)), VVt)   # (G, n, n)
        Rg = np.zeros((G, m, m))
        for i in range(m):
            Rg[:, i, i] = self.rho[i] * np.exp(np.clip(eta[:, i], -60, 60))
        self.Rg = Rg
        self.G = G

    def reset(self):
        self.pi = None; self.m_ = None; self.P = None
        return self

    def update(self, y, u=None):
        H, F = self.H, self.F
        bu = (self.B @ u) if (self.B is not None and u is not None) else 0.0
        if self.pi is None:
            self.pi = self.pi0.copy()
            self.m_ = np.linalg.lstsq(H, y, rcond=None)[0] if np.all(np.isfinite(y)) else np.zeros(self.n)
            self.P = np.eye(self.n) * (self.Rg.reshape(self.G, -1).max() + self.Qg.reshape(self.G, -1).max()) * self.n
        pi = self.pi @ self.T                                              # propagate scale posterior
        mpred = F @ self.m_ + bu
        Ppred = (F @ self.P @ F.T)[None] + self.Qg                         # (G, n, n)
        e = y - H @ mpred
        PHt = np.einsum("gij,kj->gik", Ppred, H)
        S = np.einsum("ij,gjk->gik", H, PHt) + self.Rg
        Sinv = np.linalg.inv(S)
        _, logdet = np.linalg.slogdet(S)
        maha = np.einsum("i,gij,j->g", e, Sinv, e)
        lg = -0.5 * (self.m * _LOG2PI + logdet + maha)
        w = pi * np.exp(lg - lg.max()); pi = w / w.sum()
        K = np.einsum("gik,gkl->gil", PHt, Sinv)
        Ke = np.einsum("gil,l->gi", K, e)
        Kbar = np.einsum("g,gil->il", pi, K)
        m_new = mpred + Kbar @ e
        mpost = mpred[None] + Ke
        dm = mpost - m_new
        KH = np.einsum("gil,lj->gij", K, H)
        Ppost = Ppred - np.einsum("gij,gjk->gik", KH, Ppred)
        P_new = np.einsum("g,gij->ij", pi, Ppost) + np.einsum("g,gi,gj->ij", pi, dm, dm)
        self.pi, self.m_, self.P = pi, m_new, 0.5 * (P_new + P_new.T)
        return m_new


def run_5dof(nseed=6, order=5):
    NJ, ORD = p34.NJ, p34.ORDER
    print(f"clustered per-channel grid (NO EMA), 5-DOF via {NJ} per-joint clusters, order={order}, {nseed} seeds:")
    print(f"  {'regime':13s} {'floor/orc':>10} {'grid/orc':>10}")
    for regime, tag in p34.REGIMES:
        fl = []; gr = []
        for seed in range(nseed):
            f, F, B, H, U, S, Y, jstd, pot, acc = p34.sim(seed, regime)
            oc = p34.oracle(F, B, H, U, Y, jstd, pot, acc, f.n, f.m)
            # per-joint cluster blocks
            Fb = F[:ORD, :ORD]; Gv = B[:ORD, 0]
            Hc = np.array([[1.0] + [0.0] * (ORD - 1), [0.0] * (ORD - 1) + [1.0]])
            Q0c = (p34.JERK ** 2) * np.outer(p34.G, p34.G)      # rank-1 jerk process (one mode)
            rhoc = np.array([p34.POT ** 2, p34.ACC ** 2])
            est = np.zeros((p34.T, f.n))
            for j in range(NJ):
                cg = ClusterGrid(Fb, Gv[:, None], Hc, Q0c, rhoc, order=order)
                sl = slice(j * ORD, (j + 1) * ORD); ys = slice(j * 2, (j + 1) * 2)
                for k in range(p34.T):
                    est[k, sl] = cg.update(Y[k, ys], u=U[k, j:j + 1])
            fl.append(p34.rms(f.filter(Y, U=U).mean, S) / p34.rms(oc, S))
            gr.append(p34.rms(est, S) / p34.rms(oc, S))
        print(f"  {tag:13s} {np.mean(fl):10.3f} {np.mean(gr):10.3f}")


if __name__ == "__main__":
    run_5dof()
