"""Probe 0012 -- the adaptive / sparse grid: a faithful base-reduction (not linear).

FINDINGS (see the .md):
  * An absolute-lattice sparse grid (this file's `sparse_walk`) is faithful on static
    (no drift) but UNDER-REACHES regimes -- it uses the stationary transition and so
    SHRINKS, losing the shipped filter's walking (unbounded) reach.  Wrong target.
  * The right target is the shipped filter's OFFSET window (fixed cells, walked by mu).
    Measured occupancy of that window (r=3, 125 cells): ~53 cells at eps=1e-4 (99.97%
    mass), 31 at 1e-3 (99.66%), 17 at 1e-2 (97.5%), 9 at 5e-2 (90%).  So sparsifying is
    a BASE REDUCTION 5 -> ~2.6-3.2 (a ~1.6-1.9^r speedup: ~4x at r=3, ~30-200x at
    r=6-8) at high mass fidelity -- meaningful for larger r, not the linear ideal.
  * Production plan: sparsify the mu-walked OFFSET window (keep the finding-18 walk for
    reach), prune to a kept-mass target, dense fallback when 5^r is already small.
Original absolute-lattice probe kept below for the record.

Probe 0012 -- the adaptive / sparse grid: faithful AND sub-exponential in practice.

0011 ruled out factoring (loses the process<->measurement coupling) and showed
per-component is load-bearing for state tracking.  The faithful sub-exponential path
is to keep the EXACT joint grid but instantiate only its high-weight nodes.

Representation: the scale posterior is a dict {lattice-node -> weight} over the
ABSOLUTE scale lattice (axis k at multiples of gap_k = 1.5 s).  No fixed window and no
separate walking centre mu: the active set MIGRATES to the truth on its own (nodes at 0
die, nodes at the regime spawn and survive).  Each step:
  1. transition  -- the stationary AR(1) kernel, applied SEPARABLY per axis (the
     per-axis noise sqrt(nu)=s*sqrt(1-phi^2) is < one cell, so spread is +/-1 cell);
  2. likelihood  -- the multivariate KF at each active node (the exact per-node update);
  3. reweight + PRUNE nodes below eps*max_weight (drop the negligible tail);
  4. COVERAGE    -- spawn the +/-1 neighbours of surviving nodes so the blob can crawl
     and broaden next step.
State: GPB1 collapse over the active nodes.  Faithful by construction (dropped nodes
carry < eps of the mass); the question this probe answers is the ACTIVE-SET SIZE.

Benchmarked vs the dense exact grid (0006) for faithfulness + measured |active| over time.
"""
import os
import sys
import math
import importlib.util
import itertools

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
from statfilter.core import _LOG2PI  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "p6", os.path.join(os.path.dirname(__file__), "0006_walker_nge1_and_H.py"))
p6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p6)
D, N, M, PHI, SS, LAM, RHO, H, HV = p6.D, p6.N, p6.M, p6.PHI, p6.SS, p6.LAM, p6.RHO, p6.H, p6.HV
Q_of, R_of, exact_grid = p6.Q_of, p6.R_of, p6.exact_grid

np.set_printoptions(precision=3, suppress=True)
_GAP, _SPAN, _EPS = 1.5, 3.0, 1e-4


def gen(hot, amp, T=600, seed=1):
    rng = np.random.default_rng(seed); psi = np.zeros((T, D))
    if hot is not None:
        psi[T // 3: 2 * T // 3, hot] = amp
    th = np.zeros(N); Y = np.zeros((T, M))
    for t in range(T):
        th = th + np.linalg.cholesky(Q_of(psi[t, :N]) + 1e-12 * np.eye(N)) @ rng.standard_normal(N)
        Y[t] = H @ th + np.sqrt(np.diag(R_of(psi[t, N:]))) * rng.standard_normal(M)
    return Y


def Ichar():
    P = np.eye(N) * (LAM.max() + RHO.max()); Q0 = Q_of(np.zeros(N)); R0 = R_of(np.zeros(M))
    for _ in range(400):
        Pp = P + Q0; S = H @ Pp @ H.T + R0; K = Pp @ H.T @ np.linalg.inv(S); P = Pp - K @ H @ Pp
    Pp = P + Q0; S = H @ Pp @ H.T + R0; Si = np.linalg.inv(S)
    o = [0.5 * np.trace(Si @ (LAM[k] * np.outer(HV[:, k], HV[:, k])) @ Si @ (LAM[k] * np.outer(HV[:, k], HV[:, k]))) for k in range(N)]
    for i in range(M):
        E = np.zeros((M, M)); E[i, i] = RHO[i]; o.append(0.5 * np.trace(Si @ E @ Si @ E))
    return np.array(o)


def sparse_walk(Y):
    Ich = Ichar(); Ifloor = (1 - PHI[0]) / (4 * (_SPAN * SS[0]) ** 2)
    active_ax = np.where(Ich >= Ifloor)[0]                      # only these axes vary
    r = active_ax.size
    gap = _GAP * SS[0]
    nu = max(SS[0] ** 2 * (1 - PHI[0] ** 2), 1e-12)
    phi = PHI[0]

    def full_scale(coord):
        sc = np.zeros(D)
        for a, k in enumerate(active_ax):
            sc[k] = coord[a] * gap
        return sc

    # 1-D AR(1) transition weight between lattice coords (absolute scale), mean-reverting
    def Tw(n_from, n_to):
        return math.exp(-0.5 * ((n_to * gap) - phi * (n_from * gap)) ** 2 / nu)

    active = {tuple([0] * r): 1.0}
    m = np.zeros(N); P = np.eye(N) * (LAM.max() + RHO.max()) * N
    out = np.zeros((len(Y), D)); sizes = []
    for t, y in enumerate(Y):
        # --- 1. transition, separable per active axis (+/-1 cell)
        cur = active
        for a in range(r):
            nxt = {}
            for node, w in cur.items():
                for d in (-1, 0, 1):
                    to = list(node); to[a] = node[a] + d; to = tuple(to)
                    nxt[to] = nxt.get(to, 0.0) + w * Tw(node[a], node[a] + d)
            cur = nxt
        # renormalise transition rows lumped: just normalise the whole set
        s = sum(cur.values()); active = {k: v / s for k, v in cur.items()}
        # --- 2. likelihood: KF per node
        nodes = list(active.keys()); pri = np.array([active[k] for k in nodes])
        e = y - H @ m
        lg = np.empty(len(nodes)); ms = np.empty((len(nodes), N)); Ps = np.empty((len(nodes), N, N))
        for i, nd in enumerate(nodes):
            sc = full_scale(nd); Pp = P + Q_of(sc[:N]); S = H @ Pp @ H.T + R_of(sc[N:]) + 1e-9 * np.eye(M)
            Si = np.linalg.inv(S); sgn, ld = np.linalg.slogdet(S)
            lg[i] = -0.5 * (M * _LOG2PI + ld + float(e @ Si @ e))
            Kk = Pp @ H.T @ Si; ms[i] = m + Kk @ e; Ps[i] = Pp - Kk @ H @ Pp
        w = pri * np.exp(lg - lg.max()); w /= w.sum()
        # --- 3. prune
        keep = w >= _EPS * w.max()
        nodes = [nodes[i] for i in range(len(nodes)) if keep[i]]; w = w[keep]; w /= w.sum()
        ms = ms[keep]; Ps = Ps[keep]
        active = {nodes[i]: float(w[i]) for i in range(len(nodes))}
        # --- state GPB1
        m = w @ ms; dm = ms - m
        P = np.einsum("g,gij->ij", w, Ps) + np.einsum("g,gi,gj->ij", w, dm, dm); P = 0.5 * (P + P.T)
        # reported scale = E[psi]
        sc_mean = np.zeros(D)
        for i, nd in enumerate(nodes):
            sc_mean += w[i] * full_scale(nd)
        out[t] = sc_mean
        # --- 4. coverage: spawn +/-1 neighbours of the HIGH-weight survivors, seeded
        # ABOVE the prune floor so they survive one step to catch a migrating regime.
        thresh = 0.1 * max(active.values())
        for nd, wv in list(active.items()):
            if wv < thresh:
                continue
            for a in range(r):
                for d in (-1, 1):
                    to = list(nd); to[a] = nd[a] + d; to = tuple(to)
                    if to not in active:
                        active[to] = 2.0 * _EPS * wv
        s = sum(active.values()); active = {k: v / s for k, v in active.items()}
        sizes.append(len(active))
    return out, np.array(sizes), r


if __name__ == "__main__":
    T = 600; b = slice(T // 3 + 40, 2 * T // 3 - 40); lab = ["xi1", "xi2", "eta1", "eta2"]
    _, sizes, r = sparse_walk(gen(None, 0.0, 120))
    print(f"active axes r = {r};  dense grid would be nodes^r = 5^{r} = {5**r}")
    print(f"sparse active-set size: median {int(np.median(sizes))}, max {int(sizes.max())} "
          f"(after coverage spawn)")
    for ax, nm in [(1, "xi2 hot"), (3, "eta2 hot")]:
        Y = gen(ax, 1.4, T); ref = exact_grid(Y, 5); w, sizes, _ = sparse_walk(Y)
        cr = [np.corrcoef(w[30:, k], ref[30:, k])[0, 1] if w[30:, k].std() > 1e-9 else np.nan for k in range(D)]
        print(f"{nm}: GRID {ref[b].mean(0)}  SPARSE {w[b].mean(0)}  corr {np.array(cr)}  "
              f"|active| med {int(np.median(sizes))}")
    print("\nSTATIC:", sparse_walk(gen(None, 0.0, T))[0][150:].mean(0))
