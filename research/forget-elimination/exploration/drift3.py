"""A star per axis, beneath grid-free ceilings -- a probe-side replacement for `_ClassDrift`.

    class  (phi, s) : generator G_c at the resolvability ceiling (as in the filter); fraction e^{v_c}
    split ladder    : diffusion in Whittle arclength, hop rate D / dt^2;  D = D_max e^{v_s},  D_max = L^4 / 144
    hazard ladder   : the same on its own arclength;                      D = D_max e^{v_h}

`L` is the ladder's range (pi/2, pi/3) and `D_max` the diffusion at which the steady blur
sqrt(D / I), I = 1 per step, equals the variance L^2 / 12 of knowing nothing.

The three fractions are tracked on a caltrop star: a centre and K arms either way along each
live axis, each node a full copy of the bank's weights.  A node's lead is floored at zero,
its prior log-weight is -1.125 k^2, and at most one axis recentres per step.

    use:  with drift3.installed():  f = LucidFilter(forget=None, ...)
"""
import contextlib, math
import numpy as np
from lucid.filter import lucid as L
_logsumexp = L._logsumexp
GAP, LEVEL, KARM, VMIN = L._DRIFT_GAP, L._DRIFT_LEVEL, L._DRIFT_K, L._DRIFT_VMIN
START = (0.0, 0.0, 0.0)            # where the three walks start (ln fraction); 0 = each ceiling

def _arclen_from_gain(K): return np.arccos(np.clip(1.0 - K, -1.0, 1.0))

class Drift3(L._ClassDrift):
    def __init__(self, phi_arr, s_arr, ndw, split_arr=None, hazards=None):
        super().__init__(phi_arr, s_arr, ndw, split_arr, hazards)
        nb = self.nb
        # ---- split ladder: unit-diffusion generator in arclength
        Ls = np.zeros((nb, nb)); self.Dmax_s = 0.0
        sp = None if split_arr is None else np.atleast_2d(np.asarray(split_arr, float))
        if nb > 1 and sp is not None and sp.shape[0] == nb and sp.shape[1] > 0:
            od = np.exp(sp); t = _arclen_from_gain(0.5 * (-od + np.sqrt(od * od + 4.0 * od)))
            for g in range(sp.shape[1]):
                others = np.delete(np.arange(sp.shape[1]), g)
                on = sorted([b for b in range(nb) if np.allclose(sp[b, others], sp[0, others])], key=lambda b: sp[b, g])
                for b0, b1 in zip(on[:-1], on[1:]):
                    dt = abs(t[b1, g] - t[b0, g])
                    if dt > 1e-12: Ls[b0, b1] = Ls[b1, b0] = 1.0 / dt ** 2
            self.Dmax_s = (0.5 * math.pi) ** 4 / 144.0
        self._slam, self._sV = np.linalg.eigh(Ls - np.diag(Ls.sum(1))); self._slam = np.minimum(self._slam, 0.0)
        # ---- hazard ladder
        Lj = np.zeros((self.J, self.J)); self.Dmax_h = 0.0
        if self.J > 1:
            hz = np.atleast_1d(np.asarray(hazards, float)); Pm = 0.5 * (hz + np.sqrt(hz * hz + 4.0 * hz))
            t = _arclen_from_gain(Pm / (1.0 + Pm)); order = np.argsort(t)
            for j0, j1 in zip(order[:-1], order[1:]):
                dt = abs(t[j1] - t[j0])
                if dt > 1e-12: Lj[j0, j1] = Lj[j1, j0] = 1.0 / dt ** 2
            self.Dmax_h = (math.pi / 3.0) ** 4 / 144.0
        self._jlam, self._jV = np.linalg.eigh(Lj - np.diag(Lj.sum(1))); self._jlam = np.minimum(self._jlam, 0.0)
        self._split, self._haz = bool(Ls.any()), bool(Lj.any())
        live = [self.nps > 1 and bool(self.gen.any()), self._split, self._haz]
        self.axes = [ax for ax in range(3) if live[ax]]
        nodes = [(0, 0, 0)]
        for ax in self.axes:
            for k in range(-KARM, KARM + 1):
                if k: nodes.append(tuple(k if i == ax else 0 for i in range(3)))
        self.nodes = np.array(nodes); self.index = {n: i for i, n in enumerate(nodes)}
        self.node_prior = -LEVEL * (self.nodes ** 2).sum(1).astype(float)
        self.reset()

    def reset(self):
        if not hasattr(self, "nodes"):                 # the parent's __init__ calls reset first
            return super().reset()
        self.lw = np.zeros((len(self.nodes), self.ndw * self.nc))
        self.v3 = np.array(START, float); self.v = float(self.v3[0])
        self.lead = np.zeros(len(self.nodes)); self._pri = None; self.moves = np.zeros(3, int); self.path = []

    def _Ps(self, v, a):
        """The three transition matrices at log-fractions v (length 3) over elapsed a."""
        f = np.exp(np.minimum(v, 0.0))
        Pc = np.maximum((self._V * np.exp(f[0] * a * self._lam)) @ self._V.T, 0.0)
        Ps = np.maximum((self._sV * np.exp(self.Dmax_s * f[1] * a * self._slam)) @ self._sV.T, 0.0)
        Pj = np.maximum((self._jV * np.exp(self.Dmax_h * f[2] * a * self._jlam)) @ self._jV.T, 0.0)
        return Pc, Ps, Pj

    def _chain3(self, logp, P3):
        Pc, Ps, Pj = P3
        W = np.exp(logp).reshape(self.J, self.ndw // self.J, self.nps, self.nb)
        W = np.einsum("jdpb,jk->kdpb", W, Pj); W = np.einsum("jdpb,pq->jdqb", W, Pc); W = np.einsum("jdpb,bc->jdpc", W, Ps)
        out = np.log(np.maximum(W, 1e-300)).ravel()
        return out - _logsumexp(out)

    def prior(self, flt, a):
        pri = np.empty_like(self.lw)
        for i, nd in enumerate(self.nodes):
            p = self.lw[i] - _logsumexp(self.lw[i])
            if flt._ndbase > 1: p = flt._hazard_mix(p, a)
            pri[i] = self._chain3(p, self._Ps(self.v3 + nd * GAP, a)) if (a > 0 and self.axes) else p
        self._pri = pri
        return pri

    def post(self, flt, llw, a, finite):
        pri, n = self._pri, len(self.nodes)
        lq = self.lead + self.node_prior; lq = lq - _logsumexp(lq)
        if not finite:
            self.lw = pri.copy()
            return 0.0, np.exp(lq) @ np.exp(pri)
        lp = np.array([_logsumexp(pri[i] + llw) for i in range(n)])
        bank_ll = _logsumexp(lq + lp)
        self.lw = pri + llw[None, :]
        lead = np.maximum(self.lead + lp - lp[0], 0.0); lead[0] = 0.0
        sc = lead + self.node_prior
        best = int(np.argmax(sc))
        if best and sc[best] > 1e-12:
            ax = int(np.flatnonzero(self.nodes[best])[0]); d = int(self.nodes[best][ax])
            d = min(d, int(math.floor((1e-9 - self.v3[ax]) / GAP)))               # the ceiling
            if self.v3[ax] + d * GAP < VMIN: d = 0
            if d:
                old = self.lw.copy()
                def at(k):                                                       # old node at offset k on this axis
                    k = int(np.clip(k, -KARM, KARM))
                    return old[0] if k == 0 else old[self.index[tuple(k if i == ax else 0 for i in range(3))]]
                self.lw[0] = at(d)
                for k in range(-KARM, KARM + 1):
                    if k: self.lw[self.index[tuple(k if i == ax else 0 for i in range(3))]] = at(k + d)
                self.v3[ax] += d * GAP; lead = np.zeros(n); self.moves[ax] += 1
        self.lead = lead; self.v = float(self.v3[0]); self.path.append(self.v3.copy())
        lq = lead + self.node_prior; lq = lq - _logsumexp(lq)
        post = np.zeros(self.lw.shape[1])
        for i in range(n): post += math.exp(lq[i]) * np.exp(self.lw[i] - _logsumexp(self.lw[i]))
        if a > 0 and self.axes: self._carry3(flt, post, self._Ps(self.v3, a))
        return bank_ll, post

    def _carry3(self, flt, post, P3):
        if self._where is None:
            self._where, self._plan = {}, {}
            for bi, bank in enumerate(flt._banks):
                for row, gi in enumerate(np.asarray(bank.idx)): self._where[int(gi)] = (bi, row)
        pm = np.bincount(flt._wm, weights=post, minlength=len(flt._members))
        Pc, Ps, Pj = P3; one = np.ones((1, 1))
        for d in range(flt._nb): self._mix(flt, (d,), pm, one, Pc, Ps)
        if flt._learn:
            specs = tuple(flt._nb + jj for jj in range(flt._J))
            self._mix(flt, specs, pm, Pj if self.J == len(specs) else np.eye(len(specs)), Pc, Ps)

@contextlib.contextmanager
def installed(start=None):
    global START
    keep, keep_start = L._ClassDrift, START
    if start is not None: START = tuple(start)
    L._ClassDrift = Drift3
    try: yield
    finally: L._ClassDrift, START = keep, keep_start


class DriftImplied(L._ClassDrift):
    """The split's diffusion read off each cell's own class (0021).

    A split is the difference of two log-scales; class (phi, s) asserts each moves with
    innovation variance s^2 (1 - phi^2) per step, so the split's log-odds diffuses with
    D(phi, s) = 2 s^2 (1 - phi^2).  Hop rate D / dlo^2 between neighbouring rungs, per cell.
    The class star and the hazard chain are the parent's, unchanged.
    """
    def __init__(self, phi_arr, s_arr, ndw, split_arr=None, hazards=None):
        super().__init__(phi_arr, s_arr, ndw, split_arr, hazards)
        nb = self.nb
        phis = list(dict.fromkeys(np.round(np.asarray(phi_arr, float), 12).tolist()))
        ss = list(dict.fromkeys(np.round(np.asarray(s_arr, float), 12).tolist()))
        self.Dcell = np.array([2.0 * sv * sv * (1.0 - ph * ph) for ph in phis for sv in ss])       # (nps,)
        Ls = np.zeros((nb, nb))
        sp = None if split_arr is None else np.atleast_2d(np.asarray(split_arr, float))
        if nb > 1 and sp is not None and sp.shape[0] == nb and sp.shape[1] > 0:
            for g in range(sp.shape[1]):
                others = np.delete(np.arange(sp.shape[1]), g)
                on = sorted([b for b in range(nb) if np.allclose(sp[b, others], sp[0, others])], key=lambda b: sp[b, g])
                for b0, b1 in zip(on[:-1], on[1:]):
                    dlo = abs(sp[b1, g] - sp[b0, g])
                    if dlo > 1e-12: Ls[b0, b1] = Ls[b1, b0] = 1.0 / dlo ** 2
        self._split = bool(Ls.any())
        self._ulam, self._uV = np.linalg.eigh(Ls - np.diag(Ls.sum(1))); self._ulam = np.minimum(self._ulam, 0.0)

    def _Pcell(self, a):
        """(nps, nb, nb): each class's split transition over elapsed a, [from, to]."""
        E = np.exp(a * self.Dcell[:, None] * self._ulam[None, :])                 # (nps, nb)
        return np.maximum(np.einsum("be,pe,ce->pbc", self._uV, E, self._uV), 0.0)

    def _chain(self, logp, rate, a):
        if not self._live(rate): return logp
        Pc, _, Pj = self._P(rate, a)
        W = np.exp(logp).reshape(self.J, self.ndw // self.J, self.nps, self.nb)
        if self._split: W = np.einsum("jdpb,pbc->jdpc", W, self._Pcell(a))
        W = np.einsum("jdpb,jk->kdpb", W, Pj); W = np.einsum("jdpb,pq->jdqb", W, Pc)
        out = np.log(np.maximum(W, 1e-300)).ravel()
        return out - _logsumexp(out)

    def _mix(self, flt, specs, pm, Pj, Pc, Ps):
        plan = self._group(flt, specs)
        if plan is None: return
        gids, parts = plan; nps, nb, S, N = self.nps, self.nb, len(specs), len(gids)
        if not all(isinstance(b, L._EngineBank) for b, _, _ in parts):
            return super()._mix(flt, specs, pm, Pj, Pc, Ps)
        na = parts[0][0]._m.shape[1]
        if any(b._m.shape[1] != na for b, _, _ in parts): return
        m = np.empty((N, na)); Pm = np.empty((N, na, na))
        for bank, rows, slots in parts: m[slots], Pm[slots] = bank._m[rows], bank._P[rows]
        second = (Pm + m[:, :, None] * m[:, None, :]).reshape(S, nps, nb, na * na); m4 = m.reshape(S, nps, nb, na)
        w = pm[gids].reshape(S, nps, nb); Pcell = self._Pcell(self._a_now) if self._split else None
        def push(x):
            if Pcell is not None: x = np.einsum("jpb...,pbc->jpc...", x, Pcell)
            x = np.einsum("jpb...,jk->kpb...", x, Pj)
            return np.einsum("jpb...,pq->jqb...", x, Pc)
        tot = push(w); dead = tot <= 1e-300
        if dead.all(): return
        den = np.where(dead, 1.0, tot)[..., None]
        mn = np.where(dead[..., None], m4, push(w[..., None] * m4) / den).reshape(N, na)
        Pn = np.where(dead[..., None], second, push(w[..., None] * second) / den).reshape(N, na, na)
        Pn = Pn - mn[:, :, None] * mn[:, None, :]; Pn = 0.5 * (Pn + np.swapaxes(Pn, 1, 2))
        for bank, rows, slots in parts: bank._m[rows] = mn[slots]; bank._P[rows] = Pn[slots]

    def _carry(self, flt, post, rate, a):
        self._a_now = a
        return super()._carry(flt, post, rate, a)

@contextlib.contextmanager
def implied():
    keep = L._ClassDrift; L._ClassDrift = DriftImplied
    try: yield
    finally: L._ClassDrift = keep
