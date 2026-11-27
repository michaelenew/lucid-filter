"""The dynamics departure as a class with a zero corner -- a load-time variant of the filter.

`load()` returns a module whose `LucidFilter(faults=True, forget=None)`:

  * builds one departure walker per cell (rho, s_g) of DEP_RHOS x DEP_SS, with drift
    s_g^2 rho and cap s_g^2 (the shipped ladder is the row s_g = 1);
  * has no switch between "nominal" and "departed" (`_hazard_mix` is the identity) and no
    restart (`_reprice` does nothing);
  * moves bank weight on the departure plane by the class drift: the zero corner (the nominal
    member) neighbours every cell of the smallest spread; cells neighbour along rho and along
    s_g; rates are 0.5 dt^2 / _DRIFT_LEVEL with t the Whittle arclength of the walker's steady
    gain at drift q = s_g^2 rho, scaled by the star's fraction like any class axis.

Nothing under `lucid/` is modified.  Only the plain layout (nominal + walkers, no anchors).
"""
import math, os, sys, types
import numpy as np
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "lucid", "filter", "lucid.py")
DEP_RHOS = (0.42, 0.10, 0.016, 0.0019, 2e-5)
DEP_SS = (0.125, 0.25, 0.5, 1.0, 2.0)

_OLD = """            for hzj in hazards:
                specs.append((base, F, B, _Departure(const, dirs, hzj, n, self.p)))
"""
_NEW = """            self._dep_cells = [(r_, s_) for s_ in _DEP_SS for r_ in _DEP_RHOS]
            hazards = tuple(r_ for r_, _s in self._dep_cells)
            for hzj, sgj in self._dep_cells:
                dep_ = _Departure(const, dirs, hzj, n, self.p)
                dep_.cap = np.concatenate([np.full(n, np.inf), np.full(dep_.k, sgj * sgj)])
                dep_.q_g = np.full(dep_.k, sgj * sgj * hzj)
                specs.append((base, F, B, dep_))
"""

def _arclen(q):
    q = np.asarray(q, float); P = 0.5 * (q + np.sqrt(q * q + 4.0 * q))
    return np.arccos(np.clip(1.0 - P / (1.0 + P), -1.0, 1.0))

def load():
    src = open(SRC).read()
    assert src.count(_OLD) == 1
    src = src.replace(_OLD, _NEW)
    mod = types.ModuleType("lucid_continuum"); mod.__file__ = SRC
    mod.__dict__["_DEP_RHOS"], mod.__dict__["_DEP_SS"] = DEP_RHOS, DEP_SS
    sys.modules[mod.__name__] = mod
    exec(compile(src, SRC, "exec"), mod.__dict__)
    lse = mod._logsumexp

    class DriftContinuum(mod._ClassDrift):
        def __init__(self, phi_arr, s_arr, ndw, split_arr=None, hazards=None):
            super().__init__(phi_arr, s_arr, ndw, split_arr, None)        # no hazard ladder
            cells = [(r_, s_) for s_ in DEP_SS for r_ in DEP_RHOS]
            self.J = len(cells) if (hazards is not None and ndw == 2 * len(cells)) else 1
            n1 = self.J + 1                                              # state 0 = the zero corner
            Hd = np.zeros((n1, n1))
            if self.J > 1:
                t = _arclen([s_ * s_ * r_ for r_, s_ in cells]); nr = len(DEP_RHOS)
                def link(i, j, dt):
                    Hd[i, j] = Hd[j, i] = 0.5 * dt * dt / mod._DRIFT_LEVEL
                for ks in range(len(DEP_SS)):
                    for kr in range(nr):
                        c = ks * nr + kr
                        if ks == 0: link(0, c + 1, t[c])                 # the zero corner's neighbours
                        if kr + 1 < nr: link(c + 1, c + 2, abs(t[c] - t[c + 1]))
                        if ks + 1 < len(DEP_SS): link(c + 1, c + nr + 1, abs(t[c] - t[c + nr]))
            self._haz = bool(Hd.any())
            self._dlam, self._dV = np.linalg.eigh(Hd - np.diag(Hd.sum(1))); self._dlam = np.minimum(self._dlam, 0.0)
            self.cells = cells

        def _Pd(self, tau):
            return np.maximum((self._dV * np.exp(tau * self._dlam)) @ self._dV.T, 0.0)

        def _P(self, tau, a):
            Pc, Ps, _ = super()._P(tau, a)
            return Pc, Ps, (self._Pd(tau)[1:, 1:] if self._haz else np.ones((1, 1)))

        def _live(self, rate):
            return rate > 0.0 and self.nb and (self.nps > 1 or self._split or self._haz)

        def _chain(self, logp, rate, a):
            if not self._live(rate): return logp
            Pc, Ps, _ = super()._P(rate, a)
            if not self._haz:
                W = np.exp(logp).reshape(1, self.ndw, self.nps, self.nb)
                W = np.einsum("jdpb,pq->jdqb", W, Pc); W = np.einsum("jdpb,bc->jdpc", W, Ps)
            else:
                W = np.exp(logp).reshape(self.J, 2, self.nps, self.nb)
                S = np.concatenate([W[:, 0].sum(0)[None], W[:, 1]], axis=0)       # [zero; cells]
                S = np.einsum("ipb,ik->kpb", S, self._Pd(rate))
                S = np.einsum("ipb,pq->iqb", S, Pc); S = np.einsum("ipb,bc->ipc", S, Ps)
                W = np.zeros_like(W); W[0, 0] = S[0]; W[:, 1] = S[1:]
            out = np.log(np.maximum(W, 1e-300)).ravel()
            return out - lse(out)

    mod._ClassDrift = DriftContinuum
    def _no_switch(self, logw, a=1.0): return logw - lse(logw)
    mod.LucidFilter._hazard_mix = _no_switch
    mod.LucidFilter._reprice = lambda self, spec=None: None
    return mod
