"""The memory ladder ALONE on main's source (08b925b) -- no attribution grid, no switching-rate change; the two
structural `forget` reads moved to the node budget (bit-identical at the default).  Writes the patched source.
    python 0066_memory_ladder_on_main.py <main lucid.py, e.g. from git show 08b925b:lucid/filter/lucid.py> <out.py>"""
import sys
src = open(sys.argv[1]).read()
def rep(old, new, count=1):
    global src
    assert src.count(old) == (count if count else src.count(old)) and src.count(old) >= 1, "MISSING/AMBIGUOUS: " + old[:80]
    src = src.replace(old, new)
rep('''_HAZARDS = _hazard_rungs()           # the default hazard box: 16 rungs, 0.42 down to 2.9e-7, complete to 0
''', '''_HAZARDS = _hazard_rungs()           # the default hazard box: 16 rungs, 0.42 down to 2.9e-7, complete to 0


# The MEMORY LADDER (multivariate-statfilter 0063/0064): the bank's weight memory as a nuisance the
# evidence weights, from pure Bayes down to the nominal model's own state memory.
def _memory_floor(F, H, Q, R, iters=20000, tol=1e-12):
    F = np.asarray(F, float)
    H = np.atleast_2d(np.asarray(H, float))
    Q = np.atleast_2d(np.asarray(Q, float))
    R = np.asarray(R, float)
    R = np.diag(R) if R.ndim == 1 else R
    n = F.shape[0]
    P = Q + np.eye(n)
    K = np.zeros((n, H.shape[0]))
    I = np.eye(n)
    for _ in range(iters):
        Pp = F @ P @ F.T + Q
        S = H @ Pp @ H.T + R
        K = Pp @ H.T @ np.linalg.solve(S, np.eye(S.shape[0]))
        A = I - K @ H
        Pn = A @ Pp @ A.T + K @ R @ K.T
        Pn = 0.5 * (Pn + Pn.T)
        if np.max(np.abs(Pn - P)) < tol * (1.0 + np.max(np.abs(P))):
            break
        P = Pn
    r = float(np.max(np.abs(np.linalg.eigvals((np.eye(n) - K @ H) @ F))))
    return math.inf if r >= 1.0 else 1.0 / (1.0 - r)


def _memory_rungs(F, H, Q, R, mem=None):
    mem = _LADDER_MEM if mem is None else mem
    tau = _memory_floor(F, H, Q, R)
    if not math.isfinite(tau):
        return (math.inf,)
    step = _GAP_FACTOR * math.sqrt(2.0 / mem)
    top = math.acos(1.0 - 1.0 / tau)
    J = max(int(math.ceil(top / step)), 1)
    t = (np.arange(J) + 0.5) * top / J
    return tuple(float(1.0 / (1.0 - math.cos(x))) for x in t)
''')
rep("                 phis=_PHIS, ss=_SS, forget=0.999, timestep=1.0):", "                 phis=_PHIS, ss=_SS, timestep=1.0):")
rep('''        if not 0.0 < forget <= 1.0:
            raise ValueError("forget must lie in (0, 1]")
''', '')
rep("        self.forget = float(forget)\n", "")
rep('''        probe = _WalkEngine(Q0, R0, Hm, F, B, phis[0], ss[0])
''', '''        probe = _WalkEngine(Q0, R0, Hm, F, B, phis[0], ss[0])
        self.memories = _memory_rungs(np.eye(n) if F is None else F, Hm, Q0, R0)
        self._Jm = len(self.memories)
        self._fgm = np.array([0.0 if not math.isfinite(T) else 1.0 - 1.0 / T for T in self.memories])
        self._fgm = np.where(np.isfinite(np.asarray(self.memories, float)), self._fgm, 1.0)
        self._Kmem = np.array([0.0 if not math.isfinite(T) else 1.0 / T for T in self.memories])
''')
rep('''        self._logw = np.zeros(self._ndw * self._nc)
''', '''        self._logw = np.zeros((self._Jm, self._ndw * self._nc))
        self._mlogw = np.zeros(self._Jm)
''')
rep('''        prior = self._logw - _logsumexp(self._logw)
        if self._ndbase > 1:
            prior = self._hazard_mix(prior, a)
''', '''        prior = np.stack([r - _logsumexp(r) for r in self._logw])
        if self._ndbase > 1:
            prior = np.stack([self._hazard_mix(r, a) for r in prior])
''')
rep('''        llw = llv[self._wm]
        if np.any(np.isfinite(yv)):
            bank_ll = _logsumexp(prior + llw)
            # ``forget`` is a memory PER NOMINAL STEP, so over a gap of ``a`` it is
            # ``forget**a`` -- the bank's weight memory is a duration, not a count of events.
            self._logw = (self.forget ** a) * prior + llw
        else:
            bank_ll = 0.0
            self._logw = prior
        post = np.exp(self._logw - _logsumexp(self._logw))
''', '''        llw = llv[self._wm]
        if np.any(np.isfinite(yv)):
            rung_ll = np.array([_logsumexp(prior[j] + llw) for j in range(self._Jm)])
            self._logw = (self._fgm[:, None] ** a) * prior + llw[None]
            k = self._Jm
            if k > 1:
                lam = max(1.0 - k / ((k - 1.0) * _LADDER_MEM), 0.0) ** a
                W = np.exp(self._mlogw - self._mlogw.max())
                W = lam * W + (1.0 - lam) * W.mean()
                self._mlogw = np.log(np.maximum(W, 1e-300))
            self._mlogw = self._mlogw - _logsumexp(self._mlogw) + rung_ll
            bank_ll = _logsumexp(self._mlogw)
        else:
            bank_ll = 0.0
            self._logw = prior
        mw = np.exp(self._mlogw - _logsumexp(self._mlogw))
        post = mw @ np.stack([np.exp(r - _logsumexp(r)) for r in self._logw])
        kbar = float(mw @ self._Kmem)
        self.memory = math.inf if kbar <= 0.0 else 1.0 / kbar
''')
# the two structural reads 0063 moved to the node budget (bit-identical at the default 0.999)
rep("            mem = 1.0 / max(1.0 - self.forget, 1e-12)\n", "            mem = _LADDER_MEM\n")
rep("        self.split_arr = _split_star(np.log(_rung_odds(self.forget)), len(self.groups))", "        self.split_arr = _split_star(np.log(_rung_odds(1.0)), len(self.groups))")
open(sys.argv[2], "w").write(src)
print("wrote", sys.argv[2])
