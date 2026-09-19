import os
"""Patch the filter with a MEMORY LADDER: rungs of the bank's weight memory, weight rows sharing the
member filters, each rung carrying its own weight vector and scored by its own mixture predictive
density; the rungs themselves are switching hypotheses and so leak at the derived 1/_LADDER_MEM."""
def arc_rungs(mem=1000.0, top=None):
    """The memory ladder in the gain's Whittle arclength: the split ladder's rungs read as memories
    T = 1/K, K = 1 - cos t, t uniform on [0, pi/2] (`top` caps the arclength; None = complete)."""
    import math, numpy as np
    step = 1.5 * math.sqrt(2.0 / mem)
    top = 0.5 * math.pi if top is None else top
    J = int(math.ceil(top / step))
    t = (np.arange(J) + 0.5) * top / J
    return ",".join("%.6g" % (1.0 / (1.0 - math.cos(x))) for x in t)

def patched_source(mems="10,32,100,316,1000,inf", classonly=False):
    tau_mode = mems == "tau"
    if mems.startswith("arc"):
        mems = arc_rungs(top=None if mems == "arc" else float(mems[3:]))
    if tau_mode:
        mems = "inf"     # placeholder; the rungs are computed per filter below
    src = open(os.environ.get("LUCID_SRC", "lucid/filter/lucid.py")).read()
    def rep(old, new):
        nonlocal src
        assert old in src, old[:70]
        src = src.replace(old, new, 1)
    rep("        probe = _WalkEngine(Q0, R0, Hm, F, B, phis[0], ss[0])\n",
        "        probe = _WalkEngine(Q0, R0, Hm, F, B, phis[0], ss[0])\n"
        "        self._mems = _memory_rungs(F, Hm, Q0, R0) if _TAU_MODE else _MEMS\n")
    rep("        self._logw = np.zeros(self._Ja * self._ndw * self._nc)\n",
        "        self._Jm = len(self._mems)\n"
        "        self._logw = np.zeros((self._Jm, self._Ja * self._ndw * self._nc))\n"
        "        self._mlogw = np.zeros(self._Jm)\n")
    rep("_LADDER_MEM = 1000.0",
        "_MEMS = tuple(float('inf') if m == 'inf' else float(m) for m in '%s'.split(','))\n"
        "_LADDER_MEM = 1000.0\n"
        "_CLASSONLY = %s\n"
        "_TAU_MODE = %s\n"
        "def _memory_floor(F, H, Q, R, iters=20000, tol=1e-12):\n"
        "    F = np.eye(len(np.atleast_2d(Q))) if F is None else np.asarray(F, float); H = np.atleast_2d(np.asarray(H, float)); Q = np.atleast_2d(np.asarray(Q, float))\n"
        "    R = np.asarray(R, float); R = np.diag(R) if R.ndim == 1 else R\n"
        "    n = F.shape[0]; P = Q.copy() + np.eye(n)\n"
        "    for _ in range(iters):\n"
        "        Pp = F @ P @ F.T + Q; S = H @ Pp @ H.T + R\n"
        "        K = Pp @ H.T @ np.linalg.solve(S, np.eye(S.shape[0])); Pn = Pp - K @ H @ Pp\n"
        "        if np.max(np.abs(Pn - P)) < tol * (1 + np.max(np.abs(P))): P = Pn; break\n"
        "        P = Pn\n"
        "    r = float(np.max(np.abs(np.linalg.eigvals((np.eye(n) - K @ H) @ F))))\n"
        "    return float('inf') if r >= 1.0 else 1.0 / (1.0 - r)\n"
        "def _memory_rungs(F, H, Q, R):\n"
        "    tau = _memory_floor(F, H, Q, R)\n"
        "    step = 1.5 * math.sqrt(2.0 / _LADDER_MEM)\n"
        "    top = 0.0 if not np.isfinite(tau) else math.acos(1.0 - 1.0 / tau)\n"
        "    J = max(int(math.ceil(top / step)), 1)\n"
        "    t = (np.arange(J) + 0.5) * top / J\n"
        "    out = tuple(float(1.0 / (1.0 - math.cos(x))) if x > 0 else float('inf') for x in t)\n"
        "    print('memory rungs: tau', round(tau, 1), '->', tuple(round(v, 1) for v in out))\n"
        "    return out\n"
        "def _logsumexp_axes(P, axes):\n"
        "    m = P.max(axis=axes, keepdims=True)\n"
        "    return (m + np.log(np.sum(np.exp(P - m), axis=axes, keepdims=True)))\n" % (mems, classonly, tau_mode))
    # the prior: per rung
    rep("""        prior = self._logw - _logsumexp(self._logw)
        prior = self._attribution_mix(prior, a)
        if self._ndbase > 1:
            prior = self._hazard_mix(prior, a)
""","""        prior = np.stack([r - _logsumexp(r) for r in self._logw])
        prior = np.stack([self._attribution_mix(r, a) for r in prior])
        if self._ndbase > 1:
            prior = np.stack([self._hazard_mix(r, a) for r in prior])
""")
    rep("""        llw = np.tile(llv[self._wm], self._Ja)
        if np.any(np.isfinite(yv)):
            bank_ll = _logsumexp(prior + llw)
            # ``forget`` is a memory PER NOMINAL STEP, so over a gap of ``a`` it is
            # ``forget**a`` -- the bank's weight memory is a duration, not a count of events.
            self._logw = (self.forget ** a) * prior + llw
        else:
            bank_ll = 0.0
            self._logw = prior
        post = np.exp(self._logw - _logsumexp(self._logw))
""","""        llw = np.tile(llv[self._wm], self._Ja)
        if np.any(np.isfinite(yv)):
            rung_ll = np.array([_logsumexp(prior[j] + llw) for j in range(self._Jm)])
            fg = np.array([0.0 if not np.isfinite(m) else (1.0 - 1.0 / m) for m in self._mems])
            fg = np.where(np.isfinite(np.asarray(self._mems, float)), fg, 1.0)
            if _CLASSONLY:
                # memory on the CLASS axis only: temper the class-cell marginal, keep the
                # conditional over (hazard rung, departure, copy) -- those carry their own kernels
                nr = len(self.rates)
                L = np.empty_like(prior)
                for j in range(self._Jm):
                    P = prior[j].reshape(self._Ja, self._ndw, nr, self._nc // nr)
                    marg = _logsumexp_axes(P, (0, 1, 2))
                    L[j] = ((fg[j] ** a) * marg + (P - marg)).ravel()
                self._logw = L + llw[None]
            else:
                self._logw = (fg[:, None] ** a) * prior + llw[None]
            # the rungs are SWITCHING hypotheses (which memory is right moves with the regime), so
            # they carry the derived switching kernel at 1/_LADDER_MEM, not a memory of their own
            r = 1.0 / _LADDER_MEM
            k = self._Jm
            if k > 1:
                lam = max(1.0 - r * k / (k - 1.0), 0.0) ** a
                W = np.exp(self._mlogw - self._mlogw.max())
                W = lam * W + (1.0 - lam) * W.mean()
                self._mlogw = np.log(np.maximum(W, 1e-300))
            self._mlogw = self._mlogw - _logsumexp(self._mlogw) + rung_ll
            mw = np.exp(self._mlogw - _logsumexp(self._mlogw))
            self._mw = mw
            bank_ll = _logsumexp(self._mlogw)
        else:
            bank_ll = 0.0
            self._logw = prior
            mw = np.exp(self._mlogw - _logsumexp(self._mlogw)); self._mw = mw
        post = np.einsum("j,jr->r", mw, np.stack([np.exp(r - _logsumexp(r)) for r in self._logw]))
""")
    return src
