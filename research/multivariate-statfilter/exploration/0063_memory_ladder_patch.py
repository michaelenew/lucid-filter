"""Patch the filter with a MEMORY LADDER: rungs of the bank's weight memory, weight rows sharing the
member filters, each rung carrying its own weight vector and scored by its own mixture predictive
density; the rungs themselves are switching hypotheses and so leak at the derived 1/_LADDER_MEM."""
def patched_source(mems="10,32,100,316,1000,inf"):
    src = open("lucid/filter/lucid.py").read()
    def rep(old, new):
        nonlocal src
        assert old in src, old[:70]
        src = src.replace(old, new, 1)
    rep("        self._logw = np.zeros(self._Ja * self._ndw * self._nc)\n",
        "        self._Jm = len(_MEMS)\n"
        "        self._logw = np.zeros((self._Jm, self._Ja * self._ndw * self._nc))\n"
        "        self._mlogw = np.zeros(self._Jm)\n")
    rep("_LADDER_MEM = 1000.0",
        "_MEMS = tuple(float('inf') if m == 'inf' else float(m) for m in '%s'.split(','))\n"
        "_LADDER_MEM = 1000.0" % mems)
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
            fg = np.array([0.0 if not np.isfinite(m) else (1.0 - 1.0 / m) for m in _MEMS])
            fg = np.where(np.isfinite(np.asarray(_MEMS, float)), fg, 1.0)
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
