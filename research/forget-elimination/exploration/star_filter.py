"""The rate star inside the shipped filter, by source patch (the repository's own pattern for
filter experiments: resolution-criterion/0003).  Nothing in `lucid/` is modified.

`load("star")` returns a module whose `LucidFilter` replaces the bank-weight rule

    prior <- normalise(logw) [-> hazard kernel];   logw <- forget**a * prior + ll

by five copies of the bank's weights, one per node of a walking star in v = ln(rate):

    node k:  prior_k <- normalise(logw_k) [-> hazard kernel] -> (phi, s) chain at rate e^{v + k g}
             logw_k  <- forget**a * prior_k + ll            (forget = 1 unless the caller keeps it)
    lead_k <- max(0, lead_k + lp_k - lp_centre);   node weight  exp(lead_k - 1.125 k^2)
    recentre when the posterior mode leaves the centre;  v <= 0 (the resolvability ceiling)

The chain moves weight between cells that are neighbours in phi or in s at the same split
base, at rates  KL(i, j) / 1.125  from the closed form of 0005 (Whittle, noise level 1/I = 2).
`load("shipped")` returns the unpatched source for paired runs.
"""
import math, os, sys, types
import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "lucid", "filter", "lucid.py")
C_RES = 1.5 ** 2 / 2
K = 2
V0 = 0.0          # where the star starts, in ln(rate fraction); 0 is the ceiling (the derived start)
EPS0 = 2 * math.exp(-2 * math.pi ** 2 / 1.5 ** 2)

def _spacing():
    f = lambda D: (2 * math.pi / D) * 2 * math.sqrt(math.pi * (2 * math.pi / D) / math.sinh(math.pi * 2 * math.pi / D))
    lo, hi = 0.2, 3.0
    for _ in range(200):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if f(mid) < EPS0 else (lo, mid)
    return 0.5 * (lo + hi)
G = _spacing()

_W = (np.arange(4096) + 0.5) * np.pi / 4096
def whittle(a, b, noise=2.0):
    fa = a[1] ** 2 * (1 - a[0] ** 2) / (1 + a[0] ** 2 - 2 * a[0] * np.cos(_W)) + noise
    fb = b[1] ** 2 * (1 - b[0] ** 2) / (1 + b[0] ** 2 - 2 * b[0] * np.cos(_W)) + noise
    r = fa / fb
    return 0.5 * float(np.mean(r - 1 - np.log(r)))

def cell_rates(phi_arr, s_arr):
    """HD (nc x nc): generator of the fastest resolvable drift between neighbouring cells."""
    nc = len(phi_arr)
    phis = list(dict.fromkeys(np.round(phi_arr, 12))); ss = list(dict.fromkeys(np.round(s_arr, 12)))
    nb = nc // (len(phis) * len(ss))
    Hm = np.zeros((nc, nc))
    if nb * len(phis) * len(ss) != nc:
        return Hm
    idx = lambda ip, js, b: (ip * len(ss) + js) * nb + b
    kl = {}
    def rate(p, q):
        key = (p, q) if p <= q else (q, p)
        if key not in kl: kl[key] = 0.5 * (whittle(p, q) + whittle(q, p)) / C_RES
        return kl[key]
    for ip, ph in enumerate(phis):
        for js, sv in enumerate(ss):
            nbrs = [(ip + d, js) for d in (-1, 1) if 0 <= ip + d < len(phis)] + \
                   [(ip, js + d) for d in (-1, 1) if 0 <= js + d < len(ss)]
            for (i2, j2) in nbrs:
                h = rate((ph, sv), (phis[i2], ss[j2]))
                for b in range(nb): Hm[idx(ip, js, b), idx(i2, j2, b)] = h
    tot = Hm.sum(axis=1, keepdims=True)
    Hm = np.where(tot > 0.5, Hm * 0.5 / np.maximum(tot, 1e-300), Hm)
    return Hm - np.diag(Hm.sum(axis=1))

def _lse(a):
    m = float(np.max(a)); return m + math.log(float(np.sum(np.exp(a - m))))

def _star_state(self):
    st = getattr(self, "_star", None)
    if st is None:
        n = 2 * K + 1
        st = dict(lw=np.zeros((n, self._ndw * self._nc)), v=float(V0), W=np.zeros(n), pri=None,
                  HD=cell_rates(self.phi_arr, self.s_arr), moves=0, path=[])
        st["maxleave"] = max(float(-np.diag(st["HD"]).min()), 1e-12)
        self._star = st
    return st

def _star_prior(self, a):
    st = _star_state(self); n = 2 * K + 1; pri = np.empty_like(st["lw"])
    for k in range(n):
        p = st["lw"][k] - _lse(st["lw"][k])
        if self._ndbase > 1:
            p = self._hazard_mix(p, a)
        rate = min(RATE_SCALE * math.exp(min(st["v"] + (k - K) * G, 0.0)) * a, 0.5 / st["maxleave"])
        if rate > 0.0:
            Wm = np.exp(p).reshape(self._ndw, self._nc)
            Wm = Wm + rate * (Wm @ st["HD"])
            p = np.log(np.maximum(Wm, 1e-300)).ravel(); p -= _lse(p)
        pri[k] = p
    st["pri"] = pri
    return pri

def _star_post(self, llw, a, finite):
    st = _star_state(self); n = 2 * K + 1; pri = st["pri"]
    prior_w = -C_RES * (np.arange(n) - K) ** 2
    if not finite:
        st["lw"] = pri.copy()
        lq = st["W"] + prior_w; lq -= _lse(lq)
        return 0.0, np.exp(lq) @ np.exp(pri)
    lp = np.array([_lse(pri[k] + llw) for k in range(n)])
    lq = st["W"] + prior_w; lq -= _lse(lq)
    bank_ll = _lse(lq + lp)
    st["lw"] = (self.forget ** a) * pri + llw[None, :]
    W = np.maximum(st["W"] + lp - lp[K], 0.0); W[K] = 0.0
    sc = W + prior_w; best = int(np.argmax(sc)); d = best - K if sc[best] > 1e-12 else 0
    d = min(d, int(math.floor((1e-9 - st["v"]) / G)))
    if st["v"] + d * G < -17.0: d = 0
    if d != 0:
        src = np.clip(np.arange(n) + d, 0, n - 1)
        st["lw"] = st["lw"][src]; st["v"] += d * G; W = np.zeros(n); st["moves"] += 1
    st["W"] = W; st["path"].append(st["v"])
    lq = W + prior_w; lq -= _lse(lq)
    post = np.zeros(st["lw"].shape[1])
    for k in range(n):
        post += math.exp(lq[k]) * np.exp(st["lw"][k] - _lse(st["lw"][k]))
    if MIX:
        _star_mix(self, post, min(RATE_SCALE * math.exp(min(st["v"], 0.0)) * a, 0.5 / st["maxleave"]))
    return bank_ll, post

_OLD_PRIOR = """        prior = self._logw - _logsumexp(self._logw)
        if self._ndbase > 1:
            prior = self._hazard_mix(prior, a)
"""
_NEW_PRIOR = "        prior = self._star_prior(a)\n"
_OLD_POST = """        if np.any(np.isfinite(yv)):
            bank_ll = _logsumexp(prior + llw)
            # ``forget`` is a memory PER NOMINAL STEP, so over a gap of ``a`` it is
            # ``forget**a`` -- the bank's weight memory is a duration, not a count of events.
            self._logw = (self.forget ** a) * prior + llw
        else:
            bank_ll = 0.0
            self._logw = prior
        post = np.exp(self._logw - _logsumexp(self._logw))
"""
_NEW_POST = "        bank_ll, post = self._star_post(llw, a, bool(np.any(np.isfinite(yv))))\n"
_OLD_RESET = "        self._logw = np.zeros(self._ndw * self._nc)\n"
_NEW_RESET = _OLD_RESET + "        self._star = None\n"

_OLD_MIX = "        mean = pm @ mn\n"
_NEW_MIX = _OLD_MIX + "        if getattr(self, '_dbg', None) is not None: self._dbg(self, post, mn)\n"
RATE_SCALE = 1.0     # multiplies the ceiling rates (diagnostics only; 1 is the derived value)
MIX = ()             # which member state the chain carries with the weight it moves (0009):
                     # any of "m" (state mean and covariance), "mu" (scale-walk centres and their
                     # variance).  Empty = weights only, as in 0007.

def _star_mix(self, post, rate):
    """Interacting-multiple-model step: member j's state becomes the mixture of the states of
    the members its weight just came from, mu_{i|j} = post_i P_ij / sum_i post_i P_ij, with
    P = I + rate * HD the chain at the star's centre.  Implemented for the case of one stacked
    bank whose rows are the cells (no split ladder, no fault class) -- the arm."""
    if not MIX or len(self._banks) != 1 or self._ndw != 1 or len(self._wm) != self._nc:
        return
    bank = self._banks[0]
    if not hasattr(bank, "_m") or bank._m.shape[0] != self._nc:
        return
    st = self._star
    P = np.eye(self._nc) + rate * st["HD"]
    Mx = post[:, None] * P                                   # [i, j]: weight arriving at j from i
    Mx = (Mx / np.maximum(Mx.sum(axis=0, keepdims=True), 1e-300)).T      # [j, i]
    if "m" in MIX:
        m, Pc = bank._m.copy(), bank._P.copy()
        mn = Mx @ m; d = m[None, :, :] - mn[:, None, :]      # [j, i, n]
        bank._m[...] = mn
        bank._P[...] = np.einsum("ji,iab->jab", Mx, Pc) + np.einsum("ji,jia,jib->jab", Mx, d, d)
    if "mu" in MIX:
        mu, Pm = bank.mu.copy(), bank._Pmu.copy()
        mun = Mx @ mu; d = mu[None, :, :] - mun[:, None, :]
        bank.mu[...] = mun
        bank._Pmu[...] = np.minimum(Mx @ Pm + np.einsum("ji,jid->jd", Mx, d * d), bank._Pmu_cap)

def load(variant, debug=False):
    """`debug=True` adds a no-op-unless-set hook `self._dbg(self, post, member_means)` after the
    bank's mixing step; it does not change any output."""
    src = open(SRC).read()
    if debug:
        assert src.count(_OLD_MIX) == 1
        src = src.replace(_OLD_MIX, _NEW_MIX)
    if variant == "star":
        for old, new in ((_OLD_PRIOR, _NEW_PRIOR), (_OLD_POST, _NEW_POST), (_OLD_RESET, _NEW_RESET)):
            assert src.count(old) == 1, old
            src = src.replace(old, new)
    mod = types.ModuleType("lucid_" + variant); mod.__file__ = SRC
    sys.modules[mod.__name__] = mod
    exec(compile(src, SRC, "exec"), mod.__dict__)
    if variant == "star":
        mod.LucidFilter._star_prior = _star_prior
        mod.LucidFilter._star_post = _star_post
    return mod
