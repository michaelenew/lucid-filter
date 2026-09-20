"""Stage A of the LAG-KERNEL form of patience: every class cell run once per kernel MODE; a member of
mode m walks its scales on the kernel-weighted score over a trailing buffer of its own recent scores,
the kernel a gamma in the lag with its mode at m (m = 0: today's walk, the current score only).
    python port_kernel.py <main lucid.py> <out.py> "0,1,2" [chi|gam2]"""
import sys
src = open(sys.argv[1]).read(); modes = sys.argv[3]; shape = sys.argv[4] if len(sys.argv) > 4 else "chi"
def rep(old, new, count=1):
    global src
    assert src.count(old) == count, "MISSING/AMBIGUOUS (%d): %s" % (src.count(old), old[:80])
    src = src.replace(old, new)
rep('''_HAZARDS = _hazard_rungs()           # the default hazard box: 16 rungs, 0.42 down to 2.9e-7, complete to 0
''', '''_HAZARDS = _hazard_rungs()           # the default hazard box: 16 rungs, 0.42 down to 2.9e-7, complete to 0

_MODES = tuple(int(v) for v in "%s".split(","))
_KSHAPE = "%s"
_KTAIL = 3.1e-4          # the aliasing tolerance: the buffer ends where the kernel's tail mass is below it


def _lag_kernel(mode):
    """Weights over the lag l = 0, 1, 2, ... for a member of kernel mode ``mode``: the current score
    only at mode 0; otherwise a gamma in the lag with its mode there -- chi-squared (scale 2,
    shape 1 + mode/2) or shape 2 (scale = mode) -- normalised, cut where its tail mass is < _KTAIL."""
    if mode <= 0:
        return np.array([1.0])
    if _KSHAPE == "chi":
        a, b = 1.0 + mode / 2.0, 2.0
    else:
        a, b = 2.0, float(mode)
    l = np.arange(0, 4000, dtype=float)
    w = np.where(l > 0, l ** (a - 1.0) * np.exp(-l / b), 0.0)
    w = w / w.sum()
    tail = 1.0 - np.cumsum(w)
    N = int(np.argmax(tail < _KTAIL)) + 1
    w = w[:N]
    return w / w.sum()
''' % (modes, shape))
# the members carry their mode; the stacked bank builds the kernel table and the score buffers
rep('''        self.mu, self._Pmu = st("mu"), st("_Pmu")
        self._m = np.zeros((M, self.n))
''', '''        self.mu, self._Pmu = st("mu"), st("_Pmu")
        self._wmode = np.array([int(getattr(f, "_wmode", 0)) for f in members])
        kers = [_lag_kernel(int(mm)) for mm in self._wmode]
        self._Nk = max(len(k) for k in kers)
        self._WK = np.zeros((M, self._Nk))
        for j, k in enumerate(kers):
            self._WK[j, :len(k)] = k
        self._gbuf = np.zeros((len(self._act), self._Nk, M))
        self._ibuf = np.zeros((len(self._act), self._Nk, M))
        self._nfill = np.zeros(len(self._act), int)
        self._m = np.zeros((M, self.n))
''')
rep('''            info = np.einsum("bg,bg->b", pi[:, ax], info_g) + _RIDGE
            grad = np.einsum("bg,bg->b", pi[:, ax], score)
            Kmu = self._Pmu[:, k] / (self._Pmu[:, k] + 1.0 / info)
''', '''            info = np.einsum("bg,bg->b", pi[:, ax], info_g) + _RIDGE
            grad = np.einsum("bg,bg->b", pi[:, ax], score)
            if self._Nk > 1:
                # the lag-kernel walk: the step is taken on the kernel-weighted score and
                # information over this axis's trailing buffer (mode-0 members read lag 0 only)
                self._gbuf[ax, 1:] = self._gbuf[ax, :-1]; self._gbuf[ax, 0] = grad
                self._ibuf[ax, 1:] = self._ibuf[ax, :-1]; self._ibuf[ax, 0] = info
                nf = int(min(self._nfill[ax] + 1, self._Nk)); self._nfill[ax] = nf
                Wn = self._WK[:, :nf]
                mass = Wn.sum(1, keepdims=True)
                # until the buffer reaches the kernel's support, walk on the current score
                Wn = np.where(mass > 1e-9, Wn / np.maximum(mass, 1e-300), np.eye(1, nf)[0][None, :])
                grad = np.einsum("bl,lb->b", Wn, self._gbuf[ax, :nf])
                info = np.einsum("bl,lb->b", Wn, self._ibuf[ax, :nf])
            Kmu = self._Pmu[:, k] / (self._Pmu[:, k] + 1.0 / info)
''')
# the cells: one per mode
rep('''        self.phi_arr = np.array([ph for ph in phis for _ in ss for _ in bases], float)
        self.s_arr = np.array([sv for _ in phis for sv in ss for _ in bases], float)
        cells = [(ph, sv, bq, br) for ph in phis for sv in ss for (bq, br) in bases]
''', '''        self.phi_arr = np.array([ph for ph in phis for _ in ss for _ in bases], float)
        self.s_arr = np.array([sv for _ in phis for sv in ss for _ in bases], float)
        cells = [(ph, sv, bq, br) for ph in phis for sv in ss for (bq, br) in bases]
        self.modes = _MODES
        cells = [(ph, sv, bq, br, mm) for mm in _MODES for (ph, sv, bq, br) in cells]
        self.phi_arr = np.tile(self.phi_arr, len(_MODES))
        self.s_arr = np.tile(self.s_arr, len(_MODES))
''')
rep('''                for ph, sv, bq, br in cells:
                    e = _WalkEngine(bq, br, Hm, Fs, Bs, ph, sv, fisher_Si=Si_c, prop=pr)
                    Si_c = e._fisher_Si
''', '''                for ph, sv, bq, br, mm in cells:
                    e = _WalkEngine(bq, br, Hm, Fs, Bs, ph, sv, fisher_Si=Si_c, prop=pr)
                    e._wmode = mm
                    Si_c = e._fisher_Si
''')
rep('''            for ph, sv, bq, br in cells:   # the split rides into the augmentation with the base
                Qa, Ra, _Ha, _w = dep.augment(bq, br, Hm)
                e = _WalkEngine(Qa, Ra, Ha, Fa, Ba, ph, sv, walk_axes=walk, cap=dep.cap,
                                fisher_Si=Si_c)
''', '''            for ph, sv, bq, br, mm in cells:   # the split rides into the augmentation with the base
                Qa, Ra, _Ha, _w = dep.augment(bq, br, Hm)
                e = _WalkEngine(Qa, Ra, Ha, Fa, Ba, ph, sv, walk_axes=walk, cap=dep.cap,
                                fisher_Si=Si_c)
                e._wmode = mm
''')
# the readout: weight on the modes above 0
rep('''        pm = np.bincount(self._wm, weights=post, minlength=M)      # member marginals
''', '''        self.patience = float(post.reshape(self._ndw, len(_MODES), -1)[:, 1:, :].sum())
        pm = np.bincount(self._wm, weights=post, minlength=M)      # member marginals
''')
open(sys.argv[2], "w").write(src); print("wrote", sys.argv[2], "modes", modes, "shape", shape)
