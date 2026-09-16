"""v5 = the branch's floor copy + IMM state mixing across the copies at the switching ladder's posterior-mean rate.
Per class cell, after the weight update:  x_i <- [(1-r) w_i x_i + r/(k-1) sum_{j!=i} w_j x_j] / [(1-r) w_i + r/(k-1) sum_{j!=i} w_j]
(the symmetric kernel's mixing probabilities, r the posterior-mean switching rate), P likewise with the spread term."""
import numpy as np
def patched_source(J=None):
    src = open("lucid/filter/lucid.py").read()
    if J is not None:
        o = "        self.rates = (1.0, 1.0 / _mem)\n"; assert o in src
        src = src.replace(o, "        self.rates = tuple(float(r) for r in np.geomspace(1.0, 1.0 / _mem, %d))\n" % J)
    o = "        post, patience, att_rate = self._att_marginal(post)       # over the attribution rungs\n"; assert o in src
    src = src.replace(o, o + '''        self._pm_full = post.copy(); self._llv = llv.copy(); self._last_patience, self._last_switch = patience, att_rate
        if self._nspec == 1 and len(self.rates) > 1 and att_rate > 0.0:
            # IMM mixing across the copies of each class cell at the switching rate the ladder reads
            k = len(self.rates); nc0 = M // k; r = att_rate
            Wc = post.reshape(k, nc0)                                       # weights, copy x cell
            Xc = mn.reshape(k, nc0, n); Vc = vr.reshape(k, nc0, n, n)
            tot = Wc.sum(0, keepdims=True); oth = (tot - Wc) * (r / (k - 1.0)); own = (1.0 - r) * Wc
            den = own + oth
            sx = np.einsum("kc,kci->ci", Wc, Xc)                            # sum_j w_j x_j
            Xm = (own[:, :, None] * Xc + (r / (k - 1.0)) * (sx[None] - Wc[:, :, None] * Xc)) / np.maximum(den, 1e-300)[:, :, None]
            sv = np.einsum("kc,kcij->cij", Wc, Vc + np.einsum("kci,kcj->kcij", Xc, Xc))
            Vm = (own[:, :, None, None] * (Vc + np.einsum("kci,kcj->kcij", Xc, Xc)) + (r / (k - 1.0)) * (sv[None] - Wc[:, :, None, None] * (Vc + np.einsum("kci,kcj->kcij", Xc, Xc)))) / np.maximum(den, 1e-300)[:, :, None, None] - np.einsum("kci,kcj->kcij", Xm, Xm)
            for j, f in enumerate(self._members):
                kk, cc = j // nc0, j % nc0
                if den[kk, cc] > 1e-300:
                    f._m[:] = Xm[kk, cc]; f._P[:] = 0.5 * (Vm[kk, cc] + Vm[kk, cc].T)
            mn[:] = Xm.reshape(M, n); vr[:] = Vm.reshape(M, n, n)
''')
    return src
