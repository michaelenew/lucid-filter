"""A POSTERIOR floor on each element of the memory axis (the attribution copies): after the weight update the
rung marginals are floored at eps and the floor is written back into the log-weights, so no single step can
take an element below eps.  RATES sets the elements of the axis.
    python port_floor.py <grid lucid.py> <out.py> <eps> [rates, e.g. "1,0.001" or "1,0.1,0.01,0.001"]"""
import sys
src = open(sys.argv[1]).read(); eps = float(sys.argv[2 + 1]); rates = sys.argv[4] if len(sys.argv) > 4 else "1,0.001"
def rep(old, new):
    global src
    assert src.count(old) == 1, old[:80]; src = src.replace(old, new)
rep("        self.rates = (1.0, 1.0 / _LADDER_MEM)\n", "        self.rates = tuple(float(v) for v in '%s'.split(','))\n" % rates)
rep("        post = np.exp(self._logw - _logsumexp(self._logw))\n        post, patience, att_rate = self._att_marginal(post)\n",
    """        post = np.exp(self._logw - _logsumexp(self._logw))
        if _EPS_FLOOR > 0.0:
            # the posterior floor on the memory axis: no element below _EPS_FLOOR after any one step
            nr = len(self.rates)
            P = post.reshape(self._Ja, self._ndw, nr, -1)
            marg = P.sum(axis=(0, 1, 3))                     # (nr,) the rung marginals
            tgt = np.maximum(marg, _EPS_FLOOR); tgt = tgt / tgt.sum()
            for rr in range(nr):
                if marg[rr] > 0.0:
                    P[:, :, rr, :] *= tgt[rr] / marg[rr]
                else:
                    P[:, :, rr, :] = tgt[rr] / P[:, :, rr, :].size
            post = P.ravel()
            self._logw = np.log(np.maximum(post, 1e-300))
        post, patience, att_rate = self._att_marginal(post)
""")
rep("_LADDER_MEM = 1000.0", "_EPS_FLOOR = %r\n_LADDER_MEM = 1000.0" % eps)
open(sys.argv[2], "w").write(src); print("wrote", sys.argv[2], "eps", eps, "rates", rates)
