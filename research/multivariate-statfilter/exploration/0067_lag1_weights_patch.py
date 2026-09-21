"""Stage B, minimal: the bank's weights read each step's likelihood one step LATE -- kernel w(0) = 0,
w(l) = f^(l-1) for l >= 1 (the scalar D(l) shape of 0068).  The predictive density report stays at lag 0.
    python port_lag1.py <src.py> <out.py>"""
import sys
src = open(sys.argv[1]).read()
old_main = "            self._logw = (self.forget ** a) * prior + llw\n"
assert src.count(old_main) == 1, src.count(old_main)
src = src.replace(old_main, """            # STAGE B (0067/0068): the weights read a step's evidence one step late.  The spike step
            # is scored zero (D(0) = 0): a member's response to it is what the next step scores.
            llw_use = self._llw_prev if getattr(self, "_llw_prev", None) is not None else np.zeros_like(llw)
            self._llw_prev = llw
            self._logw = (self.forget ** a) * prior + llw_use
""")
old_reset = "        self.loglik = 0.0\n"
assert src.count(old_reset) >= 1
src = src.replace(old_reset, "        self.loglik = 0.0\n        self._llw_prev = None\n", 1)
open(sys.argv[2], "w").write(src); print("wrote", sys.argv[2])
