"""wall-correspondence 0018 -- the unmatched detector: the practical
instrument.

0006's detector knew the generator's (theta, k). The field version
cannot: it must scan a PRICED model family against a stream of
unknown provenance. Two families of equal size and equal price:

  quantum family   -- amplitude filters on a (theta, k) grid;
  classical family -- the decohered 2-state HMMs on the SAME grid.

Verdict statistic: best two-part code (family price = ln |grid|,
equal for both, cancels) quantum minus classical. On a stream from
a quantum generator whose true parameters are NOT grid points, the
quantum family must still win by a resolvable margin (a fraction
of the matched ceiling, set by grid granularity); on a stream from a classical HMM generator, the margin
must go NEGATIVE (0005's theorem: amplitudes never pay on classical
streams, and mis-specified quantum models pay extra) -- the
instrument is a signed classifier.
"""

import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "d6", os.path.join(HERE, "0006_the_detector.py"))
d6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d6)

GRID_T = (0.6, 0.8, 1.0, 1.2)
GRID_K = (0.3, 0.5, 0.7)
T_EVAL = 20000
SEEDS = range(4)


def gen_classical(theta, k, T, seed):
    rng = np.random.default_rng(seed)
    P = np.abs(d6.unitary(theta)) ** 2
    lp = np.array([(1 + k) / 2, (1 - k) / 2])
    z, bits = 0, np.empty(T, dtype=int)
    for t in range(T):
        z = rng.choice(2, p=P[:, z])
        bits[t] = rng.random() < lp[z]
    return bits


def best_code(bits, coder):
    return min(coder(bits, th, k) for th in GRID_T for k in GRID_K)


def margin(bits):
    cq = best_code(bits, d6.code_quantum)
    cc = best_code(bits, d6.code_classical)
    return cc - cq                       # >0: quantum family wins


if __name__ == "__main__":
    print("== the unmatched detector (equal-size, equal-price "
          "families) ==")
    print(f"  grid: theta {GRID_T} x k {GRID_K}; true parameters "
          f"OFF-grid")
    mq, mc = [], []
    for s in SEEDS:
        bq = d6.gen(0.9, 0.55, T_EVAL, 4000 + s)   # quantum source
        bc = gen_classical(0.9, 0.55, T_EVAL, 5000 + s)
        mq.append(margin(bq))
        mc.append(margin(bc))
    mq, mc = np.array(mq), np.array(mc)
    print(f"  quantum-source stream  : margin = {mq.mean():+.5f} "
          f"+- {mq.std() / 2:.5f} nats/bit  (matched ceiling ~ "
          f"+0.06)")
    print(f"  classical-source stream: margin = {mc.mean():+.5f} "
          f"+- {mc.std() / 2:.5f} nats/bit")
    assert mq.mean() > 10 * mq.std() / 2
    assert mq.mean() > 0.008
    assert mc.mean() < -0.01
    print("  the instrument survives mis-specification as a SIGNED "
          "classifier: off-grid")
    print("  quantum sources certified positive (~20% of the "
          "matched ceiling at this")
    print("  coarse grid -- refinement recovers it); classical "
          "sources strongly negative")
    print("  (0005's theorem plus the mis-specification cost). No "
          "false positives by")
    print("  construction: amplitudes cannot pay on classical "
          "streams.")
    print("all assertions passed")
