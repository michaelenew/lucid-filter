"""wall-correspondence 0033 -- composition has two channels, and
only one of them is the source ledger.

0031 concluded that the source ledger is 'a statement about
composition' and pointed at the order channel (0009) as where it
lives. That was sloppy, and this module corrects it. 0009's order
channel composes CLASSICAL group elements -- no amplitudes anywhere.
So composition has two independent layers:

  (i)  CLASSICAL NONABELIAN composition -- the order channel. Needs
       a nonabelian group; needs no amplitudes; lives in the RECORD
       ledger.
  (ii) AMPLITUDE composition -- interference. Needs amplitudes;
       needs no nonabelian structure; lives in the SOURCE ledger.

Demonstrated by two records, each firing exactly one detector.

  s1  RECORD A: classical nonabelian. Three SU(2) elements composed
       in a random order, the composite read through noise. Order
       information is positive (measured). The generator is
       CLASSICAL by construction, so by 0005 no amplitude model can
       beat the correct classical one -- the interference channel is
       empty here.
  s2  RECORD B: quantum abelian. Two paths whose amplitudes add,
       with a known relative phase varied across trials. The
       coherent model predicts the fringe and the incoherent one
       cannot: measured code-length gap. And swapping the paths
       leaves every outcome probability BITWISE identical -- order
       information is exactly zero.
  s3  THE CORRECTION. The source ledger's observable content is
       INTERFERENCE, not composition in general. The order channel
       is a record-ledger phenomenon that happens to require
       nonabelian structure -- which is why 0009 could measure it
       with no amplitudes in sight. 0031's phrasing is corrected in
       place.
"""

import numpy as np

TH = np.linspace(1e-7, np.pi - 1e-7, 40001)
HAAR = (2 / np.pi) * np.sin(TH) ** 2


def k_heat(tau, jmax=40):
    out = np.zeros_like(TH)
    for j in np.arange(0, jmax + 0.1, 0.5):
        out += (2 * j + 1) * np.exp(-tau * j * (j + 1)) \
            * np.sin((2 * j + 1) * TH) / np.sin(TH)
    return out


_CDF = None


def sample_su2(tau, n, rng):
    global _CDF
    if _CDF is None:
        p = np.maximum(k_heat(tau), 0) * HAAR
        _CDF = np.cumsum(p) / p.sum()
    th = TH[np.searchsorted(_CDF, rng.random(n))]
    ax = rng.normal(size=(n, 3))
    ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    return np.concatenate([np.cos(th)[:, None],
                           np.sin(th)[:, None] * ax], axis=1)


def qmul(a, b):
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


def cls(q):
    return np.arccos(np.clip(q[..., 0], -1, 1))


def s1_record_A():
    print("== s1: record A -- classical nonabelian ==")
    rng = np.random.default_rng(21)
    n = 200000
    a, b, c = (sample_su2(0.2, n, rng) for _ in range(3))
    t1 = cls(qmul(qmul(a, b), c))
    t2 = cls(qmul(qmul(a, c), b))
    sig = float(np.std(cls(a)))
    sent = rng.random(n) < 0.5
    read = np.where(sent, t1, t2) + sig * rng.normal(size=n)
    aware = -0.5 * ((read - np.where(sent, t1, t2)) / sig) ** 2
    blind = np.logaddexp(-0.5 * ((read - t1) / sig) ** 2,
                         -0.5 * ((read - t2) / sig) ** 2) \
        - np.log(2)
    gap = float((aware - blind).mean())
    print(f"  order information (order-aware minus order-blind): "
          f"{gap:.4f} nats/triple")
    assert gap > 0.005
    print("  the generator is CLASSICAL -- real group elements, "
          "classical sampling, no")
    print("  amplitudes anywhere -- so by 0005 no amplitude model "
          "can beat the correct")
    print("  classical one: THE INTERFERENCE CHANNEL IS EMPTY "
          "HERE\n")
    return gap


def s2_record_B():
    print("== s2: record B -- quantum abelian ==")
    rng = np.random.default_rng(5)
    n = 200000
    a1, a2 = 0.75, 0.66
    delta = rng.uniform(0, 2 * np.pi, n)          # known per trial
    p_coh = np.abs(a1 + a2 * np.exp(1j * delta)) ** 2
    p_coh = p_coh / (a1 + a2) ** 2                # normalise to [0,1]
    p_inc = np.full(n, (a1 ** 2 + a2 ** 2) / (a1 + a2) ** 2)
    bits = rng.random(n) < p_coh
    def code(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-np.mean(np.where(bits, np.log(p),
                                       np.log(1 - p))))
    c_coh, c_inc = code(p_coh), code(p_inc)
    print(f"  coherent model {c_coh:.5f} nats/trial, incoherent "
          f"{c_inc:.5f}  ->  gap {c_inc - c_coh:+.5f}")
    assert c_inc - c_coh > 0.01
    # order: swapping the two paths in the sum
    p_swapped = np.abs(a2 * np.exp(1j * delta) + a1) ** 2 \
        / (a1 + a2) ** 2
    same = bool(np.array_equal(p_coh, p_swapped))
    print(f"  swapping the two paths: outcome probabilities bitwise "
          f"identical = {same}")
    assert same
    print("  interference is present and ORDER INFORMATION IS "
          "EXACTLY ZERO -- addition")
    print("  commutes, so there is no order to read\n")
    return c_inc - c_coh


def s3_correction(gapA, gapB):
    print("== s3: the correction ==")
    print("                          order channel    interference")
    print(f"   A classical nonabelian   {gapA:.4f}         "
          f"empty (0005)")
    print(f"   B quantum abelian        0 (exact)       "
          f"{gapB:.4f}")
    print("  each record fires exactly one detector: the two "
          "channels are INDEPENDENT.")
    print("  So the source ledger's observable content is "
          "INTERFERENCE, not composition in")
    print("  general. The order channel is a RECORD-ledger "
          "phenomenon that happens to")
    print("  require nonabelian structure -- which is why 0009 "
          "could measure it with no")
    print("  amplitudes in sight. 0031's phrasing is corrected in "
          "place\n")


if __name__ == "__main__":
    gA = s1_record_A()
    gB = s2_record_B()
    s3_correction(gA, gB)
    print("all assertions passed")
