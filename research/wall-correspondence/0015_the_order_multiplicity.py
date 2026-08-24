"""wall-correspondence 0015 -- the order channel vs multiplicity.

0009 measured the order channel at P = 3 (one parity bit). Here the
capacity curve vs multiplicity P: the boundary read of P composed
innovations can carry at most ln((P-1)!) nats of order information
(cyclic collapse). Measured: exact posterior over the (P-1)! cyclic
classes given the noisy read and the known innovations;
I = ln K - E[H(posterior)].
"""

import math

import numpy as np

TH = np.linspace(1e-7, np.pi - 1e-7, 40001)
HAAR = (2 / np.pi) * np.sin(TH) ** 2
TAU = 0.2
N = 20000


def k_heat(tau, jmax=40):
    out = np.zeros_like(TH)
    for j in np.arange(0, jmax + 0.1, 0.5):
        out += (2 * j + 1) * np.exp(-tau * j * (j + 1)) \
            * np.sin((2 * j + 1) * TH) / np.sin(TH)
    return out


CDF = None


def sample(n, rng):
    global CDF
    if CDF is None:
        p = np.maximum(k_heat(TAU), 0) * HAAR
        CDF = np.cumsum(p) / p.sum()
    th = TH[np.searchsorted(CDF, rng.random(n))]
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


def cyc_classes(P):
    """One representative order per cyclic class (fix first
    element)."""
    from itertools import permutations
    return [(0,) + p for p in permutations(range(1, P))]


if __name__ == "__main__":
    print("== order information vs multiplicity (tau = 0.2) ==")
    print("   P   classes  ceiling ln(P-1)!   I measured   fraction")
    rng = np.random.default_rng(12)
    fr = {}
    for P in (3, 4, 5):
        reps = cyc_classes(P)
        K = len(reps)
        g = [sample(N, rng) for _ in range(P)]
        cls = np.empty((K, N))
        for ci, order in enumerate(reps):
            acc = g[order[0]]
            for idx in order[1:]:
                acc = qmul(acc, g[idx])
            cls[ci] = np.arccos(np.clip(acc[..., 0], -1, 1))
        sig = float(np.std(np.arccos(np.clip(g[0][:, 0], -1, 1))))
        sent = rng.integers(0, K, N)
        read = cls[sent, np.arange(N)] + sig * rng.normal(size=N)
        ll = -0.5 * ((read[None, :] - cls) / sig) ** 2
        ll -= ll.max(0, keepdims=True)
        post = np.exp(ll)
        post /= post.sum(0, keepdims=True)
        H = -(post * np.log(np.maximum(post, 1e-300))).sum(0).mean()
        I = math.log(K) - H
        fr[P] = I / math.log(K)
        print(f"   {P}     {K:3d}      {math.log(K):.3f}          "
              f"{I:.4f}      {fr[P]:.3f}")
    assert fr[3] > 0 and fr[4] > 0 and fr[5] > 0
    assert max(fr.values()) / min(fr.values()) < 1.2
    print("  the measured information GROWS with the ceiling at a "
          "roughly CONSTANT fraction")
    print("  (~8% per scalar read at this noise): one boundary "
          "summary reads a fixed share")
    print("  of the order content however rich it gets -- the "
          "order channel scales, and")
    print("  richer boundary data multiplies it")
    print("all assertions passed")
