"""wall-correspondence 0006 -- the detector: amplitude structure in
the source, measured in nats.

0005 proved the coherent bank cannot win on classical streams (the
sibling's two-ledger theorem, operationally). Contrapositive: a
prequential win for amplitudes certifies a non-classical generator.
This module closes the loop by building the certifying stream and
measuring the win.

GENERATOR (source-side amplitude structure, discrete-time monitored
two-state system): amplitude vector psi in C^2; each step a unitary
rotation U(theta) = exp(-i theta sx/2), then a weak two-outcome
measurement of sz with strength k (Kraus M+/- = diag-sqrt of
(1 +- k)/2 terms), outcome recorded, state renormalized. The record
is a +-1 bit stream. Limits: k -> 1 is projective measurement and
the stream degenerates to a classical Markov chain; k -> 0 is no
measurement (fair coin bits at these operating points? no --
p depends on state; k -> 0 makes outcomes uninformative).

BANKS (all parameters from the generator; no fits):
  quantum   -- the amplitude filter (psi carried with phases): the
               generator's own conditional law; optimal by
               construction. THE COHERENT BANK.
  classical -- the decohered shadow: the SAME |U|^2 transition
               probabilities, the SAME outcome likelihoods, phases
               discarded -- an exact 2-state HMM filter. What any
               probability-mixing bank can know.
  markov-8  -- empirical 8th-order Markov predictor, trained on a
               SEPARATE 400k stream from the same generator
               (unbounded classical correlation capture up to lag 8;
               training cost not charged to the test stream).

The gap quantum - classical is the operational value of the phase,
in nats/bit: the detector's calibration. Scanned over (theta, k).
"""

import numpy as np

SEEDS = range(6)
T_EVAL = 50000
BURN = 200


def kraus(k):
    a, b = np.sqrt((1 + k) / 2), np.sqrt((1 - k) / 2)
    return (np.array([[a, 0], [0, b]], dtype=complex),
            np.array([[b, 0], [0, a]], dtype=complex))


def unitary(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def gen(theta, k, T, seed):
    rng = np.random.default_rng(seed)
    U = unitary(theta)
    Mp, Mm = kraus(k)
    psi = np.array([1.0, 0.0], dtype=complex)
    bits = np.empty(T, dtype=int)
    for t in range(T):
        psi = U @ psi
        pp = float(np.linalg.norm(Mp @ psi) ** 2)
        if rng.random() < pp:
            bits[t], psi = 1, Mp @ psi / np.sqrt(pp)
        else:
            bits[t], psi = 0, Mm @ psi / np.sqrt(1 - pp)
    return bits


def code_quantum(bits, theta, k):
    U = unitary(theta)
    Mp, Mm = kraus(k)
    psi = np.array([1.0, 0.0], dtype=complex)
    code = 0.0
    for t, b in enumerate(bits):
        psi = U @ psi
        pp = float(np.linalg.norm(Mp @ psi) ** 2)
        pp = min(max(pp, 1e-12), 1 - 1e-12)
        p = pp if b else 1 - pp
        M = Mp if b else Mm
        psi = M @ psi / np.linalg.norm(M @ psi)
        if t >= BURN:
            code -= np.log(p)
    return code / (len(bits) - BURN)


def code_classical(bits, theta, k):
    P = np.abs(unitary(theta)) ** 2        # decohered transition
    lp = np.array([(1 + k) / 2, (1 - k) / 2])   # P(bit=1 | state)
    w = np.array([1.0, 0.0])
    code = 0.0
    for t, b in enumerate(bits):
        w = P @ w
        like = lp if b else 1 - lp
        p = float(w @ like)
        p = min(max(p, 1e-12), 1 - 1e-12)
        if t >= BURN:
            code -= np.log(p)
        w = w * like / p
    return code / (len(bits) - BURN)


def code_markov(bits_eval, bits_train, order=8):
    ctx_tr = np.zeros(len(bits_train) - order, dtype=int)
    for i in range(order):
        ctx_tr = 2 * ctx_tr + bits_train[i:len(bits_train)
                                         - order + i]
    nxt = bits_train[order:]
    ones = np.bincount(ctx_tr, weights=nxt, minlength=2 ** order)
    tot = np.bincount(ctx_tr, minlength=2 ** order)
    p1 = (ones + 0.5) / (tot + 1.0)
    ctx = np.zeros(len(bits_eval) - order, dtype=int)
    for i in range(order):
        ctx = 2 * ctx + bits_eval[i:len(bits_eval) - order + i]
    nx = bits_eval[order:]
    p = np.where(nx == 1, p1[ctx], 1 - p1[ctx])
    return float(-np.log(np.clip(p, 1e-12, None))
                 [BURN:].mean())


if __name__ == "__main__":
    print("== the detector calibration: quantum - classical gap "
          "(nats/bit) ==")
    print("  theta   k     quantum    classical   gap "
          "(= value of the phase)")
    gaps = {}
    for theta in (0.5, 1.0):
        for k in (0.2, 0.5, 0.8, 0.98, 0.9999):
            rows = []
            for s in SEEDS:
                bits = gen(theta, k, T_EVAL, 1000 + s)
                rows.append([code_quantum(bits, theta, k),
                             code_classical(bits, theta, k)])
            R = np.array(rows)
            g = R[:, 1] - R[:, 0]
            gaps[(theta, k)] = (g.mean(), g.std() / np.sqrt(len(g)))
            print(f"  {theta:.1f}   {k:.2f}   "
                  f"{R[:, 0].mean():.5f}    {R[:, 1].mean():.5f}   "
                  f"{g.mean():+.5f} +- {g.std() / np.sqrt(len(g)):.5f}")
    print()
    print("== the unbounded-classical-memory check "
          "(theta = 1.0, k = 0.5) ==")
    tr = gen(1.0, 0.5, 400000, 77)
    rows = []
    for s in SEEDS:
        bits = gen(1.0, 0.5, T_EVAL, 2000 + s)
        rows.append([code_quantum(bits, 1.0, 0.5),
                     code_classical(bits, 1.0, 0.5),
                     code_markov(bits, tr, 8)])
    R = np.array(rows)
    print(f"  quantum {R[:, 0].mean():.5f}   2-state classical "
          f"{R[:, 1].mean():.5f}   markov-8 {R[:, 2].mean():.5f}")
    print(f"  markov-8 - quantum = "
          f"{(R[:, 2] - R[:, 0]).mean():+.5f} +- "
          f"{(R[:, 2] - R[:, 0]).std() / np.sqrt(len(R)):.5f}")
    print()
    # the structure of the calibration:
    m, se = gaps[(1.0, 0.5)]
    mp, _ = gaps[(1.0, 0.9999)]
    assert m > 10 * se                     # the win is real
    assert mp < m / 20                     # projective = classical
    assert (R[:, 2] > R[:, 0]).all()       # memory can't buy phase
    print("  the coherent bank WINS on the amplitude-source stream; "
          "the win vanishes as")
    print("  measurement turns projective (k -> 1, the classical "
          "limit); 8 lags of exact")
    print("  empirical memory do not close it. The phase's "
          "operational value is measured.")
    print("all assertions passed")
