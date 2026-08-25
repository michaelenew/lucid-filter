"""wall-correspondence 0007 -- the coupled bank: the vertex as a
shared trust node, and its transfer function.

F1 of the sibling's adoption plan (their 0079): their boundary-state
vertex / momentum propagator, reformulated. Their network tier
(their 0104) proved the lattice's shared link IS a Bayes-sharing
node between filters, so the reformulation is no longer an analogy:
two filters sharing a posterior is the vertex, on the isomorphic
tier, and the cross-stream response to an innovation is the
propagator.

DESIGN. Two observation streams; latents are independent random
walks given a SHARED log-scale lambda (the trust channel):
    lambda_t = phi lambda_{t-1} + s_l eta      (phi = 1: pinned)
    x_i,t    = x_i,t-1 + e^{lambda_t} q0 eps_i
    y_i,t    = x_i,t + s_y nu_i
The bank: Rao-Blackwellized grid filter -- per lambda grid point a
Kalman pair, joint weights updated by both streams' likelihoods.
Given lambda the streams are independent, so the ONLY cross-channel
is the shared scale posterior.

MEASUREMENT. Paired runs (identical noise); a +4 sigma innovation
spike is injected into stream 1 at t0; the transfer is stream 2's
predictive response. Two claims tested:
  1  the MEAN channel is silent (given-lambda independence: an exact
     structural null, measured);
  2  the VARIANCE channel responds, and its memory is set by the
     trust channel's mass: phi < 1 decays at rate ~(1 - phi);
     phi = 1 (the pinned root -- the sibling's massless mode) decays
     only at the DATA-INFORMATION rate: the long-memory
     cross-response that is F1's masslessness signature.
"""

import numpy as np

LGRID = np.linspace(-1.5, 1.5, 41)
NL = len(LGRID)
Q0 = 0.15
SY = 0.5
SL = 0.03
NTRIAL = 400
T = 1100
T0 = 500
WIN = 500
SPIKE = 4.0


def trans_matrix(phi):
    mu = phi * LGRID
    Tm = np.exp(-0.5 * ((LGRID[:, None] - mu[None, :]) / SL) ** 2)
    return Tm / Tm.sum(0, keepdims=True)


def gen(phi, seed):
    rng = np.random.default_rng(seed)
    lam = np.zeros(NTRIAL)
    x = np.zeros((2, NTRIAL))
    ys = np.empty((T, 2, NTRIAL))
    lams = np.empty((T, NTRIAL))
    for t in range(T):
        lam = np.clip(phi * lam + SL * rng.normal(size=NTRIAL),
                      LGRID[0], LGRID[-1])
        x += np.exp(lam) * Q0 * rng.normal(size=(2, NTRIAL))
        ys[t] = x + SY * rng.normal(size=(2, NTRIAL))
        lams[t] = lam
    return ys


def run_bank(ys, phi, spike_at=None):
    """Grid filter over lambda; returns per-step predictive mean and
    variance of stream 2 (mixture), arrays (T, NTRIAL)."""
    Tm = trans_matrix(phi)
    w = np.full((NTRIAL, NL), 1.0 / NL)
    m = np.zeros((2, NTRIAL, NL))
    P = np.ones((2, NTRIAL, NL))
    qlam = (np.exp(LGRID) * Q0) ** 2
    pm2 = np.empty((T, NTRIAL))
    pv2 = np.empty((T, NTRIAL))
    for t in range(T):
        w = w @ Tm.T
        P = P + qlam[None, None, :]
        S = P + SY ** 2
        mbar = (w * m[1]).sum(1)
        pv2[t] = (w * (S[1] + m[1] ** 2)).sum(1) - mbar ** 2
        pm2[t] = mbar
        y = ys[t].copy()
        if spike_at is not None and spike_at <= t < spike_at + 5:
            y[0] += SPIKE * SY
        ll = np.zeros((NTRIAL, NL))
        for i in range(2):
            v = y[i][:, None] - m[i]
            ll += -0.5 * (np.log(S[i]) + v ** 2 / S[i])
            K = P[i] / S[i]
            m[i] = m[i] + K * v
            P[i] = P[i] - K * P[i]
        w = w * np.exp(ll - ll.max(1, keepdims=True))
        w /= w.sum(1, keepdims=True)
    return pm2, pv2


def half_life(resp):
    ipk = int(np.argmax(resp[:40]))
    pk = resp[ipk]
    below = np.where(resp[ipk:] < pk / 2)[0]
    return (int(below[0]) if len(below) else len(resp)), pk


if __name__ == "__main__":
    print("== the coupled bank: cross-stream transfer through the "
          "shared trust node ==")
    print(f"  ({NTRIAL} paired trials; +{SPIKE:.0f} sigma innovation "
          f"into stream 1 at t0; response in stream 2)")
    hls = {}
    for phi in (0.90, 0.98, 1.00):
        ys = gen(phi, 42)
        m0, v0 = run_bank(ys, phi)
        m1, v1 = run_bank(ys, phi, spike_at=T0)
        dv = (np.log(v1) - np.log(v0))[T0 + 1:T0 + WIN].mean(1)
        dm = (m1 - m0)[T0 + 1:T0 + WIN]
        null = np.abs(dm.mean(1)).max() / (SPIKE * SY)
        rms = dm.std(1).max() / (SPIKE * SY)
        hl, pk = half_life(dv)
        area = float(dv.sum())
        tail = float(dv[200]) / pk
        hls[phi] = (hl, area, tail)
        tag = "PINNED (massless)" if phi == 1.0 else f"mass {1-phi:.2f}"
        print(f"  phi = {phi:.2f} [{tag}]")
        print(f"    mean channel: systematic |dmean2|/spike < "
              f"{null:.1e} (silent); incoherent per-trial "
              f"rms {rms:.1e}")
        print(f"    variance response: peak dlnVar2 = {pk:+.4f}, "
              f"half-life from peak = {hl} steps")
        print(f"    profile at peak + (0, 25, 50, 100, 200, 400): "
              + ", ".join(f"{dv[min(i + int(np.argmax(dv[:40])), len(dv) - 1)]:+.4f}" for i in
                          (0, 25, 50, 100, 200, 400)))
        print(f"    zero-frequency weight (area) = {area:.2f}; "
              f"tail(+200)/peak = {tail:.3f}")
        assert null < 5 * rms / np.sqrt(NTRIAL) + 1e-6
        assert pk > 0.01
    print()
    a = {p: hls[p][1] for p in hls}
    t = {p: hls[p][2] for p in hls}
    print(f"  zero-frequency transfer: phi 0.90 -> {a[0.90]:.2f}, "
          f"0.98 -> {a[0.98]:.2f}, 1.00 -> {a[1.00]:.2f}")
    print(f"  tail(+200)/peak:         phi 0.90 -> {t[0.90]:.3f}, "
          f"0.98 -> {t[0.98]:.3f}, 1.00 -> {t[1.00]:.3f}")
    assert a[0.90] < a[0.98] < a[1.00]
    assert t[1.00] > 5 * t[0.90]
    print("  the pinned trust channel is the long-memory "
          "cross-response: the propagator's")
    print("  low-frequency behaviour = the shared node's mass, "
          "measured through the vertex.")
    print("all assertions passed")
