"""wall-correspondence 0005 -- the coherent bank: does amplitude
mixing buy anything on classical data?

The sibling's Born weight is |A|^2 -- amplitudes first, squared at
readout. Their prequential tier (their 0105) proved the modulus
ledger has an operational loss (code length); the PHASE ledger has
none yet. F4 (their 0079): if a bank that mixes amplitudes with
interference beats probability mixing (IMM) prequentially on any
classical tracking task, the phase has an operational meaning
discovered from this side.

DESIGN (controlled to isolate interference):
  - Latent: a damped stochastic oscillator whose frequency switches
    between omega_0 and omega_1 (regime r_t, hazard h); phase is
    CONTINUOUS across switches. y_t = position + noise.
  - Oracle: Kalman with the true regime sequence (reference).
  - IMM: two matched Kalman filters, probability mixing with the
    true hazard matrix. No free parameters.
  - Coherent: THE SAME two filters, same readout mixture form; the
    only change is the weight dynamics: amplitudes
    alpha_i = sqrt(w_i) e^{i phi_i}, phi_i = filter i's own
    oscillator-phase belief, mixed through sqrt(M) so the transfer
    term carries cos(phi_1 - phi_2) interference; weights read out
    as |alpha|^2. No free parameters beyond IMM's.
  - Conditions: (a) plain hazard; (b) phase-gated hazard (switches
    only near zero crossings, mean rate matched to h). Control for
    (b): IMM+ = classical IMM given the true gate as a
    phase-dependent hazard through its own phase estimate -- any
    coherent win must beat IMM, and any win on (b) is only
    interference-specific if IMM+ does not already capture it.

Scoring: prequential nats/step, paired across seeds. House rule
honored: every parameter of every bank is set from the generator
(no fits); the comparison is representational, not a tuning contest.
"""

import numpy as np

OM = (0.30, 0.70)          # regime frequencies
RHO = 0.995                # oscillator damping
QP = 0.05                  # process noise
SY = 0.30                  # observation noise
H = 0.02                   # mean switch hazard
T = 60000
BURN = 500
SEEDS = range(8)
GATE = 0.25                # |sin(phase)| < GATE opens the gate


def rot(w):
    return RHO * np.array([[np.cos(w), -np.sin(w)],
                           [np.sin(w), np.cos(w)]])


F = [rot(w) for w in OM]
Q = QP ** 2 * np.eye(2)
Hobs = np.array([1.0, 0.0])


def gen(T, seed, gated):
    rng = np.random.default_rng(seed)
    s = np.array([1.0, 0.0])
    r, y, rs = 0, np.empty(T), np.empty(T, dtype=int)
    # calibrate gate hazard so the mean switch rate matches H
    hg = H / GATE if gated else H
    for t in range(T):
        phase_open = abs(s[1]) / (np.linalg.norm(s) + 1e-12) < GATE
        h_t = (hg if phase_open else 0.0) if gated else H
        if rng.random() < h_t:
            r = 1 - r
        s = F[r] @ s + QP * rng.normal(size=2)
        y[t] = s[0] + SY * rng.normal()
        rs[t] = r
    return y, rs


def kstep(s, P, Fm, y):
    s = Fm @ s
    P = Fm @ P @ Fm.T + Q
    v = y - s[0]
    S = P[0, 0] + SY ** 2
    K = P[:, 0] / S
    s = s + K * v
    P = P - np.outer(K, P[0])
    ll = -0.5 * (np.log(2 * np.pi * S) + v ** 2 / S)
    return s, P, ll


def run_oracle(y, rs):
    s, P = np.array([1.0, 0.0]), np.eye(2)
    code = 0.0
    for t in range(len(y)):
        s, P, ll = kstep(s, P, F[rs[t]], y[t])
        if t >= BURN:
            code -= ll
    return code / (len(y) - BURN)


def imm_mix(w, M):
    c = M.T @ w                       # predicted mode probs
    mu = M * w[:, None] / c[None, :]  # mu[i,j] = P(from i | now j)
    return c, mu


def coh_mix(w, phi, M):
    alpha = np.sqrt(w) * np.exp(1j * phi)
    a2 = np.sqrt(M).T @ alpha         # interference in the transfer
    c = np.abs(a2) ** 2
    c = np.maximum(c, 1e-12)
    c /= c.sum()
    # branch weights from the moduli of the contributions
    contrib = np.abs(np.sqrt(M) * alpha[:, None]) ** 2
    mu = contrib / np.maximum(contrib.sum(0), 1e-12)[None, :]
    return c, mu


def run_bank(y, kind, gated_plus=False):
    """kind: 'imm' or 'coh'. gated_plus: classical phase-dependent
    hazard (IMM+), using the bank's own phase estimate."""
    ss = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]
    Ps = [np.eye(2), np.eye(2)]
    w = np.array([0.5, 0.5])
    code = 0.0
    for t in range(len(y)):
        if gated_plus:
            sbar = w[0] * ss[0] + w[1] * ss[1]
            po = abs(sbar[1]) / (np.linalg.norm(sbar) + 1e-12) < GATE
            h_t = H / GATE if po else 1e-6
            h_t = min(h_t, 0.5)
        else:
            h_t = H
        M = np.array([[1 - h_t, h_t], [h_t, 1 - h_t]])
        if kind == "coh":
            phi = np.array([np.arctan2(s[1], s[0]) for s in ss])
            c, mu = coh_mix(w, phi, M)
        else:
            c, mu = imm_mix(w, M)
        ns, nP, ls = [], [], np.empty(2)
        for j in range(2):
            s0 = mu[0, j] * ss[0] + mu[1, j] * ss[1]
            d0, d1 = ss[0] - s0, ss[1] - s0
            P0 = (mu[0, j] * (Ps[0] + np.outer(d0, d0))
                  + mu[1, j] * (Ps[1] + np.outer(d1, d1)))
            s1, P1, ll = kstep(s0, P0, F[j], y[t])
            ns.append(s1)
            nP.append(P1)
            ls[j] = ll
        ss, Ps = ns, nP
        lmax = ls.max()
        pj = c * np.exp(ls - lmax)
        p = pj.sum()
        if t >= BURN:
            code -= np.log(p) + lmax
        w = pj / p
    return code / (len(y) - BURN)


def condition(gated):
    label = "phase-gated hazard" if gated else "plain hazard"
    rows, srate = [], []
    for seed in SEEDS:
        y, rs = gen(T, 100 + seed, gated)
        srate.append((np.diff(rs) != 0).mean())
        row = [run_oracle(y, rs), run_bank(y, "imm"),
               run_bank(y, "coh")]
        if gated:
            row.append(run_bank(y, "imm", gated_plus=True))
        rows.append(row)
    R = np.array(rows)
    m, se = R.mean(0), R.std(0) / np.sqrt(len(R))
    print(f"== {label} (T = {T}, {len(R)} seeds, realized switch "
          f"rate {np.mean(srate):.4f}, nats/step) ==")
    names = ["oracle", "IMM", "coherent"] + (["IMM+"] if gated
                                             else [])
    for i, n in enumerate(names):
        print(f"  {n:9s} {m[i]:.5f} +- {se[i]:.5f}")
    d = R[:, 2] - R[:, 1]
    print(f"  coherent - IMM = {d.mean():+.5f} +- "
          f"{d.std() / np.sqrt(len(d)):.5f}  "
          f"({'coherent WINS' if d.mean() < 0 else 'IMM wins'})")
    if gated:
        dp = R[:, 3] - R[:, 1]
        print(f"  IMM+     - IMM = {dp.mean():+.5f} +- "
              f"{dp.std() / np.sqrt(len(dp)):.5f}")
    print()
    return R


if __name__ == "__main__":
    Ra = condition(gated=False)
    Rb = condition(gated=True)
    # sanity: oracle is the floor everywhere
    assert (Ra[:, 0] <= Ra[:, 1:].min(1) + 1e-9).all()
    assert (Rb[:, 0] <= Rb[:, 1:].min(1) + 1e-9).all()
    print("all assertions passed")
