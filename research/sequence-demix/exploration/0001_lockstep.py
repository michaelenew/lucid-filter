"""Probe 0001 -- replicate the scalar lockstep and locate it in the scale-Fisher geometry.

Three questions, in order:

  Q1  Does the public filter's scale walk really move both axes identically on the hero rig?
  Q2  Is SUMMARY section 6's proportionality claim exact -- score_xi / score_eta a function of the
      current scales alone, independent of the data?
  Q3  Is the per-step scale-Fisher matrix SINGULAR in the scalar case, and what is its null
      direction?  If the null direction is the ratio, the lockstep is not a quirk of the score
      averaging: it is Proposition 1 written in coordinates, and it says exactly which directions
      a per-step walk can carry and which one it cannot.

Then the same spectrum is measured on the 5-DOF arm rig, because whatever rule decides "this
direction is unidentifiable per step" has to give the right answer there too (that is gate 2).

    python 0001_lockstep.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

from lucid import LucidFilter                      # noqa: E402
from lucid.statfilter.lucid import _WalkEngine     # noqa: E402

SEED, N, JUMP_AT, JUMP, NOISE_AT = 11, 900, 380, 9.0, 600
Q_TRUE, S2_A, S2_C = 0.02, 1.0, 9.0


def hero_series():
    rng = np.random.default_rng(SEED)
    theta = np.cumsum(rng.normal(0.0, np.sqrt(Q_TRUE), N))
    theta[JUMP_AT:] += JUMP
    sd = np.where(np.arange(N) < NOISE_AT, np.sqrt(S2_A), np.sqrt(S2_C))
    return theta, theta + rng.normal(0.0, sd)


# ------------------------------------------------------------------ Q1: the walk
def q1_walk(y):
    """Run the shipped filter and record every member's raw walk centre mu = (xi, eta)."""
    f = LucidFilter()
    mus = np.empty((N, len(f._members), 2))
    for t, v in enumerate(y):
        f.update(np.array([v]))
        for j, mem in enumerate(f._members):
            mus[t, j] = mem.mu
    return f, mus


# ------------------------------------- Q2: the per-step score ratio, member by member
def q2_scores(y):
    """One member, instrumented: the walk step on each axis, and the analytic dS ratio."""
    eng = _WalkEngine(np.eye(1), np.ones(1), np.eye(1), np.eye(1), None, 0.85, 0.45)
    rec = []
    for v in y:
        mu = eng.mu.copy()
        st = eng.update(np.array([v]))
        e = float(st.innovation[0])
        dS_xi = float(eng.lam[0] * math.exp(mu[0]) * eng.HV[0, 0] ** 2)
        dS_eta = float(eng.rho[0] * math.exp(mu[1]))
        rec.append((e, dS_xi, dS_eta, mu[0], mu[1], eng.mu[0], eng.mu[1]))
    return eng, np.array(rec)


def q2_exact_scores():
    """The score ratio at a GRID of (innovation, scale) points, from the definitions.

    score_k = 0.5 * (Si e dS_k Si e - tr(Si dS_k)); in one channel every matrix is a scalar, so
    score_k = 0.5 * dS_k * (e^2/S^2 - 1/S).  The bracket is common; the ratio is dS_xi/dS_eta.
    """
    rows = []
    for xi in (-2.0, 0.0, 1.5):
        for eta in (-2.0, 0.0, 1.5):
            Q, R = math.exp(xi), math.exp(eta)
            for P in (0.3, 3.0):
                S = P + Q + R
                for e in (-3.0, -0.4, 0.7, 5.0):
                    c = e * e / S ** 2 - 1.0 / S
                    rows.append((xi, eta, P, e, 0.5 * Q * c, 0.5 * R * c, Q / R))
    return np.array(rows)


# ------------------------------------------- Q3: the scale-Fisher matrix and its null space
def scale_fisher(F, H, Q0, R0, scale=None):
    """Full per-step scale-Fisher I_ab = 0.5 tr(Si dS_a Si dS_b) at the steady-state S.

    The same steady state the engine's `_steady_fisher` uses -- which keeps only the diagonal,
    and so cannot see the degeneracy this probe is after.
    """
    n, m = Q0.shape[0], R0.size
    lam, V = np.linalg.eigh(Q0)
    HV = H @ V
    xi = np.zeros(n) if scale is None else np.asarray(scale)[:n]
    eta = np.zeros(m) if scale is None else np.asarray(scale)[n:]
    Q = V @ np.diag(lam * np.exp(xi)) @ V.T
    R = np.diag(R0 * np.exp(eta))
    P = np.eye(n) * (lam.max() + R0.max())
    for _ in range(2000):
        Pp = F @ P @ F.T + Q
        S = H @ Pp @ H.T + R
        K = Pp @ H.T @ np.linalg.inv(S)
        P = Pp - K @ H @ Pp
    Pp = F @ P @ F.T + Q
    Si = np.linalg.inv(H @ Pp @ H.T + R)
    dS = [lam[k] * math.exp(xi[k]) * np.outer(HV[:, k], HV[:, k]) for k in range(n)]
    for i in range(m):
        E = np.zeros((m, m)); E[i, i] = R0[i] * math.exp(eta[i])
        dS.append(E)
    D = n + m
    I = np.empty((D, D))
    for a in range(D):
        SdA = Si @ dS[a]
        for b in range(a, D):
            I[a, b] = I[b, a] = 0.5 * float(np.trace(SdA @ Si @ dS[b]))
    return I, dS


def arm_rig():
    """The 0052 5-DOF arm rig's (F, H, Q0, R0) -- same constants."""
    NJ, ORDER, DT = 5, 3, 0.01
    POT, ACC, JERK = 0.06, 0.02, 0.6
    Fb = np.eye(ORDER)
    for i in range(ORDER):
        for j in range(i + 1, ORDER):
            Fb[i, j] = DT ** (j - i) / math.factorial(j - i)
    G = np.array([DT ** (ORDER - i) / math.factorial(ORDER - i) for i in range(ORDER)])
    F = np.kron(np.eye(NJ), Fb)
    Q0 = np.kron(np.eye(NJ), JERK ** 2 * np.outer(G, G) + 1e-12 * np.eye(ORDER))
    rows, rvar = [], []
    N = ORDER * NJ
    for d in range(NJ):
        for di, sd in ((0, POT), (2, ACC)):
            e = np.zeros(N); e[d * ORDER + di] = 1.0
            rows.append(e); rvar.append(sd ** 2)
    return F, np.array(rows), Q0, np.array(rvar)


def main():
    theta, y = hero_series()

    print("=" * 78)
    print("Q1  the walk on the hero rig (LucidFilter(), told nothing)")
    print("=" * 78)
    f, mus = q1_walk(y)
    d = mus[:, :, 0] - mus[:, :, 1]
    print(f"  members: {len(f._members)};  max_t,j |xi - eta| = {np.abs(d).max():.3e}")
    for j in range(5):
        print(f"    phi={f.phi_arr[j]:.2f} s={f.s_arr[j]:.2f}   "
              f"xi={mus[-1, j, 0]:+.4f}  eta={mus[-1, j, 1]:+.4f}  xi-eta={d[-1, j]:+.2e}")
    r = LucidFilter().filter(y[:, None])
    ps, ms = r.process_scale[:, 0], r.measurement_scale[:, 0]
    A, C = slice(60, JUMP_AT), slice(NOISE_AT, N)
    for name, sl, qt, rt in (("A", A, Q_TRUE, S2_A), ("C", C, Q_TRUE, S2_C)):
        Qh, Rh = np.exp(ps[sl]).mean(), np.exp(ms[sl]).mean()
        print(f"  regime {name}: learned Q={Qh:.4f} R={Rh:.4f} total={Qh + Rh:.4f} "
              f"(truth Q+R={qt + rt:.4f});  learned ratio={Qh / Rh:.4f} (truth {qt / rt:.5f})")

    print()
    print("=" * 78)
    print("Q2  is score_xi / score_eta independent of the data?")
    print("=" * 78)
    eng, rec = q2_scores(y)
    dS_xi, dS_eta = rec[:, 1], rec[:, 2]
    step_xi, step_eta = rec[:, 5] - rec[:, 3], rec[:, 6] - rec[:, 4]
    print(f"  single member (phi=0.85, s=0.45): max_t |mu_xi - mu_eta| = "
          f"{np.abs(rec[:, 5] - rec[:, 6]).max():.3e}")
    print(f"  max_t |walk step_xi - step_eta|  = {np.abs(step_xi - step_eta).max():.3e}")
    print(f"  dS_xi/dS_eta over the run: min {np.min(dS_xi / dS_eta):.6f} "
          f"max {np.max(dS_xi / dS_eta):.6f}")
    g = q2_exact_scores()
    err = np.abs(g[:, 4] / g[:, 5] - g[:, 6]).max()
    print(f"  closed form over a (xi, eta, P, e) grid ({len(g)} points): "
          f"max |score_xi/score_eta - Q/R| = {err:.3e}")
    print("  => the innovation enters BOTH scores through one common factor (e^2/S^2 - 1/S);")
    print("     it sets the STEP SIZE and never the DIRECTION.")

    print()
    print("=" * 78)
    print("Q3  the per-step scale-Fisher spectrum: which directions are identifiable?")
    print("=" * 78)
    I, _ = scale_fisher(np.eye(1), np.eye(1), np.eye(1), np.ones(1))
    w, U = np.linalg.eigh(I)
    print(f"  scalar hero class (n=1, m=1):  eigenvalues {w}")
    print(f"    rank ratio lam_min/lam_max = {w[0] / w[-1]:.3e}")
    print(f"    null direction              = {U[:, 0]}  (xi, eta)")
    print(f"    identified direction        = {U[:, -1]}")
    for xi0, eta0 in ((-3.9, 0.0), (2.0, -1.0)):
        I2, _ = scale_fisher(np.eye(1), np.eye(1), np.eye(1), np.ones(1),
                             scale=np.array([xi0, eta0]))
        w2, U2 = np.linalg.eigh(I2)
        pred = np.array([math.exp(eta0), -math.exp(xi0)])
        pred /= np.linalg.norm(pred)
        print(f"    at (xi,eta)=({xi0:+.1f},{eta0:+.1f}): lam_min/lam_max = {w2[0]/w2[-1]:.2e}, "
              f"null {np.array2string(U2[:, 0], precision=4)} vs predicted (R,-Q)/|.| "
              f"{np.array2string(pred, precision=4)}")

    print()
    F, H, Q0, R0 = arm_rig()
    I, _ = scale_fisher(F, H, Q0, R0)
    n, m = Q0.shape[0], R0.size
    eng = _WalkEngine(Q0, R0, H, F, None, 0.85, 0.45)
    act = np.flatnonzero(eng.active)
    Ia = I[np.ix_(act, act)]
    # normalise to the natural scale-per-axis metric before reading a rank off it
    dg = np.sqrt(np.diag(Ia))
    Cm = Ia / np.outer(dg, dg)
    wa, Ua = np.linalg.eigh(Cm)
    lbl = [f"xi{k}" for k in range(n)] + [("pot" if i % 2 == 0 else "acc") + str(i // 2)
                                          for i in range(m)]
    print(f"  5-DOF arm rig (n={n}, m={m}, D={n + m}); active axes: {len(act)} of {n + m}")
    print(f"    active axes: {[lbl[a] for a in act]}")
    print(f"    correlation-form eigenvalues: "
          f"{np.array2string(wa, precision=4, max_line_width=110)}")
    print(f"    rank ratio lam_min/lam_max = {wa[0] / wa[-1]:.3e}")
    print("    smallest-eigenvalue direction, largest entries:")
    for j in np.argsort(-np.abs(Ua[:, 0]))[:6]:
        print(f"      {lbl[act[j]]:>6s}  {Ua[j, 0]:+.4f}")
    # pairwise scale-Fisher correlations (0027's |C|)
    print("    pairwise |C| for joint 0's (jerk mode, pot0, acc0) triple:")
    jk = [a for a in act if a < n]
    idx = {v: i for i, v in enumerate(act)}
    for a in jk[:1] + [act[i] for i in range(len(act)) if act[i] >= n][:2]:
        row = " ".join(f"{lbl[b]}:{abs(Cm[idx[a], idx[b]]):.6f}"
                       for b in jk[:1] + [c for c in act if c >= n][:2])
        print(f"      {lbl[a]:>6s}  {row}")


    print()
    print("-" * 78)
    print("Q3b  the RAW spectrum and the 'can the walk localise it?' test")
    print("-" * 78)
    print("  The engine already owns a Fisher-valued resolution floor: `_Ifloor` = (1-phi) /")
    print("  (4 (SPAN_S s)^2), the 0010 delocalisation threshold -- the information below which a")
    print("  walk cannot hold a centre.  Read the scale-Fisher against it:")
    for tag, (Fx, Hx, Qx, Rx) in (("scalar hero", (np.eye(1), np.eye(1), np.eye(1), np.ones(1))),
                                  ("5-DOF arm", (F, H, Q0, R0))):
        Ir, _ = scale_fisher(Fx, Hx, Qx, Rx)
        e0 = _WalkEngine(Qx, Rx, Hx, Fx, None, 0.85, 0.45)
        floor = float(np.max(e0._Ifloor))
        dg = np.diag(Ir)
        info = dg > floor
        nn, mm = Qx.shape[0], Rx.size
        lb = ([f"xi{k}" for k in range(nn)] +
              ([("pot" if i % 2 == 0 else "acc") + str(i // 2) for i in range(mm)]
               if mm == 10 else [f"eta{i}" for i in range(mm)]))
        print(f"  [{tag}]  Ifloor = {floor:.4g}")
        print(f"    diag(I) range {dg.min():.3g} .. {dg.max():.3g};  "
              f"axes above the floor: {[lb[i] for i in np.flatnonzero(info)]}")
        if info.sum() >= 2:
            sub = Ir[np.ix_(info, info)]
            wq, Uq = np.linalg.eigh(sub)
            lbi = [lb[i] for i in np.flatnonzero(info)]
            print(f"    informative-block eigenvalues: "
                  f"{np.array2string(wq, precision=4, max_line_width=110)}")
            below = np.flatnonzero(wq < floor)
            print(f"    directions below the floor: {len(below)} of {len(wq)}")
            for j in below[:3]:
                top = np.argsort(-np.abs(Uq[:, j]))[:4]
                print("      u = " + ", ".join(f"{lbi[t]}:{Uq[t, j]:+.3f}" for t in top)
                      + f"   (lam = {wq[j]:.3e})")


if __name__ == "__main__":
    main()
