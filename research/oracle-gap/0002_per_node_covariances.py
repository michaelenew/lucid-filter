"""0002 -- Per-node covariances: does the evidence come back, and does the
likelihood get its curvature back?

The shipped filter is GPB1: after every step the mixture is collapsed to ONE
(m, P), so at the next step every grid node is handed the same covariance and
two nodes' predictive variances differ by one step of process noise, never by
the accumulated history of the regime they disagree about.  `0038` measured
that deletion at 75% of the Q-vs-8Q discrimination at sigma^2 = 9 and called
it a tax, because what survived was enough for a FORCED channel to close 80%
of the oracle gap.  `0001` suspects it is not a tax but the hole: compressed
evidence means a flat likelihood in the scale coordinates, and a flat
likelihood is what lets the fit land on self-confirming boundaries (0039), lets
the optimiser rather than the data pick between two reported basins
(FILTER-NOTES section 4), and caps the forced channel at 80%.

The alternative carried here is IMM: each grid node keeps its own (m_g, P_g),
mixed across nodes each step by the chain's own transition kernel -- no new
parameters, no change of model, only the collapse removed.  s_P = s_M = 0
still collapses the grid to one node, where IMM and GPB1 are the same filter
to machine precision (checked in section A).

  A  VALIDATION and COST.  IMM == plain Kalman == OdeFilter on the s = 0
     face; wall-clock per likelihood pass at order 5.

  B  THE ORACLE GAP, on `0038` section B's exact data (same seed, same
     schedule).  oracle / static / hindsight-constant / GPB1 forced / IMM
     forced.  The number that matters: gap closed by IMM at s_P = 0.8
     against GPB1's 80.0%.

  C  THE PROFILE.  Marginal likelihood in s_P through both filters on
     `0038` section D's two datasets (the x8 regime and an AR(1) log-scale
     with true s_P = 0.8).  GPB1's profile on the regime data was
     boundary-seeking from the s_P = 0 side; if the collapse is the hole,
     the IMM profile should show an interior optimum at the generating
     value.  This is the measurement that separates "tax" from "hole".

  D  PREMIUM and EXPOSURE.  What an unnecessary s_P = 0.8 costs under each
     filter (homoscedastic data), against what a missing one costs on the
     regime data.  0039's ledger was +0.0025 / +0.0872 under GPB1.

Run:  python3 0002_per_node_covariances.py
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "lucid"))

from odefilter import OdeFilter, Params  # noqa: E402
from odefilter.core import _chain, _companion  # noqa: E402

ALPHA3 = np.array([2.785218519281637, -2.6855430450862655, 0.9003245225862656])
Q0, S20 = 1.0, 9.0
LOG2PI = math.log(2.0 * math.pi)
FIG = os.path.join(HERE, "figures")


# ------------------------------------------------- data, exactly as 0038 made it
def ode(n, alpha, Qseq, rng):
    p = alpha.size
    z = np.zeros(p)
    x = np.zeros(n)
    for t in range(n):
        w = math.sqrt(Qseq[t]) * rng.standard_normal()
        z = np.concatenate([[alpha @ z + w], z[:-1]])
        x[t] = z[0]
    return x


def logscale(n, rng, phi, s):
    lam = np.zeros(n)
    nu = math.sqrt(s * s * (1.0 - phi * phi))
    lam[0] = s * rng.standard_normal()
    for t in range(1, n):
        lam[t] = phi * lam[t - 1] + nu * rng.standard_normal()
    return lam


def kalman(y, alpha, Qseq, s2, burn=0):
    """The oracle: exact filter for a KNOWN per-step Q."""
    p = alpha.size
    F = _companion(alpha)
    m = np.zeros(p)
    P = np.eye(p) * (s2 + float(np.max(Qseq))) * p
    e1 = np.zeros(p)
    e1[0] = 1.0
    nll, cal, k = 0.0, 0.0, 0
    for t, yt in enumerate(y):
        m = F @ m
        A = F @ P @ F.T
        S = A[0, 0] + Qseq[t] + s2
        e = float(yt) - m[0]
        if t >= burn:
            nll += 0.5 * (e * e / S + math.log(S) + LOG2PI)
            cal += e * e / S
            k += 1
        row = A[:, 0] + Qseq[t] * e1
        K = row / S
        m = m + K * e
        P = A
        P[0, 0] += Qseq[t]
        P -= np.outer(K, row)
    return nll / k, cal / k


# ------------------------------------------------------------- the IMM filter
def imm_run(y, pr: Params, order: int = 5, burn: int = 0):
    """Per-node covariances on the SAME grid, kernel and model as OdeFilter.

    One (m_g, P_g) per node, mixed by the transition kernel each step (IMM).
    No new parameters; s = 0 collapses the grid and recovers the shipped
    recursion exactly.  Returns (nll/pt, calibration) over t >= burn.
    """
    p = pr.p
    n = order
    lamP, wP, TP = _chain(pr.phi_P, pr.s_P, n)
    lamM, wM, TM = _chain(pr.phi_M, pr.s_M, n)
    T = np.kron(TP, TM)
    pi = np.kron(wP, wM)
    Qg = pr.Q * np.exp(np.clip(np.repeat(lamP, n), -60.0, 60.0))
    Rg = pr.s2 * np.exp(np.clip(np.tile(lamM, n), -60.0, 60.0))
    G = Qg.size
    F = _companion(np.asarray(pr.alpha))
    e1 = np.zeros(p)
    e1[0] = 1.0

    y0 = float(y[0]) if np.isfinite(y[0]) else 0.0
    m = np.full((G, p), y0)
    P = np.tile(np.eye(p) * float(Rg.max() + Qg.max()) * p, (G, 1, 1))
    nll, cal, k = 0.0, 0.0, 0

    for t, yt in enumerate(y):
        pi_pred = pi @ T                                   # (G,)
        pi_pred = np.maximum(pi_pred, 1e-300)
        mu = (pi[:, None] * T) / pi_pred[None, :]          # mu[i, j] = P(i | next=j)

        m0 = np.einsum("ij,ia->ja", mu, m)                 # mixed state per node
        dm = m[:, None, :] - m0[None, :, :]
        P0 = (np.einsum("ij,iab->jab", mu, P)
              + np.einsum("ij,ija,ijb->jab", mu, dm, dm))

        mp = m0 @ F.T                                      # (G, p)
        Ap = np.einsum("ab,jbc,dc->jad", F, P0, F)
        Ap[:, 0, 0] += Qg                                  # each node's OWN Q
        S = Ap[:, 0, 0] + Rg
        if not np.all(np.isfinite(S)) or np.any(S <= 0.0):
            return math.inf, math.nan
        e = float(yt) - mp[:, 0]

        lg = -0.5 * (np.log(S) + e * e / S)
        mx = float(lg.max())
        w = pi_pred * np.exp(lg - mx)
        Z = float(w.sum())
        ll = math.log(Z) + mx - 0.5 * LOG2PI

        ybar = float(pi_pred @ mp[:, 0])
        S_mix = float(pi_pred @ (S + (mp[:, 0] - ybar) ** 2))
        if t >= burn:
            nll -= ll
            cal += (float(yt) - ybar) ** 2 / S_mix
            k += 1

        K = Ap[:, :, 0] / S[:, None]                       # (G, p)
        m = mp + K * e[:, None]
        P = Ap - K[:, :, None] * Ap[:, None, 0, :]
        pi = w / Z

    return nll / max(k, 1), cal / max(k, 1)


def gpb1_run(y, pr: Params, order: int = 5, burn: int = 0):
    """The shipped filter, scored the same way."""
    f = OdeFilter(pr, order=order).reset()
    nll, cal, k = 0.0, 0.0, 0
    for t, v in enumerate(y):
        st = f.update(float(v))
        if t >= burn:
            nll -= st.loglik
            cal += st.innovation ** 2 / st.pred_var
            k += 1
    return nll / max(k, 1), cal / max(k, 1)


# ----------------------------------------------------------------------- A
def part_a():
    print("A.  VALIDATION and COST")
    rng = np.random.default_rng(11)
    n = 600
    y = ode(n, ALPHA3, np.full(n, Q0), rng) + math.sqrt(S20) * rng.standard_normal(n)

    pr0 = Params(ALPHA3, Q0, S20)                      # s = 0 face: one node
    nll_k, _ = kalman(y, ALPHA3, np.full(n, Q0), S20)
    nll_g, _ = gpb1_run(y, pr0)
    nll_i, _ = imm_run(y, pr0)
    print(f"    s=0 face nll/pt   kalman {nll_k:.10f}")
    print(f"                      gpb1   {nll_g:.10f}   diff {abs(nll_g-nll_k):.2e}")
    print(f"                      imm    {nll_i:.10f}   diff {abs(nll_i-nll_k):.2e}")

    pr1 = Params(ALPHA3, Q0, S20, phi_P=0.9, s_P=0.8)
    t0 = time.perf_counter()
    gpb1_run(y, pr1)
    t_g = time.perf_counter() - t0
    t0 = time.perf_counter()
    imm_run(y, pr1)
    t_i = time.perf_counter() - t0
    print(f"    cost per pass, order 5, n=600:  gpb1 {t_g*1e3:.0f} ms   "
          f"imm {t_i*1e3:.0f} ms   ratio {t_i/t_g:.1f}x")
    return dict(s0_diff_gpb1=abs(nll_g - nll_k), s0_diff_imm=abs(nll_i - nll_k),
                ms_gpb1=t_g * 1e3, ms_imm=t_i * 1e3)


# ----------------------------------------------------------------------- B
def part_b():
    rng = np.random.default_rng(4)                     # 0038 section B's data
    n, lo, hi, mult = 900, 400, 560, 8.0
    Qseq = np.full(n, Q0)
    Qseq[lo:hi] = Q0 * mult
    x = ode(n, ALPHA3, Qseq, rng)
    y = x + math.sqrt(S20) * rng.standard_normal(n)
    burn = 60

    rows = []
    nll_o, cal_o = kalman(y, ALPHA3, Qseq, S20, burn=burn)
    rows.append(("oracle Q_t", nll_o, cal_o))
    nll_f, cal_f = kalman(y, ALPHA3, np.full(n, Q0), S20, burn=burn)
    rows.append(("static Q = 1", nll_f, cal_f))
    qbar = float(np.mean(Qseq))
    rows.append((f"hindsight const Q = {qbar:.2f}",
                 *kalman(y, ALPHA3, np.full(n, qbar), S20, burn=burn)))

    for name, phi, sP in (("s_P = 0.5 forced", 0.90, 0.5),
                          ("s_P = 0.8 forced", 0.90, 0.8),
                          ("s_P = 1.2 forced", 0.95, 1.2)):
        pr = Params(ALPHA3, Q0, S20, phi_P=phi, s_P=sP)
        rows.append((f"gpb1, {name}", *gpb1_run(y, pr, burn=burn)))
        rows.append((f"imm,  {name}", *imm_run(y, pr, burn=burn)))

    span = nll_f - nll_o
    print(f"\nB.  THE ORACLE GAP -- x{mult:.0f} regime, t in [{lo}, {hi}), "
          f"same data as 0038 B")
    print(f"    {'':34s} {'nll/pt':>9s} {'calib':>7s} {'gap closed':>11s}")
    for name, nll, cal in rows:
        frac = "" if name == "static Q = 1" else f"{(nll_f-nll)/span*100:10.1f}%"
        print(f"    {name:34s} {nll:9.4f} {cal:7.3f} {frac:>11s}")
    return dict(rows=[(nm, float(a), float(b)) for nm, a, b in rows],
                span=float(span), y8=y.tolist(), Qseq=Qseq.tolist(), burn=burn)


# ----------------------------------------------------------------------- C
def part_c(b):
    y8 = np.asarray(b["y8"])
    rng = np.random.default_rng(19)                    # 0038 section D's data
    n2 = 900
    y_ar = (ode(n2, ALPHA3, Q0 * np.exp(logscale(n2, rng, 0.9, 0.8)), rng)
            + math.sqrt(S20) * rng.standard_normal(n2))

    grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.4]
    print("\nC.  THE PROFILE -- nll/pt in s_P at phi_P = 0.9, Q pinned")
    out = {}
    for tag, yy in (("x8 regime", y8), ("AR(1) log-scale, true s_P=0.8", y_ar)):
        gp, im = [], []
        for sP in grid:
            pr = Params(ALPHA3, Q0, S20, phi_P=0.9, s_P=sP)
            gp.append(gpb1_run(yy, pr)[0])
            im.append(imm_run(yy, pr)[0])
        out[tag] = dict(grid=grid, gpb1=gp, imm=im)
        amin_g, amin_i = grid[int(np.argmin(gp))], grid[int(np.argmin(im))]
        print(f"    {tag}")
        print("      s_P    " + "  ".join(f"{s:7.2f}" for s in grid))
        print("      gpb1   " + "  ".join(f"{v:7.4f}" for v in gp)
              + f"   argmin {amin_g}")
        print("      imm    " + "  ".join(f"{v:7.4f}" for v in im)
              + f"   argmin {amin_i}")
    return out


# ----------------------------------------------------------------------- D
def part_d(b):
    rng = np.random.default_rng(23)
    n = 900
    y_homo = (ode(n, ALPHA3, np.full(n, Q0), rng)
              + math.sqrt(S20) * rng.standard_normal(n))
    y8 = np.asarray(b["y8"])
    burn = b["burn"]

    print("\nD.  PREMIUM and EXPOSURE (nats/pt)")
    out = {}
    for name, run in (("gpb1", gpb1_run), ("imm", imm_run)):
        on = Params(ALPHA3, Q0, S20, phi_P=0.9, s_P=0.8)
        off = Params(ALPHA3, Q0, S20)
        prem = run(y_homo, on, burn=burn)[0] - run(y_homo, off, burn=burn)[0]
        expo = run(y8, off, burn=burn)[0] - run(y8, on, burn=burn)[0]
        out[name] = dict(premium=float(prem), exposure=float(expo))
        print(f"    {name:5s} unnecessary s_P=0.8 costs {prem:+.4f}   "
              f"missing it costs {expo:+.4f}   asymmetry {expo/max(prem,1e-9):.0f}x")
    return out


def main():
    os.makedirs(FIG, exist_ok=True)
    a = part_a()
    b = part_b()
    c = part_c(b)
    d = part_d(b)
    out = dict(A=a, B={k: v for k, v in b.items() if k not in ("y8", "Qseq")},
               C=c, D=d)
    with open(os.path.join(FIG, "gap0002.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote", os.path.join(FIG, "gap0002.json"))


if __name__ == "__main__":
    main()
