"""0021 -- The N kinds of "x jumped": how separable are the injection directions?

The parent's square was (process / measurement) x (impulse / regime).  Its
process-anomaly corner was odd: "the level jumped" and "a large process-noise
draw" are the SAME event there, with nothing to tell them apart.  The reason is
now visible.  With one state dimension there is only one direction for a
disturbance to enter, so the two descriptions cannot differ.

Once the state is (x, x', x'', ...) a disturbance has a DIRECTION:

    z_t = F z_{t-1} + u w_t

and u is a new axis.  Our model so far pins u = e_1 -- noise in the forcing
only.  Freeing it costs p-1 numbers (u up to scale), taking the model from
p+2 = 5 to 2p+1 = 7 at p=3, which is exactly the identifiable content of a
scalar-observed linear system (0001 section 3).  So the direction axis does not
over-parameterise; it saturates.  In transfer-function terms alpha is the POLES
and u is the ZEROS, and the parent had p=1, hence no zeros, hence no room for
its two descriptions to differ.

Four named corners at p = 3, three in the state and one in the observation:

    POSITION  the position moved, velocity and acceleration did not
    VELOCITY  the velocity moved (a force impulse)
    ACCEL     the acceleration moved (a force step) -- this is the forcing-noise
              direction our model currently pins
    MEASURE   the observation moved and the state did not (an outlier)

In backward-difference coordinates v = (x, Dx, D^2 x) = D z with
D = [[1,0,0],[1,-1,0],[1,-2,1]], and D is an involution (D^2 = I), so a unit
displacement of v_i is the displacement D e_i of the lag state.

What separates them.  The filter is linear, so a deterministic state
displacement produces a deterministic additive signature in the INNOVATION
sequence.  Whitening by the innovation SD, two corners with equal-energy
signatures a and b satisfy

    detection evidence  = ||a||^2 / 2                (against "nothing happened")
    attribution evidence = ||a - b||^2 / 2 = ||a||^2 (1 - cos t)

so, exactly,

    attribution / detection = 2 (1 - cos t),   t the angle between signatures.

One number per pair, and it is the direct generalisation of the parent's
mode-confusion matrix.  Everything here is exact linear algebra on the model's
own covariance -- no simulation, no fitting.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from importlib import import_module  # noqa: E402
_m2 = import_module("0002_identifying_the_dynamics")

M = 24              # post-event observations to look at
CORNERS = ["POSITION", "VELOCITY", "ACCEL", "FORCING", "MEASURE"]


def companion(a):
    p = len(a)
    F = np.zeros((p, p))
    F[0] = a
    if p > 1:
        F[1:, :-1] = np.eye(p - 1)
    return F


def diff_matrix(p):
    """v = D z maps the lag state to (x, Dx, D^2 x, ...): D[i,j] = (-1)^j C(i,j).

    Row i is the i-th backward difference of the lag vector.  D is an
    involution (D^2 = I), so the lag displacement realising a unit move of v_i
    is D e_i = COLUMN i of D -- not row i.  D is not symmetric, and using the
    rows instead gives four corners that all move x_t by 1 and therefore share
    an identical first innovation, which is how the slip announced itself."""
    D = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1):
            D[i, j] = ((-1.0) ** j) * math.comb(i, j)
    return D


def steady_state(F, Q, S2, iters=20000, tol=1e-14):
    p = F.shape[0]
    Qm = np.zeros((p, p))
    Qm[0, 0] = Q
    P = np.eye(p)
    for _ in range(iters):
        Pp = F @ P @ F.T + Qm
        S = Pp[0, 0] + S2
        K = Pp[:, 0] / S
        Pn = Pp - np.outer(K, Pp[0, :])
        if np.max(np.abs(Pn - P)) < tol:
            P = Pn
            break
        P = Pn
    Pp = F @ P @ F.T + Qm
    S = Pp[0, 0] + S2
    K = Pp[:, 0] / S
    return P, Pp, float(S), K


def signature(F, K, S, dz, meas=0.0, m=M):
    """Innovation sequence caused by a state displacement dz at k=0 (and/or an
    observation-only displacement `meas` at k=0), with everything else zero."""
    p = F.shape[0]
    z = dz.astype(float).copy()
    mh = np.zeros(p)                       # filter's state estimate
    g = np.empty(m)
    for k in range(m):
        mp = F @ mh if k > 0 else mh       # at k=0 the jump has just happened
        y = z[0] + (meas if k == 0 else 0.0)
        e = y - mp[0]
        g[k] = e
        mh = mp + K * e
        z = F @ z
    return g / np.sqrt(S)                  # whitened


def main():
    zeta, omega = 0.15, 0.35
    rho, theta = np.exp(-zeta * omega), omega * np.sqrt(1 - zeta ** 2)
    a = _m2.alpha_from_ode(rho, theta)     # unit root + damped oscillator
    p = len(a)
    F = companion(a)
    D = diff_matrix(p)
    Q = 1.0
    d_sd = _m2.simulate(a, 60000, Q, 0.0, np.random.default_rng(5))[0]
    d_sd = float(np.std(np.diff(d_sd)))

    print(f"poles: unit root, rho={rho:.4f}, theta={theta:.4f} "
          f"(period {2*np.pi/theta:.1f} steps)")
    print(f"alpha = {np.round(a, 4)}\nD =\n{D.astype(int)}\n"
          f"D^2 = I: {np.allclose(D @ D, np.eye(p))}\n")

    for kappa in (0.25, 1.0):
        S2 = (kappa * d_sd) ** 2
        P, Pp, S, K = steady_state(F, Q, S2)
        print(f"=== kappa = sigma/SD(dx) = {kappa}   "
              f"innovation SD = {np.sqrt(S):.4f}   gain = {np.round(K,4)} ===")

        sig = {}
        for i, name in enumerate(CORNERS[:3]):
            sig[name] = signature(F, K, S, D[:, i])   # unit move of v_i
        # the direction our model currently pins: w enters x_t, i.e. e_1.  In
        # difference coordinates D e_1 = (1,1,1): position, velocity and
        # acceleration all move together -- a kink.
        sig["FORCING"] = signature(F, K, S, np.eye(p)[0])
        sig["MEASURE"] = signature(F, K, S, np.zeros(p), meas=1.0)

        # detection strength per unit displacement, over the full window
        print(f"  {'corner':>9}  {'|signature|':>12}  first 6 whitened innovations")
        for name in CORNERS:
            g = sig[name]
            print(f"  {name:>9}  {np.linalg.norm(g):12.3f}  "
                  + " ".join(f"{v:+7.3f}" for v in g[:6]))

        # pairwise cos and attribution/detection ratio, against m
        ms = [1, 2, 3, 5, 8, 12, 24]
        print(f"\n  attribution / detection = 2(1 - cos t), by pair and window m")
        hdr = "  " + " " * 22 + "".join(f"{'m='+str(mm):>8}" for mm in ms)
        print(hdr)
        rows = []
        for i in range(len(CORNERS)):
            for j in range(i + 1, len(CORNERS)):
                ci, cj = CORNERS[i], CORNERS[j]
                vals = []
                for mm in ms:
                    ai, aj = sig[ci][:mm], sig[cj][:mm]
                    ni, nj = np.linalg.norm(ai), np.linalg.norm(aj)
                    if ni < 1e-12 or nj < 1e-12:
                        vals.append(np.nan)
                        continue
                    c = float(ai @ aj / (ni * nj))
                    vals.append(2.0 * (1.0 - c))
                print(f"  {ci:>9} vs {cj:<9} " +
                      "".join("     ---" if np.isnan(v) else f"{v:8.3f}"
                              for v in vals))
                rows.append(dict(kappa=kappa, a=ci, b=cj, ms=ms,
                                 ratio=[None if np.isnan(v) else float(v)
                                        for v in vals]))

        # Points needed for 99:1 ATTRIBUTION.  Each corner's event is scaled so
        # its DETECTION evidence over the window is 8 nats -- what a 4-SD
        # single-point event earns, matching the parent's ledger.  Then both
        # whitened signatures are unit vectors scaled by s with s^2 = 16, and
        #     attribution = (s^2/2) ||a_i - a_j||^2 = 16 (1 - cos t).
        print(f"\n  post-event points to reach 99:1 ATTRIBUTION (4.6 nats),"
              f" each event scaled to 8 nats of detection")
        led = []
        for i in range(len(CORNERS)):
            for j in range(i + 1, len(CORNERS)):
                ci, cj = CORNERS[i], CORNERS[j]
                need = None
                for mm in range(1, M + 1):
                    ai, aj = sig[ci][:mm], sig[cj][:mm]
                    ni, nj = np.linalg.norm(ai), np.linalg.norm(aj)
                    if ni < 1e-12 or nj < 1e-12:
                        continue
                    c = float(ai @ aj / (ni * nj))
                    if 16.0 * (1.0 - c) >= 4.6:
                        need = mm
                        break
                led.append(dict(kappa=kappa, a=ci, b=cj, points=need))
                print(f"  {ci:>9} vs {cj:<9} "
                      f"{'never within %d' % M if need is None else need}")
        print()
        if kappa == 0.25:
            keep = (sig, rows, led, S, K)

    # ------------------------------------------------------------- figure
    sig, rows, led, S, K = keep
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    ax = axes[0]
    for i, name in enumerate(CORNERS):
        ax.plot(np.arange(M), sig[name] / np.linalg.norm(sig[name]),
                marker="o", ms=3, color=ts.SERIES[i], label=name)
    ax.axhline(0.0, color=ts.INK, lw=1.0, zorder=0)
    ax.set_xlabel("steps after the event")
    ax.set_ylabel("whitened innovation, unit-norm")
    ax.set_title("The four corners have different signatures")
    ax.legend(fontsize=8)
    ts.tidy(ax)

    ax = axes[1]
    ms = rows[0]["ms"]
    for i, r in enumerate(rows):
        y = [np.nan if v is None else v for v in r["ratio"]]
        ax.plot(ms, y, marker="o", color=ts.SERIES[i % len(ts.SERIES)],
                label=f"{r['a'][:4]}/{r['b'][:4]}")
    ax.set_xscale("log")
    ax.set_xlabel("post-event points m")
    ax.set_ylabel(r"attribution / detection $= 2(1-\cos t)$")
    ax.set_title("How much of the evidence is attributable")
    ax.legend(fontsize=7, ncol=2)
    ts.tidy(ax)
    ts.save(fig, os.path.join(HERE, "figures", "fig15-injection-directions.png"))

    with open(os.path.join(HERE, "figures", "ode021.json"), "w") as f:
        json.dump(dict(alpha=a.tolist(), rho=rho, theta=theta,
                       signatures={k: v.tolist() for k, v in sig.items()},
                       pairs=rows, ledger=led, M=M), f, indent=1)


if __name__ == "__main__":
    main()
