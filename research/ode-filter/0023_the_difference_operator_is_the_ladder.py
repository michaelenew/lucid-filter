"""0023 -- Differencing acts on the direction space as (F - I).  Exactly.

0022 ordered the injection corners by the lag-1 autocorrelation of their
innovation signature and called it a ladder.  The user's observation sharpens it
from a list into an operator: successive differences point to higher
derivatives.  This finds the exact form of that, and the exact form is not the
one I first wrote down.

**The identity.**  Let the disturbance's contribution to the observation be the
one-sided sequence r_k(u) = (F^k u)_1 for k >= 0 and 0 for k < 0.  Then

    (D r)(u)  =  h_0(u) * delta  +  r((F - I) u),      h_0(u) = u_1

exactly: differencing the response of direction u gives the response of
direction (F - I)u, plus a spike whose size is the direction's immediate
observable component.  So **differencing is the linear map (F - I) on the
direction space**, and the ladder is its orbit.

Two corrections to what 0022 and my first draft of this file asserted.

1. The lag-1 autocorrelation ABOUT ZERO of a raw response is ~1 for anything
   slowly varying, so it does not order the raw responses at all.  It worked in
   0022 only because the innovation signature oscillates about zero -- the
   filter had already removed the level.  The statistic that does order the raw
   responses is **how many differences it takes to reduce the response to a
   spike**, which is the rung.
2. The claim "a measurement outlier is the first difference of a position jump"
   needs the pre-event baseline.  On an array starting at the event it is false
   -- (1,1,1,...) differences to zero.  On the one-sided sequence, which is what
   the filter actually sees, it is exactly true, because the delta comes from
   the leading edge.

**Where the ladder is exact and where it is not.**  A position jump is the
unit-root eigenvector, so (F - I) annihilates it and only the spike survives:
Delta(position step) = delta = the measurement channel, exactly.  Above that
rung the ladder FAILS, and part D shows it fails even in the continuum limit:
the ACCEL -> VELOCITY alignment saturates at 2/sqrt(10) = 0.632, not 1.

Part E says why, and it replaces the ladder with the right structure.  The
derivative basis is not the modal basis.  Decomposing each derivative-basis
corner over the eigenmodes of F:

    POSITION   offset 1.000   oscillator 0.000     <- pure, and exactly the
                                                      unit-root eigenvector
    VELOCITY   offset 0.064   oscillator 0.998     <- essentially pure
    ACCEL      offset 0.801   oscillator 0.598     <- a mongrel

so "ACCEL" is not a corner at all; it is a mixture of the other two.  The
corners are the **modes of the ODE** -- one per root of the characteristic
polynomial -- not the derivatives:

    MEASURE       outside the state
    OFFSET        the unit root; a permanent level shift
    OSCILLATOR    the complex pair; a 2-D channel (amplitude and phase)

That is the generalisation of the parent's channel axis: **which root of the
characteristic polynomial was excited.**  The parent had one root, hence one
process channel plus measurement, hence two.  It also explains 0022's ledger:
POSITION vs VELOCITY is easy because they are different modes; VELOCITY vs
ACCEL costs 4 points because ACCEL is 60% the same mode; ACCEL == FORCING
because both are mixtures of near-identical composition.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from importlib import import_module  # noqa: E402
_m2 = import_module("0002_identifying_the_dynamics")
_j = import_module("0021_injection_directions")

M = 60
STATE = ["POSITION", "VELOCITY", "ACCEL"]
ORDER = ["ACCEL", "VELOCITY", "POSITION", "MEASURE"]


def response(F, u, m=M, pad=1):
    """One-sided response: `pad` zeros, then h_k = (F^k u)_1."""
    z = np.asarray(u, dtype=float).copy()
    h = np.zeros(m + pad)
    for k in range(m):
        h[pad + k] = z[0]
        z = F @ z
    return h


def spikiness(v, tol=1e-9):
    """Fraction of the sequence's energy outside its single largest entry.

    0 for a pure spike, near 1 for anything spread out.  Scale-free, and it
    does not need the sequence to be centred, which lag-1 autocorrelation
    about zero does.
    """
    v = np.asarray(v, dtype=float)
    e = float(v @ v)
    if e < tol:
        return np.nan
    return float(1.0 - v[np.argmax(np.abs(v))] ** 2 / e)


def rungs_to_spike(v, max_m=6, thresh=1e-8):
    """How many differences reduce the sequence to a single spike (or to 0)."""
    w = np.asarray(v, dtype=float).copy()
    for m in range(max_m + 1):
        s = spikiness(w)
        if np.isnan(s) or s < thresh:
            return m
        w = np.diff(w)
    return None


def alpha_for(zeta, omega, dt):
    rho, th = np.exp(-zeta * omega * dt), omega * np.sqrt(1 - zeta ** 2) * dt
    return _m2.alpha_from_ode(rho, th), rho, th


def main():
    zeta, omega, dt = 0.15, 0.35, 1.0
    a, rho, theta = alpha_for(zeta, omega, dt)
    p = len(a)
    F = _j.companion(a)
    D = _j.diff_matrix(p)
    U = {n: D[:, i] for i, n in enumerate(STATE)}

    print(f"alpha = {np.round(a,4)}   sum = {a.sum():.6f} (unit root)")
    print(f"poles: 1, {rho:.4f} e^(+-{theta:.4f} i)\n")

    # ---------------------------------------------------------------- part A
    print("=== A. the identity  D r(u) = u_1 * delta + r((F - I) u) ===")
    worst = 0.0
    for n, u in U.items():
        lhs = np.diff(response(F, u))
        # r((F-I)u) starts at index 1, not 0: index 0 is the leading-edge delta
        rhs = response(F, (F - np.eye(p)) @ u, pad=1)[:len(lhs)].copy()
        rhs[0] += u[0]
        worst = max(worst, float(np.max(np.abs(lhs - rhs))))
        print(f"  {n:>9}: max |lhs - rhs| = {np.max(np.abs(lhs - rhs)):.3e}")
    print(f"  worst over corners: {worst:.3e}  -> exact\n")

    # ---------------------------------------------------------------- part B
    print("=== B. POSITION is the kernel of (F - I); its difference is MEASURE ===")
    kern = (F - np.eye(p)) @ U["POSITION"]
    print(f"  (F - I) * POSITION = {np.round(kern, 12)}  (norm {np.linalg.norm(kern):.2e})")
    dpos = np.diff(response(F, U["POSITION"]))
    delta = np.zeros_like(dpos)
    delta[0] = 1.0
    print(f"  max |D r(POSITION) - delta| = {np.max(np.abs(dpos - delta)):.3e}")
    print("  -> a measurement outlier IS the first difference of a position jump\n")

    # ---------------------------------------------------------------- part C
    print("=== C. how many differences reduce each corner to a spike? ===")
    R = {n: response(F, u) for n, u in U.items()}
    R["MEASURE"] = np.zeros(M + 1)
    R["MEASURE"][1] = 1.0
    print(f"  {'corner':>9}  {'rungs':>6}   spikiness after m differences")
    tab = {}
    for n in ORDER:
        row = []
        w = R[n].copy()
        for m in range(5):
            row.append(spikiness(w))
            w = np.diff(w)
        tab[n] = row
        rg = rungs_to_spike(R[n])
        print(f"  {n:>9}  {str(rg):>6}   " + "  ".join(
            "  ----" if np.isnan(x) else f"{x:6.4f}" for x in row))
    print("  exact for POSITION and MEASURE; approximate above, because F has"
          "\n  distinct eigenvalues rather than a Jordan block.\n")

    # ---------------------------------------------------------------- part D
    print("=== D. the shift is exact only in the continuum limit ===")
    print("  alignment |cos| between (F - I) u_j and u_{j-1}, against sampling")
    print(f"  {'dt':>7} {'|z|':>7} {'theta':>7}   {'ACCEL->VEL':>11} {'VEL->POS':>10}")
    rows = []
    for dts in (2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02):
        aa, rr, tt = alpha_for(zeta, omega, dts)
        FF = _j.companion(aa)
        DD = _j.diff_matrix(3)
        Im = np.eye(3)
        cs = []
        for j in (2, 1):
            v = (FF - Im) @ DD[:, j]
            w = DD[:, j - 1]
            cs.append(abs(float(v @ w / (np.linalg.norm(v) * np.linalg.norm(w)))))
        rows.append(dict(dt=dts, rho=rr, theta=tt, acc_vel=cs[0], vel_pos=cs[1]))
        print(f"  {dts:7.2f} {rr:7.4f} {tt:7.4f}   {cs[0]:11.4f} {cs[1]:10.4f}")
    print("  -> the ladder is a Jordan chain in the continuum limit and an"
          "\n     approximation at finite sampling; the error is set by how far"
          "\n     the poles sit from 1, i.e. by how smooth the sampled process is.")

    # ---------------------------------------------------------------- part E
    print("\n=== E. the derivative basis is not the modal basis ===")
    ev, V = np.linalg.eig(F)
    Vi = np.linalg.inv(V)
    kroot = int(np.argmin(np.abs(ev - 1.0)))
    print(f"  eigenvalues: {np.round(ev, 4)}")
    print(f"  {'corner':>9}  {'offset mode':>12}  {'oscillator mode':>16}")
    modal = {}
    for n in STATE:
        c = np.abs(Vi @ U[n])
        c = c / np.linalg.norm(c)
        osc = float(np.sqrt(sum(c[m] ** 2 for m in range(p) if m != kroot)))
        modal[n] = dict(offset=float(c[kroot]), osc=osc)
        print(f"  {n:>9}  {c[kroot]:12.3f}  {osc:16.3f}")
    print("  (F - I) is diagonal in the modal basis, entries "
          f"{np.round(ev - 1, 4)},\n  so the modal basis is (F-I)-invariant and "
          "the derivative basis is not.")

    # --------------------------------------------------------------- figures
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    ax = axes[0]
    for i, n in enumerate(ORDER):
        v = R[n][:16]
        s = np.max(np.abs(v))
        ax.plot(np.arange(len(v)) - 1, v / (s if s > 0 else 1.0),
                marker="o", ms=3.5, color=ts.SERIES[i], label=n)
    ax.axvline(0, color=ts.GRID, lw=1.2)
    ax.axhline(0.0, color=ts.INK, lw=1.0, zorder=0)
    ax.set_xlabel("steps relative to the event")
    ax.set_ylabel("observation response (scaled)")
    ax.set_title("Each rung integrates the one below")
    ax.legend(fontsize=8)
    ts.tidy(ax)

    ax = axes[1]
    for i, n in enumerate(ORDER):
        ax.plot(np.arange(5), tab[n], marker="o", color=ts.SERIES[i], label=n)
    ax.set_yscale("symlog", linthresh=1e-9)
    ax.set_xticks(range(5))
    ax.set_xlabel(r"differences applied, $m$")
    ax.set_ylabel("spikiness (0 = a pure spike)")
    ax.set_title("Differencing walks down the ladder")
    ax.legend(fontsize=8)
    ts.tidy(ax)

    ax = axes[2]
    ax.plot([r["dt"] for r in rows], [r["acc_vel"] for r in rows], marker="o",
            color=ts.SERIES[0], label=r"ACCEL $\to$ VELOCITY")
    ax.plot([r["dt"] for r in rows], [r["vel_pos"] for r in rows], marker="s",
            color=ts.SERIES[1], label=r"VELOCITY $\to$ POSITION")
    ax.axhline(1.0, color=ts.INK, lw=1.0, ls="--", zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel(r"sampling interval $\Delta t$")
    ax.set_ylabel(r"$|\cos|$ between $(F-I)u_j$ and $u_{j-1}$")
    ax.set_title("Exact only in the continuum limit")
    ax.legend(fontsize=8)
    ts.tidy(ax)
    ts.save(fig, os.path.join(HERE, "figures", "fig16-difference-ladder.png"))

    with open(os.path.join(HERE, "figures", "ode023.json"), "w") as f:
        json.dump(dict(alpha=a.tolist(), rho=rho, theta=theta,
                       responses={k: v.tolist() for k, v in R.items()},
                       spikiness={k: [None if np.isnan(x) else x for x in v]
                                  for k, v in tab.items()},
                       sampling=rows, modal=modal,
                       eigenvalues=[[float(z.real), float(z.imag)]
                                    for z in ev]), f, indent=1)


if __name__ == "__main__":
    main()
