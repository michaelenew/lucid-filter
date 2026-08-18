"""0025 -- Are the two axes orthogonal, or am I splitting one concept in two?

The caution, from the user: a sudden force producing a shift in x that does NOT
persist over subsequent measurements **is** process noise.  They collapse to the
same value.  What differs is that a true process change maintains the implied
finite-difference relationship over time, whereas a velocity anomaly recovers
according to the existing dynamics.

Taken seriously that is a real hazard for 0021-0024, because every disturbance
measured there was a one-off kick.  If "a velocity anomaly" and "process noise"
are the same object, then the four corners are not four modes -- they are one
mode (process noise) refined by direction, plus measurement noise.  Which is
exactly two concepts, the parent's two, not four.

So the question is whether the DIRECTION axis and the PERSISTENCE axis are
orthogonal.  The prediction, if they are:

  impulsive kick, any direction    a transient in the innovation MEAN, decaying
                                   to zero; second moments unchanged afterwards
  dynamics change (alpha -> alpha')  no unconditional mean effect at all, because
                                   the mismatch (alpha - alpha_hat) . z is
                                   proportional to a RANDOM state; a permanent
                                   change in innovation VARIANCE and
                                   AUTOCORRELATION

If that is what happens, the axes are orthogonal: one lives in the innovation's
first moment, the other in its second.  That is the parent's own split -- its
amplitude conservation law is the first moment, its scale conservation law the
second -- and the direction axis refines the parent's process-anomaly corner
rather than adding modes to it.

If instead a dynamics change also produces a mean transient, or an impulsive
kick also shifts the second moments, the two are entangled and 0022's prism is
the wrong shape.
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

T0, PRE, POST = 400, 120, 60


def run_filter(y, F, K, S, x_kick=None):
    """Steady-state filter; returns the innovation sequence."""
    p = F.shape[0]
    mh = np.zeros(p)
    e = np.empty(len(y))
    for t, yt in enumerate(y):
        mp = F @ mh
        e[t] = yt - mp[0]
        mh = mp + K * e[t]
    return e


def simulate(a_pre, a_post, n, Q, S2, rng, t0, kick=None):
    """AR(p) with an optional coefficient switch at t0 and/or a state kick.

    The kick displaces the state used to generate the FUTURE, applied after the
    observation at t0 has been emitted.  Editing x[t0-1], x[t0-2] in place --
    which a naive lag-vector displacement does -- would retroactively rewrite
    observations that had already been made, and it inflates the pre-event
    variance, which is what made an earlier version of this probe report a
    variance ratio below 1 for a pure kick.
    """
    p = len(a_pre)
    z = np.zeros(p)                     # lag state, z[0] = x_t
    x = np.zeros(n)
    for t in range(n):
        a = a_pre if t < t0 else a_post
        xn = a @ z + np.sqrt(Q) * rng.standard_normal()
        z = np.concatenate([[xn], z[:-1]])
        x[t] = xn                       # emitted before any displacement
        if kick is not None and t == t0:
            z = z + kick                # affects the future only
    return x, x + np.sqrt(S2) * rng.standard_normal(n)


def main():
    zeta, omega = 0.15, 0.35
    rho, theta = np.exp(-zeta * omega), omega * np.sqrt(1 - zeta ** 2)
    a = _m2.alpha_from_ode(rho, theta)
    p = len(a)
    F = _j.companion(a)
    D = _j.diff_matrix(p)
    Q = 1.0
    d_sd = float(np.std(np.diff(
        _m2.simulate(a, 60000, Q, 0.0, np.random.default_rng(5))[0])))
    kappa = 0.25
    S2 = (kappa * d_sd) ** 2
    _, _, S, K = _j.steady_state(F, Q, S2)
    sd = np.sqrt(S)

    # a dynamics change of comparable "size": shift the oscillator pole
    a2 = _m2.alpha_from_ode(0.88, 0.45)
    F2 = _j.companion(a2)
    print(f"pre  alpha = {np.round(a, 4)}   poles 1, {rho:.4f} e^(+-{theta:.3f}i)")
    print(f"post alpha = {np.round(a2, 4)}  poles 1, 0.8800 e^(+-0.450i)")
    print(f"innovation SD at truth = {sd:.4f}\n")

    n, R = T0 + POST + 40, 4000
    master = np.random.default_rng(8675309)
    idx = np.arange(T0 - PRE, T0 + POST)

    scen = {}
    # --- impulsive kicks, one per direction, sized to 4 innovation SD at k=0
    for i, name in enumerate(["OFFSET", "OSCILLATOR"]):
        u = D[:, i]
        # scale so the immediate observable displacement is 4 innovation SD,
        # or, when u_1 = 0, so the peak of the response is 4 SD
        h = np.array([(np.linalg.matrix_power(F, k) @ u)[0] for k in range(20)])
        scale = 4.0 * sd / np.max(np.abs(h))
        E = np.empty((R, len(idx)))
        for r in range(R):
            rng = np.random.default_rng(master.integers(2 ** 63))
            x, y = simulate(a, a, n, Q, S2, rng, T0, kick=scale * u)
            E[r] = run_filter(y, F, K, S)[idx]
        scen["kick " + name] = E

    # --- dynamics change at T0, filter keeps the old alpha
    E = np.empty((R, len(idx)))
    for r in range(R):
        rng = np.random.default_rng(master.integers(2 ** 63))
        x, y = simulate(a, a2, n, Q, S2, rng, T0)
        E[r] = run_filter(y, F, K, S)[idx]
    scen["dynamics change"] = E

    # --- control: nothing happens
    E = np.empty((R, len(idx)))
    for r in range(R):
        rng = np.random.default_rng(master.integers(2 ** 63))
        x, y = simulate(a, a, n, Q, S2, rng, T0)
        E[r] = run_filter(y, F, K, S)[idx]
    scen["nothing"] = E

    # ------------------------------------------------------------- measure
    pre = slice(0, PRE)
    post = slice(PRE, PRE + 40)
    late = slice(PRE + 20, PRE + POST)
    # A correctly specified filter produces WHITE innovations, so lag-1
    # autocorrelation is the clean mismatch statistic; the variance ratio also
    # moves when the process itself gets quieter, which is not mismatch.
    print(f"{'scenario':>18} {'peak |mean|':>12} {'late |mean|':>12} "
          f"{'var ratio':>10} {'autocorr':>12}")
    print("-" * 70)
    rows = []
    for k, E in scen.items():
        mu = E.mean(0)
        se = E.std(0) / np.sqrt(E.shape[0])
        peak = float(np.max(np.abs(mu[post])) / sd)
        latem = float(np.max(np.abs(mu[late])) / sd)
        vr = float(E[:, late].var() / E[:, pre].var())
        Lt = E[:, late]
        prod = (Lt[:, :-1] * Lt[:, 1:]).ravel()
        ac = float(prod.mean() / Lt.var())
        ac_se = float(prod.std() / np.sqrt(prod.size) / Lt.var())
        rows.append(dict(scenario=k, peak_mean_sd=peak, late_mean_sd=latem,
                         var_ratio=vr, autocorr=ac, autocorr_se=ac_se,
                         mean_se_sd=float(np.mean(se) / sd),
                         mean=mu.tolist()))
        print(f"{k:>18} {peak:12.3f} {latem:12.3f} {vr:10.3f} "
              f"{ac:+8.4f}+-{ac_se:.4f}")
    print(f"\n  means are over {R} seeds; the Monte Carlo SE on each mean is "
          f"~{np.mean(scen['nothing'].std(0) / np.sqrt(R)) / sd:.4f} SD")
    print("  'peak |mean|' and 'late |mean|' are in innovation SD; var ratio and")
    print("  autocorr are measured 20+ steps after the event.")

    nothing = [r for r in rows if r["scenario"] == "nothing"][0]
    dyn = [r for r in rows if r["scenario"] == "dynamics change"][0]
    kicks = [r for r in rows if r["scenario"].startswith("kick")]
    print("\n=== verdict ===")
    print(f"  impulsive kicks: peak mean {min(k['peak_mean_sd'] for k in kicks):.2f}"
          f"-{max(k['peak_mean_sd'] for k in kicks):.2f} SD, "
          f"late mean {max(k['late_mean_sd'] for k in kicks):.3f} SD, "
          f"var ratio {min(k['var_ratio'] for k in kicks):.3f}"
          f"-{max(k['var_ratio'] for k in kicks):.3f}")
    print(f"  dynamics change: peak mean {dyn['peak_mean_sd']:.3f} SD, "
          f"autocorr {dyn['autocorr']:+.4f}+-{dyn['autocorr_se']:.4f}")
    print(f"  control        : peak mean {nothing['peak_mean_sd']:.3f} SD, "
          f"autocorr {nothing['autocorr']:+.4f}+-{nothing['autocorr_se']:.4f}")
    kac = max(abs(k["autocorr"]) for k in kicks)
    z_dyn = abs(dyn["autocorr"] - nothing["autocorr"]) / dyn["autocorr_se"]
    print(f"  kicks' worst |autocorr| {kac:.4f}; dynamics change is "
          f"{z_dyn:.1f} SE from the control")

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8))
    ax = axes[0]
    ks = idx - T0
    for i, (k, E) in enumerate(scen.items()):
        ax.plot(ks, E.mean(0) / sd, color=ts.SERIES[i], lw=1.8, label=k)
    ax.axvline(0, color=ts.GRID, lw=1.2)
    ax.axhline(0, color=ts.INK, lw=1.0, zorder=0)
    ax.set_xlabel("steps relative to the event")
    ax.set_ylabel("mean innovation / SD")
    ax.set_title("First moment: only kicks move it")
    ax.legend(fontsize=8)
    ts.tidy(ax)

    ax = axes[1]
    names = [r["scenario"] for r in rows]
    xs = np.arange(len(rows))
    ax.bar(xs - 0.2, [r["var_ratio"] for r in rows], width=0.4,
           color=ts.SERIES[0], label="variance ratio")
    ax.bar(xs + 0.2, [r["autocorr"] for r in rows], width=0.4,
           color=ts.SERIES[1], label="lag-1 autocorrelation")
    ax.axhline(1.0, color=ts.INK, lw=1.0, ls="--", zorder=0)
    ax.axhline(0.0, color=ts.INK, lw=1.0, zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
    ax.set_ylabel("late innovations, 20+ steps after")
    ax.set_title("Second moments: only a parameter change moves them")
    ax.legend(fontsize=8)
    ts.tidy(ax)
    ts.save(fig, os.path.join(HERE, "figures", "fig17-event-vs-parameter.png"))

    with open(os.path.join(HERE, "figures", "ode025.json"), "w") as f:
        json.dump(dict(rows=rows, R=R, kappa=kappa, T0=T0, PRE=PRE, POST=POST,
                       alpha_pre=a.tolist(), alpha_post=a2.tolist()), f, indent=1)


if __name__ == "__main__":
    main()
