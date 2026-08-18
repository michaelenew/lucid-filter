"""0032 -- One hard series, both filters, everything at once.

A scripted stress series that exercises every axis the two filters claim to
handle, so the difference between them can be looked at rather than tabulated:

  events (impulsive, one-off)      a POSITION kick, a VELOCITY kick and an
                                   ACCEL kick -- the same disturbance carried
                                   through 0, 1 and 2 integrations before it
                                   reaches the observation (0021, 0024).  Per
                                   0025 all three ARE process noise: they move
                                   the innovation mean and leave no trace in
                                   its second moment.
  heteroscedasticity               a measurement-noise regime and, separately,
                                   a process-noise regime.  This is the
                                   parent's own axis and is inherited unchanged.
  evolving dynamics                two jumps in alpha itself -- the one thing
                                   neither filter models.  odefilter reports it
                                   (whiteness); the parent cannot even express
                                   it.

Both filters are fitted on the BASELINE STRETCH ONLY -- everything before the
first jump in alpha -- and then run over the whole series.  That is the honest
deployment setting: you learn from clean history and meet the surprises live.
It also matters for what the picture means.  Fitting across all three alpha
regimes at once was tried first and inflates Q sevenfold (7.29 against a truth
of 1.0), because one static alpha has to compromise between three; the filter
then trusts the observations too much and the comparison is between two
badly-fitted filters rather than between two models.  The measurement-noise
regime falls inside the fitting window, so both filters' scale channels see it;
the process-noise regime and both alpha jumps are genuinely out of sample.

The layout puts the aggregate (rows 1-4) above the detail (row 5) above the
scoreboard (row 6), and every panel shares the timeline.
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "lucid"))
sys.path.insert(0, os.path.join(ROOT, "lucid"))
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from odefilter import OdeFilter, difference_matrix  # noqa: E402
import statfilter  # noqa: E402

N = 1100
H = 10                       # forecast horizon scored throughout

# (z-1)(z^2 - 2 rho cos th z + rho^2): a damped oscillator plus a constant
# offset, which is the class this workstream targets
def alpha3(rho, th):
    c = 2.0 * rho * math.cos(th)
    return np.array([c + 1.0, -(rho * rho + c), rho * rho])


REGIMES = [(0, 620, 0.9489, 0.3462),      # baseline
           (620, 880, 0.9750, 0.6200),    # jump 1: faster, less damped
           (880, N, 0.9200, 0.2400)]      # jump 2: slower, more damped
JUMPS = [620, 880]
KICKS = [(200, 0, "POSITION"), (300, 1, "VELOCITY"), (400, 2, "ACCEL")]
MEAS_REGIME = (470, 600)                  # measurement noise x6
PROC_REGIME = (720, 850)                  # process noise x8
Q0, S20 = 1.0, 9.0


def companion(a):
    F = np.zeros((3, 3))
    F[0] = a
    F[1:, :-1] = np.eye(2)
    return F


def simulate(rng):
    """The scripted series.  Kicks displace the state AFTER emission (0025)."""
    D = difference_matrix(3)
    z = np.zeros(3)
    x = np.zeros(N)
    qmul = np.ones(N)
    smul = np.ones(N)
    qmul[PROC_REGIME[0]:PROC_REGIME[1]] = 8.0
    smul[MEAS_REGIME[0]:MEAS_REGIME[1]] = 6.0

    kick_at = {}
    for t0, i, name in KICKS:
        a = alpha3(*[r[2:] for r in REGIMES if r[0] <= t0 < r[1]][0])
        u = D[:, i]                                  # the i-th derivative corner
        F = companion(a)
        resp = np.array([(np.linalg.matrix_power(F, k) @ u)[0] for k in range(40)])
        # size every kick to the same observable excursion: 6 measurement SD
        kick_at[t0] = (6.0 * math.sqrt(S20) / np.max(np.abs(resp))) * u

    for t in range(N):
        a = alpha3(*[r[2:] for r in REGIMES if r[0] <= t < r[1]][0])
        z = np.concatenate([[a @ z + math.sqrt(Q0 * qmul[t]) * rng.standard_normal()],
                            z[:-1]])
        x[t] = z[0]
        if t in kick_at:
            z = z + kick_at[t]
    y = x + np.sqrt(S20 * smul) * rng.standard_normal(N)
    return x, y, qmul, smul


def run(f, y, parent):
    """Filtered posterior, and the h-step forecast made h steps earlier."""
    f.reset()
    m = np.empty(len(y))
    v = np.empty(len(y))
    fc = np.full(len(y), np.nan)
    for t, val in enumerate(y):
        st = f.update(val)
        m[t], v[t] = st.mean, st.var
        if t + H < len(y):
            fc[t + H] = f.predict(1 if parent else H)[0]
    return m, v, fc


CACHE = os.path.join(HERE, "figures", "ode032_fit.json")


def fit_and_cache(y):
    """Stage 1, cached: the expensive part.  `y` is the baseline stretch.

    Fitting p=3 on this many points takes minutes, and this environment reaps
    long background jobs, so the fitted parameters are written out and reused.
    Delete figures/ode032_fit.json to refit.
    """
    if os.path.exists(CACHE):
        d = json.load(open(CACHE))
        return (OdeFilter.from_dict(d["ode"]),
                statfilter.AdaptiveFilter.from_dict(d["parent"]))
    t0 = time.time()
    fo = OdeFilter.fit(y, p=3, order=5, max_iter=250)
    t1 = time.time()
    print(f"  ode fitted in {t1-t0:.0f}s", flush=True)
    fp = statfilter.AdaptiveFilter.fit(y, order=5, max_iter=250)
    print(f"  parent fitted in {time.time()-t1:.0f}s", flush=True)
    with open(CACHE, "w") as f:
        json.dump(dict(ode=fo.to_dict(), parent=fp.to_dict()), f, indent=1)
    return fo, fp


def main():
    rng = np.random.default_rng(20260801)
    x, y, qmul, smul = simulate(rng)
    fo, fp = fit_and_cache(y[:JUMPS[0]])
    print(f"  ode roots {np.round(fo.params.roots, 4)}")
    print(f"  ode   Q={fo.params.Q:.3f} s2={fo.params.s2:.2f} "
          f"s_P={fo.params.s_P:.3f} s_M={fo.params.s_M:.3f}")
    print(f"  parent Q={fp.params.Q:.3f} s2={fp.params.s2:.2f} "
          f"s_P={fp.params.s_P:.3f} s_M={fp.params.s_M:.3f}")

    ro, rp = fo.filter(y), fp.filter(y)
    mo, vo, fco = run(fo, y, parent=False)
    mp, vp, fcp = run(fp, y, parent=True)

    fo.reset()
    dv = np.empty((N, 4))
    for t, val in enumerate(y):
        fo.update(val)
        dm, dP = fo.derivatives()
        dv[t] = [dm[1], math.sqrt(max(dP[1, 1], 0.0)),
                 dm[2], math.sqrt(max(dP[2, 2], 0.0))]

    # ------------------------------------------------------------- scoring
    phases = [("baseline", 60, 190), ("kicks", 195, 460),
              ("meas. regime", 470, 600), ("dyn. jump 1", 620, 720),
              ("proc. regime", 720, 850), ("dyn. jump 2", 880, 980),
              ("all", 60, N)]
    rows = []
    print(f"\n  {'phase':>14} {'track ode':>10} {'track par':>10} {'ratio':>7}"
          f"   {'fc ode':>9} {'fc par':>9} {'ratio':>7}")
    for nm, a, b in phases:
        s = slice(a, b)
        to_ = float(np.mean((mo[s] - x[s]) ** 2))
        tp_ = float(np.mean((mp[s] - x[s]) ** 2))
        ko = float(np.nanmean((fco[s] - x[s]) ** 2))
        kp = float(np.nanmean((fcp[s] - x[s]) ** 2))
        rows.append(dict(phase=nm, lo=a, hi=b, track_ode=to_, track_parent=tp_,
                         fc_ode=ko, fc_parent=kp))
        print(f"  {nm:>14} {to_:10.2f} {tp_:10.2f} {to_/tp_:7.3f}   "
              f"{ko:9.1f} {kp:9.1f} {ko/kp:7.3f}")

    # ------------------------------------------------------------- figure
    fig = plt.figure(figsize=(14.6, 17.0))
    gs = fig.add_gridspec(6, 4, height_ratios=[3.0, 1.5, 1.1, 1.5, 2.3, 1.5],
                          hspace=0.42, wspace=0.24)
    T = np.arange(N)

    def bands(ax, legend=False):
        ax.axvspan(*MEAS_REGIME, color=ts.SERIES[3], alpha=0.10, lw=0, zorder=0)
        ax.axvspan(*PROC_REGIME, color=ts.SERIES[5], alpha=0.10, lw=0, zorder=0)
        for j in JUMPS:
            ax.axvline(j, color=ts.SERIES[7], lw=1.3, ls="--", zorder=0)
        for tk, _, _ in KICKS:
            ax.axvline(tk, color=ts.INK2, lw=0.9, ls=":", zorder=0)

    # --- row 1: the series
    ax = fig.add_subplot(gs[0, :])
    ax.scatter(T, y, s=3.5, color=ts.INK2, alpha=0.30, linewidths=0, zorder=2,
               label="observed")
    ax.plot(T, x, color=ts.INK, lw=1.3, zorder=4, label="truth")
    for m, v, c, nm in ((mp, vp, ts.SERIES[1], "parent"),
                        (mo, vo, ts.SERIES[0], "odefilter")):
        sd = np.sqrt(np.maximum(v, 0.0))
        ax.fill_between(T, m - 2 * sd, m + 2 * sd, color=c, alpha=0.18, lw=0,
                        zorder=1)
        ax.plot(T, m, color=c, lw=1.2, zorder=3, label=nm)
    bands(ax)
    yl = ax.get_ylim()
    for tk, _, nm in KICKS:
        ax.annotate(nm, (tk, yl[1]), xytext=(0, -11), textcoords="offset points",
                    ha="center", fontsize=7.5, color=ts.INK2, rotation=90,
                    va="top")
    for j in JUMPS:
        ax.annotate("$\\alpha$ jump", (j, yl[1]), xytext=(0, -11),
                    textcoords="offset points", ha="center", fontsize=7.5,
                    color=ts.SERIES[7], rotation=90, va="top")
    ax.annotate("measurement\nregime ×6", (np.mean(MEAS_REGIME), yl[0]),
                xytext=(0, 8), textcoords="offset points", ha="center",
                fontsize=7.5, color=ts.SERIES[3])
    ax.annotate("process\nregime ×8", (np.mean(PROC_REGIME), yl[0]),
                xytext=(0, 8), textcoords="offset points", ha="center",
                fontsize=7.5, color=ts.SERIES[5])
    ax.set_ylabel("level")
    ax.set_title("One series, everything at once: the estimate profiles")
    ax.legend(fontsize=8.5, ncol=4, loc="upper left")
    ts.tidy(ax)

    # --- row 2: forecast error
    ax = fig.add_subplot(gs[1, :])
    k = 21
    ker = np.ones(k) / k
    for fc, c, nm in ((fcp, ts.SERIES[1], "parent"),
                      (fco, ts.SERIES[0], "odefilter")):
        e2 = (fc - x) ** 2
        sm = np.convolve(np.nan_to_num(e2), ker, mode="same")
        ax.plot(T, sm, color=c, lw=1.4, label=nm)
    bands(ax)
    ax.set_yscale("log")
    ax.set_ylabel(f"$h={H}$ sq. forecast\nerror ({k}-pt mean)")
    ax.set_title(f"Forecasting at $h={H}$: the gap, and where it closes")
    ax.legend(fontsize=8.5, ncol=2)
    ts.tidy(ax)

    # --- row 3: whiteness
    ax = fig.add_subplot(gs[2, :])
    ax.plot(T, ro.whiteness, color=ts.SERIES[0], lw=1.5, label="odefilter")
    ax.axhline(0, color=ts.INK, lw=1.0, zorder=0)
    bands(ax)
    ax.set_ylabel("whiteness\n(lag-1 autocorr)")
    ax.set_title("The diagnostic: only a change in $\\alpha$ moves it "
                 "(the parent has no analogue)")
    ts.tidy(ax)

    # --- row 4: scale coordinates
    ax = fig.add_subplot(gs[3, :])
    ax.plot(T, ro.measurement_scale, color=ts.SERIES[0], lw=1.3,
            label="ode: measurement")
    ax.plot(T, ro.process_scale, color=ts.SERIES[2], lw=1.3, label="ode: process")
    ax.plot(T, rp.measurement_scale, color=ts.SERIES[1], lw=1.1, ls="--",
            label="parent: measurement")
    ax.plot(T, rp.process_scale, color=ts.SERIES[4], lw=1.1, ls="--",
            label="parent: process")
    ax.axhline(0, color=ts.INK, lw=1.0, zorder=0)
    bands(ax)
    ax.set_ylabel("log-scale (nats)")
    ax.set_title("The inherited channels: odefilter's process channel is "
                 "flat at zero ($s_P$ fitted to 0) and its measurement "
                 "channel absorbs both regimes")
    ax.legend(fontsize=8, ncol=4)
    ts.tidy(ax)

    # --- row 5: zooms, in error space
    zooms = [(KICKS[0][0], "POSITION kick"), (KICKS[1][0], "VELOCITY kick"),
             (KICKS[2][0], "ACCEL kick"), (JUMPS[0], "$\\alpha$ jump 1")]
    # Levels are useless at this zoom: the tracking error is ~2.5 against a
    # local range of ~150, so all three curves lie on top of each other.  Show
    # the error itself, which is the quantity the comparison is about.
    for i, (tc, nm) in enumerate(zooms):
        ax = fig.add_subplot(gs[4, i])
        w = slice(max(tc - 40, 0), min(tc + 70, N))
        tt = T[w]
        ax.axhline(0, color=ts.INK, lw=1.0, zorder=0)
        for m, c, lab in ((mp, ts.SERIES[1], "parent"),
                          (mo, ts.SERIES[0], "odefilter")):
            ax.plot(tt, m[w] - x[w], color=c, lw=1.2, zorder=3, label=lab)
        ax.axvline(tc, color=ts.SERIES[7] if "jump" in nm else ts.INK2,
                   lw=1.2, ls="--" if "jump" in nm else ":", zorder=0)
        ax.set_title(nm, fontsize=9.5)
        ax.set_xlabel("t")
        if i == 0:
            ax.set_ylabel("estimate $-$ truth")
            ax.legend(fontsize=7.5)
        ts.tidy(ax)

    # --- row 6: the scoreboard
    ax = fig.add_subplot(gs[5, :])
    sel = [r for r in rows if r["phase"] != "all"]
    xs = np.arange(len(sel))
    tr = [r["track_ode"] / r["track_parent"] for r in sel]
    fr = [r["fc_ode"] / r["fc_parent"] for r in sel]
    ax.bar(xs - 0.2, tr, width=0.4, color=ts.SERIES[2], label="tracking MSE")
    ax.bar(xs + 0.2, fr, width=0.4, color=ts.SERIES[0],
           label=f"$h={H}$ forecast MSE")
    ax.axhline(1.0, color=ts.INK, lw=1.2, ls="--", zorder=0)
    for i, v in enumerate(fr):
        ax.annotate(f"{v:.2f}", (i + 0.2, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7.5,
                    color=ts.SERIES[0])
    ax.set_xticks(xs)
    ax.set_xticklabels([r["phase"] for r in sel], fontsize=8.5)
    ax.set_ylabel("odefilter ÷ parent\n(below 1 is better)")
    ax.set_title("The scoreboard, by phase")
    ax.legend(fontsize=8.5, ncol=2)
    ts.tidy(ax)

    for a in fig.axes[:4]:
        a.set_xlim(0, N)
    ts.save(fig, os.path.join(HERE, "figures", "fig21-a-hard-series.png"))

    with open(os.path.join(HERE, "figures", "ode032.json"), "w") as f:
        json.dump(dict(N=N, H=H, phases=rows, jumps=JUMPS,
                       kicks=[[a, b, c] for a, b, c in KICKS],
                       meas_regime=list(MEAS_REGIME),
                       proc_regime=list(PROC_REGIME),
                       ode=dict(Q=fo.params.Q, s2=fo.params.s2,
                                s_P=fo.params.s_P, s_M=fo.params.s_M,
                                phi_P=fo.params.phi_P, phi_M=fo.params.phi_M),
                       parent=dict(Q=fp.params.Q, s2=fp.params.s2,
                                   s_P=fp.params.s_P, s_M=fp.params.s_M,
                                   phi_P=fp.params.phi_P, phi_M=fp.params.phi_M),
                       roots=[[float(z.real), float(z.imag)]
                              for z in fo.params.roots]), f, indent=1)

    # the raw series, so the artifact can draw it interactively
    np.savez_compressed(os.path.join(HERE, "figures", "ode032_series.npz"),
                        x=x, y=y, mo=mo, vo=vo, mp=mp, vp=vp, fco=fco, fcp=fcp,
                        white=ro.whiteness, dv=dv)


if __name__ == "__main__":
    main()
