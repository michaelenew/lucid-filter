"""The README hero figure, regenerated on the PUBLIC filter: LucidFilter vs an oracle-tuned Kalman.

Same rig as README-001, one important difference: the lucid filter here is the shipped
`from lucid import LucidFilter` -- constructed with NO fit, NO history, no arguments at all.
Three regimes on one series:

  A  STEADY STATE, matched to the Kalman filter's tuning.  The Kalman filter is handed the
     TRUE Q and sigma^2 of this segment.  This is its home ground, and the fair place to ask
     what adaptivity costs when nothing is changing.
  B  A LEVEL JUMP.  Inside the random-walk class (a large process innovation), far outside
     what a fixed gain expects: a constant-gain filter averages it away over ~1/K points; a
     filter whose process scale can spike absorbs it in a few.
  C  A MEASUREMENT-NOISE REGIME CHANGE.  The sensor gets 3x noisier.  The correct response is
     to lower the gain and widen the reported uncertainty; a filter tuned for regime A does
     neither.

Writes figures/hero-lucidfilter-vs-kalman.png and .json (the measured claims the README cites).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                              # research/random-walk-filter
REPO = os.path.dirname(os.path.dirname(ROOT))             # repository root
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

import theory_style as ts                                 # noqa: E402
import matplotlib.pyplot as plt                           # noqa: E402
from lucid import LucidFilter                             # noqa: E402

FIG = os.path.join(ROOT, "figures")

SEED = 11
N = 900
JUMP_AT = 380
JUMP = 9.0
NOISE_AT = 600
Q_TRUE = 0.02        # process variance, all three regimes
S2_A = 1.0           # measurement variance in regimes A and B
S2_C = 9.0           # ... and after the sensor degrades


def generate():
    """The series in the figure: quiet, then a jump, then a noisier sensor."""
    rng = np.random.default_rng(SEED)
    theta = np.cumsum(rng.normal(0.0, np.sqrt(Q_TRUE), N))
    theta[JUMP_AT:] += JUMP
    sd = np.where(np.arange(N) < NOISE_AT, np.sqrt(S2_A), np.sqrt(S2_C))
    return theta, theta + rng.normal(0.0, sd)


def kalman(y, Q, R):
    """A textbook scalar local-level Kalman filter, handed the true Q and R."""
    m, P = y[0], R
    mean = np.empty_like(y)
    var = np.empty_like(y)
    for t, v in enumerate(y):
        P = P + Q
        S = P + R
        K = P / S
        m = m + K * (v - m)
        P = (1.0 - K) * P
        mean[t], var[t] = m, P
    return mean, var


def rise_time(est, theta, start, jump, frac=0.1):
    """Steps after the jump until the estimate has closed 90% of it."""
    err = np.abs(est[start:] - theta[start:])
    below = np.flatnonzero(err < frac * abs(jump))
    return int(below[0]) if below.size else len(err)


def main():
    theta, y = generate()

    r = LucidFilter().filter(y[:, None])       # the public filter: no fit, no history, no arguments
    lucid_m, lucid_v = r.mean[:, 0], r.var[:, 0, 0]

    kal_m, kal_v = kalman(y, Q_TRUE, S2_A)     # told the truth about regime A

    A = slice(60, JUMP_AT)                     # skip both filters' cold start
    C = slice(NOISE_AT, N)
    rmse = lambda e, s: float(np.sqrt(np.mean((e[s] - theta[s]) ** 2)))
    calib = lambda e, v, s: float(np.mean((e[s] - theta[s]) ** 2 / v[s]))

    stats = {
        "seed": SEED, "n": N,
        "truth": {"Q": Q_TRUE, "s2_A": S2_A, "s2_C": S2_C, "jump": JUMP},
        "steady_state": {"lucid_rmse": rmse(lucid_m, A), "kalman_rmse": rmse(kal_m, A)},
        "after_noise_change": {"lucid_rmse": rmse(lucid_m, C), "kalman_rmse": rmse(kal_m, C)},
        "calibration_after_noise_change": {
            "lucid": calib(lucid_m, lucid_v, C), "kalman": calib(kal_m, kal_v, C)},
        "jump_rise_time_steps": {
            "lucid": rise_time(lucid_m, theta, JUMP_AT, JUMP),
            "kalman": rise_time(kal_m, theta, JUMP_AT, JUMP)},
    }
    ss = stats["steady_state"]
    stats["steady_state"]["lucid_penalty_pct"] = 100 * (ss["lucid_rmse"] / ss["kalman_rmse"] - 1)
    nc = stats["after_noise_change"]
    stats["after_noise_change"]["kalman_penalty_pct"] = 100 * (nc["kalman_rmse"] / nc["lucid_rmse"] - 1)

    print(json.dumps(stats, indent=2))
    with open(os.path.join(FIG, "hero-lucidfilter-vs-kalman.json"), "w") as fh:
        json.dump(stats, fh, indent=2)

    # ------------------------------------------------------------------ figure
    t = np.arange(N)
    BLUE, ORANGE = ts.SERIES[0], ts.SERIES[1]
    DOT = "#c9c8c3"
    fig, axes = plt.subplots(3, 1, figsize=(11.4, 9.0), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 1.5], "hspace": 0.16})

    for ax, (est, var, colour, name) in zip(
            axes[:2],
            [(lucid_m, lucid_v, BLUE, "lucid filter"),
             (kal_m, kal_v, ORANGE, "Kalman filter")]):
        ts.tidy(ax)
        ax.axvspan(NOISE_AT, N, color="#f3f2ee", zorder=0)
        ax.plot(t, y, ".", color=DOT, ms=2.6, zorder=1, label="measurements")
        sd = np.sqrt(var)
        ax.fill_between(t, est - 2 * sd, est + 2 * sd, color=colour, alpha=0.18,
                        lw=0, zorder=2, label="filter ±2σ (its own claim)")
        ax.plot(t, theta, color=ts.INK, lw=1.5, zorder=3, label="true value")
        ax.plot(t, est, color=colour, lw=1.7, zorder=4, label=name)
        ax.set_ylabel("level")
        ax.set_ylim(min(y.min(), theta.min()) - 1.0, max(y.max(), theta.max()) * 1.24)
        ax.legend(loc="upper left", ncol=2, fontsize=8.4, columnspacing=1.4,
                  handletextpad=0.6)
        for x in (JUMP_AT, NOISE_AT):
            ax.axvline(x, color=ts.INK2, lw=0.8, ls=(0, (4, 3)), zorder=1.5)

    axes[0].set_title("The lucid filter is told nothing. The Kalman filter is told the truth. "
                      "Same data.", loc="left", pad=30)

    top = axes[0]
    for x0, x1, lab in [(0, JUMP_AT, "A · steady state — the Kalman tuning is exactly right here"),
                        (JUMP_AT, NOISE_AT, "B · a level jump"),
                        (NOISE_AT, N, "C · the sensor gets 3× noisier")]:
        top.annotate("", xy=(x0 / N, 1.045), xytext=(x1 / N, 1.045),
                     xycoords="axes fraction", textcoords="axes fraction",
                     annotation_clip=False,
                     arrowprops=dict(arrowstyle="|-|,widthA=0.35,widthB=0.35",
                                     color=ts.INK2, lw=0.9))
        top.text((x0 + x1) / (2 * N), 1.075, lab, transform=top.transAxes,
                 ha="center", va="bottom", fontsize=8.5, color=ts.INK2,
                 clip_on=False)

    step_word = "step" if stats["jump_rise_time_steps"]["lucid"] == 1 else "steps"
    axes[0].annotate("absorbs the jump in %d %s"
                     % (stats["jump_rise_time_steps"]["lucid"], step_word),
                     xy=(JUMP_AT + 4, theta[JUMP_AT + 4]),
                     xytext=(JUMP_AT - 235, theta[JUMP_AT] + 3.4),
                     fontsize=8.6, color=ts.INK2,
                     arrowprops=dict(arrowstyle="->", color=ts.INK2, lw=0.9))
    axes[1].annotate("takes %d to catch up" % stats["jump_rise_time_steps"]["kalman"],
                     xy=(JUMP_AT + 12, kal_m[JUMP_AT + 12]),
                     xytext=(JUMP_AT - 235, theta[JUMP_AT] + 3.4),
                     fontsize=8.6, color=ts.INK2,
                     arrowprops=dict(arrowstyle="->", color=ts.INK2, lw=0.9))
    axes[1].annotate("tuned for regime A, so it keeps chasing\nnoise it should now be ignoring",
                     xy=(NOISE_AT + 160, kal_m[NOISE_AT + 160]),
                     xytext=(NOISE_AT + 25, theta.max() + 5.4),
                     fontsize=8.6, color=ts.INK2,
                     arrowprops=dict(arrowstyle="->", color=ts.INK2, lw=0.9))

    ax = ts.tidy(axes[2])
    ax.axvspan(NOISE_AT, N, color="#f3f2ee", zorder=0)
    ax.plot(t, np.sqrt(lucid_v), color=BLUE, lw=1.8, label="lucid filter")
    ax.plot(t, np.sqrt(kal_v), color=ORANGE, lw=1.8, label="Kalman filter")
    for x in (JUMP_AT, NOISE_AT):
        ax.axvline(x, color=ts.INK2, lw=0.8, ls=(0, (4, 3)))
    ax.set_ylabel("reported σ")
    ax.set_xlabel("t")
    ax.legend(loc="upper right", ncol=2, fontsize=8.4)
    ax.set_ylim(bottom=0.0)
    ax.annotate("the Kalman filter's σ is flat by construction — it was decided at tuning\n"
                "time and cannot see the data. The lucid filter's rises at every event,\n"
                "and stays raised once the sensor degrades.",
                xy=(770, float(np.sqrt(kal_v[770]))), xytext=(300, 0.06),
                fontsize=8.4, color=ts.INK2, va="bottom",
                arrowprops=dict(arrowstyle="->", color=ts.INK2, lw=0.9))
    ax.set_title("“How sure am I?” — the Kalman filter's answer cannot depend on the data; "
                 "the lucid filter's does.", loc="left", fontsize=9.5, pad=6)

    ts.save(fig, os.path.join(FIG, "hero-lucidfilter-vs-kalman.png"))


if __name__ == "__main__":
    main()
