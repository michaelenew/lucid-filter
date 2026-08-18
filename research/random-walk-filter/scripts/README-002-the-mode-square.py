"""The README's founding-insight figure: the four failure modes are one square.

A monitoring system usually ships four detectors -- one for outliers, one for
level jumps, one for drift changes, one for noise-level changes -- each with its
own threshold, each able to fire when it should not.

They are not four things.  Each noise channel (process, measurement) carries a
log-scale that is an AR(1), and an AR(1) has two ends: impulsive (phi -> 0, a
one-off excursion) and persistent (phi -> 1, a carried-over level).  Two
channels crossed with the two ends of persistence gives four corners of ONE
continuous square, and the filter reports a point inside it at every step.
No thresholds exist because nothing is ever being decided.

The corner geometry is the one measured in THEORY-005 / fig14-deviation-square,
where the shading is the exact expected posterior over (a, phi).  Here the same
square carries a real trajectory, so the claim can be checked rather than
asserted.

The overlaid coordinates are derived from the filter's own four reported mode
coordinates, which are SIGNED log-scale nats splitting each channel's scale into
a new-at-t part and a carried-over part.  Writing

    wPA, wPR, wMA, wMR = |process_anomaly|, |process_regime|,
                         |measurement_anomaly|, |measurement_regime|
    E = wPA + wPR + wMA + wMR

the position plotted is the share of total excitation on each axis:

    a   = (wPA + wPR) / E      0 = all measurement, 1 = all process
    phi = (wPR + wMR) / E      0 = all one-off,     1 = all carried over

Writes figures/hero-mode-square.png.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(ROOT))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "lucid"))

import theory_style as ts                                     # noqa: E402
import matplotlib.pyplot as plt                               # noqa: E402
from statfilter import AdaptiveFilter                         # noqa: E402

import importlib.util                                         # noqa: E402
spec = importlib.util.spec_from_file_location(
    "hero", os.path.join(HERE, "README-001-hero-lucid-vs-kalman.py"))
hero = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hero)

FIG = os.path.join(ROOT, "figures")

CORNERS = {
    "MA": (0.0, 0.0, "an outlier —\none bad reading"),
    "PA": (1.0, 0.0, "a jump —\nthe level really moved"),
    "MR": (0.0, 1.0, "the sensor\ngot noisier"),
    "PR": (1.0, 1.0, "the drift rate\nchanged"),
}


def main():
    theta, y = hero.generate()
    f = AdaptiveFilter.fit(hero.history())
    r = f.filter(y)

    wPA, wPR = np.abs(r.process_anomaly), np.abs(r.process_regime)
    wMA, wMR = np.abs(r.measurement_anomaly), np.abs(r.measurement_regime)
    E = np.maximum(wPA + wPR + wMA + wMR, 1e-12)
    a = (wPA + wPR) / E
    phi = (wPR + wMR) / E

    quiet = slice(60, hero.JUMP_AT - 20)
    jump = hero.JUMP_AT
    noisy = slice(hero.NOISE_AT + 40, hero.N)

    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    ts.tidy(ax)
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)

    # the square itself: a flat, recessive field -- every interior point is
    # as legitimate as every corner, which is the whole claim
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#eef4fc",
                               edgecolor="#cddcf0", lw=1.2, zorder=0))

    ax.plot(a[quiet], phi[quiet], ".", ms=3.4, color="#9d9c97", alpha=0.6,
            zorder=2, label="every step, quiet stretch")
    ax.plot(a[noisy], phi[noisy], ".", ms=3.4, color=ts.SERIES[2], alpha=0.7,
            zorder=3, label="every step, after the sensor degrades")
    ax.plot([a[jump]], [phi[jump]], "*", ms=18, color=ts.SERIES[1],
            mec=ts.SURFACE, mew=1.2, zorder=5, label="the step where the level jumped")

    for key, (cx, cy, plain) in CORNERS.items():
        ax.plot([cx], [cy], "o", ms=9, mfc=ts.SURFACE, mec=ts.INK, mew=1.6, zorder=4)
        ax.text(cx + (-0.05 if cx < 0.5 else 0.05),
                cy + (-0.06 if cy < 0.5 else 0.06),
                "%s\n%s" % (key, plain),
                ha="right" if cx < 0.5 else "left",
                va="top" if cy < 0.5 else "bottom",
                fontsize=9.2, color=ts.INK, linespacing=1.4, zorder=6)

    ax.set_xlim(-0.52, 1.52)
    ax.set_ylim(-0.42, 1.42)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xticklabels(["the sensor", "", "the process"])
    ax.set_yticklabels(["one-off", "", "carried over"])
    ax.set_xlabel("$a$   —   which channel", labelpad=6)
    ax.set_ylabel("$\\varphi$   —   how persistent", labelpad=6)
    ax.set_title("The four failure modes are corners of one smooth square.\n"
                 "The filter reports a point inside it at every step — so there is "
                 "nothing to threshold.", loc="left", pad=16, fontsize=11.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, fontsize=8.8)

    fig.text(0.5, 0.012,
             "Two channels × two ends of persistence = four corners. The ODE filter adds "
             "a third channel — the dynamics —\nand the square becomes a prism with six.",
             ha="center", va="bottom", fontsize=8.6, color=ts.INK2, linespacing=1.5)
    fig.subplots_adjust(bottom=0.24)

    fig.savefig(os.path.join(FIG, "hero-mode-square.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/hero-mode-square.png")


if __name__ == "__main__":
    main()
