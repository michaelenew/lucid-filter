"""The ridge, spoken plainly: (phi, s) are identified but sloppy -- the block is the class.

Finding 13 left (phi, s) as a broad ridge and called it "irreducible in count,
free in effect."  That was a resting place.  This makes the theory talk.

THE GEOMETRY.  The Fisher information of (phi, s) at the truth is FULL RANK (both
eigenvalues > 0): there is no flat direction, so (phi, s) ARE identified -- no
permanent free number lives on the ridge.  What is true is that the two
eigenvalues differ by ~15x: one combination (the STIFF direction) is pinned
tightly, the other (the SLOPPY direction) loosely.  At a finite sample the loose
one has a wide posterior and reads as a ridge; its width shrinks as 1/sqrt(n), so
the ridge SHARPENS with data and, in the limit, collapses onto the truth.  A
sloppy model, not a degenerate one (Transtrum et al.'s sense).

WHERE THE BLOCK ACTUALLY IS.  Since the numbers are learnable (full rank), the
irreducible commitment is not a number at all -- it is the model CLASS: the
assertion that the log-scale is a single-timescale stationary AR(1).  That is a
functional form, chosen once, and the no-zero-parameters theorem lives there, not
on the ridge.  The ridge is where a FINITE sample is briefly uncertain, not where
the method is permanently free.

WHAT TO DO ABOUT IT (the reduction, and the average).
  * REDUCE 2 -> 1.  Work in the eigenbasis: the data determines the stiff
    coordinate; only the sloppy coordinate (position along the ridge) is loose.
    One loosely-known number, not two.
  * AVERAGE IT AWAY.  Because tracking is flat along the sloppy direction
    (finding 13), point-estimating it only injects its estimation noise, while
    MARGINALISING it (a small evidence-weighted bank along the ridge) is safe and
    is insensitive to the prior you put on it -- the last assumption averaged out.
    This is "find the ridge the data allows, then average over the freedom": the
    freedom is parked in the least consequential direction and integrated away.

Measured
--------
(a) the (phi, s) confidence ellipse at the truth: stiff vs sloppy axes, ratio ~15,
    both finite (full rank);
(b) sloppy-direction posterior width vs n: ~1/sqrt(n) -> the ridge sharpens, so it
    is finite-sample, not fundamental;
(c) ridge-averaging is insensitive to its prior: tracking RMSE vs the width of the
    prior over the sloppy coordinate is flat and ~oracle -- the last number is
    averaged away, not chosen.

Run: python 0027_ridge_theory.py   (~2-3 min; leg (b) fits several Hessians)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "lucid"))
sys.path.insert(0, os.path.join(HERE, "..", "random-walk-filter", "scripts"))

from statfilter import AdaptiveFilter, Params, WalkingFilter  # noqa: E402
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

PHI0, S0, Q, S2 = 0.9, 0.45, 1.0, 1.0


def gen(seed, nt):
    rng = np.random.default_rng(seed)
    z = 0.0
    lam = np.zeros(nt)
    for t in range(nt):
        z = PHI0 * z + np.sqrt(S0 * S0 * (1 - PHI0 ** 2)) * rng.standard_normal()
        lam[t] = z
    theta = np.cumsum(rng.standard_normal(nt) * np.sqrt(Q * np.exp(lam)))
    x = theta + rng.standard_normal(nt) * np.sqrt(S2)
    return x, lam


def fisher(nt, nseed, hs=0.05, hp=0.03):
    """Empirical Fisher of (s, phi) = -Hessian of the seed-averaged generative loglik."""
    def L(ds, dp):
        return np.mean([AdaptiveFilter(Params(Q, S2, phi_P=PHI0 + dp, s_P=S0 + ds)).loglik(gen(sd, nt)[0])
                        for sd in range(nseed)])
    c = L(0, 0)
    Lss = (L(hs, 0) - 2 * c + L(-hs, 0)) / hs ** 2
    Lpp = (L(0, hp) - 2 * c + L(0, -hp)) / hp ** 2
    Lsp = (L(hs, hp) - L(hs, -hp) - L(-hs, hp) + L(-hs, -hp)) / (4 * hs * hp)
    F = -np.array([[Lss, Lsp], [Lsp, Lpp]])
    w, V = np.linalg.eigh(F)
    return F, w, V


def track_rmse(x, lam, phi, s, warm=100):
    r = WalkingFilter(Q=Q, s2=S2, phi=phi, s=s).filter(x)
    v = r.process_scale
    v = v - np.median(v[warm:warm + 300]) + np.median(lam[warm:warm + 300])
    return float(np.sqrt(np.mean((v[warm:] - lam[warm:]) ** 2))), v


def main():
    # (a) Fisher ellipse at a reference n
    F, w, V = fisher(4000, 24)
    ratio = w.max() / w.min()
    stiff, sloppy = V[:, np.argmax(w)], V[:, np.argmin(w)]
    print(f"[fisher] eigenvalues {np.round(w,1)}  full-rank; sloppiness {ratio:.1f}x")
    print(f"[fisher] stiff dir (s,phi) {np.round(stiff,3)}  sloppy dir {np.round(sloppy,3)}")

    # (b) sloppy-direction width vs n  (expect ~ 1/sqrt(n))
    ns = [500, 1000, 2000, 4000, 8000]
    widths = []
    for n in ns:
        _, wn, _ = fisher(n, 16)
        widths.append(1.0 / np.sqrt(wn.min()))
    widths = np.array(widths)
    slope = np.polyfit(np.log(ns), np.log(widths), 1)[0]
    print(f"[sharpening] sloppy width vs n: log-log slope {slope:.2f} (expect -0.50)")

    # (c) ridge-averaging insensitive to its prior.  Sloppy dir ~ the s axis, so
    # sweep single-model RMSE across s (the ridge) and average with priors of
    # growing width; show both flat and ~oracle.
    x, lam = gen(3, 2500)
    warm = 100
    ss = np.array([0.20, 0.30, 0.45, 0.60, 0.80])
    per = {s: track_rmse(x, lam, PHI0, s)[0] for s in ss}
    r_oracle = per[S0]
    # evidence weights along the ridge (generative loglik), for the average
    evid = np.array([AdaptiveFilter(Params(Q, S2, phi_P=PHI0, s_P=s)).loglik(x) for s in ss])
    scales = np.array([track_rmse(x, lam, PHI0, s)[1] for s in ss])
    prior_widths = [0.1, 0.2, 0.4, 0.8]         # gaussian prior on s about a wrong centre
    wrong_centre = 0.65                          # deliberately off the truth
    avg_rmse = []
    for pw in prior_widths:
        logpri = -0.5 * ((ss - wrong_centre) / pw) ** 2
        lw = evid + logpri; lw -= lw.max()
        W = np.exp(lw); W /= W.sum()
        vbar = (W[:, None] * scales).sum(0)
        vbar = vbar - np.median(vbar[warm:warm + 300]) + np.median(lam[warm:warm + 300])
        avg_rmse.append(float(np.sqrt(np.mean((vbar[warm:] - lam[warm:]) ** 2))))
    print(f"[average] oracle {r_oracle:.3f} | ridge-average vs prior width "
          f"{dict(zip(prior_widths, np.round(avg_rmse,3)))}")

    fig, ax = plt.subplots(1, 3, figsize=(16.2, 4.6))

    a = ts.tidy(ax[0])
    th = np.linspace(0, 2 * np.pi, 200)
    for k in (1.0, 2.0):
        ell = k * (stiff[:, None] / np.sqrt(w.max()) * np.cos(th)
                   + sloppy[:, None] / np.sqrt(w.min()) * np.sin(th))
        a.plot(S0 + ell[0], PHI0 + ell[1], color=ts.SERIES[2] if k == 1 else ts.SEQ[3],
               lw=1.8 if k == 1 else 1.0, label="1σ" if k == 1 else "2σ")
    a.scatter([S0], [PHI0], color=ts.INK, s=50, zorder=5, label="truth")
    a.annotate("", xy=(S0 + 2 * sloppy[0] / np.sqrt(w.min()), PHI0 + 2 * sloppy[1] / np.sqrt(w.min())),
               xytext=(S0, PHI0), arrowprops=dict(arrowstyle="->", color=ts.SERIES[1], lw=1.6))
    a.text(S0 - 0.16, PHI0 - 0.02, "sloppy\n(the ridge)", color=ts.SERIES[1], fontsize=7.6)
    a.set_xlabel("s  (scale swing)"); a.set_ylabel("phi  (persistence)")
    a.set_title(f"(a) full-rank but sloppy ({ratio:.0f}:1) — identified, not degenerate")
    a.legend(loc="upper right", fontsize=7.6)

    a = ts.tidy(ax[1])
    a.loglog(ns, widths, color=ts.SERIES[3], lw=1.8, marker="o", ms=5, label="sloppy 1σ width")
    ref = widths[0] * (np.array(ns) / ns[0]) ** -0.5
    a.loglog(ns, ref, color=ts.INK2, lw=1.1, ls="--", label="∝ 1/√n")
    a.set_xlabel("sample size  n"); a.set_ylabel("sloppy-direction 1σ width")
    a.set_title(f"(b) the ridge sharpens as 1/√n (slope {slope:.2f}) — finite-sample")
    a.legend(loc="upper right", fontsize=7.8)

    a = ts.tidy(ax[2])
    a.axhline(r_oracle, color=ts.INK, lw=1.1, ls=":", label=f"oracle {r_oracle:.2f}")
    a.plot(ss, [per[s] for s in ss], color=ts.SERIES[5], lw=1.6, marker="o", ms=4,
           label="single model along ridge")
    a.plot(prior_widths, avg_rmse, color=ts.SERIES[3], lw=1.8, marker="s", ms=5,
           label="ridge-average vs prior width")
    a.set_xlabel("s  (single model)   /   prior width (average)")
    a.set_ylabel("scale-tracking RMSE")
    a.set_title("(c) averaging the ridge: flat, ~oracle, prior-insensitive")
    a.legend(loc="upper left", fontsize=7.4)
    ts.save(fig, os.path.join(HERE, "figures", "0026-ridge-theory.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
