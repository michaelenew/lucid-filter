"""Grid the nuisance: a bank of models, its resolvability landscape, and averaging.

The last commitment is the pair (phi, s).  Rather than assert a point, do here what
the whole programme does one level down -- GRID it, run a bank of models, and
compare.  Two landscapes come out, and they answer two different questions.

1. RESOLVABILITY -- "where does the dip (a dead zone) appear?"
   A model's grid resolves a truth cleanly only while its node spacing stays under
   the dead-zone bound (~2 s, finding 11); past that, the grid-shift score sags or
   inverts between nodes -- a gap the filter cannot see through.  Panel (a) grids
   the SPACING at a fixed kernel and maps the score across the true log-scale, so
   you can read exactly where gaps open as the model coarsens.  This is a property
   of each model's geometry, not of any data.

2. EVIDENCE -- "which (phi, s) does the DATA prefer, and how sharply?"
   Panel (b) maps the exact GENERATIVE marginal log-likelihood over (phi, s) (the
   process channel run at each grid point).  Measured, it is not a peak but a
   broad RIDGE: persistence and swing trade off (high phi + small s buys nearly
   the same log-innovation autocovariance as low phi + large s), so the data only
   weakly identifies the pair at a moderate sample -- the argmax wanders off the
   truth by a realisation.  A broad prior therefore does NOT wash out to a point.

3. TRACKING COST -- "does the weak identification matter?"
   Panel (c) maps the WalkingFilter's scale-tracking RMSE over the same grid.  It
   is nearly FLAT (a ~10% spread across the whole bank), and flattest exactly
   along the evidence ridge.  So the direction the data cannot pin is the
   direction that barely changes the answer: the commitment is irreducible in
   count (2 numbers) but nearly free in effect.  A causal model average over the
   bank tracks at ~oracle, confirming you need not commit to a point at all.

Run: python 0026_grid_the_nuisance.py   (~1-2 min)
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

from gridlab import simulate, uniform_grid, run_channel  # noqa: E402
from statfilter import WalkingFilter, AdaptiveFilter, Params  # noqa: E402
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

PHI_STAR, S_STAR, Q, S2 = 0.9, 0.45, 1.0, 1.0


# ---------------------------------------------------- (a) resolvability landscape
def score_curve(gap, s=0.30, base=2.0, nseed=80, nt=400):
    """Grid-shift score vs true log-scale offset, for a fixed kernel s at spacing gap.

    Grid centred at `base` (loud, clean).  Score>0 = 'truth is above, slide up'.
    A sag or sign flip between nodes is a dead zone -- the dip.
    """
    lam, w0, T = uniform_grid(0.9, s, 4.0, gap)
    offs = np.linspace(-1.6, 1.6, 41)
    out = np.zeros(offs.size)
    for k, t in enumerate(offs):
        X = np.array([simulate(np.random.default_rng(sd), base + t, 1.0, 1.0, nt)
                      for sd in range(nseed)])
        out[k] = run_channel(X, lam, w0, T, 1.0, 1.0, mu=base)["score"].mean()
    return offs, out, lam


# ------------------------------------------------------------- data for (b), (c)
def make(seed, nt=2500):
    rng = np.random.default_rng(seed)
    z = 0.0
    lam = np.zeros(nt)
    for t in range(nt):
        z = PHI_STAR * z + np.sqrt(S_STAR * S_STAR * (1 - PHI_STAR ** 2)) * rng.standard_normal()
        lam[t] = z
    theta = np.cumsum(rng.standard_normal(nt) * np.sqrt(Q * np.exp(lam)))
    x = theta + rng.standard_normal(nt) * np.sqrt(S2)
    return x, lam


def stream(x, phi, s):
    """Per-step loglik and process-scale for one model."""
    f = WalkingFilter(Q=Q, s2=S2, phi=phi, s=s)
    f.reset()
    ll = np.empty(x.size); sc = np.empty(x.size)
    for i, v in enumerate(x):
        st = f.update(v)
        ll[i] = st.loglik; sc[i] = st.process_scale
    return ll, sc


def main():
    # (a) resolvability: fixed kernel s=0.3, spacing gridded coarse -> fine
    gaps = [0.45, 0.75, 1.05, 1.50]      # 1.5s=0.45 (safe) ... up to 5s (coarse)
    curves = {g: score_curve(g) for g in gaps}

    # data, shared by (b) and (c)
    PHIS = np.array([0.7, 0.8, 0.9, 0.95])
    SS = np.array([0.20, 0.30, 0.45, 0.60, 0.80])
    x, lam = make(3)
    warm = 100

    # (b) GENERATIVE evidence landscape (exact marginal loglik of the process channel)
    eland = np.array([[AdaptiveFilter(Params(Q, S2, phi_P=phi, s_P=s)).loglik(x)
                       for s in SS] for phi in PHIS])
    eland = eland - eland.max()
    ebi = np.unravel_index(np.argmax(eland), eland.shape)
    print(f"[evidence] generative argmax phi={PHIS[ebi[0]]:.2f}, s={SS[ebi[1]]:.2f}  "
          f"(truth {PHI_STAR}, {S_STAR}); range {eland.min():.0f} nats over {x.size} pts "
          f"= {eland.min()/x.size:.4f}/pt -> a broad ridge")

    # (c) WalkingFilter tracking-RMSE landscape + causal model average
    keys = [(phi, s) for phi in PHIS for s in SS]
    LL = np.empty((len(keys), x.size)); SC = np.empty((len(keys), x.size))
    for i, (phi, s) in enumerate(keys):
        LL[i], SC[i] = stream(x, phi, s)

    def align(v):
        return v - np.median(v[warm:warm + 300]) + np.median(lam[warm:warm + 300])
    def rmse(v):
        return float(np.sqrt(np.mean((align(v)[warm:] - lam[warm:]) ** 2)))

    rland = np.array([[rmse(SC[keys.index((phi, s))]) for s in SS] for phi in PHIS])
    cum = np.cumsum(LL, axis=1)
    cum_prev = np.concatenate([np.zeros((len(keys), 1)), cum[:, :-1]], axis=1)
    W = np.exp(cum_prev - cum_prev.max(0, keepdims=True)); W /= W.sum(0, keepdims=True)
    r_avg = rmse((W * SC).sum(0))
    r_oracle = rmse(SC[keys.index((PHI_STAR, S_STAR))])
    print(f"[tracking] RMSE spread {rland.min():.3f}..{rland.max():.3f} "
          f"({100*(rland.max()/rland.min()-1):.0f}% across the bank)")
    print(f"[tracking] oracle {r_oracle:.3f} | model-average {r_avg:.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(16.2, 4.6))

    a = ts.tidy(ax[0])
    cols = [ts.SERIES[3], ts.SERIES[5], ts.SERIES[2], ts.SERIES[1]]
    a.axhline(0.0, color=ts.INK, lw=0.7)
    for g, c in zip(gaps, cols):
        offs, sc, lamn = curves[g]
        mark = " — dead zone" if g > 0.62 else ""
        a.plot(offs, sc, color=c, lw=1.7, label=f"gap {g:.2f} = {g/0.30:.1f}s{mark}")
    a.set_xlabel("true log-scale offset from a node  (nats)")
    a.set_ylabel("grid-shift score  (>0: slide up)")
    a.set_title("(a) resolvability: the dip opens as spacing passes ~2s")
    a.legend(loc="upper left", fontsize=7.2, title="spacing (kernel s=0.30)")

    a = ts.tidy(ax[1])
    im = a.imshow(eland, origin="lower", aspect="auto", cmap="viridis",
                  extent=[SS[0], SS[-1], PHIS[0], PHIS[-1]])
    a.scatter([S_STAR], [PHI_STAR], marker="*", s=210, color="white",
              edgecolors=ts.INK, lw=0.8, label="truth", zorder=5)
    a.scatter([SS[ebi[1]]], [PHIS[ebi[0]]], marker="o", s=70, facecolors="none",
              edgecolors="white", lw=1.8, label="argmax", zorder=5)
    a.set_xlabel("s  (scale swing)"); a.set_ylabel("phi  (persistence)")
    a.set_title("(b) evidence: a broad ridge, φ and s trade off")
    a.legend(loc="lower left", fontsize=7.2)
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    cb.set_label("relative log-evidence (nats)", fontsize=7.5)

    a = ts.tidy(ax[2])
    im2 = a.imshow(rland, origin="lower", aspect="auto", cmap="magma_r",
                   extent=[SS[0], SS[-1], PHIS[0], PHIS[-1]])
    a.scatter([S_STAR], [PHI_STAR], marker="*", s=210, color="white",
              edgecolors=ts.INK, lw=0.8, label="oracle", zorder=5)
    a.set_xlabel("s  (scale swing)"); a.set_ylabel("phi  (persistence)")
    a.set_title(f"(c) tracking RMSE: nearly flat ({100*(rland.max()/rland.min()-1):.0f}% spread)")
    a.legend(loc="lower left", fontsize=7.2)
    cb2 = fig.colorbar(im2, ax=a, fraction=0.046, pad=0.04)
    cb2.set_label("scale-tracking RMSE (nats)", fontsize=7.5)
    a.text(0.03, 0.96, f"model-average RMSE {r_avg:.2f}  ≈  oracle {r_oracle:.2f}",
           transform=a.transAxes, fontsize=7.6, va="top", color="white")
    ts.save(fig, os.path.join(HERE, "figures", "0025-grid-the-nuisance.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
