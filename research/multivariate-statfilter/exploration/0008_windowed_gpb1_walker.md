# 0008 — the windowed / GPB1 walker: walking-only is viable

Resolves the 0007 bracket. The walker carries a scale posterior `(mu, Sigma)`
[diagonal] and uses **2D+1 sigma points** — the linear-in-D realisation of "simplex
for direction + a marginal". Each step: place sigma points from the predicted
`(mu, Sigma)`, **run the state KF at each** (mixing it over the scale window),
reweight by likelihood, and moment-match to update the scale (a 2D+1-particle GPB1);
collapse the state over the same weights. Model: n=2, m=2, correlated PD Q, mixing H
(from 0006).

## The support-width thesis, confirmed

Reach needs support that spans the truth. Narrowing the **grid's** span under-reaches
the truth (1.4):

| grid order | span | eta2 hot-band |
|---|---|---|
| 3 | ±0.60 | 0.53 |
| 5 | ±1.20 | 0.94 |
| 7 | ±1.80 | 0.95 |

A *moving* window narrows and loses that far support — which is why the first
windowed run (small window-growth `q`) under-reached (0.02→0.22).

## The under-reach was `q`-tuning, not fundamental

Cranking the window-growth `q` (unbounded predict, `Sigma <- Sigma + q`) reaches:

| q | eta2 hot-band | corr vs grid |
|---|---|---|
| 0.02 | 0.22 | 0.22 |
| 0.10 | 0.82 | 0.52 |
| 0.30 | 1.38 | 0.90 |
| 1.00 | 1.49 | **0.99** |

With enough `q` the windowed walker **reaches the true 1.4 and tracks the grid at
corr 0.99**, at linear (2D+1) cost. So **walking-only multivariate is viable** — the
earlier "brackets but neither matches" (0007) was the small-`q` corner.

## Two residual items (well-defined)

1. **Reach vs stability is a `q` trade-off** (the finding-18 tension). Larger `q`
   tracks better but drifts/leaks more on static data (q=1.0 static: `eta1≈-0.25`;
   q=0.3 static: `eta2≈0.21`). The right `q` is a critical-damping choice — the
   **finding-18 analogue for this windowed loop**, to be *derived*, not tuned. (The
   scalar `q_mu = K*²/(I_char(1-K*))` ≈ 0.01 here is the under-reaching corner, so the
   windowed loop's derived `q` is a different, larger constant — an open derivation.)
2. **Residual cross-axis leakage** from the process↔measurement coupling (0003's ~0.2
   cross-term): hot xi2 leaks into eta2 (~0.6 at q=0.3), which the *diagonal*
   sigma-points don't fully de-mix. Candidate fix: joint sigma points in the one
   process↔measurement 2-block (still linear-ish), or accept it as bounded bias.

## State of the construction

Walking-only multivariate per-component deduction **works** (reaches, tracks the
exact grid at corr up to 0.99, linear cost, stable with the mixing) once `q` is set
in the reaching regime. Remaining before production: derive `q` (finding-18 analogue),
handle/bound the one coupling block, and produce the shares/saturation marginal from
the sigma-point posterior (already available as the weighted `S`-decomposition).

Code: `0008_windowed_gpb1_walker.py`.
