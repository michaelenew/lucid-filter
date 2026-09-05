# 0005 — the faithful multivariate walker works (diagonal, linear in D)

> **⚖️ ATTRIBUTION —** _Shows the walker works once it uses the expected (not observed) Fisher accumulated in a Kalman recursion — a standard natural-gradient / recursive-estimation assembly; "diagonal suffices" is a measured finding on this rig._ Prior art: expected Fisher / natural gradient — Amari 1998; recursive prediction-error estimation — Ljung & Söderström 1983. Status: RECOMBINATION.

Fixes the two failures 0004 diagnosed by copying what the scalar walking filter
actually does: the **expected** Fisher (deterministic given S, not a single-sample
Hessian), an **unbounded** μ-walk (no reversion), and the finding-18 gain
`K*=(1-φ)/4` via a fixed drift variance `q_mu = K*²/(I_char(1-K*))`.

Analytic, for a log-scale axis k multiplying a covariance piece (predictive S,
innovation e):
```
dS_k    = dS/dpsi_k
score_k = 0.5 ( eᵀ S⁻¹ dS_k S⁻¹ e − tr(S⁻¹ dS_k) )        (gradient)
F_kl    = 0.5 tr( S⁻¹ dS_k S⁻¹ dS_l )                       (EXPECTED Fisher)
```

## Result (n=1, m=2, H=[[1],[1]]; sensor 1 hot at eta_1=1.6 mid-stream)

| | hot-band `[xi, eta1, eta2]` | sep `eta1-eta2` | corr vs grid `[xi,eta1,eta2]` |
|---|---|---|---|
| exact grid (ref) | `[-0.02, 1.06, 0.10]` | 0.96 | — (span-capped at ~1.2) |
| **walker, diagonal** | `[-0.18, 1.58, 0.22]` | 1.35 | **[0.90, 0.99, 0.96]** |
| walker, block | `[-0.25, 1.58, 0.20]` | 1.38 | [0.87, 0.99, 0.93] |

## What it settles

1. **The practical multivariate walker works.** It isolates the hot sensor
   (`eta_1 → 1.6`, others near 0), tracks the grid's trajectory (corr > 0.95 on the
   sensors), and — being **unbounded** — reaches the true 1.6 where the fixed grid
   is span-capped at ~1.06. Per-component deduction, achieved.
2. **Diagonal is sufficient; block is not needed here.** DIAG ≈ BLOCK (corr 0.991
   vs 0.988 on the hot sensor). The ~0.2 process↔measurement coupling (0003) does
   not require the full-Fisher natural gradient for this case — the diagonal
   accumulated walker is as good, at **linear cost in D**. (Promote to the block
   only if a higher-coupling regime shows leakage that matters.)
3. **Expected Fisher is the load-bearing choice.** Using `E[-d²loglik]` (a trace in
   S⁻¹ and dS) instead of the finite-difference observed Hessian is what makes the
   natural gradient stable — the deterministic-given-S property, exactly as the
   scalar loop relies on.

## Residual / next

- Small biases (quiet `eta_1 ≈ 0.17`, hot `xi ≈ -0.18`) from running the state KF
  at the point estimate (no GPB1 mixing over a window) and the transient — within
  the 7–14% off-diagonal budget; block doesn't remove them.
- **Not yet:** n>1 process eigenmode axes in the walker (only the scalar `xi`
  tested); the shares/saturation marginal outputs from the walker; a real
  representative H beyond `[[1],[1]]`. These are the production-build items.

Code: `0005_accumulated_fisher_walker.py` (reuses 0004's grid reference + generator).
