# 0020 — The tangle (F5): the pair-only parameter, and monogamy as positive-definiteness

> **AI-generated, not peer-reviewed.** The last unstarted front.
> Code: `0020_the_cross_spectral.py`.

- **The marginal-invisible parameter, tracked**: two streams with
  exactly white marginals whatever θ (verified: all marginal
  statistics carry |corr| < 0.15 with θ), θ carried only in the
  cross-spectral phase. The cross-spectral tracker follows a
  wandering θ_t at RMSE 0.052 vs blind scale 0.423 (8×). The
  parameter lives only in the pair — the filter-side tangle.
- **Monogamy = PSD**: with ρ₂₃ = 0, the correlation matrix is
  positive-definite iff ρ₁₂² + ρ₁₃² ≤ 1 — in information form
  **e^{−2I(1;2)} + e^{−2I(1;3)} ≥ 1**, exact at the Cholesky
  boundary (1.000000) and traced by measured MI on generated banks
  (0.9997). Sharing with one partner caps sharing with another: the
  filter-side CKW, exact in the Gaussian tier.

Open: the non-Gaussian budget (does the bound tighten toward
Tsirelson-like values under amplitude structure? — the bridge to
their P4→Tsirelson debt); tangle detection as an instrument
(third sibling to 0006/0009).
