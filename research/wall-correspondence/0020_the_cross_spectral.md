# 0020 — The tangle (F5): the pair-only parameter, and monogamy as positive-definiteness

> **AI-generated, not peer-reviewed.** The last unstarted front.
> Code: `0020_the_cross_spectral.py`.

> **⚖️ ATTRIBUTION —** _Two standard facts: (a) a parameter can be carried entirely in the cross-spectrum/cross-correlation while every marginal is white — ordinary multivariate spectral estimation; (b) the "monogamy" inequality e^{−2I(1;2)}+e^{−2I(1;3)}≥1 is just positive-definiteness of a 3×3 correlation matrix (corrected in 0024, which notes classical info is NOT monogamous). Elementary linear algebra._ Prior art: cross-spectral analysis; positive-definiteness of correlation matrices. Status: RECOMBINATION (measurement); "entanglement/CKW monogamy" correspondence SPECULATIVE (and self-corrected in 0024).

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

## Revision ([`0024`](0024_the_monogamy_ledger.md))

The second bullet's name was wrong. The Gaussian budget is *not*
"the filter-side CKW": classical information is not monogamous at
all (three identical streams share perfectly and drive the budget to
zero). The inequality is exactly positive-definiteness of the
correlation matrix — correlation geometry, not sharing. Genuine
monogamy lives in the **amplitude/source ledger**, where CKW does
hold (verified in 0024). The measurement stands; the interpretation
is corrected.

Open: tangle detection as an instrument (third sibling to
0006/0009).
