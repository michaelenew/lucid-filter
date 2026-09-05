# 0024 — Monogamy belongs to the source ledger (a correction to 0020)

> **AI-generated, not peer-reviewed.** 0020's open, pushed — and the
> correction is worth more than the original claim. Code:
> `0024_the_monogamy_ledger.py`.

> **⚖️ ATTRIBUTION —** _An honest correction plus a reproduction: classical mutual information is genuinely not monogamous (three identical streams share freely), so 0020's Gaussian "budget" was only positive-definiteness of a correlation matrix, not a sharing law; true monogamy (CKW) is verified numerically on 400 random 3-qubit pure states (W saturates, GHZ extremal). The CKW verification is REPRODUCTION of a known theorem._ Prior art: Coffman–Kundu–Wootters monogamy 2000; W/GHZ states. Status: REPRODUCTION (CKW check) + NEGATIVE-RESULT (the classical non-monogamy correction); "two-ledger" framing SPECULATIVE.

0020 measured e^{−2I(1;2)} + e^{−2I(1;3)} ≥ 1 at the Gaussian
boundary and called it "the filter-side CKW". **That name was
wrong.**

- **Classical information is not monogamous at all.** Three
  identical streams share perfectly: I₁₂ = I₁₃ = H, so the "budget"
  is 2e^{−2H} → 0 — violated by as much as you like (measured:
  0.500, 0.031, 0.0005 for alphabets 2, 8, 64). Copying is free;
  nothing in the record ledger forbids sharing.
- **What the Gaussian budget actually is**: under joint Gaussianity
  mutual information is a function of correlation alone, so the
  inequality is exactly positive-definiteness of the correlation
  matrix — true and useful, but a statement about *correlation
  geometry*, not sharing. Exhibited: a sign-copy triple with modest
  pairwise correlations (0.64) whose streams nonetheless share a
  full bit with each other and with the first.
- **The amplitude sector IS monogamous.** For three-party amplitude
  states the CKW inequality τ_{A(BC)} ≥ C²_AB + C²_AC holds —
  verified on 400 random pure states (min slack +0.0195), saturated
  exactly by **W** (0.8889 = 0.8889) and extremal for **GHZ**
  (τ = 1 with zero pairwise concurrence).

**The lesson**: the record ledger shares freely, the source ledger
is budgeted. That is the two-ledger split (0005/0006) appearing in
the *sharing structure itself* — and it relocates the sibling's
monogamy/Tsirelson row from their record ledger to their source
ledger, where the coherent-bank instruments (0006) apply.
