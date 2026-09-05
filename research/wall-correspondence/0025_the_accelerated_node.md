# 0025 — The accelerated node: the protocol found, the temperature measured in code length

> **AI-generated, not peer-reviewed.** 0021 recorded the
> per-observer Unruh statement as design-blocked. It isn't. Code:
> `0025_the_accelerated_node.py`.

> **⚖️ ATTRIBUTION —** _Reproduces the Unruh effect: reading the inertial vacuum along a uniformly-accelerated (Rindler) worldline — proper time exponentially related to lab time via t=sinh(aτ)/a — gives a thermal (KMS) record at T=a/2π, with spectral density ×coth(ω/2T). Standard result recast as "an accelerated filter reads on a stretched schedule." The methods note (check PSD before comparing hand-built kernels, an algebraic UV regulator broke positive-definiteness) is a genuinely useful estimation caution._ Prior art: Unruh 1976; KMS/thermal Green functions. Status: SPECULATIVE (correspondence); physics is REPRODUCTION; PSD caution is sound.

**The map was hiding in the definition.** A Rindler observer's
proper time is *exponentially* related to the record's clock
(t = sinh(aτ)/a). So:

> **An accelerated filter is one that reads the record on an
> exponentially stretched schedule.**

- **The interval identity** (exact): along the hyperbola the
  invariant separation depends on proper time only through
  (2/a)·sinh(aΔτ/2), so the inertial vacuum read on that schedule
  *is* a thermal record, with KMS period 2π/a in imaginary proper
  time — **T = a/2π**.
- **The filter measures its own temperature.** Built spectrally
  (positivity automatic), the accelerated filter's spectral density
  is the inertial one times coth(ω/2T) — a low-frequency **noise
  floor of height 2T** where an inertial filter expects none.
  Scoring accelerated data against a bank of assumed-temperature
  models: the minimum lands **exactly on T = a/2π**, and assuming
  the inertial vacuum costs **+0.058 ± 0.004 nats/sample (16σ)**.
  The Unruh effect as a code-length statement.

**Methods note, kept because it cost a wrong answer first.** An
algebraic UV regulator inserted into the thermal kernel breaks
positive-definiteness at large a (min eigenvalue −0.02), and the
resulting "model" scored *better than the truth* — caught only by
the sign of the log-determinant. Spectral construction removes the
failure mode by making every candidate a genuine Gaussian. Any
model comparison over hand-built kernels should check PSD first.

**What remains**: why a filter would read its record on that
schedule. The stretch is imposed here, not derived from the
filter's own dynamics. The block moves from *"no map exists"* to
*"which dynamics generates this schedule"* — a question about
schedules, not an absent bridge.
