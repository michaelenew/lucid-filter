# 0005 — The coherent bank: amplitude mixing loses on classical data, and that is the theorem

> **AI-generated, not peer-reviewed.** F4 of the sibling program's
> adoption plan (their 0079) — flagged there as the highest-upside
> experiment. Run here under house rules: every parameter of every
> bank set from the generator, no fits; prequential nats/step, 8
> paired seeds. Code: `0005_the_coherent_bank.py`.

**Question.** The sibling's Born weight mixes *amplitudes* and
squares at readout. If a bank that carries complex amplitudes with
interference beats probability mixing (IMM) on any classical
tracking task, the phase acquires an operational meaning from the
data side.

**Design.** Damped stochastic oscillator, frequency switching
between two regimes, phase continuous across switches; two matched
Kalman trackers in every bank; identical readout mixture. The only
difference: IMM propagates weights by the Bayes recursion; the
coherent bank propagates α_i = √w_i·e^{iφ_i} (φ_i = filter i's own
phase belief) through √M, so transfer carries a
cos(φ₁−φ₂) interference term, and reads out w = |α|². Conditions:
plain hazard (realized switch rate 0.0200), and phase-gated hazard —
switches only near zero crossings (realized rate 0.0129) — with an
IMM+ control that is *given* the true gate as a classical
phase-dependent hazard.

**Result: the coherent bank loses everywhere, cleanly.**

```
plain hazard:   oracle 0.3261   IMM 0.3848   coherent 0.3959
                coherent − IMM = +0.0111 ± 0.0003 nats/step
phase-gated:    oracle 0.3260   IMM 0.3716   coherent 0.3888
                coherent − IMM = +0.0173 ± 0.0003
                IMM+  − IMM = +0.0041 ± 0.0004  (the gate, known
                exactly, also does not help classically)
```

Three readings, in increasing order of importance:

1. **The local one.** Interference perturbs the weight recursion
   away from Bayes; on data whose generator is a classical switching
   process, that deviation is pure KL cost, paid every step. Even
   the engineered phase-sensitive condition gives interference
   nothing to grip: inter-hypothesis phase carries no information a
   classical stream can supply. (Side finding: IMM+ shows hard
   gates + estimation noise are *fragile* — assigning near-zero
   hazard while the gate is actually open misses switches
   catastrophically, costing more than the soft misspecified hazard.
   Structure known exactly can still lose to a robust blur.)
2. **The honest negative as boundary-stone.** F4's upside is dead on
   classical data, and not for a fixable reason: the Born
   structure's utility cannot be discovered by classical tracking.
   Whatever the square buys, it does not buy inference power against
   classical generators.
3. **The port back — this is the sibling's two-ledger theorem,
   operationally.** Their 0086 proved phase lives entirely in the
   *source* ledger and never reaches the *record*; their 0105
   proved the record ledger's loss is prequential code length. Put
   together, those predict exactly this experiment's outcome: no
   record-side score can ever pay the phase, on any classical
   stream. The negative here is the filter-side face of that
   theorem — measured, not argued. Contrapositive, which is the
   real import: **if amplitude structure ever pays prequentially,
   the stream's generator is not classical.** The coherent bank is
   thereby not an inference upgrade but a *detector of source-side
   amplitude structure* — a falsifiable instrument, pointed at
   data, that operationalizes "is this stream quantum?" as a code-
   length comparison.

## Honest limits

- One task family (switching oscillators), one coherent
  construction (√M transfer with model-phase amplitudes). Other
  coherent liftings exist; the two-ledger argument says they share
  the verdict on classical generators, but that is argued, not
  scanned.
- The gated condition's realized rate (0.0129) sits below the
  banks' modeled hazard (0.02) — conservative misspecification,
  shared by all banks equally; the paired comparison is unaffected.
- IMM is the collapsed approximation to exact Bayes, not exact
  Bayes; the coherent bank could in principle have beaten it as a
  different approximation. It did not, anywhere, at 30σ+.

## Open

1. Run the detector version: a stream generated WITH source-side
   amplitude structure (the sibling can supply one from their
   lattice objects) — the coherent bank should win there, and the
   margin is the operational value of the phase. That closes F4's
   loop from the other side.
2. The remaining filter-first queue: F1 (coupled-bank transfer
   function — the boundary-vertex doppelganger), the prequential
   floor for the sibling's level selection, experiment 7 (nodes as
   search barriers).
