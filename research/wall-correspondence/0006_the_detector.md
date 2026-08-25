# 0006 — The detector: the phase's operational value, measured

> **AI-generated, not peer-reviewed.** The other half of 0005: it
> proved amplitude mixing cannot pay on classical streams; here is
> the stream where it must pay, and the measurement of how much.
> No fitted parameters anywhere. Code: `0006_the_detector.py`.

**Generator with source-side amplitude structure**: a discrete-time
weakly monitored two-state system — unitary rotation U(θ) each step,
then a weak two-outcome measurement of strength k; the record is a
bit stream. **Banks**: the coherent bank (the amplitude filter —
the generator's own conditional law); the decohered shadow (same
|U|² transitions, same outcome likelihoods, phases discarded — an
exact 2-state HMM: everything probability mixing can know); and an
8th-order empirical Markov predictor trained on a separate 400k
stream (unbounded classical correlation capture, training not
charged).

**The calibration** (nats/bit, 6 seeds; gap = coherent's win = the
operational value of the phase):

| θ | k = 0.2 | 0.5 | 0.8 | 0.98 | 1.0 (projective) |
|---|---|---|---|---|---|
| 0.5 | +0.0117 | +0.0672 | +0.0622 | +0.0133 | — |
| 1.0 | +0.0101 | +0.0606 | **+0.0866** | +0.0201 | **+0.0001** |

- **The coherent bank wins on the amplitude-source stream,
  everywhere, decisively** (100σ+ at the operating points) — the
  contrapositive of 0005 realized: a prequential win for amplitudes
  is possible, and only the generator's non-classicality supplies it.
- **The win is interference, not memory**: markov-8, allowed to
  memorize eight lags of the true process exactly, recovers most of
  the naive HMM's loss but leaves **+0.0110 ± 0.0001 nats/bit** it
  cannot touch. Phase is not a correlation you can tabulate.
- **The projective limit is the classical limit, exactly**: at
  k = 1 the gap is +0.0001 — the stream degenerates to the Markov
  chain on |U|², and probability mixing is optimal again. The
  detector's null is built in.

**The instrument, stated**: run the coherent bank and its decohered
shadow on any stream; the paired code-length difference is a
calibrated, falsifiable statistic for source-side amplitude
structure. Zero on every classical generator (0005, and the
two-ledger theorem behind it); positive only when the source
carries phases. F4 is closed: the Born structure's utility is not
inference power — it is *fidelity to non-classical sources*, and
the margin is measurable in nats.

## Honest limits

- The detector is matched (it knows U and k). An unmatched version
  — scanning a model family against a stream of unknown provenance —
  needs the family priced; the matched version establishes the
  principle and the ceiling.
- Two-state generator only; the monotone rise of the gap with k up
  to ~0.8 and collapse thereafter (information–disturbance
  trade-off) is measured, not derived.

## Open

1. Port back: the sibling's record streams (the ledger with live
   source phases) run through this instrument — does their derived
   measure sit at the matched ceiling?
2. The unmatched detector with a priced model family (the practical
   instrument).
