# 0007 — The coupled bank: the vertex as a shared trust node, its propagator measured

> **AI-generated, not peer-reviewed.** F1 of the sibling's adoption
> plan — the boundary-vertex/propagator doppelganger — now on the
> proven network tier (their 0104: a shared node *is* Bayes-sharing
> between filters, not an analogy). Code: `0007_the_coupled_bank.py`;
> 400 paired trials, no fitted parameters.

**The object.** Two streams, latents independent *given* a shared
log-scale λ (the trust channel; AR(1) with persistence φ, φ = 1 the
pinned root = the sibling's massless mode). The bank is a
Rao-Blackwellized grid filter: per-λ Kalman pairs, jointly
reweighted. The only cross-channel is the shared scale posterior —
the vertex. A +4σ five-step innovation packet enters stream 1; the
propagator is stream 2's response.

**Measured:**

- **The mean channel is systematically silent** (|Δmean| consistent
  with zero at the 400-trial floor), with a real but *incoherent*
  per-trial jitter (rms ~4e−2 of the spike, sign-random) — the
  vertex transfers no signal, only *confidence*. In the sibling's
  terms: the shared node moves the metric, not the state.
- **The variance channel is the propagator**, and its memory is the
  channel's mass:

| φ | peak Δln Var₂ | half-life | tail(+200)/peak | zero-freq weight |
|---|---|---|---|---|
| 0.90 | +0.012 | 16 | 0.000 | 0.27 |
| 0.98 | +0.062 | 43 | 0.046 | 4.35 |
| **1.00 (pinned)** | **+0.138** | 52 | **0.095** | **12.13** |

The massless signature is the low-frequency end, exactly as F1
posed it: the pinned channel's zero-frequency transfer is **45×**
the massive one's and still responding at +400 steps, decaying only
at the data-information rate (new observations slowly re-pin the
scale posterior), not at any dynamical rate. Mass 1−φ reads off the
decay; masslessness reads off the surviving tail.

**Ports back to the physics:**

1. The vertex propagator's long-range behaviour = the shared trust
   channel's persistence; a massless (conserved/pinned) scale mode
   at the vertex ⇒ long-memory cross-response between the streams it
   couples. The sibling's masslessness theorem (pinned root, their
   0096) predicts exactly the φ = 1 column.
2. The channel decomposition is a prediction for the boundary-state
   vertex: *confidence transfers, state does not* — their vertex
   should propagate precision (metric) between subsystems while the
   mean channel stays silent, and the two-ledger split says which
   ledger rides it (the record/modulus one).
3. The instrument: injected innovation packets + paired runs give
   the propagator without solving intertwiner combinatorics — the
   escalation 0079 planned for the F1 stall, now operational.

## Honest limits

- One shared-channel topology (one λ for two streams); the physics
  vertex couples more legs with structure — the multi-leg version
  is a direct extension of the same harness.
- The λ grid filter is exact only up to discretization (41 points);
  the paired design cancels discretization bias in the response.
- "Decays at the data-information rate" is observed (non-exponential
  tail), not yet fit to a rate law; the sibling's capacity results
  suggest ~1/t — worth a dedicated fit with longer windows.

## Open

1. Multi-leg vertex (three+ streams, one shared node): does the
   response split by the sibling's fusion rules?
2. Fit the massless tail's decay law against the information-rate
   prediction.
3. Feed the measured transfer function to the sibling as the
   boundary-vertex propagator's target shape.
