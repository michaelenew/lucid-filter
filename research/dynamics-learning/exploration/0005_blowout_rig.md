# 0005 — the blowout rig: 18 ms to detect, side pinned instantly, healthy wheel comes home at 1% — the asymmetric case passes the same frame

The safety rung (`0005_blowout_rig.py`): differential-drive robot at 50 Hz, waypoint
autopilot on measurements, mocap sensing (x, y, heading).  At t* the left wheel's
radius collapses to 0.30 r0 — one column of B, asymmetric, adversarial to a symmetric
prior.  Machine: 0004's stack re-anchored — {nominal, blowL-anchor, blowR-anchor,
walker} × {q, 4q}, anchors at 0.40 r0 (class guess, deliberately not the truth),
walker = augmented EKF on (state, rL, rR) with jump-class drift, cap, shared-event
variance restart.  Radii are control-effectiveness gains, so the dynamics are already
linear in θ — the linearizing coordinate is the physical one.  20 seeds × 4 scenarios.

## Scorecard

| criterion | measured | target | |
|---|---|---|---|
| detection delay | **0.9 ± 0.1 steps = 18 ms** | D* = 1.2 steps (25 ms) | on the frontier |
| side attribution | pinned at 0.9 ± 0.1 steps; **0% wrong-side in the attribution window** | side frontier 0.2 | pass |
| recovery | 1.204 ± 0.022 in [0,50); 1.014 by 200; settled 1.011 ± 0.001 | ≤ 1.2× | pass (first window at the line) |
| recovered radii | rL 0.303 r0 (true 0.30), **rR 1.010 r0** (leak ≈ 1%) | healthy wheel home | pass |
| BLOWR symmetry | identical numbers mirrored | — | pass |
| CALM / GUST | 1.0001 / 1.0024; false-detect 5%/5%, none persistent | ≈ 1.00 | pass |
| cost | 0.83 ms/step numpy, 8 members, n=3+2 | embedded budget | pass |

Frozen-nominal pays **5.2–6.5×** oracle forever — on this rig the latency argument is
visceral: a blowout is a ~29 nats/step event, so the information frontier is ~1 step,
and the machine takes it.  The safety-critical scenario the SUMMARY named ("a filter
that takes 100 steps to notice has already put the state estimate somewhere
dangerous") resolves at less than one control period of latency, with zero tuned
constants: the 30× KL rate (vs the drone's 0.3) buys the 30× faster detection, exactly
as the frontier scaling says it must.

## What is new relative to 0004

- **Side attribution is itself frontier-priced.**  The blowL-vs-blowR llr edge is
  enormous (44 nats/step — opposite yaw signatures), so pinning the side is as fast as
  detection; measured 0% wrong-side decisions during the attribution window, on both
  sides.  The 0052-style leak onto the healthy wheel after the shared-event restart
  (both wheels' variance re-priced to cap, since the event does not say which side) is
  ~1% and transient — the data pins the block immediately.  Block-sparse recovery
  needs no block-sparse *prior* here; the excitation does the sparsification.
- **A readout degeneracy worth recording**: the side sub-competition (anchor weights
  renormalized over static members) is meaningful only while the anchors are live
  competitors.  Once the walker takes over (~300 steps in), the static weights sit at
  the mixing floor and the readout is vestigial noise — 45% of seeds show late
  "wrong-side crossings" of a statistic that no longer means anything, while the
  walker's estimate (rL 0.30, rR 1.01) is the actual attribution.  For 0006: the
  reported fault state after takeover should be read from the refined walker, and the
  sub-competition readout should carry a validity flag (anchor mass above the mixing
  floor), not because the filter needs it — it never thresholds — but because the
  *consumer* of the diagnostic does.

## Ladder status after 0005

Both named acceptance rigs pass the SUMMARY's frame: detection on the derived
(masking-corrected) frontier, recovery ≤1.2× refit-oracle within the system's memory,
calm ≈ 1.00, never worse than frozen, embedded cost, zero tuning constants (hazard
ρ = 1/T, class scale 0.5·θ0, caps = class prior — all labeled commitments).  What
remains for 0006 (unification into `LucidFilter(dynamics=None)`):

- wire the real scale-walk machinery (0052 engine) beside the dynamics bank in place
  of the {q, 4q} toy axis;
- `dynamics=None` (no F0 at all) vs `dynamics≈F0` — the probes all had an F0; the
  None cell needs the walker to start from a generic prior (integrator + cap);
- the 0004 residual (settled 1.075 on the drone from carrying P_θ in the gain; the
  jump-hold θ-prior candidate);
- the detection-readout validity flag above;
- profiler regimes WEIGHT/BLOWOUT and the demo gif.
