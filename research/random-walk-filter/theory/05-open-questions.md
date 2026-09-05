# 05 — What this does not settle

> **⚖️ ATTRIBUTION —** _This file is an honest self-audit of the framework's assumptions and unresolved gaps (Gaussianity, known event time, oracle noise, single-event framing, undetermined ω, unverified √-nats generalisation, no fast rate for the scale plane)._ Prior art: these are limitations of the standard machinery being used, not new claims; the value is the candour and the specific numbers. Status: NEGATIVE-RESULT (a catalogue of measured/argued limitations).

## Assumptions that are doing real work

**Everything is Gaussian and the event time is known.** Both the orthogonality of
the location and scale blocks and the closed-form KLs depend on Gaussianity. The
block orthogonality is robust (it follows from any location-scale family with
symmetric noise), but the *magnitudes* in fig10 are not — heavy tails would move
the location/scale boundary substantially, and heavy tails are one of the nine
probes. Untested here.

**The event time $t_0$ is given.** Every number in [03](03-four-deviation-modes.md)
and [04](04-nats-trust-influence.md) conditions on knowing when the event
happened. A real detector scans over $t_0$ and pays a multiple-comparisons cost
of roughly $\log(\text{window length})$ nats — about 3–5 nats for a window of
20–150. **That is comparable to the entire 99:1 threshold**, and it is larger
than the Occam cost for the event size. It would push most of the "**2**" entries
in the confirmation ledger to 3–4 and could erase the 3-SD row entirely. This is
the single most important gap and it should be closed before anything is built.

> **⚖️ ATTRIBUTION —** _The t₀-scan (changepoint-location) multiple-comparisons cost is ~log(window) nats, comparable to the whole detection threshold — flagged here, computed exactly in 07._ Prior art: the ~log n penalty for scanning an unknown changepoint location is classic (GLR / scan statistics, Willsky & Jones 1976; Siegmund 1985). Status: REPRODUCTION.

**Pre-event $(Q,\sigma^2)$ are known exactly.** Oracle framing, as requested. In
practice they carry the uncertainty computed in [01](01-information-accounting.md),
which at $n$=200 is ±45% on $Q$. Propagating that into the LLR matrix will hurt
the scale plane much more than the location plane, because the scale plane's
discrimination is already only 0.2–0.4 nats at $m$=2.

**One event at a time.** The four modes are treated as alternatives to each other
and to $H_0$. Real series have a jump *during* a $\sigma^2$ regime, which is a
point in the interior of the 4-space, not one of its axes. The score vector
handles this correctly by construction (it is a vector, not a classification),
but the confirmation ledger does not — it is a pairwise-alternative table.

## Things I could not derive

**$\omega$, the hyper-drift rate.** [02](02-relevance-decay-and-the-tail.md)
converts the tail length $L$ into a statement about $\omega$, which is progress —
$\omega$ has units, is estimable, and is the same *kind* of object as $Q$ — but
it is still a free quantity, and asking for *its* drift rate reopens the regress.
The honest claim is only that this level of the regress is cheaper than the
previous ones: the loss surface is flat in $L$ (a 2× error in $\omega$ costs 25%),
whereas $c$ and $\nu$ sat on steep surfaces. It is a better parameter, not the
absence of one.

**Whether $\sqrt{\text{nats}}$ generalises past the location channel.** The
amplitude/energy relation $a_k\propto\sqrt{\Delta\text{nats}_k}$ is verified
exactly for the level, where the estimator is linear in the data. For $Q$ and
$\sigma^2$ the estimator is quadratic in the data, and the corresponding relation
is untested. My expectation is that it holds with the squared data as the
"observations", but that is a conjecture, not a result.

**A rate for the scale plane.** The location plane resolves in 2 points because
its two modes separate at $m$=2. The scale plane also becomes non-singular at
$m$=2 but its evidence accrues at ~0.2 nats/point at $q$=0.05, so it needs tens.
There is no reformulation in this document that speeds that up, and I do not
currently believe one exists — at $q$=0.05, $Q$ genuinely contributes 2.4% of
what is observed. If the process noise is small, the process noise is hard to
see, and no basis change fixes that.

## The one thing I'd want checked before building

The ledger says a 4-SD jump earns 99:1 in two points. That is the oracle bound
with a known $t_0$. Stacking the two known corrections — Occam for $\delta$ (1.6
nats) and the scan over $t_0$ (3–5 nats) — leaves 10.5 − 1.6 − 4 ≈ 5 nats, which
still clears 99:1 but only just, and the fluctuation SD at that level is ±3.
**So the headline result survives, narrowly, and only for large jumps.** Before
any of this becomes a filter, the $t_0$-scan cost should be computed exactly
rather than estimated, because it is the term that decides whether the two-point
rule is real.

## What the next construction should be

Not a filter. The sequential form: an e-process on each of the four axes, with
influence gated by $\Lambda^{\text{robust}}$ through $\sqrt{\sigma(\Lambda)}$
rather than by a threshold crossing. Two properties make it worth building over
the gates that were abandoned:

- it has **no coefficient to choose** — Bayes fixes the log-odds weight at 1, and
  the whitened score removes the unit mismatch that forced $c$, $a_j$ and the 6.0
  into existence;
- it **degrades to $H_0$ safely** — the row-minimum definition returns zero
  evidence exactly when a mode is unidentifiable, so the failure mode is
  inaction, not a wrong action. Every previous gate's failure mode was 8–11× MSE
  in the wrong regime, which is the opposite.

The separation of concerns the analysis forces: anomalies get a fast, saturating,
two-point test that touches only the level; regime changes get a slow accumulator
that touches only the noise parameters; and the two never interact, because their
score blocks are orthogonal.
