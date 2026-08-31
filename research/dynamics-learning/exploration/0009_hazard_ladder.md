# 0009: the hazard is not a labeled prior — ladder it and read the regime off the data

**The charge (review, 2026-08-31).**  The class commitment carried "one labeled prior, the
hazard ρ" since 0001, defended as being of the same standing as `forget`.  The review broke
that defense with a test worth keeping: *unless accuracy is monotonic in the parameter, it is
a tuning parameter, not a compute budget.*  ρ fails it — two measured, monotone, opposing
effects:

- calm accuracy degrades monotonically in ρ (0006: ρ = 1/7000 → calm floor 1.148,
  ρ = 1/50000 → 1.066);
- detection delay improves monotonically in ρ (0001: `D* = log(1/ρ)/KL`).

Two opposing monotone effects is a Pareto trade-off, and a parameter on a trade-off is a
knob.  `forget` survives the same scrutiny for a different reason: it is the *practical
escape rate on the one maintained assumption* (a constant AR family — how fast that
assumption is allowed to be violated), and tracking is measured-insensitive to any value
near 1.  ρ had neither defense.  Worse, it had the caller telling the filter the regime,
where every other nuisance here — the `(phi, s)` box, the split ladder, the offset classes —
is gridded and weighted by evidence so the filter tells the caller.

The in-repo precedent is `research/random-walk-filter/original_chat.md`: a fixed-hazard sweep
there produced a clean U-shape in MSE (interior optimum — the signature of a knob), with the
optima for two information measures 30× apart; and the second knob of that thread, the fault
kernel's own drift, is this channel's `q_g = σ²ρ`.

**Two candidate mechanisms, one rejected.**  (B) Learn the rate online by the
Gamma–Poisson conjugate update (Jeffreys ρ(λ) ∝ λ^(-1/2), soft counts) — REJECTED at review:
its effective hazard `α/(β+t)` anchors to the filter's own start time, so two filters switched
on at different moments disagree about the same world; time-translation invariance is not
negotiable.  (A) The house rule: a hazard LADDER mixed by running predictive likelihood,
exactly as the `(phi, s)` box — no special time, no soft-count circularity.  Built and
measured here.

## The derived ladder (shipped as `_hazard_ladder`)

- **Top = 1/2 per step**: the fault class's own persistence boundary.  The class says a fault
  *persists*; above 1/2 the dynamics would leave a hypothesis more often than stay — that is
  not a rare-large-persistent world, so it is not a rung of this class.  (It is also exactly
  the memoryless kernel at bank size 2, the largest hazard whose uniform-leak chain stays
  stochastic for every k ≥ 2.)
- **Bottom = one fault per weight memory** `1/T`, `T = 1/(1−forget)`: the most quiet the
  filter's memory can ever claim to have witnessed.  This is *load-bearing*, not
  tail-trimming: the calm equilibrium slides onto ever-lower rungs (each decade down cuts the
  hedging tax but deepens the detection launch by log 10), so an unbounded ladder would drag
  the launch without the weights ever having accumulated the quiet that justifies it.
  Measured below: one extra decade costs ~+13 steps of delay for ~nothing in calm.
- **Decade rungs, uniform initial weights** = the log-uniform reference prior over the rate.
  Refining the spacing is a compute budget in the `order` sense: monotone, no trade-off.
- Default (`forget = 0.999`): rungs (0.5, 0.05, 5e-3, 5e-4).

Mechanics: the posterior lives on (rung j) × (hypothesis d) × (noise cell c); each rung runs
the Shiryaev kernel at its own ρ_j and carries its own departure walker with the rung's own
second moment `q_g = σ²ρ_j`; the nominal and the anchors carry no rate, so those member
filters are shared across rungs exactly (weight rows differ, recursions do not — the `_wm`
dedup).  The gap kernel is now the exact chain power (`λ2^a` in the shared eigenframe) —
the old `1-(1-ρ)^a` closed form is its O(ρ²) approximation and leaves the simplex above
ρ = (k−1)/k, which the top rung reaches.  Pinning (`faults=ρ`) is the length-1 ladder and
reproduces the old implementation to machine precision at unit gaps (measured: max mean
diff 8.9e-16, loglik diff 0.0 over 400 steps).

## Measured (scalar 0001-style rig, A 0.90 → 0.55 at t* = 1500, two sensors, 20 seeds)

All arms 20 seeds (change world seeds 100–119, no-change 900–919); delay = first
`fault > 0.5` after t* (the 0001 reporting convention); false = crossing fraction on
[300, t*); calm rmse from the no-change arm; hz = the `r.hazard` readout.

| arm | delay | false% | calm rmse | recov | settled | hz(calm) | hz(end) |
|---|---|---|---|---|---|---|---|
| **ladder (faults=True)** | **82.0 ± 8.3** | **1.38%** | **0.32460 ± 0.00219** | 0.2902 | 0.2924 | 1.13e-3 | 1.77e-3 |
| ladder + decade below | 100.5 ± 10.9 | 1.01% | 0.32455 ± 0.00220 | 0.2909 | 0.2927 | 6.2e-4 | 1.4e-3 |
| pinned 0.5 | 1.4 ± 0.3 | 35.31% | 0.34310 ± 0.00247 | 0.2890 | 0.2978 | 0.5 | 0.5 |
| pinned 0.05 | 11.4 ± 2.3 | 16.80% | 0.33199 ± 0.00244 | 0.2889 | 0.2977 | 0.05 | 0.05 |
| pinned 5e-3 | 39.1 ± 4.4 | 3.51% | 0.32563 ± 0.00224 | 0.2893 | 0.2949 | 5e-3 | 5e-3 |
| pinned 5e-4 | 91.8 ± 8.1 | 1.15% | 0.32440 ± 0.00219 | 0.2910 | 0.2916 | 5e-4 | 5e-4 |

What the table says:

1. **The pinned sweep exposes the knob exactly as charged** — delay 1.4 → 91.8 and false
   35.3% → 1.15% fall, calm cost 0.3431 → 0.3244 rises, monotonically across three decades.
   No pinned value is defensible.
2. **The ladder sits at its derived bottom rung's operating point without being told**:
   calm RMSE 0.32460 ± 0.00219 vs the bottom pin's 0.32440 ± 0.00219 (indistinguishable),
   false 1.38% vs 1.15%, delay 82.0 ± 8.3 vs 91.8 ± 8.1 — slightly FASTER, because the upper
   rungs' standing fault mass shallows the mixture's launch.  Cost: ~2.5× the members (the
   shared-nominal dedup keeps it (nb+J)/(nb+1), not J×).
3. **The bottom end is priced, not asserted**: one extra decade below the floor slides part
   of the calm weight down (hz(calm) 1.13e-3 → 6.2e-4) and pays the predicted deeper launch
   — delay 82.0 → 100.5, ≈ +log10/KL at the measured partial migration — for a calm gain of
   0.00005 RMSE (nothing).  An unbounded ladder would keep sliding; the `forget`-derived
   floor is where the slide is stopped by construction, at the most quiet the weight memory
   can actually have witnessed.
4. **The regime is read, not told**: calm settles the readout near the ladder floor
   (1.13e-3), the one fault lifts it (1.77e-3), and in the fault-RICH world below it climbs
   to 4.4e-3 against a true event rate of 8/1600 = 5e-3 — the filter reports, within 13%,
   the rate of the world it is actually in.  A pinned hazard reports its own input back.

**The fault-rich world** (A alternates 0.90 ↔ 0.55 every 150 steps, 8 events, 20 seeds;
delay per A0 → A1 edge, censored at 150):

| arm | first-event delay | late-event delays | rmse | hz(end) |
|---|---|---|---|---|
| **ladder (faults=True)** | 66.6 ± 8.2 | **59.0 ± 7.4** | 0.3120 | **4.4e-3** |
| pinned 5e-4 | 84.6 ± 7.8 | 85.0 ± 6.9 | 0.3129 | 5e-4 |
| pinned 5e-3 (the oracle-rate pin) | 47.5 ± 7.8 | 47.5 ± 5.7 | 0.3120 | 5e-3 |

The pin cannot adapt (85.0 late = 84.6 first, by construction).  The ladder climbs: later
events are caught 31% faster than the bottom pin (59.0 vs 85.0), two-thirds of the way to
the oracle-rate pin's 47.5, and its state RMSE matches the oracle-rate pin's.  Even its
FIRST event is faster here (66.6 vs the single-fault world's 82.0): only 400 calm steps
precede it, so the weights have not yet settled as deep — the launch depth tracks the quiet
actually witnessed, which is the mechanism working as derived.

## What this does NOT claim

- The ladder does not beat every pin on every rig — it cannot; pins span a Pareto frontier.
  It matches the operating point its own derivation selects, with zero told, and moves when
  the world's rate moves.  (Same shape as the random-walk-filter finding: "matches a
  reasonable guess while requiring none.")
- The acceptance rigs (0004 drone, 0005 blowout, 0008) are measured PINNED; re-measuring
  them under `faults=True` is an open (SUMMARY).  Expected from the frontier arithmetic:
  ~12% faster detection (launch 7.6 vs 8.7 nats), calm within noise, ~2.5× members.
- The offset channel's hazard got the same treatment (the ladder crossed into its class
  rungs — a `(width, rate)` pair per rung); its acceptance numbers in the README are from
  the pinned era and were not re-measured here beyond the test suite.

## Opens

- The rung walkers currently form one stacked bank PER RUNG (the `_bank_key` splits on
  `id(F)`); sharing the augmented `F`/cap objects across rungs would stack all J walkers
  into one bank and reclaim most of the ladder's constant factor.
- The jump-hold exact theta-prior open (SUMMARY) now has a cleaner home: rung j's walker is
  the diffusion surrogate of rung j's jump process; the exact two-state prior would replace
  the surrogate *per rung* and should retire the 0003 restart.
- The t = 0 initialization puts uniform weight on (rung × hypothesis), so `fault` starts at
  1/2 and burns in over the first ~300 steps — inherited from 0001's construction,
  unchanged by this probe, and the tests' burn-in convention stands.
