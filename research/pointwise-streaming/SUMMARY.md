# pointwise-streaming: the input is a stream of `(sensor, timestamp, value)` points

**The goal.** `LucidFilter` took a matrix of rows: every sensor reporting on every row, at
a fixed and unstated rate. Both assumptions are wrong about real sensor sets. A 5 Hz
absolute fix sits beside a 200 Hz IMU; a camera drops frames; a bus arbitrates; a device
answers when it is polled. **The general thing that happens is that one sensor reports, at
some time.** This workstream makes that the filter's native input and the vector row the
special case — without giving up a single float of what the vector row already did.

**Status: delivered.** Shipped in `LucidFilter` as `observe(sensor, value, t=)` /
`stream(points)`, partial rows through `update`/`filter`, and `timestep=` fixing the time
unit. The ladder is `exploration/0001–0005`; the acceptance rig (0005) passes; a
synchronous full row at the nominal step is **bit-for-bit** what it was.

## What this is not

It is *not* missing-data handling. The old filter had a missing-data case — an all-`NaN`
row, propagate and do not correct — and the parent workstream carried "partial
missingness" as a corner to patch
([`multivariate-statfilter` opens](../multivariate-statfilter/SUMMARY.md#open-items)).
That framing was wrong, and the probes say why: treating a partly-observed instant as a
*lost* instant does not lose a little accuracy at the margin, it throws away every reading
that did not coincide with the slowest sensor's schedule — 416 readings against 32 at a
1-in-25 duty cycle (0002) — and the cost lands, undiluted, on whatever state the discarded
sensor was carrying (**6.8×** on velocity there, against 1.2× on position).

## The settled design (each element pinned by a numbered probe)

The commitment: **an event is a set of readings sharing an instant, and one reading is the
general case.** Everything else follows from taking that seriously.

1. **Sub-select, never impute** (0002). The sensors that reported are selected out of `H`
   and `r`; the correction, the predictive density and the scale evidence are all over
   that subset. Cost falls with the subset — `G(2n²m_o + 2n m_o² + m_o³)`, so a
   single-sensor event has no `m³` term at all.
2. **Structural activation stays keyed to the full `H`** (0002). Observability is a
   property of the sensor suite, not of this event. What varies per event is *evidence*,
   not *identifiability* — the 0024 activation rule is unchanged.
3. **An axis with no evidence drifts; it is never frozen and never updated on nothing**
   (0001 §4). This is not an addition to the walk: `P_mu ← (1−K_mu)P_mu + q_mu` at
   `info → 0` **is** `P_mu ← P_mu + q_mu`, measured 9.3e-06 apart. Applying it on the
   all-missing row too removes a discontinuity in the sensor count — a sensor absent while
   others report would otherwise drift, and the same sensor absent while none report would
   not. Free when the scale did not move across the gap, and **1.5%/5.0% better (3–4σ)**
   when it moved during a blackout. ~~Recorded cost, not a win: 1.9%/5.9% *worse* at 8σ~~ —
   that was measured before the split ladder and the split ladder retired it (0001 §4).
4. **Time enters as a rate, everywhere** (0001, 0004). `timestep` fixes the unit;
   everything supplied about the model and every class timescale is per nominal step, and
   an event `a = dt/timestep` steps later takes each of them to that power:
   `F(a) = exp(a log F)`, `Phi(a)Phi(1)⁻¹B` for the forcing, `Q → Q·a`, `phi → phi^a`,
   `forget → forget^a`, `rho → 1−(1−rho)^a`, `q_mu → q_mu·a`. The departure channel's
   drift `q_g = sigma²rho` rides `Q`'s scaling and needs no separate rule. **`R` is not
   scaled** — a measurement variance belongs to the reading, not to the gap before it.
5. **`F(a)` is exact, and it is a matrix logarithm rather than an eigendecomposition**
   (0001 §1). A supplied `F` is the `a = 1` sampling of a fixed generator; `exp(a log F)`
   by inverse scaling-and-squaring is exact to the last bit on the constant-velocity block
   — which is **defective**, one eigenvector for two states, and which
   `W diag(µ^a) W⁻¹` gets wrong by 5e-2 with no warning. Dynamics with no real generator
   are refused with an error naming the fix, not approximated. The factorisation is lazy,
   so a uniformly-sampled filter never computes a logarithm.
6. **`Q(a)` is exact, recovered from the generator the propagator already holds** (0004,
   0005). ~~`Q(a) = Q·a` is the one deliberate approximation, and it is a division of
   labour: an error in `F` is absorbed by nothing, while correcting the process-noise
   magnitude online is the scale walk's whole job.~~ **Retired — it was not a division of
   labour, it was a bug.** Linear scaling is exact only for `F = I`. A double integrator
   accumulates position variance as `t³`, so a half-length gap gets **4×** and a
   tenth-length gap **100×** the position noise it should, and the walk cannot absorb it
   because the misfit is a different multiple at every gap. The continuous spectral density
   is now recovered once per member by inverting the Van Loan map on the symmetric basis,
   and `Q(a)` is exact to 1e-16; the scale's *departure* from it stays linear in the gap, so
   `dQ/dξ` and the walk are unchanged. Falls back to linear scaling where the recovery is
   unavailable — no real generator, a singular map, a non-PSD density, or a per-step
   linearisation, which has no fixed generator at all. **Probe 0004 missed this**: its rig
   carries a direct position-diffusion term that masks the cubic, so the approximation
   measured fine there and failed on 0005.
7. **Every rate in the engine composes over the gap, including the ones that arrived
   later.** The split ladder's group revert
   ([`sequence-demix`](../sequence-demix/SUMMARY.md)) relaxes a confounded pair's log-odds
   toward its member's hypothesis at the class's persistence *per nominal step*, so over a
   gap it is `revert**a` and at `a = 0` it is the identity. Left per-arrival it would
   revert `m` times for one instant delivered as `m` points, which is the same category of
   error as item 4 and was caught by the same rule: **if it is a rate, it takes the gap's
   power; if it is a property of the reading, it does not.**
8. **A partial event may move a direction it can see, and may not move one it cannot**
   (0005). With one sensor reporting, `S` is a **scalar**: a process mode that sensor sees
   and the sensor's own noise enter it additively, so their scale scores are exactly
   proportional and the split between them is invisible at every such step. That is
   Proposition 1 reached through the *packetisation* rather than through the model — a joint
   row constrains the split because the same process jump would also have to explain the
   other sensors' small innovations, and a single reading has no such other sensors. So the
   event moves each confounded pair's **total**, which its scalar `S` does see, and the
   **split is held** at whatever identifiable evidence already made it. Nothing is chosen
   here: no anchor, no rate, no threshold — the null direction of this event's score is
   simply not a direction this event writes to.

   Two things this is *not*. It is not the split ladder's revert
   ([`sequence-demix`](../sequence-demix/SUMMARY.md)), which relaxes a pair toward the
   member's **base** split: that is the right anchor for a pair the model confounds, and the
   wrong one for a pair only this event confounds — when a sensor really is 100× bad,
   reverting toward the base hands about half of a genuinely elevated total to the process
   axis, and it sticks (measured: tail 3.61× with it, 1.23× without). And it is not a bound
   on the walk: the walk's post-burst behaviour is *correct* — with the split held, the
   scale it settles on is right.
9. **The walk's step budget is per nominal step OF A FULL ROW** (0005). The clip is one
   grid spacing — what stops a single Newton step against a near-singular Fisher becoming a
   verdict. An event carrying `m_o` of `m` sensors carries that share of the evidence that
   would contradict such a step, so it gets that share of the budget. Worth 15.2× → 2.7× in
   the acceptance rig's hot window on its own.
10. **The process-scale score uses the LIVE process time, not the gap** (0003). The engine's
   score is the local one: it keeps `Q`'s dependence on `xi` and drops the prior
   covariance's. At a zero gap `Q(0) = 0`, so that score is not small but *structurally
   absent*, and the first-arriving reading of an instant took all of the process-scale
   evidence while the other `m − 1` were discarded. Using the time over which the `Q` now
   in the prior was injected restores the leading term. The two are the same number
   whenever the gap is non-zero, so nothing already working moves by a bit.

## Acceptance results (the SUMMARY's definition of done, measured)

- **No regression, bit-for-bit** (0001 §3). Six rigs — scalar level, kinematic, control
  map, `dynamics=None`, `faults=` with a named anchor, 3-DOF — against the filter at the
  parent commit: worst |Δ| = **0**, Δloglik = **0**, on every output field. Not "close":
  every reduction to the old case (no clock, full rows, `a = 1`) returns the same floats,
  which is what makes the general path a generalisation rather than a second filter. A
  seventh rig, whole-row gaps, differs — deliberately, by design item 3.
- **Partial observation converts a schedule problem into an identifiability one** (0002) —
  and the sweep says so in two tables that disagree. **Velocity**, read by the sensor that
  never stops reporting, is held at the all-sensors value at every duty cycle (better past
  1-in-5), while dropping incomplete rows degrades it to **11.4×**. **Position**, pinned by
  the *slow* sensor, goes the other way at moderate duty: dropping rows is **2× better** at
  1-in-2, because a complete row identifies the process/sensor split and a partial one does
  not (item 8), and only past 1-in-10 do the extra readings win (1.51× at 1-in-25). So the
  claim is not "partial beats dropping" — it is that partial keeps every reading and pays
  for it in identifiability, which is unambiguously the right trade when the fast sensor
  carries the state you care about, and a genuine trade otherwise. Model-free half: 416
  readings used against 32. The **diagnosis is what it does not cost**: a sensor failing ×8
  is named at 3.34 of a true 4.16 nats at one row in ten, and the leak onto the healthy
  sensor is *smaller* under partial delivery (+0.03) than under complete rows (+0.48).
- **The pointwise decomposition tracks the joint row and declines to learn what it learns**
  (0003). One row fed as `m` points at one timestamp costs **6–27% of state RMSE**
  (ratio 1.062–1.270 across n = m = 2…10) and 8–97 nats of predictive density — not
  numerical error but the identifiability a row has and a point does not, by item 8.
  ~~The same filter, 1.018–1.100~~ — that was measured before the split-holding rule, which
  widened this deliberately. Arithmetic per instant falls to 0.44–0.65× (the `m³` solve is
  gone); numpy wall time rises 1.9–5.0×, more than before, because each event now carries an
  exact `Q(a)` as well as its own interpreter pass.
- **The clock is worth what it costs** (0004). Under irregular arrivals with the same mean
  rate, supplying the timestamps puts the filter **on** an oracle Kalman filter told the
  true schedule *and* the true noise — ratio 1.001, 1.001, 1.003 at gap spreads 0.25, 0.5,
  1.0 — while assuming uniformity costs 1.10×, 1.40×, 1.54×. At zero spread the two are
  the same filter.
- **The asynchronous rig** (0005), the acceptance benchmark — a 100 Hz rate gyro, a 5 Hz
  absolute fix and a 12 Hz jittered second absolute, none phase-locked, one failing ×10
  mid-run, 1877 events, driven entirely through `observe`. Lucid **1.158×** an oracle told
  the true schedule and the true noise (1.027× calm, 1.434× with the sensor hot) against a
  fixed-noise filter at 2.103× (1.000× calm — the adaptation costs nothing when there is
  nothing to adapt to — and **3.357×** once a sensor degrades). The failing sensor's chip
  rises +4.74 nats on a truth of 4.61. **The old API's route — bin onto the fast grid,
  drop any incomplete row — keeps 11 of 1600 rows (0.7%) and pays 21.3×**, and that number
  gets worse as sensor rates get less commensurable, not better. 1.47 ms per event in pure
  numpy. This is the probe that found design items 6, 8 and 9; the row-wise rigs could not
  have.

## A note on the second engine

`LucidFilter` runs its members through **two** copies of the same recursion: the looped
`_WalkEngine.update` and the stacked `_EngineBank.update` that main added, which runs
structurally-identical members as one batched pass and is the path a default filter
actually takes. Everything above is implemented in both, and `test_bank_matches_the_looped_members`
pins them to each other — now on partial rows and non-nominal gaps as well as the paths it
already covered.

That guard earned its keep immediately. The stacked copy's axial-reweight loop used `a` as
its loop variable, which the elapsed gap now also uses; after the rename the gap read
`len(act) - 1` for the whole rest of the step, so `Q`, the walk drift and the scale scores
were all scaled by a small integer. The looped copy uses `i` there and was unaffected —
which is exactly the shape of bug a two-implementation guard exists to catch, and exactly
the shape one loses by trusting that "the stacked path is the same arithmetic" without
checking.

## Open items

- ~~**A gap-dependent `Q` misfit the walk cannot reach** (0004) — the fix is to let
  `process=` be declared as a continuous spectral density; recovering `Qc` from a supplied
  one-step `Q0` instead is an `n²×n²` inverse with no PSD guarantee, which is why it is not
  done.~~ **Done, and the reason not to do it was wrong.** The inverse is a solve on the
  symmetric basis, `n(n+1)/2` square, built from Van Loan on that basis and computed once
  per member — 3 exponentials at `n = 2`, 120 at `n = 15`, lazily and only when a
  non-nominal gap actually arrives. It is checked for PSD and round-trip and falls back to
  the linear scaling when either fails, so the escape hatch the open asked for is the
  fallback rather than the design. What remains open is only the callable path: a per-step
  linearisation has no fixed generator, so it keeps the linear map.
- **The residual on the acceptance rig** (0005). 1.220× oracle overall against 1.147× for
  the pre-merge filter on the narrower box, and 1.665× in the hot window against 1.247×.
  Calm is *better* (1.036 vs 1.063) and the tail shows no latch (1.08×), so what is left is
  concentrated in the burst itself — the price of a wider `(phi, s)` box under evidence
  that arrives one sensor at a time. Whether the box should be narrower when delivery is
  pointwise is a question for the box, not for the streaming path, and it is not answered
  here.
- **The no-information drift's saturation** (0001 §4). Over a long unobserved gap the walk
  covariance drifts to the window-localisation bound `(3s)²`. The scale's own stationary
  variance is `s²`, and saturating there instead is the alternative; 0001 §4's table is what
  it would have to beat, and that bar moved when the split ladder turned the drift from a
  1.9%/5.9% cost into a 1.5%/5.0% gain.
- **The residual pointwise/joint gap** (0003). 2–10% above a 0.6–2.0% frozen-walk floor:
  `m` successive GPB1 collapses and `m` smaller walk steps on the same information. It is
  a property of the caltrop-plus-GPB1 construction, not of the decomposition, and it is
  the same object the [`sequence-demix`](../sequence-demix/SUMMARY.md) workstream is
  taking apart from the other side.
- **Attribution between two sensors reading the same state** (0005). The failing
  absolute sensor's chip is right (+4.49 of 4.61) but leaks +0.90 onto the *other*
  absolute sensor, which reads the same state and is therefore partly collinear with it.
  The state estimate needs only the total and is unaffected; this is the standing
  [`0027`](../multivariate-statfilter/exploration/0027_confound.md) confound showing up in
  a second, cleaner instance, not a streaming defect.
- **Per-sensor `timestep`.** One nominal step serves the whole filter. A suite whose rates
  differ by 40× has one class `(phi, s)` box measured in *that* unit, and whether the
  scale class should be per-sensor rather than per-filter is untested.
- **Out-of-order arrival.** Timestamps must be non-decreasing; a late packet is rejected
  rather than re-processed. The Bayes-correct treatment is a fixed-lag smoother over a
  reorder buffer, which is a different object from this filter.
