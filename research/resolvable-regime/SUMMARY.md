# Current state

> **AI-generated, not peer-reviewed** — produced by an AI system, not
> independently verified. Treat as provisional.

## Verdict

**The axiom as first written is refuted; the axiom as intended was never tested,
because it was aimed at the wrong process.** [`0009`](exploration/0009_the_axiom_relocated.md)
relocates it and measures the number it runs on.

The constraint belongs on **how fast the state moves relative to the sensor** —
what the process/sensor demix actually reads — not on how fast the noise level
moves, which is where [`0001`](exploration/0001_the_axiom.md) put it. In its
correct form: *anything white at our sampling rate **is** measurement noise*, by
definition rather than by approximation, since "white at `Delta`" is exactly "no
slope in the variogram at `Delta`". The step-change objection that killed the
first version does not touch this one — a step in `Q` or `R` is a regime event,
while this axiom concerns the structure of the innovation sequence at fixed
`(Q, R)`.

The measurements below all concern the **noise-scale** channel and are unaffected
by the relocation. They stand, including the one recommendation with product
consequence.

The thread asked whether the filter's founding assumption — that the log-scales
are a **stationary** AR(1) family — could be replaced by *resolvability*: the data
we receive is meaningful, so we would not get a different answer at a different
sampling rate. Five probes were run. **Four of the five predictions the reframing
made are false**, and the one that survives points at a much smaller change than
the axiom.

## The confidence ledger

Every claim this workstream has made, with what it now rests on. This table is the
workstream's uncertainty read-out; nothing below it should be cited without it.

| claim | status | evidence |
|---|---|---|
| `I(lambda) = 1/2` per observation, blur width `sqrt(2)` nats | **established** | elementary; numerically 0.49935 on 2e6 samples ([`0001_box_ends.py`](exploration/0001_box_ends.py)) |
| AR(1) is forced by time-homogeneity + continuous paths (Doob), not assumed | **established** | standard theorem; not re-derived here |
| **C2** — the identification ridge is the mean-reversion direction | **FALSE** | the flat direction holds `gamma_0 = s^2` (0.067–0.238) and the stiff one moves it (1.394–1.412 of a maximum 1.4142); the shipped `(phi, s)` chart is the diagonalising one to within 3.2°–9.7° ([`0002`](exploration/0002_the_ridge.md)) |
| **C5** — members above `sqrt(2)` are inert and can be pruned | **FALSE** | the exact 13-member prune costs +10.0% on the sensor-degradation window and +5.7% on calibration; patched-module pin against the shipped filter is exact, 0.000e+00 ([`0005`](exploration/0005_the_seam.md)) |
| **C3** — dropping stationarity helps where the world wanders | **FALSE** | on a `kappa = 0` truth the correctly-specified wandering class scores 1.118 ± 0.142 and 0.896 ± 0.114 against the stationary class — parity — while losing 15–24% on stationary truths ([`0003`](exploration/0003_what_stationarity_costs.md)) |
| **R2 as first written** (no regime structure below `Delta`) is compatible with this filter's purpose | **FALSE** | a regime change is a step, hence sub-`Delta` at every sampling rate; that R2 forbids exactly the events the repository's headline results are about ([`0005`](exploration/0005_the_seam.md)). **Withdrawn and replaced** — it constrained the wrong process ([`0009`](exploration/0009_the_axiom_relocated.md)) |
| the split channel carries Fisher information **1 per step** in `t = arccos(1-K)`, on the compact interval `[0, pi/2]` | **established** | measured 0.99–1.15 over a 400× range in `q`, along the ladder's own direction ([`0009`](exploration/0009_the_axiom_relocated.md)); confirms `sequence-demix` 0002's `AUDIT[derived+proxy]` claim |
| the relocated axiom (whiteness at `Delta` **defines** measurement noise) | **untested as a filter change** | stated and motivated in [`0009`](exploration/0009_the_axiom_relocated.md) |
| the scale channel's box is the split channel's ladder one level up | **derivation closes, application FALSE** | the closed form `phi = sqrt(cos t)`, `s = sqrt(2(sec t - 1))` is exact to 1e-16 and needs no constants — and loses 22% on regime C; re-keying the window to the channel's predictive width loses on 4 of 5 columns ([`0010`](exploration/0010_the_derivation.md)) |
| the `(phi, s)` window is a **resolution** grid | **FALSE** | scaling the window geometry by `c` has an **interior** optimum that differs by column (steady wants large `c`, settling wants small, regime C peaks at 1.4–2.0); a resolution-only argument gives `c < 1` throughout, the losing direction ([`0010`](exploration/0010_the_derivation.md) §3) |
| **no resolution argument can derive `_PHIS`/`_SS`** | **established, negative** | follows from the row above; AUD-2 needs a two-sided argument pricing reach against resolution, and §3's sweep is the first map of it |
| **C6** — resolvability is checkable by decimation | **supported** | inferred log-scale paths at `Delta` and `2Delta` disagree monotonically with the log-scale's speed, 0.22 → 7.84 nats over a 35× sweep ([`0008`](exploration/0008_the_indicator.md)) |
| the per-step **step ratio** is a usable confidence read-out | **half-established, one-sided** | rank correlation 1.000 with decimation disagreement below the seam; **saturates and decreases above it**, so a low reading is not a clean bill of health ([`0008`](exploration/0008_the_indicator.md)) |
| `_PHIS` does not reach near enough to 1 | **measured on one rig** | shipped-like reach costs 2.9× and 3.5× on wandering log-scales, rail-pinned on every seed, and costs nothing on a stationary one ([`0007`](exploration/0007_phi_reach.md)) |
| `sqrt(2)` is a **seam** (drift below, jump above), not a ceiling | **interpretation of a measurement** | consistent with [`0005`](exploration/0005_the_seam.md)'s +10.0%; the mechanism is not separately measured |
| **C1** — `Q(a) = Q·a` breaks refinement covariance | **untested here** | algebraic; the misfit is already measured elsewhere (45% of `Q` at `‖A‖a ≈ 1.2`, `pointwise-streaming`) |

## Why the swap fails, in one paragraph

The axiom's second half — *no regime structure below the sampling interval* —
sounds like a statement about sampling fast enough. It is not. **A regime change
is a step**, and a step is sub-interval structure at every sampling rate; you
cannot sample your way to resolving a discontinuity. So R2 forbids the sensor
failing ×200, the crate picked up mid-flight, the tire blowing out — the
repository's headline results. [`0005`](exploration/0005_the_seam.md) measures the
consequence directly: the two class members the axiom would delete are worth 10%
of the sensor-degradation window. And the axiom's first half buys nothing either,
because [`0003`](exploration/0003_what_stationarity_costs.md) shows a broad
stationary class with `phi` near 1 already contains wandering behaviour to within
what a finite record can distinguish — while [`0007`](exploration/0007_phi_reach.md)
shows a nearly-non-stationary *stationary* member actually **beats** the
correctly-specified non-stationary one (0.266 against 0.351), because it is
anchored and the wandering class is not.

## What the thread produced anyway

Three things, in descending order of confidence.

1. **The box's orientation and shape are now derived, not measured.**
   [`0002`](exploration/0002_the_ridge.md): the observed information's
   eigenvectors align with the `(phi, s)` axes to within 3.2°–9.7°, with `s` stiff
   and `phi` flat. So the product grid is the diagonalising chart, and spending
   more nodes on `s` (5) than on `phi` (3) is the right way round. AUD-2 graded
   this `measured`; it is now a computation. **The ends are still not derived.**

2. **A requirement on the box's top end.**
   [`0005`](exploration/0005_the_seam.md): the `s` ladder must reach *above*
   `sqrt(2)` in per-step increment SD, because that is what separates drift
   hypotheses from jump hypotheses, and having no above-seam member costs +10.0%
   RMSE and +5.7% calibration on a rig with a ×3 sensor change. A requirement, not
   a derivation of 3.20.

3. **A recommendation, and it is the only thing here with product consequence.**
   [`0007`](exploration/0007_phi_reach.md): add a rung near 1 to `_PHIS` —
   `(0.70, 0.85, 0.95, 0.995)`. Worth 2.9–5.6× on a slowly-wandering noise
   environment, free on a reverting one, costs one bank cell on the flat axis.
   **Measured on the scalar grid filter only. Not yet run through the shipped
   filter or any acceptance gate.**

## What was wrong with the reasoning, so it is not repeated

C2 confused *what one step sees* with *what the record sees*. The per-step
likelihood does see the log-scale's increment — that governs the **walk**. But the
bank is doing class identification over a whole record, and over a record the
stationary variance is pinned far harder than any increment; `nu = gamma_0(1 -
phi^2)` is the badly-conditioned *derived* combination, not the primitive one. The
reframing was built on the wrong one of the two.

## Nothing was changed in the filter

Three candidate changes were built and tested; none clears the repository's bar.
The table in [`0010`](exploration/0010_the_derivation.md) §5 records why. The
closest is `_PHIS` reaching to 0.995 — worth 2.9–3.5× on a wandering noise level
in [`0007`](exploration/0007_phi_reach.md)'s *research* grid filter, and exactly
neutral on the shipped filter's own hero gate, whose regime changes are steps
rather than a wandering level. A rig that wanders, run through the shipped filter,
would settle it.

## Next, if this is picked up

0. **The two-sided box argument.** [`0010`](exploration/0010_the_derivation.md) §3
   shows the window trades reach against resolution with an interior optimum, and
   that the resolution half alone is the losing direction. That sweep is the
   starting point for the argument AUD-2 actually needs. Recorded but not derived:
   on the repo's own gate, `c = 2` beats the shipped `c = 1` on regime C (1.001 vs
   1.078) and in steady state, paying in settling time (49.2 vs 35.0).

1. **Run [`0007`](exploration/0007_phi_reach.md)'s recommendation through the
   shipped filter, on a wandering-scale rig** — the hero gate, the arm rig, the drone rig, with
   `phis=(0.70, 0.85, 0.95, 0.995)`. This is the only item with a product payoff
   and it is cheap.
2. **Close the indicator's blind spot** — the step ratio saturates exactly where
   it matters. A second statistic that rises where this one gives up (innovation
   whiteness in the window is the obvious untested candidate) would make it
   two-sided and shippable.
3. **Sweep the seam** — [`0005`](exploration/0005_the_seam.md) measured one jump
   size (×3 = 2.20 nats). If `sqrt(2)` is really the seam, the cost of pruning
   above-seam members should grow with jump size and vanish below it.
4. **Do not write C3's minimax proof.** Its class is the wandering one, and
   [`0003`](exploration/0003_what_stationarity_costs.md) says that class is not
   better on any truth tested.

## Layout

- `exploration/` — numbered, later is more recent.
  [`0001`](exploration/0001_the_axiom.md) is the original argument and is now
  largely superseded; read it for the reasoning, and this file for what survived.
  [`0002`](exploration/0002_the_ridge.md), [`0003`](exploration/0003_what_stationarity_costs.md),
  [`0005`](exploration/0005_the_seam.md), [`0007`](exploration/0007_phi_reach.md),
  [`0008`](exploration/0008_the_indicator.md) carry the measurements.
- `output/` — empty. Nothing here stands on its own yet.
