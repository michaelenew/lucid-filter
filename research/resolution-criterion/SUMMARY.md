# resolution-criterion: a rigorous replacement for the Sparrow proxy

> **AI-generated, not peer-reviewed** — produced by an AI system, not
> independently verified. Treat as provisional.

## Verdict

**Each of the three resolution sites has now been taken to its own coordinate — the one in
which one step of evidence is worth the same everywhere — and judged there.** One site was
already there (the split ladder), one was not and has been moved (the hazard ladder, now
shipped), and one is exact above its floor and wrong at it (the walk grid), which is where
the arm's e^{30} comes from and where the open now sits.

| site | its coordinate | status | in the filter |
|---|---|---|---|
| split ladder (`_rung_odds`) | Whittle arclength `t = arccos(1 − K)`, `[0, π/2]` | **verified** exact to 2nd order; code length saturated at the shipped spacing ([`0005`](exploration/0005_the_split_ladder_verified.md)) | unchanged; AUD-5 closed |
| hazard ladder (`_HAZARDS`) | the same `t`, via the walker's steady gain `K(ρ)`, `[0, π/3]` | **derived + moved**: shipped log-ρ ladder was 3.4× non-uniform and incomplete; uniform-`t` ladder measured equivalent ([`0004`](exploration/0004_the_hazard_ladder_in_its_coordinate.md)) | **shipped**, 16 rungs, complete to `ρ = 0`; AUD-3 closed |
| walk grid (`_GAP_FACTOR`, `_SPAN_S`) | Fisher arclength of `log q`: `λ/√2` above the floor, compact below | **split verdict**: 0002's theorem is exact above the floor; at the floor log-scale is wrong and the arm sits there on all 15 process axes ([`0006`](exploration/0006_the_walk_coordinate.md)) | unchanged; AUD-1's open restated as a floor problem |

A surprise worth recording: the hazard ladder and the split ladder are literally the same
grid. A hazard rung is a diffusion rate, hence a steady gain, and the Whittle metric on a
gain `K` equals the Bernoulli arcsine metric on a rate `h = K/2` (checked to five digits).
The "event probability" and "diffusion rate" readings of the hazard have one geometry.

### The earlier closure, kept

**The spacing half of AUD-1 is closed by a theorem; the span half stays open.**

The filter's three resolution constants — the walk grid (`_GAP_FACTOR`), the
hazard ladder (`_HAZARD_GAP`) and the split ladder (`_rung_odds`) — were each
`gap = 1.5 × (local blur width)`, with `1.5` the Sparrow optical two-point
analogy, conceded unproven. They are now one statement:

> **Uniform quadrature of a Gaussian.** A grid at spacing `c·b` on a Gaussian of
> SD `b` has relative aliasing `2·exp(−2π²/c²)` (Poisson summation), the moments
> sharing the rate. So `c = 1.5` reproduces the blur-width Gaussian to
> `ε₀ = 3.1×10⁻⁴` — **one tolerance at all three sites**
> ([`0002`](exploration/0002_the_aliasing_theorem.md), verified to every printed
> digit).

That is the sharp statement the audit recorded as missing. It changes the
**grade, not the value**: finer gridding is monotonically more accurate at node
cost only, so `c` is a *budget* in the audit's exact sense, and the theorem is
the derived price it buys. All three sites and their audit rows are regraded
`proxy → derived + budget`; the change is comment-only and the suite is green
(55 passed).

**A measured number got its first derivation.** The walk grid, unlike the
ladders, must let its centre *cross* the grid, which needs the between-node
score to keep its sign. Using the exact log-scale likelihood spectrum
`|ĝ(ω)| = 1/√(u·cosh πω)` ([`0001`](exploration/0001_the_likelihood_spectrum.md)),
the score alias at `ε₀` gives an **absolute** bound `gap ≤ 0.89` nats. adaptive-
grid finding 11 had *measured* the walk's dead-zone onset at ~0.7–0.8 nats with no
derivation. 0.89 is consistent with it — not a bullseye, and the first
derivation it has had.

**Enforcing the absolute bound does not ship**
([`0003`](exploration/0003_the_cap_experiment.md)). With reach preserved it wins
the scalar jump (−4%, `t = −2.35`) and the async rig (hot 1.44 → 1.34), but
regresses the coupled 15-DOF arm **6× on its sensor-burst regime** at 4.5× cost —
the same wall span-6 hit, one step short of NaN. The bank already captures most
of the bound's benefit (small-`s` members supply the fine grid, large-`s` members
the reach).

## Where it stands

| site | was | now | value |
|---|---|---|---|
| `_GAP_FACTOR` (walk grid) | proxy (Sparrow) | **derived + budget** | 1.5, unchanged |
| `_HAZARD_GAP` (hazard ladder) | proxy (Sparrow) | **derived + budget** | 1.5, unchanged |
| `_rung_odds` (split ladder) | derived + proxy | **derived + budget** | 1.5, unchanged |
| `_SPAN_S` (reach) | proxy (±3σ support) | proxy — **open** | 3.0, unchanged |

## The confidence ledger

| claim | status | evidence |
|---|---|---|
| `|ĝ(ω)| = 1/√(u cosh πω)`, decay `e^{−π|ω|/2}` | **established** | analytic + FFT ≤ 4e-5; DC and Γ identities to 1e-8 ([`0001`](exploration/0001_the_likelihood_spectrum.py)) |
| aliasing theorem `err(c) = 2Σe^{−2π²m²/c²}`; `c=1.5 ⇔ ε₀=3.1e-4` | **established** | Poisson summation; worst-case-over-phase measurement matches every digit ([`0002`](exploration/0002_the_aliasing_theorem.py)) |
| the three Sparrow sites are one tolerance | **established** | each is `gap = 1.5·blur` with a Gaussian-ish local posterior ([`0002`](exploration/0002_the_aliasing_theorem.md)) |
| the walk's absolute bound `gap ≤ 0.89` nats at `ε₀` | **derived** | score alias of the exact spectrum ([`0002`](exploration/0002_the_aliasing_theorem.md)); consistent with finding 11's measured 0.7–0.8 |
| enforcing the absolute bound improves the filter | **FALSE on the arm** | +4% scalar jump, better async, but arm SENSOR 2.70 → 16.49× at 4.5× cost ([`0003`](exploration/0003_the_cap_experiment.md)) |
| the likelihood's resolution is absolute, not `∝ s` | **established** | `u`-independent bandwidth ([`0001`](exploration/0001_the_likelihood_spectrum.md)); confirmed by the bound binding at large `s` |

## What remains open, precisely

**The floor.** The exact per-step Fisher of a process-noise scale is `I_q(x) = x²(x+2)/(2(x(x+4))^{3/2})`,
`x = q/r` (closed form from the differenced spectrum, [`0006`](exploration/0006_the_walk_coordinate.md)):
½ at the top, so log-scale is exactly linear there and the e^19 node is a legitimate
hypothesis of the class prior; `√x/8 → 0` at the floor, so the axis is *compact* there and a
window with reach in nats creates far hypotheses the data cannot resolve. On the arm every
process axis is at its floor, and the shipped filter carries `e^{30}` scale hypotheses on
them, pulled up by the window's outer node during bursts — measured. Placing the window's
nodes by information distance (the naive transform back) goes singular on the arm; the
budget is not the lever (flat). The rigorous fix has the shape the split ladder already
has for a singly-read pair — a complete, compact ladder on the confounded direction —
extended to pairs the sensors read twice. That is structural.

The **span** (reach), as before. Twice now — span-6 ([`resolvable-regime/0016`](../resolvable-regime/exploration/0016_the_full_battery.md)–`0017`)
and the cap here — a grid change that helps a 1-D or partial-event rig has hurt
the coupled 15-DOF arm, and both times the mechanism is the reach/node structure
on many coupled axes, not the spacing rule. The concrete form of the open is now
**per-axis (or per-member) node count**: the rectangular axial array forces one
`K` on every axis, so any fine grid or wide reach goes to all 30 of the arm's
axes at once. Relaxing that would let the fine grid go only where the walk must
move and let reach be set in absolute nats. That is a structural change, not a
constant, and it is the next thing to build if this is picked up.

## Layout

- `exploration/` — numbered, later is more recent.
  [`0001`](exploration/0001_the_likelihood_spectrum.md) the spectrum;
  [`0002`](exploration/0002_the_aliasing_theorem.md) the theorem and regrade;
  [`0003`](exploration/0003_the_cap_experiment.md) the enforcement test;
  [`0004`](exploration/0004_the_hazard_ladder_in_its_coordinate.md) the hazard ladder, derived and shipped;
  [`0005`](exploration/0005_the_split_ladder_verified.md) the split ladder verified, AUD-5 closed;
  [`0006`](exploration/0006_the_walk_coordinate.md) the walk's coordinate: exact above the floor, the arm at it.
- `output/` — empty; the theorem lives in `0002` and in the filter's audit.
