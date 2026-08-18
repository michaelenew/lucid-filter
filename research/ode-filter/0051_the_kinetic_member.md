# 0051 — The persistence axis holds: the kinetic member fixes calibration, not points

From [`0050`](0050_the_persistence_axis.py); raw numbers in
`figures/ode050.json`. Discharges `0047` §4 item 2.

## What was built

The $\tau$ kernel's missing persistent end, as a **kinetic grid**: nodes
$(\tau_j,\dot\tau_r)$ whose transition *advects* mass along $\tau$ at each
node's velocity (fractional shifts split between bracketing nodes — first-order
upwind transport), plus a velocity-switching mass $\kappa$ and the restart
channel. Three members Bayes-mixed online by prequential likelihood: FLAT
$(0,0)$, DIFFUSE $(0.03, 10^{-2})$ (the `0046` best), KINETIC. Regret bound
$\log 3$; the velocity grid and $\kappa$ are the member's structural constants
and would join the hyper-grid in a full treatment.

## What was measured

| | jump+ramp run | static run |
|---|---|---|
| ramp coverage-90 | **0.93** (was 0.61) | — |
| overall coverage-90 | 0.97 | — |
| KINETIC mass on the ramp | **0.998** | — |
| FLAT mass, pre-jump / final | 0.50 | **0.9998** |
| $\dot\tau$ posterior mean on ramp | −0.0024 (truth −0.0037) | −0.0002 (truth 0) |
| ramp RMS error | 0.103 — unchanged | — |
| relocation latency | 5 points (was 4) | — |
| $\Lambda$ slope (directed info) | 0.567 — unchanged | — |

Four conclusions:

1. **The likelihood endorses persistence exactly where it is true.** KINETIC
   takes 0.998 of the mass during the ramp and nowhere else; the static run
   keeps 0.9998 on FLAT with $\dot\tau\approx0$ — no hallucinated drift. The
   choice is made online, with no thresholds, at a total structural cost under
   $\log 3$ nats.
2. **The fix is calibration, not point accuracy** — ramp RMS is unchanged at
   0.103. This is `0047` §3's diagnosis confirmed from the other side: the
   staircase's *point* error was already near the evidence-rate bound; what the
   missing axis cost was honest *width*. The kinetic member widens and tilts
   the band along the drift instead of overcommitting between restarts.
3. **The offset's velocity is now a readable** — sign and order of magnitude
   right, magnitude 35% low, resolution-limited by the 5-node velocity grid
   (truth sits between nodes and the $\dot\tau=0$ node keeps mass through the
   kernel's switching floor).
4. **The member level has no forgetting, and it shows.** FLAT is *permanently*
   discredited by the single jump (static Bayes over members accumulates
   forever), after which KINETIC's $\dot\tau{=}0$ nodes serve as the de-facto
   FLAT. Harmless here — the nested member absorbs the role — but it is the
   parent's persistence question reappearing at the hyper level, and a
   fixed-share / switching prior over members is the standard repair when it
   matters.

## Standing

The `0047` §4 stack after this: ~~1 tube~~ (`0048`), ~~2 persistence~~ (here),
**3 joint $(\alpha,\tau)$** — next, **4** a self-consistency score for the
trusted distribution, **5** negative $\tau$ via deferred updates.
