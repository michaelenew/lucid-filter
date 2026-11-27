# 0022 — the dynamics departure as a class with a zero corner

> **AI-generated, not peer-reviewed.** Code [`continuum_fault.py`](continuum_fault.py),
> [`0022_the_departure_as_a_class.py`](0022_the_departure_as_a_class.py), output
> [`0022_output.txt`](0022_output.txt), predictions written first in
> [`0022_predictions.md`](0022_predictions.md). A load-time variant; nothing under `lucid/`
> is changed. One rig, 4 seeds.

## Why

Everywhere else in the filter a change is a point on a continuum with the jump at one
corner: a noise level's class `(phi, s)` runs from "never moves" to "jumps", and nothing
counts events. The dynamics channel is the exception. It carries a fault *hazard* — a
ladder of event rates — with three event-shaped pieces around it: a separate nominal member
with a per-rung switch to "departed", a restart of the walker when a fault is confirmed,
and a belief over the rates that needs a memory of its own. `0014`–`0020` spent their
effort on that memory and ended at a trade-off the data cannot settle.

A hazard rung was already a diffusion rate for the departure (`resolution-criterion/0004`).
The rest is historical: the machinery went in with the dynamics channel (late August), for
the drone-payload and blowout cases.

## The object

The departure `g` gets a class of its own — a memory and a spread. Cell `(rho, s_g)`:
coefficients drift with variance `s_g²·rho` per step, capped at `s_g²`. The shipped ladder
is the single row `s_g = 1`. "No departure" is the zero corner, the edge `s_g → 0`, and is
the nominal member. Weight moves on this plane by the class drift like any class axis:
neighbours along `rho` and along `s_g`, the zero corner neighbouring every cell of the
smallest spread, rates from the walker's Whittle arclength, scaled by the star's fraction.

**Removed:** the switch (`_hazard_mix`), the restart (`_reprice`), the hazard as an event
rate, and with it the hazard's memory.

Box: `rho` ∈ {0.42, 0.10, 0.016, 0.0019, 2e-5} × `s_g` ∈ {⅛, ¼, ½, 1, 2}.

## Result

AR(1) state, two sensors, dynamics 0.9 → 0.3 after a calm. With no event to flag, every arm
is scored on its dynamics estimate.

| calm | arm | steps until `F̂` < 0.6 and stays | `F̂` wander in calm | RMSE calm | RMSE, 300 after | `F̂` at the end (truth 0.3) |
|---|---|---|---|---|---|---|
| 600 | default | 124 | 0.041 | 0.3239 | 0.2928 | 0.209 |
| 600 | filter as it stands | 126 | 0.034 | 0.3240 | 0.2922 | 0.213 |
| 600 | **continuum** | 138 | 0.070 | 0.3239 | **0.2876** | **0.250** |
| 4,000 | default | 92 | 0.034 | 0.3216 | 0.2728 | 0.272 |
| 4,000 | filter as it stands | 94 | 0.029 | 0.3221 | 0.2726 | 0.277 |
| 4,000 | **continuum** | 140 | 0.038 | 0.3230 | **0.2724** | **0.283** |

| | prediction | verdict |
|---|---|---|
| K1 | it runs and finds the change on every seed | **confirmed** |
| K2 | delay within 1.5× of the default's after the short calm | **confirmed** — 1.11× |
| K3 | the delay does not depend on the length of the calm beyond 25% | **confirmed** — 138 and 140 steps. Against the default it is 1.11× and 1.52×, because the default got *faster* on the longer record |
| K4 | without the restart: state error after the change within 5% of the default's, `F̂` as close | **confirmed, and better** — 0.2876 against 0.2928; `F̂` ends nearer the truth in both |
| K5 | false alarms in calm ≤ 1% | **FALSE as defined** — the read-out "weight away from the zero corner" is 0.87–0.95 throughout the calm |
| K6 | after the change the weight sits on wide, slow cells | **FALSE** — it sits on `s_g = ¼` (about half) and `⅛`, across all rates |

## Reading

1. **The switch and the restart are not needed on this rig.** With both removed and
   nothing put in their place but a spread dimension, the state error after the change is
   as good or better, the dynamics estimate ends closer to the truth, and the calm costs
   nothing in state error (within 0.4%). The post-jump miscalibration the restart was added
   for (`dynamics-learning` 0003) is what a single spread could not represent.

2. **There is no hazard memory left to choose.** The delay is the same after 600 and after
   4,000 calm steps. The question of `0014`–`0020` — how fast may the belief about
   fault-proneness move — does not arise; what remains is the class star's fraction, which
   every class axis already uses.

3. **It is slower to move the dynamics estimate than the default** — 11% after a short
   calm, 52% after a long one on this rig. The default's hedge buys that speed (the
   "detection-speed subsidy" of `dynamics-learning` finding 2), and its `forget` makes the
   longer record faster.

4. **"Fault" needs a new definition.** On a continuum, cells of small spread are
   indistinguishable from no departure, so the bank spreads its calm weight over the bottom
   row (0.59 on the slowest, smallest cell after 4,000 steps) and the zero corner holds
   5–13%. Which cell holds the weight is not the event. The read-out has to be about the
   departure itself — how far `F̂` has moved against how well it is known — which is a
   reporting convention, as the ½-crossing already was.

5. **A new box.** `(rho, s_g)` is a 5 × 5 grid chosen here, as `_PHIS × _SS` was; 390
   members against 255 and 13 against 9 ms per step. After the change the weight is on the
   two smallest spreads, so the box's top rows did nothing on this rig.

## Limits

One scalar rig, four seeds, one size of change. Not run: the drone rig, named anchors, the
learned-dynamics mode (`dynamics=None`), the offset channel (which has its own hazard), or
the suite's fault tests — several of which assert the `fault` read-out and would need the
new definition first. The rates on the departure plane use the walker's arclength at drift
`s_g²·rho` in class units; that the spread enters only through the drift is an assumption.
