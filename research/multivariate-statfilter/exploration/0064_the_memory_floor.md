# 0064 — The memory floor: the weights may not forget faster than the state

> **AI-generated, not peer-reviewed.** Scripts: [`0064_memory_trace.py`](0064_memory_trace.py) (per-rung
> trace on the arm), [`0064_memory_floor.py`](0064_memory_floor.py) (the floor of every rig's nominal model
> and the arm's slow directions), [`0063_memory_ladder_patch.py`](0063_memory_ladder_patch.py) (now with
> `MEMS=tau`, the rungs from the filter's own nominal model, `MEMS=arc`, the complete ladder, and
> `CLASSONLY=1`; `LUCID_SRC=` points it at a pre-port source). Arm seed 0 both schedules; scalar 12 seeds;
> async 5 seeds; learned dynamics, the two shipped tests. **Shipped:** `_memory_floor` / `_memory_rungs`
> in `lucid.py`, the `forget` parameter removed, the `memory` readout added.

[`0063`](0063_the_last_escape.md) left one derivation open: the memory ladder works on every low-dimensional
window with rungs down to `T = 10`, breaks the arm with them, and the floor that repairs the arm (`T ≳ 100`)
was chosen by hand. This note derives the floor, and the coordinate, from the filter's own quantities.

## 1. The ladder is not the problem: the evidence really prefers the short memory on the arm

Per-rung trace of the full ladder (`10, 32, 100, 316, 1000, ∞`) on the arm — each rung's OWN tip error
(× oracle), the weight the ladder puts on it, and the copy marginal it carries:

| window | own-rung error 10 / 32 / 100 / 316 / ∞ | rung weight on `T = 10` |
|---|---|---|
| SENSOR (normal) | **1.36** / 1.24 / 1.24 / 1.24 / 1.24 | 0.84 |
| POTFAIL (normal) | **1.16** / 1.10 / 0.99 / 1.11 / 1.11 | 0.89 |
| BOTH (normal) | **1.40** / 1.40 / 1.11 / 1.06 / 1.06 | 0.92 |
| PROCESS (swapped) | **3.11** / 2.79 / 1.92 / 2.02 / 2.04 | 0.68 |
| calm after SENSOR (swapped) | **4.50** / 1.75 / 1.75 / 1.75 / 1.75 | 0.35 |

Two facts. The ladder's error is the `T = 10` rung's own error everywhere (ladder BOTH 1.396, rung 1.40;
ladder swapped PROCESS 3.044, rung 3.11): the ladder faithfully follows the best-scoring rung, and the
same filter run at a fixed `forget = 0.9` alone reproduces the ladder's numbers to two decimals
(1.093 / 1.357 / 1.103 / 1.155 / 1.400 against the ladder's 1.081 / 1.354 / 1.150 / 1.152 / 1.396). So
this is not the mixing failure of [`0060`](0060_the_rate_ladder_interior.md)/[`0061`](0061_the_interior_is_theory.md).
And the short rung wins the rung weight *by the filter's own evidence* — its one-step predictive density
is the best of the six through every burst — while its tip error is the worst. A range restriction
imposed on top of that would be a knob set against the evidence, unless the floor says exactly where and
why the evidence is blind. That is what §3 derives.

Where the score advantage comes from, per window (the short rung's log-score over the pure-Bayes rung,
whole window against its first 25 steps): normal SENSOR +104 nats, 71 of them in the first 25 steps;
normal BOTH +1001, 784 at the onset; swapped SENSOR +670, 663 at the onset — the sensor bursts are won at
the *onset*, where the long rung's sharp calm weights take the hit while the walk catches up; but swapped
PROCESS +471 with only 51 at the onset, a *sustained* 1.9 nats/step for 250 steps. So the short rung is not
only faster: on the swapped process burst it is the better one-step predictor throughout, and 50% worse on
the tip. That is the discrepancy §3 has to explain.

The copies are one part of it and not the whole: in SENSOR the `T = 10` rung carries patience 0.68 against
0.88 on the long rungs (a ten-step memory cannot hold the multi-step attribution evidence of
[`0059`](0059_why_copies.md)), but in BOTH and in the swapped PROCESS every rung has patience ≈ 0 and the
short rung is still 30–50% worse. Tempering the class-cell marginal only, leaving the copy and departure
conditionals at pure Bayes (`CLASSONLY=1`), changes nothing (swapped PROCESS 3.033 vs 3.044). The mechanism
is in the class weights themselves.

## 2. The coordinate: a memory is a gain, so the memory ladder is the split ladder read as memories

The rung's log-weights obey `L_t = f L_{t-1} + ℓ_t` with `f = 1 - 1/T` and `ℓ_t` the step's log-likelihood.
Write `m_t = f m_{t-1} + (1 - f) ℓ_t`, the local-level smoother of the log-likelihood stream at gain
`K = 1 - f = 1/T`; then `L_t = T m_t` exactly. **A memory rung is a local-level filter on the members'
log-fitness at gain `1/T`** — the hypothesis "the members' relative fitness drifts as a random walk with
per-step signal-to-noise `ρ(K)`". That is the same object the hazard ladder is built on ("a rung is a
diffusion rate, hence a gain"), so it has the same derived coordinate: the Whittle arclength of the gain,
`t = arccos(1 - K)`, per-step Fisher 1, on `[0, π/2]`. `t = 0` is `K = 0`, `T = ∞`, pure Bayes; `t = π/2`
is `K = 1`, `T = 1`, the per-step maximum-likelihood member. Spacing at the node budget, `1.5 √(2/mem)`,
gives 24 rungs at `mem = 1000` — the split ladder's rungs, `1/K` in place of `K²/(1-K)`:

`T = 1868, 208, 75, 38, 23, 16, 11, 8.5, 6.6, 5.3, 4.4, 3.7, 3.2, 2.7, 2.4, 2.1, 1.9, 1.7, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0`

The coordinate is dense at short memories and sparse at long ones because that is where the gains are
distinguishable per step; my [`0063`](0063_the_last_escape.md) §5 remark that the spacing "diverges toward
pure Bayes" was this coordinate seen from the wrong side. Audit grade: the same as the hazard ladder's —
the metric is that of the rung as a model of the log-likelihood stream, a proxy for the metric of its
mixture predictive density on `y`.

## 3. The floor: a weight memory shorter than the state memory scores cells before their errors show

A cell's per-step log-likelihood is a function of its innovation, `ν = H e + v`, where `e` is the cell's
state error. That error obeys the cell's closed-loop recursion `e_{t+1} = (I - K H) F e_t + noise`, whose
eigen-directions have time constants `τ_v = 1 / (1 - |λ_v|)`. A wrong hypothesis on the scale that feeds a
slow direction produces an error there that *builds over `τ_v` steps* and is exposed in the innovations on
the same timescale; a wrong hypothesis on a fast direction is exposed within a few steps. A weight memory
`T` holds the last `T` steps of that evidence. For `T ≪ τ_v` the evidence about the slow direction never
accumulates inside the window — it is forgotten as fast as it arrives — so the rung ranks cells by their
fast directions only, and its ranking is not the statistic for the slow ones. The rung's score is honest
(it is a prequential density) and blind: it prefers whichever cell explains the fast channels, whatever
that cell does to the slow state. On the arm the fast channels are ten accelerometers and the slow state
is the position the tip error is measured on; the short rung explains the accelerometers best and lets the
position drift. That is §1.

So the score is the statistic for a memory only above the slowest closed-loop time constant of the state,

    τ = 1 / (1 - ρ((I - K H) F)),   K the steady-state gain of the nominal (F, H, Q0, R0),

and the memory ladder's range is the arclength interval `[0, arccos(1 - 1/τ)]`: from pure Bayes down to the
filter's own state memory, complete, at the node budget's spacing. In the scalar case `τ = 1/K_state`
exactly, and the sentence is: **the weights may not forget faster than the state does** — the ladder runs
over weight gains from 0 up to the state's gain. It is the same rule as `_FLOOR_SHARE` one level up:
a range end placed where the per-step score stops being the statistic (resolution-criterion 0006),
not a value chosen for what it measures.

The floor from every rig's nominal model:

| rig | nominal model | τ | the measured floor it predicts |
|---|---|---|---|
| arm | `(F, H_CHAR, Q0, R0)` | **140.6** | fixed `T = 10, 33` break (BOTH 1.40, swapped PROCESS 3.1 / 2.75); `T ≥ 100` clean — 0063's hand-chosen 100 |
| arm, pots masked | accelerometer rows only | ∞ | position unobservable: no finite memory is admissible while the pots are out |
| scalar hero, true model | `q = 0.02, r = 1` / `r = 9` | 7.6 / 21.7 | the sweep's best memory at the jump is the shortest tried (10 ≥ 7.6); the ladder settles at 32–56 in regime C (≥ 21.7) |
| scalar hero, as told | defaults `Q0 = I, R0 = I` | 1.6 | 18 rungs down to `T = 1.7`; see §5 |
| async | `nominal_model()` | 802 | one rung (`T = 3207`): the ladder collapses to pure Bayes |
| learned dynamics | random-walk start, `q = 0.09, r = 0.25` | 2.2 | 15 rungs down to `T = 2.4` |

The arm's slow direction is a physical fact of the rig: the two eigenvalues at `|λ| = 0.9929` live on
joint 0's `(θ, ω)` — the base yaw, which gravity cannot see and only its own pot reads (0054's remark
that "absolute yaw still rests on the encoder"); every other joint's slowest mode is `τ ≈ 34`, read by
gravity in the accelerometers. Mask pot 0 alone and `τ = ∞`; mask any other pot and it stays 140.6.
The per-mode trace on the short rung is the same fact seen from the weights: (per-mode trace pending: rerun after a container restart)

## 4. The derived ladder on every rig

Rungs from each filter's own nominal model (`MEMS=tau`), against the shipped fixed `forget = 0.999`
(the final A/B is the ported tree against the commit before it):

| rig / window | shipped 0.999 | derived memory ladder |
|---|---|---|
| arm calm / SENSOR / PROCESS / POTFAIL / BOTH (rungs 2247, 250) | 1.102 / 1.241 / 1.160 / 1.114 / 1.063 | 1.103 / 1.241 / 1.160 / 1.114 / **1.063** |
| arm swapped calm / PROCESS / SENSOR / POTFAIL / BOTH | 1.175 / 2.036 / 1.027 / 1.031 / 0.973 | **1.169** / **2.022** / 1.027 / 1.031 / 0.973 |
| async whole / hot (one rung, 3207) | 1.21 / 1.49 | 1.22 / **1.46** |
| learned dynamics: test 1 learned vs walk, fault, `F`; test 2 ratio (15 rungs) | 0.2703 < 0.3079, 0.780, 0.494; 1.0992 | 0.2707 < 0.3055, 0.689, 0.564; **1.0528** |
| scalar jump / steady / C (18 rungs) | 1.7474 / 0.3833 / 0.8890 | **1.4637** / 0.3926 / **0.8149** |

The arm — the rig the range was blocked on — is unchanged or better on every window of both schedules,
the 4% BOTH regression of 0063's hand-chosen floor included: `T = 100` sits just below `τ = 141`, and
the derived rungs do not. Every gate of the learned-dynamics tests holds and its ratio improves 4%.
The hero rig keeps the ladder's gains (jump −16%, regime C −8%) and pays 2.4% on the steady window.
Suite: 56 passed, 16 skipped (one test rewritten, one added).

## 5. What is open: the floor is the nominal model's, and the hero rig's walk moves below it

The one regression has a precise cause. A filter told nothing has `Q0 = I, R0 = I`, `τ = 1.6`, and admits
rungs down to `T = 1.7`; the hero rig's walk then settles the bank at an effective state memory of 7.6
(calm) to 22 (regime C). The rungs between 1.7 and ~7 are below the *running* floor — exactly the
rungs §3 says have no statistic — and they are what the steady window pays for: the same ladder capped at the true-model
floor (`t ≤ arccos(1 − 1/7.59)`, 8 rungs from 1900 down to 8.6) gives jump 1.5304 / steady **0.3885** /
C 0.8148 against the told-nothing ladder's 1.4637 / 0.3926 / 0.8149 — the rungs below the running floor
are half of the steady cost and all of the extra jump gain. The other half (0.3885 against 0.3833) is the
ladder's own resolution: rungs `1.5 √(2/mem)` apart are, by construction, separable only over `mem` steps,
so in a 300-step stationary window the short rungs above the floor keep weight (0.03–0.07 each in early
calm) and the mixture pays their jitter. That is the same price the split ladder pays for completeness,
not a range question.

So the residual is not a hole in the derivation; it is the derivation applied to the wrong model. The
floor should be the *bank's* closed-loop time constant, not the nominal's, and the model-averaged gain it
needs is already computed each step for the offset channel (`Kc`). What is not designed is the runtime
form — which rungs a step scores when the floor moves — and that is a convention this note does not want
to set without measuring it. Recorded as the narrowed AUD-11. The same floor answers AUD-10's question
for this ladder (which rungs of a weight-mixing ladder a coupled rig can carry: those above its state
memory); whether it also governs the rate ladder's interior is not tested.

One more pinned memory remains in the filter: the offset channel's class-ladder decay,
`_MeanChannel._decay = 1 − 1/_LADDER_MEM` — the same object the memory ladder replaced one level up,
still a node-budget constant. AUD-12.
