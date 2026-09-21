# 0063 — The last escape: `forget` is a knob by this repo's own test, and gridding it is the remedy

> **AI-generated, not peer-reviewed.** Scripts: [`0063_what_the_escape_buys.py`](0063_what_the_escape_buys.py)
> (the same filter at several `forget`, every rig; `FGS=` to sweep), [`0063_memory_ladder_patch.py`](0063_memory_ladder_patch.py)
> (the ladder as a patch), [`0063_memory_ladder_hero.py`](0063_memory_ladder_hero.py) and
> [`0063_memory_ladder_rigs.py`](0063_memory_ladder_rigs.py). Arm seed 0 both schedules; scalar 12 seeds;
> async 5 seeds; learned dynamics, the two shipped tests.

## 1. Nothing structural reads it any more (shipped)

Three constructions still read `forget`, one of them newly introduced by the attribution grid:

| read | was | now |
|---|---|---|
| the grid's walk rates and switching rate | `min(1/(1−forget), _LADDER_MEM)` | `_LADDER_MEM` |
| the offset channel's class floor `V/T` | `T = 1/(1−forget)` (diverges at 1) | `T = _LADDER_MEM` |
| the split ladder's rung count | `_rung_odds(forget)` | `_rung_odds(1.0)` |

All three are no-ops at the default, and every other ladder (`_hazard_rungs`) already read the node
budget directly for exactly this reason. The filter's structure — members, rungs, rates, weight rows —
is now **identical from `forget = 0.9` to `forget = 1.0`**, which is the invariant
[`dynamics-learning/0009`](../../dynamics-learning/exploration/0009_hazard_ladder.py) established and
which the grid had quietly broken.

## 2. What the escape still buys: one window

The same filter at `forget = 0.999` and at the nominal `forget = 1` (pure Bayes):

| rig | 0.999 | 1.0 |
|---|---|---|
| arm, five regimes | 1.102 / 1.241 / 1.160 / 1.114 / 1.063 | 1.101 / 1.241 / 1.160 / 1.114 / 1.063 |
| arm swapped, five regimes | identical to three decimals | identical |
| learned dynamics (RMSE, oracle ratio) | 0.2703, 1.0992 | 0.2674, 1.0955 |
| async whole / hot | 1.21 / 1.49 | 1.21 / 1.46 |
| scalar jump / steady | 1.7474 / 0.3833 | 1.7445 / 0.3832 |
| **scalar regime C** | **0.8890** | **0.9225** |

One number, 3.8%: the window after the hero rig's sensor noise triples. The escape's whole surviving
job is letting the bank re-weight across its **static** hypotheses when the noise class changes — and
the rig where that shows is the one rig on which the attribution grid is inactive (its only axis pair
is a confounded pair, owned by the split ladder).

## 3. It is a knob, by the test this repo already uses

Sweeping the hero rig (12 seeds, three windows):

| `forget` | memory | jump | steady | regime C |
|---|---|---|---|---|
| 0.9 | 10 | **1.3538** | 0.4407 | 1.1347 |
| 0.99 | 100 | 1.6569 | 0.3857 | **0.8072** |
| 0.997 | 333 | 1.7196 | 0.3837 | 0.8432 |
| **0.999 (shipped)** | 1000 | 1.7474 | 0.3833 | 0.8890 |
| 0.9995 | 2000 | 1.7469 | 0.3832 | 0.9057 |
| 1.0 | ∞ | 1.7445 | **0.3832** | 0.9225 |

Two opposing monotone effects: jump response improves monotonically as the memory shortens (23% at
`T = 10`), steady-state precision improves monotonically as it lengthens. **The shipped 0.999 is
optimal on none of the three.** That is exactly the test [`dynamics-learning/0009`](../../dynamics-learning/exploration/0009_hazard_ladder.py)
used to condemn a pinned fault hazard — "two opposing monotone effects is a trade-off, and a parameter
on a trade-off is a knob". The parameter's own doc claims (from adaptive-grid 0029) that "tracking
[is] identical across {1.0, 0.999, 0.99} and any value near 1 is free"; that is **refuted** on this rig
and these windows, and the doc needs correcting.

## 4. The remedy is the house rule: grid it, let the evidence weight it

A **memory ladder**: rungs of the bank's weight memory, weight rows sharing the member filters (so the
cost is weights, not filters), each rung carrying its own weight vector and scored by its own mixture
predictive density. The rungs are *switching* hypotheses — which memory is right moves with the regime —
so they carry the derived switching kernel at `1/_LADDER_MEM` from [`0062`](0062_the_switching_rate.md),
not a memory of their own.

Full range, rungs `10, 32, 100, 316, 1000, ∞`:

| rig | fixed 0.999 | ladder |
|---|---|---|
| scalar jump / steady / C | 1.7474 / 0.3833 / 0.8890 | **1.5346** / 0.3851 / **0.8235** |
| async whole / hot | 1.21 / 1.49 | **1.18** / **1.36** |
| learned dynamics ratio | 1.0992 | **1.0557** |
| arm calm / SENSOR / PROCESS / POTFAIL / BOTH | 1.102 / 1.241 / 1.160 / 1.114 / 1.063 | 1.081 / **1.354** / 1.150 / **1.152** / **1.396** |
| arm swapped calm / PROCESS | 1.175 / 2.036 | **1.622** / **3.044** |

It works — the rung weights swing from the long memories in calm to `T = 10` at the jump and `T = 32`
in regime C, from evidence alone — and it **breaks the arm**, the same 1-D-helps / coupled-rig-hurts
pattern this workstream has hit at every grid change.

Restricting to the long rungs `100, 316, 1000, ∞` keeps almost all of the gain and repairs the arm:

| rig / window | fixed 0.999 | long-only ladder |
|---|---|---|
| scalar jump / steady / C | 1.7474 / 0.3833 / 0.8890 | 1.6610 / 0.3837 / **0.8156** |
| async whole / hot | 1.21 / 1.49 | 1.20 / **1.38** |
| arm calm / SENSOR / PROCESS / POTFAIL / BOTH | 1.102 / 1.241 / 1.160 / 1.114 / 1.063 | 1.078 / 1.241 / 1.160 / **1.020** / 1.106 |
| arm swapped calm / PROCESS | 1.175 / 2.036 | 1.169 / **1.913** |
| arm swapped SENSOR / POTFAIL / BOTH | 1.027 / 1.031 / 0.973 | identical |

Better on seven windows, unchanged on five, worse on one (arm BOTH, +4%). Rung count is not monotone
(3 / 6 / 10 rungs give jump 1.5698 / 1.5346 / 1.5551 and C 0.8186 / 0.8235 / 0.8047) but the spread is
1–2% and every count beats the fixed constant.

## 5. Status: not shipped, and the blocker is one derivation

What is shipped is §1 — the escape is now structurally inert, which it was supposed to be already.

What is not shipped is the ladder, and the reason is precise: **its range is not derived.** The
evidence wants short memories (they win every low-dimensional window); short memories destabilise the
coupled arm; and the floor that makes it work, `T ≳ 100`, is a number chosen by hand — replacing one
hand-picked constant with another. The obvious derived candidate does not bind: a memory shorter than
the class's own slowest relaxation `1/(1 − φ_max) = 20` steps cannot see one relaxation of the process
it weighs, but the ladder wants memories well above that floor on the arm and well below it on the
hero rig.

So the escape's status has changed from "a free parameter the caller sets, whose value is claimed not
to matter" to "a nuisance with a measured ladder, one 4% regression, and one missing derivation — the
admissible memory range". That is a strictly better place to be, and it is the same open that
[`0060`](0060_the_rate_ladder_interior.md)/[`0061`](0061_the_interior_is_theory.md) reached from the
rate ladder: which rungs of a weight-mixing ladder a coupled high-dimensional rig can carry.
