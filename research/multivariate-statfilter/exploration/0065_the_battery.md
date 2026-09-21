# 0065 — The battery, `main` against the branch: the windows improve, two aggregates do not

> **AI-generated, not peer-reviewed.** Scripts: [`0065_run_battery.py`](0065_run_battery.py) (any rig's `main()`
> on any filter source), [`0065_arm_battery_per_seed.py`](0065_arm_battery_per_seed.py),
> [`0065_drone_no_crate_per_seed.py`](0065_drone_no_crate_per_seed.py). Rigs: the README battery — the arm
> (0054, 3 seeds), the drone (dynamics-learning 0008, 5 fault seeds + 3 no-crate), the scalar hero
> (random-walk-filter README-004, 1 seed). Sources: `main` = 08b925b (pre-grid), `grid` = 3806147 (attribution
> grid + derived switching rate, before the memory ladder), `head` = the branch.

The regime tables of 0059–0064 score windows that **skip each regime's onset** (`SKIP` steps on the arm,
120 on the drone). The README battery also reports aggregates that do not skip them. Those disagree.

## 1. Arm (3 seeds, tip RMSE ratio to an oracle told the noise schedule)

| regime | main | head |
|---|---|---|
| calm | 0.94 | 0.91 |
| SENSOR (accelerometers ×15) | 2.77 | **1.16** |
| PROCESS (vibration ×20) | 1.11 | 1.11 |
| POTFAIL (one pot ×15) | 1.07 | 1.08 |
| BOTH | 2.10 | 1.90 |
| **tip RMSE over every burst step, metres** | 0.0190 ± 0.0040 | **0.0684 ± 0.0522** |

Per seed (burst RMSE in metres; worst tip error and where):

| seed | main | grid | head |
|---|---|---|---|
| 0 | 0.0123; 0.043 m at BOTH+168 | 0.0178; **0.384 m at BOTH+0** | 0.0126; 0.068 m at BOTH+33 |
| 1 | 0.0187; 0.069 m at BOTH+65 | 0.1758; **1.083 m at SENSOR+0, 41 steps > 0.2 m** | 0.1728; **1.083 m at SENSOR+0, 40 steps** |
| 2 | 0.0262; 0.105 m at BOTH+59 | 0.0198; 0.094 m | 0.0199; 0.094 m |

The aggregate regression is one thing: **the attribution grid puts a ~1 m tip excursion at the first
step of the first sensor burst on seed 1**, lasting 40 steps, on a rig whose oracle error there is 3 mm.
It is the grid's (present at `grid`, absent at `main`); the memory ladder neither causes nor cures it
(it does remove the grid's 0.38 m excursion at seed 0's BOTH onset, and takes seed 1's windowed SENSOR
from 8.00 to 1.02 — the windows never see the onset). This is the "first burst off the floor" open of
AUD-10, now with a size: the eager copy attributes the burst within a step while the patient copy holds,
and until the copies' evidence separates the mixture straddles two states a metre apart at the tip.

## 2. Drone (5 fault seeds; 3 no-crate; position RMSE ratio to an oracle told noise and payload)

| window | main | grid | head |
|---|---|---|---|
| pre-pick-up | 0.98 | 0.98 | 0.98 |
| WIND · carrying | 1.03 | 1.03 | 1.05 |
| calm · carrying | 1.02 | 1.01 | 1.01 |
| MULTIPATH (GPS ×12) · carrying | 1.37 | 1.33 | 1.31 |
| VIBRATION (gyro ×12) · empty | 0.98 | 0.99 | 0.99 |
| calm · empty | 1.09 | 1.11 | 1.08 |
| detection; recovered mass carrying / after | 2.8 ± 0.4 steps; 1.529 / 1.100 | 2.6 ± 0.4; 1.528 / 1.100 | 2.8 ± 0.4; 1.528 / 1.101 |
| **no-crate control, RMSE / oracle** | 1.024 ± 0.044 | **1.563 ± 0.237** | **1.554 ± 0.234** |

The crate mission is unchanged. The **no-crate mission regresses from 2% to 55% above the oracle**, and it
is the grid's again (identical at `grid` and `head`; the memory ladder is inert on this rig — its nominal
model's `τ` gives a single rung, reported `memory = 3430`). Per seed, main → head: 1.090 → 1.729,
1.041 → 1.843, 0.941 → 1.091; the damage is in the **GPS multipath window and the calm after it** —
MULTIPATH 2.35 → 10.70, 1.08 → 4.08, 0.92 → 1.71; the following calm 1.09 → 2.37, 1.24 → 3.68, 0.94 → 0.83.
With the crate aboard the same window is 1.37 → 1.31. So the grid's copies mis-handle a sensor burst on
the GPS position channels when there is no payload departure for the departure walker to be busy with.
Mechanism not yet traced (the patience trace per phase is the next measurement).

## 3. Scalar hero (README-004, one seed, told nothing, against a Kalman filter told the truth)

| | main | head |
|---|---|---|
| steady-state penalty over the oracle | 3.5% | **7.2%** |
| after the sensor noise triples, lucid RMSE (Kalman 0.7748) | 0.8816 | **0.8397** |
| calibration there (1 honest; Kalman 4.56) | 0.81 | 0.68 |
| jump rise time (Kalman 16) | 3 steps | **4** |

This one is the memory ladder's (the grid is inert on the scalar rig): the AUD-11 steady cost on the
README's seed, and a one-step-slower jump on it — the 12-seed jump window improves 16% (0064), so the
single seed is not the rig, but the README's "3 steps" claim would need re-measuring before it is kept.

## 4. Status

The regime windows of every rig are unchanged or better on the branch, and three aggregates say the
branch is not mergeable as it stands: the arm's whole-burst RMSE (×3.6, the grid's onset excursion), the
drone's no-crate mission (1.02 → 1.55, the grid's multipath handling), and the hero's steady penalty
(3.5 → 7.2%, the memory ladder's told-nothing floor). The first two are the attribution grid's, both
already named in AUD-10 as opens, now with sizes that the windowed tables hid; the third is AUD-11.

## 5. The two mechanisms, as far as the per-copy traces take them

[`0065_per_copy_trace.py`](0065_per_copy_trace.py): each copy's weight, its OWN error from its own members' states,
its scales, per phase — at the grid commit and, as the control, `main`'s single filter on the same seed.

**Arm, seed 1, the first sensor burst.** In the onset's first 40 steps the eager copy holds weight 1.00 and
its own tip error is **144× the oracle (1.083 m)**, with the accelerometer burst booked on the sensors at +7.70
*and on the process axes at +2.84* (log-scale, all modes); the patient copy is 33× (0.78 m) and takes the weight
by step 290 (0.96, own error 1.00×). `main`'s single filter at the same onset: 2.95× (0.033 m), sensors +4.68,
process +1.69 — the same misattribution at a third of the size, and no excursion. **The eager copy is not
`main`'s filter**, although its walk is the same rule at rate 1: something the grid shares between the copies
(the bank's collapse, the star window, or the per-rung restart — not the offset channel, which this rig does not
run) carries the patient copy's held state into the eager copy's step at the onset, and the excursion is the
product. Which shared piece is not yet isolated; it is the next measurement, and it is the same question 0061
asked of the rate ladder's interior (weights mixed, states not mixed) from the other side.

**Drone, no-crate seed 0, the GPS multipath window.** Held axes on this rig: process modes 0–8 and sensor
channels 9–11. Through the window the patient copy holds weight 0.98 with its own error **10.8× the oracle**;
the eager copy is 2.35× (= `main`) at weight 0.02. Both copies have the GPS-position scales at +4.99 (= 2 ln 12,
correct). The patient copy's held process modes sit at their nominal 0.01 and its held sensor channels at 0.01 —
*the oracle's own model* — while the eager copy carries attitude-mode process scales at +2.4 (raised in WIND,
never brought back) and channels 9–11 at +1.3. So the copy with the right scales is five times worse than the
copy with the wrong ones, and the evidence prefers it by 0.14 nats/step. A correctly-scaled Kalman filter that
is 10× a correctly-scaled oracle differs from it in its *state*, not its noise model: the candidates are the
patient copy's departure walker (this rig runs the dynamics channel; the copies' departure rows are duplicated
at rate 1 — AUD-10) and whatever the shared piece above is. The score prefers it because position under
GPS ×12 is a slow direction — the score cannot see the position error, exactly the blindness `_memory_floor`
derives. Not yet isolated.

**What this means for the branch.** The memory ladder (0063/0064) is independent of the grid: its derivation
reads only the nominal model, and on every rig its effect is the one 0064 measured. The attribution grid
(0059–0062) carries two defects of size on the README battery that its windowed acceptance hid, and both are
in the same place — the copies share something they should not, or the score ranks copies on a slow direction
it cannot see. The grid should not merge until that piece is found; the memory ladder can be carried onto
`main` without it and re-measured there.
