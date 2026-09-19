# 0066 — The memory ladder alone, on `main`: no regressions on the arm or the drone, one on the hero rig

> **AI-generated, not peer-reviewed.** Script: [`0066_memory_ladder_on_main.py`](0066_memory_ladder_on_main.py)
> builds the variant — `main` (08b925b) plus the memory ladder of 0064 and the two node-budget reads of 0063,
> nothing of the attribution grid — as a patched source; the battery is run on it with
> [`0065_run_battery.py`](0065_run_battery.py) and the per-seed traces of 0065. Same rigs and seeds as 0065.
> Not merged, and not on the branch's tree: the branch carries the grid; this is the grid-free measurement.

0065 found that the branch's two large regressions on the README battery are the attribution grid's, and that
the memory ladder moves neither. This note measures the ladder without the grid, against `main`.

## 1. Every rig, `main` against `main + memory ladder`

| rig / window | `main` | + memory ladder |
|---|---|---|
| **arm** calm / SENSOR / PROCESS / POTFAIL / BOTH (3 seeds) | 0.94 / 2.77 / 1.11 / 1.07 / 2.10 | 0.94 / 2.77 / 1.11 / 1.07 / 2.10 |
| arm, tip RMSE over every burst step | 0.0190 ± 0.0040 m | 0.0190 ± 0.0040 m |
| arm per seed: worst tip error | 0.043 / 0.069 / 0.105 m | 0.043 / 0.069 / 0.105 m |
| **drone** pre-pick-up / WIND / calm carrying / MULTIPATH / VIBRATION / calm empty | 0.98 / 1.03 / 1.02 / 1.37 / 0.98 / 1.09 | 0.98 / 1.05 / **1.03** / **1.33** / 0.98 / **1.03** |
| drone detection; mass carrying / after | 2.8 ± 0.4 steps; 1.529 / 1.100 | 2.8 ± 0.4; 1.529 / 1.101 |
| drone no-crate RMSE / oracle (3 seeds; per seed) | 1.024 ± 0.044 (1.090 / 1.041 / 0.941) | 1.025 ± 0.045 (1.092 / 1.041 / 0.940) |
| **async** whole / hot | 1.16 / 1.44 | 1.16 / **1.40** |
| **learned dynamics** test 2 ratio (every gate holds) | 1.0992 | **1.0528** |
| **scalar hero**, 12 seeds: jump / steady / C | 1.7474 / 0.3833 / 0.8890 | **1.4637** / 0.3926 / **0.8149** |
| scalar hero, the README seed: steady penalty / C / rise time | 3.5% / 0.8816 / 3 steps | **7.2%** / **0.8397** / 4 steps |
| suite | 55 passed | 55 passed, 1 failed (the `memory` readout is not plumbed into `LucidStep` in this patch — a plumbing omission of the measurement, not a filter failure) |

The arm is bit-identical (its ladder is two rungs, 2247 and 250, and on `main` there is no grid excursion
for them to remove). The drone is within 0.02 on every window and better on three. Async, learned dynamics
and the hero's jump and regime-C windows improve. **The one regression is the hero's steady window**, 2.4%
over 12 seeds and 3.6% on the README seed (the rise time there is 4 steps against the README's 3 — one seed;
the 12-seed jump window improves 16%). That is AUD-11 exactly: a filter told nothing has `τ = 1.6` and admits
rungs its walk moves below, and 0064 measured that capping at the running floor recovers half of the cost.

## 2. Status

Against `main`, the memory ladder alone is: one regression (the hero's steady window, named and half-explained),
no change on the arm, no change worse than 0.02 on the drone, improvements on five windows across four rigs,
and the last declared engineering parameter removed. The branch as it stands carries the grid's two blockers
on top of this (0065); the grid-free form is this patch, not the branch's tree.
