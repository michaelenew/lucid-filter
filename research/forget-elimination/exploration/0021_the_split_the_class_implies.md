# 0021 — the split's drift read off the class

> **AI-generated, not peer-reviewed.** Code [`drift3.py`](drift3.py) (`DriftImplied`),
> [`0021_the_split_the_class_implies.py`](0021_the_split_the_class_implies.py), output
> [`0021_output.txt`](0021_output.txt), predictions written first in
> [`0021_predictions.md`](0021_predictions.md). Probe-side; nothing under `lucid/` is changed.
> This probe was not asked for; it follows from reading `0020`'s results.

## The idea

`0020` treated the split's rate of drift as an unknown and paid for it with a long walk.
But the split is the difference of two log-scales, and a cell of the bank already asserts
how fast a log-scale moves: class `(phi, s)` is an innovation variance `s²(1 − phi²)` per
step. Two scales moving independently move their difference — the split's log-odds — with

    D(phi, s) = 2·s²·(1 − phi²)   per step.

Each cell then carries its own split diffusion (hop rate `D/Δlo²` between neighbouring
rungs). No ceiling, no star, no constant, no grid. The class's mean reversion is left out,
because reverting needs an anchor and the ladder has none.

## Result

RMSE over the oracle's (hero: absolute).

| rig | default | filter as it stands | star per axis (`0020`) | **class-implied** |
|---|---|---|---|---|
| ramp, `_LADDER_MEM` 250 / 1000 / 4000 | 1.113 / 1.118 / 1.118 | 1.031 / 1.033 / 1.107 | – / 1.081 / 1.081 | **1.057 / 1.059 / 1.060** |
| hero log-likelihood (24 seeds) | −1721.7 | **−1704.6** | −1717.0 | −1709.0 |
| hero jump / steady / loud | 1.815 / 0.374 / 0.928 | 1.759 / 0.374 / **0.747** | 1.460 / 0.445 / 0.928 | **1.315** / 0.389 / 0.788 |
| calm | 1.004 | 1.005 | 1.058 | 1.041 |
| calm, log-lik/step − default, ×1000 | 0 | −2.1 | −10.7 | −7.7 |
| stale | 1.177 | 1.124 | **1.092** | 1.096 |
| manoeuvre | 1.305 | 1.236 | – | **1.081** |
| flicker 25 | 2.024 | **1.395** | 1.856 | 2.880 |
| flicker 100 | **1.289** | 1.391 | 1.750 | 1.872 |
| flicker 400 | 1.456 | **1.268** | – | 1.508 |
| outliers | 1.150 | **1.137** | – | 2.212 |

| | prediction | verdict |
|---|---|---|
| J1 | independent of the ladder's size | **confirmed** — 1.057 / 1.059 / 1.060 |
| J2 | ramp better than the default, 1.05–1.09 | **confirmed** — 1.059 |
| J3 | calm RMSE within 2% of the default; log-likelihood 2–5e-3 below | **FALSE** — +3.7%; 7.7e-3 below |
| J4 | hero log-likelihood above the default's | **confirmed** — +12.7 nats (`t` = 12.5), short of the filter-as-it-stands' +17.1 |
| J5 | flicker no worse than the filter as it stands | **FALSE, badly** — 2.88 against 1.40 at 25 steps; outliers 2.21 against 1.14 |

## Reading

1. **It is the cleanest of the constant-free constructions and it is not good enough.**
   Grid-free by construction, nothing chosen, and better than `0020`'s star per axis on
   five of the six rigs they share. But it loses to the filter as it stands on eight of
   eleven rows, and to the default on five.

2. **The pattern is sharp.** A fast-moving split reads change as *process*. Where the
   change is in the process it is the best rule measured — manoeuvre 1.081 against 1.236,
   the hero jump 1.31 against 1.76. Where the change is in the *sensor* and brief —
   outliers, a 25-step flicker — it is the worst, because it lets a burst of sensor noise
   move the split toward process and then chases it.

3. **So the split's timescale is not a nuisance constant.** Slow favours sensor-side
   events, fast favours process-side events, and no record here says which the world will
   send. `0020` found the same thing for the hazard (a point on a delay / false-alarm
   curve). In both ladders the number that `forget` used to supply is a *preference between
   kinds of error*, which is why resolvability cannot derive it.

4. **What the idea is missing.** Taking the class literally without its mean reversion
   makes every cell's split wander without bound, which is the calm cost (7.7e-3
   nats/step). The class-consistent model has two levels — a long-run split that is a
   constant of the world, and a deviation about it that reverts at the class's own rate —
   and would bound the wander by `√2·s`. That needs the bank's weights over (long-run
   split, current split) jointly. Not built.
