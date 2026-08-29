# 0002 — partial rows against dropping the row: the cost lands on the state the discarded sensor reads

`0002_partial_is_the_operating_condition.py`. Before this workstream the filter handled
exactly one kind of absence: an all-`NaN` row (propagate, do not correct). A row where
*some* sensors read had no path through the engine, so the only way to feed a multi-rate
sensor set was to discard every reading that did not land on the schedule the slowest
sensor kept — most of them, and the good ones as often as the bad.

Rig: position + velocity, `dt = 0.1`, read by the asymmetric pairing this is actually
about — a **coarse absolute** sensor (σ = 0.30) beside a **precise rate** sensor
(σ = 0.02, finer than one step of the velocity's own motion, so every reading of it is
worth having). The absolute sensor reports one row in `k`; the rate sensor reports on
every row. 8 seeds, T = 400.

## A. The duty-cycle sweep

| k | readings used, partial / drop | all sensors | partial (new) | drop the row (old) | old/new |
|---|---|---|---|---|---|
| 1 | 800 / 800 | 0.2056 | 0.2056 ± 0.0037 | 0.2056 ± 0.0037 | 1.00× |
| 2 | 600 / 400 | 0.2056 | 0.2330 ± 0.0050 | 0.2331 ± 0.0050 | 1.00× |
| 5 | 480 / 160 | 0.2056 | 0.2662 ± 0.0081 | 0.2667 ± 0.0080 | 1.00× |
| 10 | 440 / 80 | 0.2056 | 0.2883 ± 0.0147 | 0.2915 ± 0.0144 | 1.01× |
| 25 | 416 / 32 | 0.2056 | 0.2982 ± 0.0235 | 0.3588 ± 0.0216 | **1.20×** |

*position RMSE (m)*

And the same runs, read on **velocity** — the state the discarded sensor actually reads:

| k | all sensors | partial (new) | drop the row (old) | old/new |
|---|---|---|---|---|
| 1 | 0.0241 | 0.0241 | 0.0241 | 1.00× |
| 2 | 0.0241 | **0.0241** | 0.0417 | **1.73×** |
| 5 | 0.0241 | **0.0241** | 0.0703 | **2.92×** |
| 10 | 0.0241 | **0.0241** | 0.1023 | **4.25×** |
| 25 | 0.0241 | **0.0241** | 0.1632 | **6.77×** |

The prediction was that the two would separate as the duty cycle fell, and they do — but
the sharp reading is the second table, and it corrects the first. **Partial filtering
holds velocity at the all-sensors value exactly, at every duty cycle**, because the rate
sensor never stops reporting and is never thrown away; dropping incomplete rows degrades
it as `1/duty`, to 6.8× at k = 25. Position moves much less because it is pinned by the
absolute sensor either way, and only starts to care once the velocity error has had 25
steps to integrate between fixes. **The cost of "drop the row" is not diffuse — it is
concentrated on whatever the discarded sensor was carrying**, which is exactly why a
position-only headline understates it and why the general statement has to be in terms of
readings used: 416 against 32.

## B. The diagnosis survives

The learned per-sensor log-scale is the filter's live diagnosis — the chip that says *this
sensor is the one that is failing*. Sensor 0 degrades ×8 over a 120-row window while
reporting one row in `k`; the lift is the mean learned log-scale inside the burst minus
outside, truth `log 8² = 4.16`:

| duty | lift | leak onto the healthy sensor |
|---|---|---|
| 1 row in 1 | 4.12 ± 0.09 | −0.02 |
| 1 row in 2 | 4.19 ± 0.12 | −0.02 |
| 1 row in 5 | 3.59 ± 0.21 | −0.02 |
| 1 row in 10 | 1.84 ± 0.17 | −0.01 |

Down to one row in five the failing sensor is still named at close to full strength, and
the leak onto the healthy sensor is flat at −0.02 throughout — the attribution does not
smear as the rate falls, it just runs out of readings. At one row in ten the burst is only
12 readings long and the walk, which moves at most `1.5 s` per reading, has not finished
climbing before the burst ends: 1.84 of 4.16. **The limit is arithmetic, not a defect** —
a scale `Δ` nats away needs `≳ Δ / 1.5s` readings of that sensor to reach, whenever they
arrive.
