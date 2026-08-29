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

| k | readings used, partial / drop | all sensors | partial | drop the row | drop/partial |
|---|---|---|---|---|---|
| 1 | 800 / 800 | 0.1337 | 0.1337 ± 0.0110 | 0.1337 ± 0.0110 | 1.00× |
| 2 | 600 / 400 | 0.1337 | 0.1613 ± 0.0132 | 0.0791 ± 0.0031 | **0.49×** |
| 5 | 480 / 160 | 0.1337 | 0.2083 ± 0.0077 | 0.1484 ± 0.0083 | **0.71×** |
| 10 | 440 / 80 | 0.1337 | 0.2439 ± 0.0128 | 0.2264 ± 0.0177 | 0.93× |
| 25 | 416 / 32 | 0.1337 | 0.2809 ± 0.0220 | 0.4237 ± 0.0194 | **1.51×** |

*position RMSE (m)*

And the same runs read on **velocity** — the state the discarded sensor actually reads:

| k | all sensors | partial | drop the row | drop/partial |
|---|---|---|---|---|
| 1 | 0.0235 | 0.0235 | 0.0235 | 1.00× |
| 2 | 0.0235 | **0.0236** | 0.0515 | **2.18×** |
| 5 | 0.0235 | **0.0215** | 0.0927 | **4.32×** |
| 10 | 0.0235 | **0.0202** | 0.1384 | **6.85×** |
| 25 | 0.0235 | **0.0199** | 0.2260 | **11.35×** |

**The two tables disagree, and the disagreement is the result.** Velocity is what the
prediction expected: partial filtering holds it at the all-sensors value at every duty
cycle (better, in fact, past k = 5 — the coarse absolute sensor's readings are worth
slightly less to velocity than the split they cost), while dropping incomplete rows
degrades it as `1/duty`, to **11.4×**. Position does not: at k = 2 and k = 5 **dropping
rows is twice as good**, and only past k = 10 does keeping the readings win.

Both halves follow from one fact. A **complete** row identifies the process/sensor split —
a process mode reaches several entries of `S` while a sensor reaches one — and a **partial**
one does not, so the filter holds the split rather than guessing it
([`SUMMARY.md`](../SUMMARY.md) item 8). Dropping incomplete rows therefore buys
identifiability with readings; keeping them buys readings with identifiability. Which trade
wins depends on what the discarded readings were carrying: velocity is read directly by the
sensor that never stops reporting, so keeping it always wins there; position is pinned by
the *slow* sensor, whose complete rows are exactly the ones that carry the split.

So the honest claim is narrower than "partial beats dropping", and more useful: **partial
observation converts a schedule problem into an identifiability one.** It is unambiguously
right when the fast sensor carries the state you care about (the asynchronous case, and
[`0005`](0005_the_asynchronous_rig.md)'s rig), and it is a genuine trade when the slow one
does. The model-free half is not a trade at all: 416 readings used against 32.

## B. The diagnosis survives

The learned per-sensor log-scale is the filter's live diagnosis — the chip that says *this
sensor is the one that is failing*. Sensor 0 degrades ×8 over a 120-row window while
reporting one row in `k`; the lift is the mean learned log-scale inside the burst minus
outside, truth `log 8² = 4.16`:

| duty | lift | leak onto the healthy sensor |
|---|---|---|
| 1 row in 1 | 4.58 ± 0.11 | +0.48 |
| 1 row in 2 | 4.22 ± 0.12 | +0.04 |
| 1 row in 5 | 3.99 ± 0.17 | +0.03 |
| 1 row in 10 | 3.34 ± 0.19 | +0.06 |

**The diagnosis is what partial observation does not cost.** Down to one row in ten the
failing sensor is still named at 3.34 of a true 4.16, and the leak onto the healthy sensor
is *smaller* under partial delivery (+0.03 to +0.06) than under complete rows (+0.48) —
holding a split it cannot see is also declining to blame the wrong sensor for it. What is
left is arithmetic: at one row in ten the burst is only 12 readings long, and the walk moves
at most `1.5 s` per reading, so it has not finished climbing before the burst ends. **The limit is arithmetic, not a defect** —
a scale `Δ` nats away needs `≳ Δ / 1.5s` readings of that sensor to reach, whenever they
arrive.
