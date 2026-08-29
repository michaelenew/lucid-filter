# 0004 — the clock: what indexing time by arrival count costs, and what the linear-`Q` map leaves behind

`0004_the_clock.py`. A filter that counts arrivals instead of measuring time is making a
claim about the world — that the gap before every reading is the same — and a stream of
`(sensor, timestamp, value)` points is exactly the case where it is false. Two errors
follow and they do **not** cancel: over a longer-than-nominal gap the state has moved
further and accumulated more process noise, so the filter is over-confident and
under-predicts; over a shorter one it is under-confident and over-predicts.

Rig: a continuous double integrator `dx = A x dt + dW` with spectral density `Qc`,
sampled at gamma-distributed gaps with **mean equal to the nominal step** (0.1 s) and a
controlled coefficient of variation, so the sweep isolates *irregularity* from *rate*.
The truth is the exact Van Loan discretisation at each instant, and the oracle is a Kalman
filter told the true schedule **and** the true noise. 6 seeds, T = 400.

## A. Timestamps supplied against assumed uniform

| gap spread | timestamped | assumed uniform | oracle | uniform / ts | ts / oracle |
|---|---|---|---|---|---|
| 0.00 | 0.1350 ± 0.0028 | 0.1350 ± 0.0028 | 0.1347 | 1.00× | 1.002 |
| 0.25 | 0.1240 ± 0.0058 | 0.1364 ± 0.0076 | 0.1240 | 1.10× | **1.000** |
| 0.50 | 0.1232 ± 0.0047 | 0.1729 ± 0.0120 | 0.1231 | 1.40× | **1.000** |
| 1.00 | 0.1228 ± 0.0013 | 0.1883 ± 0.0143 | 0.1224 | 1.53× | **1.003** |

*position RMSE*

At zero spread the two filters are the same filter — the check that the clock is a
generalisation and not a second code path. From there, ignoring the timestamps costs
10%, 40%, 53%. What is worth more than the penalty column is the last one: **the
timestamped filter sits on the oracle to three decimal places at every irregularity
level**, an oracle that was told both the schedule and the noise. Irregular sampling is
not a source of loss for this filter once it is told the times; it is only a source of
loss when it is not.

## B/C. The one deliberate approximation, measured

`F(a)` is exact and the forcing map is exact. `Q(a) = Q·a` is not: it is exact for the
random-walk default (`F = I`, where the whole construction is `Q` accumulating linearly),
and first order in `‖A‖a` otherwise against `∫₀^a exp(Aτ) Qc exp(A'τ) dτ`.

| `‖A‖` stiffness | `a` / nominal | `‖A‖a` | relative error of `Q·a` |
|---|---|---|---|
| 1 | 0.25 | 0.025 | 3.8% |
| 1 | 1.00 | 0.100 | **0** (by construction) |
| 1 | 4.00 | 0.400 | 15.0% |
| 3 | 0.25 | 0.075 | 11.3% |
| 3 | 4.00 | 1.200 | 45.0% |
| 10 | 0.25 | 0.250 | 37.5% |
| 10 | 4.00 | 4.000 | 91.0% |

It is left approximate **on purpose**, and the reason is a division of labour rather than
convenience: an error in `F` is absorbed by nothing, while correcting the process-noise
magnitude online is the scale walk's entire job. It is exact at the nominal gap, so the
error is zero where a uniformly-sampled filter lives, and it is bounded by `‖A‖a` — small
wherever a stream is fine-sampled enough to be asynchronous in the first place. Panel A's
rig sits at `‖A‖a ≲ 0.4` and the filter is on the oracle there.

**Where this is honest about its limit:** the walk can absorb a *constant* multiplicative
misfit in `Q` (it walks the log-scale to wherever the data says), but the linear-`Q` error
is **gap-dependent** — a different multiplier per event — so the walk can only absorb its
average, not its variation. On a stiff generator sampled with wide gaps (`‖A‖a ≳ 1`,
bottom rows) the residual is a per-event miscalibration the walk cannot reach. Exact
`Q(a)` needs the continuous spectral density `Qc`, and recovering it from a supplied
one-step `Q0` is an `n²×n²` inverse of the Van Loan map with no guarantee of returning a
PSD answer — which is why it is not done here. **Open**, with this table as the acceptance
target: a `process=` that may be declared as a continuous spectral density directly, for
callers who have one.
