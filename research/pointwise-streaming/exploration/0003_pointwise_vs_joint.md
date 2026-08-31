# 0003 — the pointwise decomposition, and the zero-gap hole it found in the process-scale score

`0003_pointwise_vs_joint.py`. If `(sensor, timestamp, value)` is to be the filter's native
input then the vector row must be a *special case* of it and not a different object:
feeding a row as `m` points sharing one timestamp has to mean what feeding the row means.
For a plain Kalman filter with diagonal `R` that is an exact identity, and the probe
carries it as a control — `m` sequential scalar corrections equal the joint one to
**1e-15** at every size. Here it is not exact, because between the sub-updates the engine
does two things a Kalman filter does not: it GPB1-collapses the caltrop star back to one
(mean, covariance), and it takes a walk step on every scale axis the sub-event can see.

**The prediction was that the two would agree to within a small fraction of the filter's
own error. The first run said otherwise**, and the miss was the useful part of this probe.

## What the decomposition costs, on the current engine

| n = m | RMSE joint | RMSE pointwise | ratio | Δ log-likelihood over the run |
|---|---|---|---|---|
| 2 | 0.08024 | 0.10192 | **1.270** | −8.4 nats |
| 4 | 0.09169 | 0.10232 | **1.116** | −26.3 |
| 6 | 0.11884 | 0.12617 | **1.062** | −48.2 |
| 10 | 0.10552 | 0.11637 | **1.103** | −97.0 |

A synchronous row delivered as `m` points costs **6–27% of state RMSE**, and the reason is
not numerical: it is that the two routes have different information. A joint row identifies
the process/sensor split because a process mode reaches several entries of `S` while a
sensor reaches one. Each point's `S` is a **scalar**, in which the two are exactly
proportional, so the filter holds the split rather than guessing it
([`SUMMARY.md`](../SUMMARY.md) item 8). The state is what survives that — it only ever
needed the total — and the log-likelihood ledger is what does not.

**So the honest claim is not "the same filter".** It is that the pointwise route tracks what
the row tracks and declines to learn what the row learns, and the gap between them is the
identifiability a row has and a point does not. Which is precisely why the route is safe on
a stream that has no rows to begin with: there is nothing there to decline.

## Cost per instant

| n = m | ms/instant joint | ms/instant pointwise | multiply-adds joint | pointwise | ratio |
|---|---|---|---|---|---|
| 2 | 25.9 | 49.5 | 680 | 442 | 0.65 |
| 4 | 78.7 | 217.3 | 10,560 | 5,412 | 0.51 |
| 6 | 175.3 | 623.6 | 52,920 | 24,990 | 0.47 |
| 10 | 770.4 | 3851.5 | 405,000 | 179,010 | **0.44** |

The **arithmetic falls** — the joint `G(2n²m + 2nm² + m³)` becomes `m · G(2n² + 2n + 1)`,
the `m³` solve is gone, and the advantage grows with `m`. The **wall time rises**, and by
more than it used to (1.9–5.0× against 1.7–3.8× before the exact process accumulation
landed): `m` python-level events replace one, and each now carries an exact `Q(a)` instead of
a scalar multiply. Both numbers are real and they point opposite ways — in a compiled
implementation the first column governs, in numpy today the second does. It sharpens the
standing open on a lean profile rather than answering it.

## The failure this probe found first, and where it was

| n = m | ratio pointwise/joint | Δ log-likelihood over the run |
|---|---|---|
| 2 | 1.056 | −198 nats |
| 4 | **1.293** | −825 |
| 6 | **1.309** | −1347 |
| 10 | **1.377** | −2323 |

A 30–38% RMSE penalty and a log-likelihood ledger that did not telescope — and it grew
with `m`, which pointed straight at "something is divided among the sensors". Three
controls localised it:

| variant | ratio at n = m = 10 |
|---|---|
| plain Kalman filter, joint vs sequential | 1.000 (1e-15) |
| LucidFilter, **scale walk frozen** at `mu = 0` | **1.006** |
| LucidFilter, one bank member, walk live | 1.164 |
| LucidFilter, full bank, walk live | 1.377 |

**With the walk frozen the two routes agree to 0.6%.** The repeated collapse and the
repeated bank reweighting cost essentially nothing; the entire gap was in the scale walk.

The mechanism, once looked at directly: the engine's scale score is the *local* one — it
keeps `Q`'s own dependence on `xi` and drops the prior covariance's. At a zero gap
`Q(0) = 0`, so `dS/dxi = 0` identically for every **process** axis, and the score is not
merely small but structurally absent. The consequence is an attribution accident: of the
`m` readings taken at one instant, only the one that happened to arrive first — the one
carrying the elapsed gap — contributed anything at all to the process-scale walk, and the
other `m − 1` innovations, which say just as much about the same `Q`, were discarded. A
single instant inspected by hand shows it exactly: joint `process_scale` `[−0.630, −0.636]`
against pointwise `[−0.630, −0.611]`, the second mode short because the sensor that reads
it arrived at a zero gap.

## The fix

The score for a process mode uses the **live process time** — the gap over which the `Q`
now sitting in the prior covariance was injected — rather than the gap since the last
event. The two are the same number whenever the gap is non-zero, so nothing that was
already working changes by even a bit; at a zero gap it restores the leading term of the
`dP/dxi` the local score drops, instead of leaving zero in its place.

| n = m | ratio before | **ratio after** | Δ loglik before | **after** |
|---|---|---|---|---|
| 2 | 1.056 | **1.018** | −198 | **−11.8** |
| 4 | 1.293 | **1.075** | −825 | **−20.4** |
| 6 | 1.309 | **1.066** | −1347 | **−29.7** |
| 10 | 1.377 | **1.100** | −2323 | **−42.6** |

The ledger telescoped to within 40–50× of where it was, and the residual 2–10% sat just
above the 0.6–2.0% floor the frozen-walk control measured.

**Both numbers are historical.** They were taken before the split-holding rule
([`SUMMARY.md`](../SUMMARY.md) item 8), which came out of
[`0005`](0005_the_asynchronous_rig.md) and deliberately widened this gap: a partial event
now declines to move a split it cannot see, so a row delivered as points no longer converges
to the row's attribution at all. The live numbers are at the top of this file. The
zero-gap fix recorded here is still load-bearing — it is what stops the process-scale
evidence of an instant being handed to whichever sensor happened to arrive first — it is
simply no longer the last word on how close the two routes get.

