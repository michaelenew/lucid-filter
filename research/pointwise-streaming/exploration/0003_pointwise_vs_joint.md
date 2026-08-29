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

## The failure, and where it was

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

The ledger now telescopes to within 40–50× of where it was, and the residual 2–10% sits
just above the 0.6–2.0% floor the frozen-walk control measured — i.e. what is left is the
`m` successive collapses and the walk taking `m` smaller steps on the same information,
which is a property of the caltrop-plus-GPB1 construction and not of the decomposition.
**Recorded as the residual, not claimed as zero.**

## Cost per instant

| n = m | ms/instant joint | ms/instant pointwise | multiply-adds joint | pointwise | ratio |
|---|---|---|---|---|---|
| 2 | 5.8 | 9.7 | 680 | 442 | 0.65 |
| 4 | 9.8 | 26.7 | 10,560 | 5,412 | 0.51 |
| 6 | 15.2 | 51.6 | 52,920 | 24,990 | 0.47 |
| 10 | 34.6 | 133.4 | 405,000 | 179,010 | **0.44** |

The **arithmetic falls** — the joint `G(2n²m + 2nm² + m³)` becomes `m · G(2n² + 2n + 1)`,
and the `m³` solve is gone, so a pointwise instant is 0.44–0.65× the multiply-adds of the
joint one and the advantage grows with `m`. The **wall time rises**, 1.7–3.8×, because `m`
python-level events replace one and this is pure numpy: the same interpreter-overhead
story the repo's cost model already tells (`0053` §5, README). Both numbers are real and
they point opposite ways — in a compiled implementation the first column is the one that
governs, in numpy today the second is.
