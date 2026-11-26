# 0010 — the battery for `LucidFilter(forget=None)`

> **AI-generated, not peer-reviewed.** Code [`0010_the_battery.py`](0010_the_battery.py),
> [`0010b_suite_under_drift.py`](0010b_suite_under_drift.py); outputs
> [`0010_output_scalar_async.txt`](0010_output_scalar_async.txt),
> [`0010_output_arm.txt`](0010_output_arm.txt), [`0010b_output.txt`](0010b_output.txt);
> predictions written first in [`0010_predictions.md`](0010_predictions.md). The class drift
> is now in `lucid/filter/lucid.py` as an opt-in; these probes use the filter directly.

## What is being tested

`forget=None` selects the class drift (`_ClassDrift`): the resolvability ceiling between
neighbouring cells, the fraction beneath it found by a five-node walking star, and member
state carried with the weight. The default (`forget = 0.999`) is unchanged bit for bit and
is the comparison throughout.

**Three versions were run; only the last is reported in the tables.** The first two are
kept under [`v1/`](v1/) and [`v2/`](v2/) because what changed between them was found by
the battery itself:

| version | change | why |
|---|---|---|
| v1 | as ported from the load-time patch | — |
| v2 | the chain over an elapsed time is the exact `exp(τG)`, replacing a first-order step with a row cap | the cap made every node of the star identical over long gaps (`0012`, `gappy`) |
| **v3** | the split ladder gets its own derived chain | with the split left as pure Bayes the drift lost a slow ramp by 9% and irregular gaps by 27% (`0012`, `0013`) |

## The shipped suite

| run | result |
|---|---|
| default path, with the new tests | **65 passed**, 16 skipped (the kernel tests, not built here) |
| the same suite with `forget=None` as the default | **63 passed, 1 failed** (1 deselected: it asserts the shipped default) |

The one failure is `test_pointwise_decomposition_tracks_the_joint_row`: the ratio of
pointwise to joint-row RMSE is 1.32 against a bound of 1.25 (default: 1.16). On that seed
the drift improves the joint row (0.0879 → 0.0817) and slightly worsens the pointwise
stream (0.1022 → 0.1081). On four seeds of the same rig the pointwise RMSE averages 0.0907
under the drift against 0.0921 under the default, so this is the bound of a one-seed ratio
being crossed, not a structural break. It would need re-pinning if the default changed.

## Hero rig — 24 seeds, paired

| | jump RMSE | steady RMSE | loud RMSE | log-likelihood |
|---|---|---|---|---|
| default | 1.815 | 0.3740 | 0.928 | −1721.7 |
| `forget=None` | 1.759 | 0.3738 | **0.747** | **−1704.6** |
| paired `t` | −0.91 | −0.37 | **−7.04** | **+19.85** (+17.1 nats) |

## Asynchronous rig — 8 seeds, paired

| | whole (× oracle) | hot (× oracle) | worst hot | log-likelihood |
|---|---|---|---|---|
| default | 1.149 | 1.530 | 2.79 | 1735.6 |
| `forget=None` | 1.112 | 1.430 | 2.65 | **1766.0** |
| paired `t` | −0.74 | −0.51 | | **+10.99** (+30.4 nats) |

## Arm — 6 seeds, tip RMSE over the oracle's

| seed | | calm | SENSOR | PROCESS | POTFAIL | BOTH | log-likelihood | worst 100 steps |
|---|---|---|---|---|---|---|---|---|
| 0 | default / drift | 1.141 / 1.130 | 2.700 / 1.562 | 1.160 / 1.312 | 1.114 / 1.114 | 1.063 / 0.858 | 30,574 / 33,792 | 5.9× / 4.1× |
| 1 | | 1.102 / 0.897 | 3.480 / 1.401 | 1.027 / 1.076 | 1.160 / 1.051 | 1.723 / 0.962 | 29,722 / 33,409 | 3.6× / 1.7× |
| 2 | | 0.563 / 0.541 | 2.144 / 1.910 | 1.157 / 1.158 | 0.941 / 1.175 | 3.516 / 1.265 | 30,299 / 33,755 | 14.8× / 3.7× |
| 3 | | 1.154 / 1.067 | 2.008 / 1.312 | 1.430 / 1.222 | 1.223 / 1.056 | 1.341 / 1.505 | 30,922 / 33,955 | 4.2× / 3.9× |
| 4 | | 1.142 / 1.021 | 1.503 / 1.451 | 1.548 / 1.269 | 1.105 / 1.040 | 0.928 / 1.091 | 30,160 / 33,599 | 4.9× / 4.7× |
| 5 | | 1.113 / 0.992 | 2.895 / 1.656 | 1.169 / 1.331 | 1.106 / 1.120 | 1.025 / 1.139 | 31,020 / 33,897 | 3.1× / 1.8× |
| **mean** | default | 1.036 | 2.455 | 1.248 | 1.108 | 1.599 | 30,450 | 14.8× |
| **mean** | drift | **0.941** | **1.549** | 1.228 | 1.093 | 1.137 | **33,734** | **4.7×** |
| | paired `t` (log ratio) | **−3.34** | **−3.30** | −0.16 | −0.23 | −1.23 | | |

No non-finite output on any run. Cost on the arm: 78 against 79 ms/step.

## Predictions against the run

| | prediction | verdict |
|---|---|---|
| B1 | the filter's own path reproduces the patch on the arm to within 1% | **FALSE in the letter** — seed 0 BOTH came out 0.964 against the patch's 0.911 (5.8%), the other windows within 1%. The state step was re-implemented and the records diverge at the last digits; no window changed sign |
| B2 | hero with state carried: log-likelihood > +10 nats at `t` > 3; loud better; jump and steady within noise | **confirmed** — +17.1 nats, `t` = 19.9; loud −19%, `t` = −7.0 |
| B3 | async: RMSE within noise, log-likelihood higher | **confirmed** |
| B4 | at most three of the 55 behavioural tests fail under the drift as default, none non-finite | **confirmed** — one, by a one-seed ratio bound |

## Reading

1. **By the filter's own loss the drift is better on every rig**, with paired `t` of 20, 11
   and (arm) +11% on every seed.
2. **In state error it is never resolvably worse**: better in the hero loud window, the
   arm's calm and SENSOR windows; within noise everywhere else. The arm's worst
   100-step stretch falls from 14.8× to 4.7× the oracle.
3. **Per seed there are losses** — PROCESS on seeds 0 and 5 (+13%, +14%), BOTH on seeds 3,
   4 and 5 (+11–18%), POTFAIL on seed 2 (+25%). They are profiled in
   [`0011`](0011_profile_worse_regimes.md).
4. **Cost.** On the scalar rig the drift step costs 3.4× the default's: 1.09 → 3.69
   ms/step on an idle machine. Most of it is the state step over 360 cells (15 classes × 24
   split rungs) done densely; a separable form exists and is not implemented. No
   measurable cost on the arm (15 cells).
