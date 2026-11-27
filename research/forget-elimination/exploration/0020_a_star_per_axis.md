# 0020 — the interval as yardstick, and a star per axis

> **AI-generated, not peer-reviewed.** Code [`drift3.py`](drift3.py) (`Drift3`),
> [`0020_a_star_per_axis.py`](0020_a_star_per_axis.py), output
> [`0020_output.txt`](0020_output.txt), predictions written first in
> [`0020_predictions.md`](0020_predictions.md). Probe-side; nothing under `lucid/` is changed.

`0019` made the ladders' drift independent of their grid and thereby exposed a diffusion
constant that resolvability cannot fix. This note tries the two constant-free routes named
there, together.

## The construction

1. **Grid-free ladders.** The split and hazard beliefs diffuse in Whittle arclength, hop
   rate `D/Δt²`.
2. **The interval as yardstick.** A ladder spans a bounded range `L` with information 1 per
   step. Its ceiling is the diffusion at which the steady blur `√D` equals the variance of
   knowing nothing, `L²/12`: `D_max = L⁴/144` — 0.042 for the split, 0.0084 for the hazard.
3. **A star per axis.** Class, split and hazard each have their own fraction beneath their
   own ceiling, on a caltrop star (a centre and two arms each way per axis, up to 13 copies
   of the bank's weights). All start at their ceilings.

Nothing in it is chosen. Today's behaviour sits at fractions of about `2e-4` (split) and
`1e-3` (hazard) of these ceilings.

## Result

**Ramp** (6 seeds, × oracle):

| `_LADDER_MEM` | default | filter as it stands | **star per axis** | split fraction at step 1000 / 3000 / end |
|---|---|---|---|---|
| 1000 | 1.118 | 1.033 | **1.081** | 5e-2 / 1e-2 / 2.4e-3 |
| 4000 | 1.118 | 1.107 | **1.081** | 5e-2 / 1e-2 / 2.1e-3 |

**Faults** (4 seeds):

| calm | arm | hazard believed before the fault | false alarms: first 500 / after | steps to flag | held 50 | RMSE after |
|---|---|---|---|---|---|---|
| 600 | default | 1.1e-3 | 0.4% / 0% | 60 | 92 | 0.2928 |
| 600 | as it stands | 9.5e-4 | 0.1% / 0% | 68 | 104 | 0.2922 |
| 600 | **star per axis** | 4.5e-2 | **8.1%** / 0.3% | **18** | 48 | 0.2881 |
| 4,000 | default | 3.3e-4 | 0.4% / 0.4% | 82 | 94 | 0.2728 |
| 4,000 | as it stands | 3.9e-4 | 0.1% / 0.2% | 92 | 98 | 0.2726 |
| 4,000 | **star per axis** | 5.2e-3 | **5.6% / 2.0%** | **54** | 72 | 0.2710 |

**Hero** (24 seeds, 900 steps) and **stress** (8 seeds):

| | default | as it stands | **star per axis** |
|---|---|---|---|
| hero log-likelihood | −1721.7 | −1704.6 | −1717.0 |
| hero RMSE jump / steady / loud | 1.815 / 0.374 / 0.928 | 1.759 / 0.374 / 0.747 | 1.460 / **0.445** / 0.928 |
| calm (× oracle) | 1.004 | 1.005 | **1.058** |
| calm, log-lik/step − default, ×1000 | 0 | −2.1 | **−10.7** |
| stale | 1.177 | 1.124 | **1.092** |
| flicker 25 | 2.024 | 1.395 | 1.856 |
| flicker 100 | 1.289 | 1.391 | **1.750** |

## Predictions against the run

| | prediction | verdict |
|---|---|---|
| H1 | grid independence: ramp at 1000 and 4000 within 2% | **confirmed** — identical to three digits |
| H2 | ramp within 7% of the filter as it stands, better than the default | **confirmed** — +4.6%; 1.081 against 1.118 |
| H3 | the split star within two e-folds of `2e-4` by the end of the ramp | **FALSE** — `2.4e-3`, 2.5 e-folds short after 6,000 steps |
| H4 | faults: flagging within 25% of the default; believed hazard within 3×; early false alarms higher | **flagging is faster, not slower** (18 against 60; 54 against 82); **hazard 15–40× higher**; false alarms 5–20× the default's, and not only early |
| H5 | hero: log-likelihood between the default's and the filter-as-it-stands' | **confirmed** — and the steady window is 19% worse, the loud-window gain is gone |
| H6 | the walk down is the price everywhere | **confirmed** — calm log-likelihood 5× further below the default than the filter as it stands |

## Reading

1. **It is what was asked for and it works as a construction.** No constant, no grid
   dependence, and it improves with the length of the record (the one long mode, `stale`,
   is its best: 1.092 against 1.124).

2. **The walk down is too long to be a default.** From a ceiling that means "nothing is
   learned" the split has seven to nine e-folds to descend. The evidence that a lower rate
   is better goes as `√D` — strong at the top, vanishing near the answer — so the star
   takes 6,000 steps to get within 2.5 e-folds and every record shorter than that is spent
   mostly too fast. On 900 steps the steady-state error is 19% worse than the default's.

3. **The hazard has no answer to walk to.** Its star stays high, so the bank believes the
   world 15–40× more fault-prone than the default does: faults are flagged two to three
   times sooner and false alarms are five to twenty times more frequent. State error is
   unchanged. That is a point on the detection-delay / false-alarm curve, and the data
   cannot pick the point: with no fault in the record, every rate predicts the calm almost
   equally well. **What the earlier notes called a defect of the pure-Bayes rungs — the
   report going "stale" — is the other end of the same curve**, the end where a long calm
   is believed.

4. **So the interval is a true ceiling and a poor yardstick.** It bounds the rate without
   locating it, and the star's job from there is the whole problem.

## Where this leaves the three options of `0019`

Options 1 and 2 are built and measured: sound, constant-free, and worse than the filter as
it stands on every record shorter than several thousand steps. Option 3 (declare `D`)
stands. A fourth appeared while reading these results — the split's rate need not be an
unknown at all — and is tested in [`0021`](0021_the_split_the_class_implies.md).
