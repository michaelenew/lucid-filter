# 0008 — arm seed 2: where the weight is when the estimate spikes

> **AI-generated, not peer-reviewed.** Code [`0008_the_bad_seed.py`](0008_the_bad_seed.py),
> output [`0008_output.txt`](0008_output.txt), predictions written first in
> [`0008_predictions.md`](0008_predictions.md). The filter is patched at load
> ([`star_filter.py`](star_filter.py)); `lucid/` is untouched.

`0007` left one failure: on arm seed 2 the star's tip error spikes for ~100 steps after
two bursts end (steps 500–600 and 1400–1500), taking the calm window from 0.56× to 6.84×
the oracle. Two causes were proposed — rates too high, and weight moved onto members whose
state is poor. This probe separates them.

## Step 1 — the weight and the members

Per 50-step block: the bank's weight on each `s` class (summed over `phi`) and the tip
error of each class's own estimate.

| block | filter | tip error | weight on `s` = 0.2 / 0.4 / 0.8 / 1.6 / 3.2 | class error, same order |
|---|---|---|---|---|
| 450–500 | shipped | 0.0063 | 0 / 0 / **1.000** / 0 / 0 | – / – / 0.0057 / – / – |
| | star | 0.0061 | 0.980 / 0.019 / 0.001 / 0 / 0 | 0.0057 / 0.0053 / 0.0044 / 0.027 / – |
| **500–550** | shipped | 0.0083 | 0 / 0 / **1.000** / 0 / 0 | – / – / 0.0080 / – / – |
| | star | **0.294** | 0.019 / 0.002 / 0.021 / **0.958** / 0 | 0.0071 / 0.0049 / 0.012 / **0.302** / 0.012 |
| 550–600 | star | 0.139 | 0.690 / 0.064 / 0.004 / 0.242 / 0 | 0.0023 / 0.0020 / 0.0024 / **0.286** / 0.0065 |
| 600–650 | star | 0.0044 | 0.870 / 0.075 / 0.049 / 0.006 / 0 | all 0.003–0.006 |
| **1450–1500** | shipped | 0.066 | 0 / 0 / **1.000** / 0 / 0 | – / – / 0.063 / – / – |
| | star | **0.472** | 0.001 / 0.693 / 0.226 / 0 / 0.080 | 0.067 / 0.071 / 0.030 / 0.079 / **1.65** |

At the worst step of each spike the star has 0.998 of its weight on `s = 1.6` (step 518)
and 1.000 on `s = 3.2` (step 1453). Other `s = 1.6` members at step 518 have an error of
0.003 — the weight is on one particular bad member, not on a bad class.

## Step 2 — dose

| ceiling rates × | calm | SENSOR | PROCESS | POTFAIL | BOTH | log-likelihood | error 500–600 | error 1400–1500 |
|---|---|---|---|---|---|---|---|---|
| shipped | 0.563 | 2.144 | 1.157 | 0.941 | 3.516 | 30,299 | 0.0064 | 0.047 |
| 1 | 6.84 | 1.327 | 1.198 | 1.138 | 2.782 | 32,794 | 0.230 | 0.334 |
| 0.3 | 7.04 | 1.410 | 1.203 | 1.132 | 2.797 | 32,771 | 0.232 | 0.333 |
| 0.1 | 7.26 | 1.470 | 1.207 | 1.130 | 2.821 | 32,740 | 0.234 | 0.333 |
| 0.03 | 7.53 | 1.526 | 1.212 | 1.132 | 2.845 | 32,699 | 0.237 | 0.333 |

## Predictions against the run

| | prediction | verdict |
|---|---|---|
| G1 | during the spikes the star holds far more weight on the wide classes than shipped | **confirmed** — 0.96–1.00 against shipped's 0 |
| G2 | those members' own estimates are the bad ones | **confirmed** — 0.30 and 1.65 against 0.005–0.08 for the rest |
| G3 | the spike shrinks with the rate scale and is gone by 0.1 | **FALSE** — unchanged to three digits across a 30× range |
| G4 | the SENSOR gain survives at 0.1 | **confirmed** (1.47), and moot |

## Reading

1. **The spike is the bank choosing a member with a bad state estimate.** For ~50 steps
   after a burst ends one wide member predicts the *data* better than the narrow ones and
   takes the weight, while its *state* is 30–200× off. Its predictive density is good and
   its mean is not, which is why the log-likelihood rises while the tip error spikes.

2. **The rates are not the cause.** Thirty times slower changes nothing. Any rule that
   keeps the losing members alive at all — a floor of any size — lets that member win on
   evidence when its moment comes. A member-level clamp (`0001`) would do the same.

3. **The shipped filter is not selecting on the arm at all.** Its entire weight sits on one
   class, `s = 0.8`, in every block, burst or calm. With `forget = 0.999` the evidence it
   holds for that class is thousands of nats and nothing dislodges it. This is why
   `forget = 1` and `forget = 0.999` agree to three decimals on the arm (`0007`), and why
   the arm has never shown this failure: the bank has been one member. The star's SENSOR
   gain (2.14 → 1.33) is what re-selection is worth there; the spike is what it costs when
   the selected member's state is bad.

4. **So the defect is in the members, and it was hidden.** A wide-`s` member running on
   its own for hundreds of steps accumulates a poor state; the bank never looked at it
   before. The remedy that follows is to stop members running on their own: when weight
   moves into a member, its state should come from where the weight came from
   ([`0009`](0009_state_moves_with_weight.md)).
