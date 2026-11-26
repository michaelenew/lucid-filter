# 0009 — the chain carries member state with the weight it moves

> **AI-generated, not peer-reviewed.** Code
> [`0009_state_moves_with_weight.py`](0009_state_moves_with_weight.py), outputs
> [`0009_output.txt`](0009_output.txt) (seed 2, all variants) and
> [`0009b_output.txt`](0009b_output.txt) (seeds 0, 1), predictions written first in
> [`0009_predictions.md`](0009_predictions.md). Patched at load
> ([`star_filter.py`](star_filter.py), `MIX`); `lucid/` is untouched.

> **⚖️ ATTRIBUTION —** _Mixing member states in proportion to the weight that flows
> between them is the interaction step of the interacting-multiple-model filter (Blom &
> Bar-Shalom 1988)._ Status: REPRODUCTION of the method; the measurements are new.

## What was tested

`0008` traced the star's failure on arm seed 2 to the bank selecting a wide member whose
own state estimate was 30–200× off. The remedy: when the chain moves weight from member
`i` to member `j`, `j`'s state becomes the mixture of where its weight came from,

    mu_{i|j} = post_i · P_ij / Σ_i post_i · P_ij,       P = I + rate · HD at the star's centre.

Three variants of *which* state is carried: the state mean and covariance (`m`); the
scale-walk centres and their variance (`mu`); both.

## Seed 2, all variants

| | calm | SENSOR | PROCESS | POTFAIL | BOTH | log-likelihood | worst 100-step block |
|---|---|---|---|---|---|---|---|
| shipped | 0.563 | 2.144 | 1.157 | 0.941 | 3.516 | 30,299 | 14.8× |
| star, weights only (`0007`) | 6.841 | 1.327 | 1.198 | 1.138 | 2.782 | 32,794 | 104.9× |
| **star + `m`** | **0.505** | 2.067 | 1.152 | 1.171 | **1.859** | **33,615** | **6.1×** |
| star + `mu` | 88.7 | 1.374 | 1.078 | 0.910 | 13.6 | 12,703 | 283.5× |
| star + `m` + `mu` | 2.180 | 1.145 | 2.924 | 1.014 | 13.7 | 27,399 | 21.6× |

## Star + `m` on three seeds

| seed | filter | calm | SENSOR | PROCESS | POTFAIL | BOTH | log-likelihood | worst block |
|---|---|---|---|---|---|---|---|---|
| 0 | shipped | 1.141 | 2.700 | 1.160 | 1.114 | 1.063 | 30,574 | 5.9× |
| 0 | weights only | 1.184 | 1.854 | 1.095 | 1.232 | 1.384 | 32,664 | 5.7× |
| 0 | **star + `m`** | 1.154 | 1.719 | 1.318 | 1.114 | 0.911 | 33,661 | 4.9× |
| 1 | shipped | 1.102 | 3.480 | 1.027 | 1.160 | 1.723 | 29,722 | 3.6× |
| 1 | weights only | 1.058 | 1.365 | 1.065 | 1.142 | 1.690 | 33,960 | 42.9× |
| 1 | **star + `m`** | 0.950 | 1.317 | 1.027 | 1.054 | 0.915 | 33,343 | 1.8× |
| 2 | shipped | 0.563 | 2.144 | 1.157 | 0.941 | 3.516 | 30,299 | 14.8× |
| 2 | weights only | 6.841 | 1.327 | 1.198 | 1.138 | 2.782 | 32,794 | 104.9× |
| 2 | **star + `m`** | 0.505 | 2.067 | 1.152 | 1.171 | 1.859 | 33,615 | 6.1× |
| mean | shipped | 0.935 | 2.775 | 1.115 | 1.072 | 2.101 | 30,198 | |
| mean | **star + `m`** | **0.870** | **1.701** | 1.166 | 1.113 | **1.228** | **33,540** | |

## Predictions against the run

| | prediction | verdict |
|---|---|---|
| H1 | carrying both `m` and `mu` removes the spikes (calm within 2× of shipped) | **FALSE as stated** — both together is poor (calm 2.18, BOTH 13.7). **Carrying `m` alone does it**: calm 0.505 against shipped's 0.563, worst block 6.1× against 104.9× |
| H2 | `mu` alone does better than `m` alone | **FALSE, badly** — `mu` alone is the worst variant by far (calm 88.7, log-likelihood 12,703) |
| H3 | the SENSOR gain survives (< 2.0 on seed 2) | **FALSE on seed 2** (2.067 against shipped's 2.144) and **true on seeds 0 and 1** (1.72, 1.32 against 2.70, 3.48). Mean 1.70 against 2.78 |
| H4 | log-likelihood stays above shipped | **confirmed** — +3,000 to +3,600 on every seed, and above the weights-only star on two of three |

## Reading

1. **The bad seed is explained and fixed.** The spike was the bank selecting a member that
   had been running on its own long enough for its state to go bad; handing a member the
   state of the members its weight came from removes it. The worst 100-step stretch is now
   *below shipped's* on all three seeds (4.9 / 1.8 / 6.1× against 5.9 / 3.6 / 14.8×). Seed 1
   had the same defect under the weights-only star — a 42.9× block that the window
   averages in `0007` did not show.

2. **Against shipped, on three seeds:** better in calm (0.87 against 0.94), SENSOR (1.70
   against 2.78) and BOTH (1.23 against 2.10); slightly worse in PROCESS (+5%) and POTFAIL
   (+4%); log-likelihood +11%. No window is worse by more than 14% on any seed except
   POTFAIL on seed 2 (+24%).

3. **Scale-walk state must not be carried this way.** Mixing the scale centres across
   members is destructive. The likely reason is structural rather than a bug: each
   member's window posterior is stored relative to its own centre on its own spacing
   (`gap ∝ s`), so moving the centre without re-expressing the window leaves the two
   inconsistent. Not investigated further; `m` alone is the working form.

4. **Some of the SENSOR gain is given back on one seed.** With state shared, the wide
   member can no longer run ahead on its own, which is what hurt in calm and what helped in
   the burst on seed 2 (1.33 → 2.07). The other two seeds keep or improve the gain.

## Limits

- **Arm only.** The mixing step is implemented for one stacked bank whose rows are the
  cells — no split ladder, no fault class. On the hero and async rigs it is a no-op, so
  their `0007` numbers are for the weights-only star. Extending it means mixing within a
  split rung and within a dynamics hypothesis; not done.
- Three seeds, one burst schedule.
- The rates still use the rig's constant noise level in the formula (`0008` showed the
  result is insensitive to them over 30×).
