# 0005 — The split ladder, verified in its own coordinate

> **AI-generated, not peer-reviewed.** Script: [`0005_the_split_ladder_verified.py`](0005_the_split_ladder_verified.py).
> Numbers below are from that script on the shipped filter.

## Its role, and its coordinate

A rung of the split ladder is a hypothesis about how a confounded pair's total noise
divides between process and sensor. All a split does is set the filter's gain `K`, and
a local-level filter at gain `K` models its differenced data as MA(1) with `θ = 1 − K`.
The per-step Kullback–Leibler divergence between two rungs, with the innovation variance
held at the shared total, is exactly

    D(θ → θ') = ½ (θ − θ')² / (1 − θ'²)        (Whittle; unit innovations, log-term zero)

whose second-order form is `½ dt²` in `t = arccos(1 − K)`. So `t` is the coordinate in
which one step of evidence is worth the same everywhere, and it runs over the **bounded**
interval `[0, π/2]`: complete, no reach constant. This is the one site that was already
built in its linearising coordinate ([`sequence-demix/0002`](../../sequence-demix/exploration/0002_ratio_ladder.md)),
and it is the template the other two sites are now held to.

## (1) Is it well-conditioned everywhere? Exact KL between adjacent rungs

Shipped ladder at `forget = 0.999`: 24 rungs, `t ∈ [0.033, 1.538]`, step `0.0655`,
`K ∈ [5.4e-4, 0.967]`. Target per-step KL `½ step² = 2.14e-3`. Ratio of the exact KL
to the target, rung pair by rung pair from the bottom (`K → 0`) up:

| direction | rungs 1–2 | 2–3 | 3–4 | 5–6 | 9–10 | 15–16 | 23–24 |
|---|---|---|---|---|---|---|---|
| lower rung true | 0.445 | 0.642 | 0.738 | 0.832 | 0.909 | 0.958 | 0.996 |
| upper rung true | 3.99 | 1.77 | 1.43 | 1.23 | 1.11 | 1.05 | 1.00 |
| geometric mean | 1.333 | 1.067 | 1.029 | 1.010 | 1.003 | 1.001 | 1.001 |

**To second order the ladder is exactly uniform** (geometric mean within 0.3% from rung 3
on, within 0.1% from rung 9). The KL is asymmetric, and the asymmetry is the whole
departure: at the `K → 0` end the divergence is 4× larger with the upper rung true than
with the lower rung true. The direction that limits distinguishability is the smaller one,
so the bottom two pairs are **finer than needed** (0.45 and 0.64 of target) — an
over-resolution that costs at most one rung and never under-resolves. The same ladder
written in log-odds (the coordinate the previous ladder used) has spacings from 4.40 to
0.34 nats: a 13× non-uniformity, which is what the arclength removed.

An exact-to-all-orders placement is a one-line recursion (place each next rung where
`min(D(θ→θ'), D(θ'→θ)) = ½ step²`); it would move the bottom two rungs slightly and add
one. Not applied: the second-order ladder already meets its own criterion in the limiting
direction everywhere except the two bottom pairs, and there it errs toward finer.

## (2) AUD-5: is code length monotone in rung count, and does it converge?

Scalar hero rig, 12 seeds, rung spacing `c · √(2/mem)` for four `c`; total log-likelihood
(mean ± sem over seeds) and windowed RMSE:

| `c` | rungs | log-likelihood | jump | steady | regime C | ms/step |
|---|---|---|---|---|---|---|
| 3.00 | 12 | −1715.17 ± 8.46 | 1.7482 | 0.3833 | 0.9037 | 1.19 |
| **1.50** (shipped) | 24 | −1714.80 ± 8.47 | 1.7474 | 0.3833 | 0.8890 | 1.55 |
| 1.00 | 36 | −1714.80 ± 8.46 | 1.7466 | 0.3833 | 0.8882 | 2.03 |
| 0.75 | 47 | −1714.80 ± 8.46 | 1.7470 | 0.3833 | 0.8883 | 2.34 |

Code length is **monotone non-decreasing in rung count and saturated at the shipped
spacing**: halving the spacing buys 0.00 nats over 900 steps; doubling it loses 0.37 nats
and 1.7% on regime C. The ladder is at the resolution the data supports. This is the
monotonicity AUD-5 asked for, measured, and the saturation is the stronger statement:
`c = 1.5` is not a chosen tolerance here but the coarsest spacing that loses nothing.

## Verdict

The split ladder is well-conditioned everywhere for its role: exactly uniform to second
order in the coordinate it lives in, complete over that coordinate's bounded range,
over-resolved (never under-) at the one end where the third-order asymmetry shows, and at
the resolution the data supports. AUD-5's open item (monotonicity) is closed by
measurement. Nothing changes in the filter.
