# 0003 — the channel that carries the split, and why reading it directly is worse than reading it through a filter

Script: `0003_variogram_channel.py`.

## 1. The identity

From the theory workstream (`random-walk-filter/theory/02`, and the derivation in
`original_chat`):

    V(k) := E[(y_t - y_{t-k})^2] = k Q + 2 sigma^2

**A process variance accumulates over a lag; a measurement variance does not.**  Slope `Q`,
intercept `2 sigma^2`.  This is the different signal SUMMARY §6 asks for: one step sees only
`Q + sigma^2` — Proposition 1, exactly — but a *tail* of prior points sees the slope and the
intercept separately, with no fit, no EMA and no threshold.  Everything else in the workstream is
a way of reading this line: the lag-1 triple is its shortest chord (and 96x inefficient for it,
because three adjacent points are a high-pass filter and all the information about `Q` sits at
zero frequency); the Gauss-Markov weights `w_k ∝ 1/(V(k)^2 k)` are its inverse variances; the
`1/k^3` decay is the "remaining information" of an old point.

It is also, and this is the point of this note, exactly what the ladder of
[`0002`](0002_ratio_ladder.md) reads.  A rung runs its own filter over the same tail, and a
filter's one-step predictive densities multiply to the exact joint likelihood of everything it has
seen.  The tail is not an extra channel the ladder is missing; it is the channel the ladder is
made of, in sufficient form.

## 2. Reading it directly, against the ladder

Three reads of the split's log-odds on the hero series, at the events the rig contains
(truth: −3.91 in regime A, −6.11 after the sensor triples at t = 600; a level jump at t = 380):

| | t=300 | t=379 | t=400 | t=500 | t=599 | t=650 | t=750 | t=899 |
|---|---|---|---|---|---|---|---|---|
| truth | −3.91 | −3.91 | −3.91 | −3.91 | −3.91 | −6.11 | −6.11 | −6.11 |
| **ladder (the engine)** | −4.27 | −3.90 | −2.90 | −3.24 | −3.48 | −3.35 | −3.67 | −4.26 |
| variogram, tail 200, raw | −3.77 | −2.76 | −0.41 | −0.61 | −4.04 | −2.89 | −7.21 | **−29.90** |
| variogram, tail 200, rectified | −1.30 | −1.23 | −0.30 | −0.44 | −1.30 | −1.21 | −1.28 | −1.35 |
| variogram, tail 400, rectified | – | – | −1.04 | −0.98 | −0.95 | −1.11 | −1.43 | −1.61 |

RMS error, in nats of log-odds:

| read | regime A | regime C |
|---|---|---|
| **ladder** | **0.64** | **2.44** |
| variogram, tail 200, raw | 0.59 | 14.88 |
| variogram, tail 200, rectified | 2.63 | 4.82 |
| variogram, tail 400, rectified | – | 4.72 |

**In regime A the two are the same read** — 0.59 against 0.64 nats, i.e. the ladder is already
extracting the tail at the precision a direct variogram gets.  (Both are near the floor: the
split's per-step information is ≈0.0154 in `(log q)^2`, so over regime A's ~320 steps the
Cramér–Rao SD is 0.45 nats and the ladder is within 1.4x of it — the same 55–77% efficiency band
the theory workstream measured for the variogram itself.)

**In regime C the direct read is much worse, in either of its two forms, and for two separate
reasons:**

* *Raw*, it is unstable — the GLS slope wanders to zero and the log-odds fall off a cliff (−29.9).
  That is the known behaviour: a fitted slope and intercept are both variances and least squares
  does not know it.
* *Rectified* (`FILTER-010`'s own positive-root correction at the scale of the standard error), it
  is stable and badly biased — the SE floor inflates `Q̂` by more than an order of magnitude at
  `q = 0.02`, which is the "$\hat Q$ looks big → high apparent SNR → short window → big floor"
  spiral the original derivation ran into and closed as a known failure.
* And in both forms it is **poisoned by the level jump**: `(y_t − y_{t−k})^2` takes the jump at
  full size for every lag shorter than the time since it happened, so the read goes −3.77 → −0.41
  across t = 380 and takes hundreds of steps to recover.  The rungs' own filters simply absorb the
  jump, because a jump is exactly what their state means.

## 3. What this settles

The identity is right and it is the reason the split is learnable at all.  Reading it *directly*
is not an improvement on reading it through anchored per-rung filters: the filters are the
sufficient form of the same tail, they are robust to the events the tail statistic is not, and
they are what the bank already knows how to weight.  Adding the direct read on top of them makes
the filter worse — measured in [`0004`](0004_four_negatives.md) §4 — because it double-counts data
the rungs have already scored, with a less efficient and less robust statistic.

The residual it does highlight is real, and it is the workstream's open item: in regime C the
ladder sits at −3.5 when the truth is −6.11, while the raw variogram at t = 750 has reached −7.21.
The direct read is *more responsive and much noisier*; the ladder is *steady and slow*.  Neither is
the right object.  What the C gate needs is a posterior over the split whose memory for holding a
verdict is not the same number as its memory for revising one — see [`0004`](0004_four_negatives.md)
§3 for the one derivation of that number that was tried and failed, and why.
