# 0004 — four things that should have helped the split and did not

All four were run on the hero rig with `LucidFilter()` told nothing, against the build of
[`0002`](0002_ratio_ladder.md) (steady state **1.031x**, regime C **1.101x**, jump rise 7,
calibration 1.44 / 0.78).  Each is filed because the reasoning behind it was sound and the
measurement was not; the reasons are the useful part.

## 1. Memoryless windows on the confounded axes — worse, in every form

**The reasoning.** A confounded group's two axes span the split, and the split has exactly zero
per-step Fisher (0001).  A window posterior on them therefore accumulates something the data never
said: after a sensor degrades, "it was all process" fits every single step exactly as well as the
truth, so a persistent window can lock into it.  The only honest kernel on a coordinate with no
per-step information looked like the prior itself — a one-step hedge that carries nothing forward.

| variant | ss | regime C | rise |
|---|---|---|---|
| build (window memory kept) | 1.031x | **1.101x** | 7 |
| both axes memoryless | 1.045x | 1.251x | 6 |
| process axis only | 1.031x | 1.277x | 7 |
| sensor axis only | 1.046x | 1.219x | 3 |

**Why it fails.** The window's memory is not carrying an illegitimate verdict; it is carrying the
part of the split that genuinely moves inside a run.  In regime C the true split really does drop
by a factor of nine, and the ladder's own verdict cannot follow that fast (see §4 below and 0002
§4).  The window is what covers the gap.  Stripping the *sensor* axis's memory is the most
instructive: it buys the level jump (rise 3, because the process axis then dominates every
response) and pays for it in exactly the regime where the sensor is the answer.

## 2. Pairing the confounded axes into one star axis — worse

**The reasoning.** The star's two axial windows are each an extreme ("the whole change was mine").
They fit any change to `S` equally well, so the collapse splits a total change between them about
half and half, whatever the member's hypothesis about the split is.  The hypothesis the caltrop is
missing is the corner where both move together — a change in the TOTAL at the member's own split —
which for a confounded pair is the one that is usually right (0013's no-corners caveat, exactly
where it should bite).

| variant | ss | regime C | rise |
|---|---|---|---|
| build (two axial windows) | 1.031x | **1.101x** | **7** |
| pair REPLACES the two singles | **1.027x** | 1.237x | 11 |
| pair ADDED to the two singles | 1.035x | 1.377x | 8 |

**Why it fails.** Replacing them gives the best steady state measured anywhere in this workstream
(1.027x) and loses everything else: with no split window, a level jump can only be absorbed by the
bank swinging to a high-gain rung, which takes eleven steps, and the fast partial re-split in
regime C is gone too.  Adding the corner is worse still — it does not replace the wrong
attribution, it dilutes the right one, because all three hypotheses fit the step equally and the
corner simply takes a third of the mass.  The caltrop's missing corner is real; it is not the
remedy here.

## 3. Deriving the ladder's memory from the class — much worse

**The reasoning**, and it is the most attractive of the four.  `theory/02` §C solves the tail-length
question in closed form: under drifting parameters the optimal rectangular window is
`L* = sqrt(3 d / (omega^2 tr I_1))`, and both inputs are already known to a member — `I_1 = 1`
exactly in the arclength coordinate `t = arccos(1 - K)` (the coordinate the MA(1) divergence
`0.5 dt^2` is unit-curvature in), and `omega` is the member's own class, a log-scale AR(1)
innovation `s sqrt(1 - phi^2)` converted into `t` at its own rung.  The numbers that falls out are
`L* = 25 .. 225` steps across the shipped box — squarely on 0041's Lorden frontier, and exactly
the horizon regime C needs, where the bank's `forget` gives 1000.

Measured, with the ladder's forgetting set per member to `1 - 1/L*`:

| | ss | regime C | rise | calib A |
|---|---|---|---|---|
| build (`forget` = 0.999 for every member) | **1.031x** | **1.101x** | 7 | 1.44 |
| per-member derived `L*` | 1.294x | 1.453x | 4 | 0.82 |

**Why it fails.** `omega` is the drift of the *scales*, and the scales' drift is already carried —
by the walk and the window.  What the ladder holds is the residual, structural part of the split,
which under the class does not drift at all; feeding the class's `omega` into its memory
double-counts and shortens the tail far below what the data supports.  At `L* ~ 60` the verdict's
own SD is about one nat of log-odds — a factor of three in `Q/R` — and the steady state pays for
it immediately.  0053's lesson (b) survives the challenge: the verdict does belong on the bank's
timescale.  The open question it leaves is sharper than before, and it is stated in 0002 §4: the
memory that is right for holding a verdict is wrong for revising one, and nothing measured here
separates the two.

## 4. The variogram as a second evidence stream — worse

**The reasoning.** `V(k) = E[(y_t - y_{t-k})^2] = k Q + 2 sigma^2` — a process variance accumulates
over a lag and a measurement variance does not, so a tail of prior points separates them exactly
where one step cannot.  Adding that as a per-lag Gaussian density on a Fibonacci ladder (weighted
`1/k`, the independent-pair count, the same `1/k` in `w_k ∝ 1/(V(k)^2 k)`) should sharpen the
rung weights with a statistic the rungs' own adaptation cannot game.

| | ss | regime C | rise | calib A |
|---|---|---|---|---|
| build | **1.031x** | **1.101x** | 7 | 1.44 |
| + variogram evidence on the rung weights | 1.107x | 1.433x | 7 | 1.06 |

**Why it fails** — two reasons, both measured in [`0003`](0003_variogram_channel.md).  It
double-counts: a rung's own one-step predictive densities already multiply to the exact joint
likelihood of everything seen, so the variogram is a function of data that has been scored, and a
less efficient function of it.  And it is not robust to the level jump the rig contains: the jump
enters `(y_t - y_{t-k})^2` at full size for every lag shorter than the time since it happened, so
the direct read is poisoned for hundreds of steps where the rungs' own filters simply absorb it.
The identity is the reason the ladder works; reading it directly is not an improvement on reading
it through a filter.
