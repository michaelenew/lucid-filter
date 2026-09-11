# 0009 — The axiom was aimed at the wrong process. Relocating it, and the number it runs on.

> **AI-generated, not peer-reviewed.** Code
> [`0009_split_channel_information.py`](0009_split_channel_information.py).
> Supersedes [`0001`](0001_the_axiom.md)'s R2 and the objection raised against it
> in [`0005`](0005_the_seam.md) §"the flaw in the axiom".

## The correction

[`0001`](0001_the_axiom.md) put the resolvability constraint on the **noise
level's** motion — "no regime structure below `Delta`". [`0005`](0005_the_seam.md)
then killed it with the observation that a regime change is a step, hence
sub-interval at every sampling rate.

That objection is correct and it does not apply to the intended axiom, because
the intended axiom constrains a **different process**. It is not about how fast
the noise level moves. It is about **how fast the state moves relative to the
sensor** — the thing the process/sensor demix actually reads.

> **The assumption: the process is not moving so fast that it looks like sensor
> noise.**
>
> Equivalently, and this is the form worth carrying: *anything white at our
> sampling rate **is** measurement noise.* Not approximately — by definition.

That is not a hedge, it is the only definition of the split that is covariant
under the sampling rate, because "white at `Delta`" is exactly "no slope in the
variogram at `Delta`", and the slope is the whole of the process channel.

## The machinery it names, which is already in hand

[`sequence-demix`](../../sequence-demix/SUMMARY.md), and it is worth restating
because [`0001`](0001_the_axiom.md) did not use it.

**The identity** ([`0003`](../../sequence-demix/exploration/0003_variogram_channel.md)):

```
V(k) := E[(y_t - y_{t-k})^2] = k Q + 2 sigma^2
```

A process variance **accumulates over a lag**; a measurement variance does not.
Slope `Q`, intercept `2 sigma^2` — the classical variogram, with measurement noise
as the nugget. Three adjacent points are the shortest chord of that line: under
process noise the next point centres on the last point, so a line of best fit
through three points is not flat; under measurement noise it centres on the true
value, so it is. (Three adjacent points are a *high-pass* filter, and all the
information about `Q` sits at zero frequency, so the lag-1 triple is 96× less
efficient than the full tail — which is why the engine reads the tail through a
bank of filters rather than through a local statistic.)

**Per step, nothing** ([`0001`](../../sequence-demix/exploration/0001_lockstep.md)):
the per-step scale-Fisher is exactly rank 1 with null direction `(R, -Q)`. Only
the total `Q + R` is visible in one step. That is Proposition 1 in coordinates.

**Over a sequence, everything**: the split acts only through the gain `K`, the
differenced series is MA(1), and the split coordinate is

```
t = arccos(1 - K),      t in [0, pi/2]
```

## The number the relocated axiom runs on, now measured

The repo claims the Whittle MA(1) KL gives `0.5 dt^2` in this coordinate — i.e.
**Fisher information exactly 1 per step in `t`** — and grades it
`AUDIT[derived+proxy]`. Measured here along the ladder's own direction (split
varied at **fixed total**, which is the null direction of the per-step Fisher),
paired over 6 seeds of 300 000 steps:

| `q` | `K` | `t` | measured `I_t` (dt=0.06) | (dt=0.12) | Richardson → 0 |
|---|---|---|---|---|---|
| 0.005 | 0.068 | 0.372 | 1.088 | 1.210 | 1.05 |
| 0.02 | 0.132 | 0.519 | 1.099 | 1.158 | 1.08 |
| 0.10 | 0.270 | 0.753 | 1.080 | 1.110 | 1.07 |
| 0.50 | 0.500 | 1.047 | 1.003 | 1.031 | 0.99 |
| 2.00 | 0.732 | 1.300 | 1.158 | 1.183 | 1.15 |

**`I_t = 1` holds to within 0–15% over a 400× range in `q`.** The claim stands.

> ⚠️ A first run of this probe varied `q` at **fixed `R`**, which moves the total
> as well as the split, and measured `I_t` drifting to 2.6 at `q = 0.5`. That was
> the wrong path through the parameter space, not a refutation. The direction
> matters: the ladder spans the split at fixed total, and that is where the metric
> is flat.

## Why this coordinate is the right place to put an axiom

Compare the two channels the filter carries:

| | coordinate | range | information rate | free parameters |
|---|---|---|---|---|
| the **split** | `t = arccos(1-K)` | **compact**, `[0, pi/2]` | **1 per step**, flat | none — 24 rungs follow |
| the **noise scale** | `lambda` (log-variance) | unbounded | `1/2` per event | `_PHIS`, `_SS` (AUDIT[measured]) |

The split channel is completely pinned with no free parameters *because its
coordinate is compact and its metric is flat*: the rung count is
`(pi/2) / (1.5 sqrt(2(1-forget)))` = 24, and nothing was chosen. The noise-scale
channel is the one carrying the repository's unjustified constants, and the reason
is structural — `lambda` is unbounded, which is why `_SS`'s top end needs a "reach"
argument instead of a derivation.

**That is the asymmetry to exploit.** If the log-scale class's timescale can be
tied to *the split channel's* resolution rather than assumed as a stationary AR(1)
box, it inherits the compactness and the flat metric — and the box's ends stop
being a convention.

## The derivation sketch, and it is only a sketch

The split's blur after `N` steps is `sqrt(2/N)`. If the split coordinate drifts at
`sigma_t` per step, staleness over a window of `N` is `sigma_t sqrt(N)`; balancing
it against the noise floor gives

```
N* = sqrt(2) / sigma_t,        achieved resolution = 2^{1/4} sqrt(sigma_t)
```

and requiring that resolution to be smaller than the coordinate's own range
`pi/2` gives `sigma_t < (pi/2)^2 / sqrt(2) = 1.745` per step — *the split may not
traverse its own range within one sampling interval*, which is the axiom said
back in its own units.

**Marked as a sketch, not a result.** The balance assumes a random-walk drift and
an unweighted window, neither of which is what the bank does. What is worth
carrying is not the number 1.745 but the *shape*: a compact coordinate with a flat
unit metric supports a resolution argument that an unbounded log-scale does not.

## What is now superseded

- [`0001`](0001_the_axiom.md)'s **R2** — "no regime structure below `Delta`" — is
  withdrawn, and so is [`0005`](0005_the_seam.md)'s step-change objection *as
  aimed at the intended axiom*. Both remain correct about the version
  [`0001`](0001_the_axiom.md) actually wrote.
- [`0002`](0002_the_ridge.md), [`0003`](0003_what_stationarity_costs.md),
  [`0005`](0005_the_seam.md), [`0007`](0007_phi_reach.md) are **unaffected**.
  They measured the noise-scale channel, and every one of those results stands —
  including the recommendation that `_PHIS` reach nearer 1.
- The **AR(1) shape derivation** stands and is independent: among time-homogeneous
  continuous-path Gaussian Markov processes the linear SDE is the only family
  (Doob), and its `Delta`-sampling is AR(1) with `phi = e^{-kappa Delta}`. So AR(1)
  — and the `phi -> phi^a` rule the filter already uses for irregular gaps — is
  *derived from sampling covariance* rather than assumed.

## Next

1. **Write the relocated axiom properly** — R1 (refinement covariance) is
   unchanged and still does useful work; R2 is replaced by the whiteness
   definition above. State what it forces and what it forbids, with the
   `sequence-demix` machinery as the mechanism rather than an afterthought.
2. **Tie the log-scale class timescale to the split channel's resolution** — the
   sketch above, done properly against the bank's actual weighting. This is the
   route by which the noise-scale channel's box ends could stop being
   `AUDIT[measured]`, and it is the first thing that would justify the whole
   workstream.
3. **Test the axiom's own failure mode** — the relative-degree rigs. The README's
   standing open (a jerk disturbance reaching a rate gyro through `dt^2/2` is
   blamed on the sensors, at 103× the oracle) is *exactly* this axiom being
   violated: the process contribution is white at that sensor's sampling rate, so
   the machinery books it as measurement noise, correctly by the axiom's own
   definition and uselessly for the user. A per-channel variogram-slope test would
   name it.
