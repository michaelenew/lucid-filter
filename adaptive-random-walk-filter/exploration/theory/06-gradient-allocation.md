# 06 — The modes as a smooth square; allocation without detection

Scripts: `scripts/THEORY-005-gradient-allocation.py`, `scripts/THEORY-006-continuous-allocator.py`
Figures: fig14–fig18

Two objections drove this file, and both are sustained:

1. *"The two-point rule isn't a rule — it's happenstance past the threshold where
   the minimum sample size becomes sufficient evidence."*
2. *"Event detection is a happenstance too. The right approach differentiates the
   modes along a gradient and handles all modes at once, allocating change
   optimally among the perceived change modes."*

## What was wrong with [03](03-four-deviation-modes.md)

The two anomalies were parameterised as **mean** shifts and the two regime
changes as **covariance** changes. Gaussian mean-scores and covariance-scores are
exactly orthogonal, so the $4\times4$ Fisher matrix came out block diagonal and I
reported "location and scale events are never confusable" as a structural fact.

It isn't one. It is a consequence of giving the anomaly a *known direction* —
i.e. of being the oracle. A non-oracle must marginalise the event size, and
marginalising $\delta\sim N(0,\tau^2)$ turns the mean shift into a rank-one
covariance bump $\tau^2 e_0e_0^\top$. Once that happens all four modes are
covariance perturbations and the block structure is gone.

## The unified family

$$\Sigma(a,\varphi,A)=\Sigma_0 + A\sum_{t}\varphi^{\,t}\Big[a\,G^P_t+(1-a)\,G^M_t\Big]$$

$$G^P_t=e_te_t^\top \quad\text{(process-noise excess at }t),\qquad
G^M_t=(e_t-e_{t+1})(e_t-e_{t+1})^\top \quad\text{(measurement-noise excess)}$$

Two continuous coordinates and an amplitude:

- $a\in[0,1]$ — **channel**: which noise source the excess came from
- $\varphi\in[0,1]$ — **persistence**: impulse at 0, step at 1, and a genuine
  exponential of time constant $1/|\log\varphi|$ in between
- $A>0$ — how much, with $A=0$ meaning "nothing happened"

The four named modes are the **corners**, and they are exactly representable
(asserted in the script, not approximated):

| | $\varphi=0$ (impulse) | $\varphi=1$ (step) |
|---|---|---|
| $a=1$ | **PA** $=A\,e_0e_0^\top$ | **PR** $=A\,\mathbb 1$ |
| $a=0$ | **MA** $=A\,uu^\top$, $u=e_0-e_1$ | **MR** $=A\,\tilde T$ |

$\mathrm{PR}=\sum_t G^P_t$ and $\mathrm{MR}=\sum_t G^M_t$ — **a regime change is
literally the temporal accumulation of its own anomaly.** That is what makes the
anomaly↔regime axis a continuum rather than a dichotomy, and it is why $\varphi$
is the right coordinate for it.

## The Gram matrix that replaces the orthogonality claim (fig14)

Fisher correlations between the four corner directions, evaluated at $H_0$:

| $m$=2 | PA | MA | PR | MR | | $m$=20 | PA | MA | PR | MR |
|---|---|---|---|---|---|---|---|---|---|---|
| **PA** | 1 | 0.256 | **0.787** | 0.312 | | **PA** | 1 | 0.038 | 0.134 | 0.022 |
| **MA** | | 1 | 0.326 | **0.798** | | **MA** | | 1 | 0.093 | 0.221 |
| **PR** | | | 1 | 0.698 | | **PR** | | | 1 | 0.317 |
| **MR** | | | | 1 | | **MR** | | | | 1 |

**The confusable axis is persistence, not channel.** Within a channel, spike vs
shift correlates at 0.79/0.80 at $m$=2. Across channels it is 0.26–0.33. The
previous framing had this exactly backwards in emphasis: process-vs-measurement
was never the hard problem; impulse-vs-step is.

## Detection is fast; attribution is not; and they separate cleanly (fig18)

For a 4-SD jump, non-oracle (amplitude profiled, worst case over rivals):

| $m$ | vs $H_0$ *(anything?)* | vs MA *(which channel?)* | vs PR *(how persistent?)* | vs MR |
|---|---|---|---|---|
| 2 | **8.95** | **7.19** | 0.79 | 1.40 |
| 4 | 10.61 | 10.16 | 2.12 | 4.38 |
| 8 | 11.34 | 11.15 | 4.08 | 9.91 |
| 20 | 11.47 | 11.31 | 7.36 | 11.37 |
| 32 | 11.47 | 11.31 | **9.00** | 11.41 |

Three questions with three different answers, and this is the central result of
the reframe:

- **"Did something happen?"** — 9 nats immediately, saturates by $m$=5. Bounded.
- **"Which channel?"** ($a$) — 7 nats immediately, saturates by $m$=5. Bounded.
- **"How persistent?"** ($\varphi$) — 0.79 nats at $m$=2 and still climbing at
  $m$=32. Unbounded, slow, never finished.

So the oracle behaviour originally described — *absorb the excursion into the
level at once, but leave $Q$ and $\sigma^2$ and their confidences untouched* —
is not a policy choice. It is **forced**, because the level update depends only
on $a$ (resolved at $m$=2) and the $Q$ update depends on $\varphi$ (never
resolved). Allocate on the coordinates the data has pinned; stay spread on the
ones it hasn't. No threshold is involved in either.

## The "two-point rule" was an oracle artifact, and worse than I said

Under the honest formulation, a 4-SD jump earns **0.79 nats of worst-case
attribution evidence at $m$=2**, not 10.5. It reaches 99:1 at $m\approx9$. And
at $m$=2 you would need a **5.1-SD** excursion for even 1 nat.

Jump size needed to reach a given worst-case evidence (fig17, right):

| $m$ | 1 nat | 2.2 (90%) | 4.6 (99%) |
|---|---|---|---|
| 2 | 5.1 SD | >8 | >8 |
| 4 | 2.3 | 4.4 | >8 |
| 8 | 1.9 | 2.6 | 4.8 |
| 16 | 1.6 | 2.3 | 3.4 |
| 32 | 1.6 | 2.3 | 3.0 |

These are contours of one smooth surface. **Nothing switches on anywhere.** The
99:1-at-two-points claim was one level set, drawn on a surface, under an oracle
that knew the event size — exactly the "happenstance past a threshold" reading.
My Occam correction last turn (1.2–2.1 nats) accounted only for not knowing
$\delta$ while keeping the known *direction* of a mean shift; the full non-oracle
cost is far larger, because a variance bump of unknown size says much less than a
mean shift of unknown size.

The $m=1$ singularity from [03](03-four-deviation-modes.md) survives — it is rank
deficiency, not a threshold — but it is now the statement that a single point
cannot place you anywhere on the square, which is a fact about dimension.

## The allocator (fig16, fig17 left)

No event time, no test, no gate. At each step it returns three continuous fields
over what was actually observed, by exact Bayesian model averaging over the
square (a 4,901-component mixture on $(a,\varphi,A)$ including $A=0$):

$$E[\Delta\theta\mid d],\qquad E[a\mid d,\ \text{deviation}],\qquad E[\varphi\mid d,\ \text{deviation}]$$

Fraction of an observed excursion $d_0$ allocated to the level:

| $\lvert d_0\rvert$ / SD | no reversal ($d_1=0$) | full reversal ($d_1=-d_0$) | Gaussian filter |
|---|---|---|---|
| 0.5 | 0.100 | 0.011 | 0.048 |
| 1 | 0.114 | 0.013 | 0.048 |
| 2 | 0.220 | 0.029 | 0.048 |
| 3 | 0.562 | 0.071 | 0.048 |
| 4 | 0.746 | 0.086 | 0.048 |
| 6 | 0.818 | 0.087 | 0.048 |
| 8 | 0.851 | 0.086 | 0.048 |

**The Gaussian filter is a flat plane at 0.048** — a constant fraction whatever it
sees. That is the whole reason a plain Kalman filter cannot react to a jump, and
it is visible as a geometric fact in fig16 rather than as a tuning failure.

**The mixture is a smooth ridged surface.** Absorption rises through the 2–4 SD
band as a sigmoid in magnitude and falls off continuously with reversal ratio.
The "reversal test" that discriminated jumps from outliers is now just the
orientation of the ridge — a feature of a smooth field, not a rule applied to it.
There is no magnitude and no $m$ at which anything switches.

## Where the allocation correctly refuses to commit (fig14, fig15)

Posterior-mean allocation on the square, exact expected posterior:

| truth | $m$ | $E[a]$ (channel) | $E[\varphi]$ (persistence) |
|---|---|---|---|
| PA | 20 | 0.685 | 0.247 |
| MA | 20 | 0.355 | 0.256 |
| MR | 20 | 0.278 | 0.906 |
| **PR** | 20 | **0.509** | **0.522** |

Three corners migrate to the right place. **PR sits at the centroid of the square
at every $m$** — a 3× change in $Q$ at $q$=0.05 produces essentially no
information, so the allocator declines to move. That is the correct behaviour and
it arrives without a rule enforcing it: a flat likelihood gives a flat posterior
gives a centred allocation. The failure mode of this construction is *inaction*,
which is the opposite of every gate the project has abandoned.

fig15 maps this as a landscape — the weakest Fisher eigenvalue over the square
spans 2.4 decades, with the flat region running along the low-$a$/mid-$\varphi$
band. Identifiability is terrain, not a yes/no, and the allocation should stay
spread wherever the terrain is flat.

## What this changes going forward

- **Delete the confirmation ledger from [04](04-nats-trust-influence.md) as an
  operational object.** It is a table of level-set crossings on a smooth surface,
  under oracle knowledge. It remains useful as an upper bound.
- **The state to carry is a distribution on the square**, not a mode label. Its
  two marginals are the two things a filter actually needs: $E[a]$ gates how much
  of an excursion moves the level, $E[\varphi]$ gates how much it moves $Q$ and
  $\sigma^2$. Both are continuous, both are always defined, neither needs a
  threshold.
- **$\varphi$ and $\omega$ are the same object seen at two scales.** $\varphi$ is
  the persistence of a single deviation; $\omega$ from
  [02](02-relevance-decay-and-the-tail.md) is the long-run rate at which the
  parameters wander. A prior on $\varphi$ implies one on $\omega$, which sets
  $L^*$. That is the first time in this project that the tail length, the gate,
  and the noise model have come from one quantity.

## Still open

- The allocator here is exact but built on a fixed 2-increment window with the
  event at the window edge. Sliding it, so every point is simultaneously the
  possible start of a deviation, is the $t_0$-scan problem from
  [05](05-open-questions.md) and is still uncosted.
- Everything remains Gaussian-conditional-on-$(a,\varphi,A)$. The mixture gives
  heavy tails as an emergent property, which is the right way round, but the
  tail index it produces has not been checked against the heavy-tail probe.
- The amplitude grid caps at $A=100$, so absorption saturates at 0.85 rather than
  approaching 1. A scale-invariant prior on $A$ would fix this and is the obvious
  next refinement.
