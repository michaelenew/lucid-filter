# 0001 — Resolvability as the founding axiom, in place of stationarity

> **AI-generated, not peer-reviewed.**
>
> ⚠️ **SUPERSEDED — the axiom proposed here is refuted.** C2 is false
> ([`0002`](0002_the_ridge.md)), C5 is false ([`0005`](0005_the_seam.md)), C3's
> premise is false ([`0003`](0003_what_stationarity_costs.md)), and R2 itself is
> incompatible with the filter's purpose because a regime change is a step, hence
> sub-interval at every sampling rate. See [`../SUMMARY.md`](../SUMMARY.md) for
> what survived. This file is kept as the record of the reasoning, including the
> mistake: it confused what one STEP sees (the increment) with what the RECORD
> sees (the stationary variance), and built on the wrong one.

## 1. The swap

**What the filter assumes today.** The per-component log-scales are a
*stationary* AR(1) family, `(phi, s)` per channel, and because those two numbers
are the class rather than parameters within it, the filter runs a bank across a
broad `(phi, s)` box (`_PHIS`, `_SS`) and lets the evidence average it out.
`forget < 1` sits outside that theory: its own parameter doc names it "the
engineering escape for the stationarity assumption ITSELF being violated".

**What this thread proposes instead.** One assumption, in two parts:

> **(R1) Refinement covariance.** There is a single continuous-time latent
> object — state, log-scales, fault kernel. The model at sampling spacing `Δ` is
> its exact `Δ`-marginal, and inference commutes with decimation: what we infer
> from every other sample is the `2Δ`-marginal of what we infer from all of them.
>
> **(R2) Resolution closure.** No latent *regime* structure exists below `Δ`.
> Variation faster than the record can see is not regime; it is shape at fixed
> scale.

Plainly: *the data we receive is meaningful.* We would not get a different answer
at a different sampling rate — so the answer is a property of the world, not of
the schedule we happened to sample it on.

R1 is a statement about the generator. R2 is a statement about where the
generator's bandwidth sits relative to ours. Neither one mentions stationarity.

## 2. What is actually being dropped

Stationarity is a claim about the *marginal law* of `lambda`. Time-homogeneity is
a claim about its *generator*. The old axiom asserted both and used the first;
R1 asserts only the second.

By Doob's theorem (1942), the only continuous-path, time-homogeneous Gaussian
Markov process is the linear SDE

```
d lambda = -kappa (lambda - mu) dt + sigma dW
```

whose `Δ`-sampling is exactly AR(1) with `phi = e^{-kappa Δ}` and per-step
increment variance `sigma^2 (1 - e^{-2 kappa Δ}) / (2 kappa)`.

**So AR(1) is not dropped — it is derived.** What is dropped is the requirement
`kappa > 0` with the stationary initial marginal. The admitted new member is
`kappa = 0`: a Brownian log-scale, no mean reversion, the stationary variance
`s^2 = sigma^2 / (2 kappa)` divergent while `sigma^2` stays finite.

That divergence is the point. **The old coordinates `(phi, s)` blow up exactly at
the boundary the box most needs to reach**, which is why `_SS` has to climb to
3.20 and why the reach argument in the audit is the only part of that constant
with a derivation attached. In `(kappa, sigma^2)` the boundary is an interior
point of the parameterisation: `kappa = 0`, `sigma^2` finite.

Note also what the shipped filter already advertises: scale walks with
**unbounded reach**. That is `kappa ≈ 0` behaviour. The suspicion worth carrying
through this thread is that the new axiom describes the product *better than the
old one does*.

## 3. Consequences, ordered by confidence

### C1 — the clock rules are the unique semigroup solutions. *(high)*

R1 says the `a`-step transition is a one-parameter semigroup in `a`, which forces
exponentials and nothing else:

| object | rule | status in the repo |
|---|---|---|
| dynamics | `F(a) = exp(a log F)` | shipped, exact |
| log-scale persistence | `phi(a) = phi^a` | shipped, exact |
| fault kernel | `M^a` (exact chain power) | shipped, exact |
| weight memory | `forget^a` | shipped |
| process noise | `Q(a) = ∫_0^a e^{As} Q_c e^{A^T s} ds` | **not shipped** — `Q(a) = Q·a` |

The repo derived the first four one at a time, each from "carry the class
timescale to the gap". R1 gives all five from one condition — and the fifth
fails. `Q(a) = Q·a` is exact for the random-walk default and first order in
`‖A‖a` otherwise (45% of `Q` misfit measured at `‖A‖a ≈ 1.2`,
`pointwise-streaming`). **Under the new axiom that is not an accuracy issue, it
is an axiom violation**: a filter whose `Q` does not compose correctly under
decimation gives a different answer at a different rate, by construction. The
open item "let `process=` be declared as a continuous spectral density" is
promoted from a refinement to a prerequisite.

### C2 — the sloppy ridge should be the `kappa` direction. *(medium, testable)*

`adaptive-grid` findings 13–16 measured `(phi, s)` to be identified but **sloppy**
— a flat identification ridge that the bank marginalises. In `(kappa, sigma^2)`
there is an obvious candidate for that ridge: per step the likelihood sees the
*increment* variance `sigma^2 Δ`; `kappa` enters only through mean reversion
accumulated over many steps.

**Prediction.** Re-expressed in `(kappa, sigma^2)`, the stiff direction is
`sigma^2` and the sloppy one is `kappa`. If that holds, the `phi` axis of the box
is not averaging over a class dimension — it is averaging over a direction the
data cannot see, and the *least-committal* member on it is `kappa = 0` (C3).
**The bank drops from 15 members to ~5, by argument rather than by measurement.**

**Test.** Fisher/sloppiness analysis of the profile likelihood in both
coordinate systems on the scalar rig; the ridge direction is an eigenvector, so
this is a direct computation, not a sweep.

### C3 — Theorem C re-runs on a one-moment class. *(medium, needs the proof)*

`optimality-proof`'s own diagnosis of why layer 2 cannot be closed: *"the
obstruction is not the choice of loss — it is that the class is too big."*
Prescribing `gamma_0, gamma_1` leaves risk varying across members under either
loss, with the filter's model interior rather than at a maximum. And the stated
repair — "shrinking the class to the AR(1) family" — was rejected as
"nearly tautological", because nothing *forced* the shrink.

R2 forces it. Under R2 the class is the log-scale *increments* with one
prescribed second moment per unit physical time, and nothing else. Max entropy
subject to one second moment per increment is independent Gaussian increments —
a Gaussian random walk, `kappa = 0`, flat initial. Code length is affine in the
constrained statistic, so Theorem C's three lines (equalizer, Bayes at one
member, weak duality) run verbatim on the new class.

**Two predictions follow, and they are the reason this thread is worth running.**

1. `0017`'s result — the max-entropy member is not least favourable under MSE,
   with risk monotone in `gamma_2` in opposite directions in two regimes — was a
   statement about a *third* autocovariance the class left free. R2 does not
   leave `gamma_2` free; the semigroup fixes the whole autocovariance sequence
   from one number. So gap 1 does not get bounded, it gets **dissolved**.
2. `0023`'s result — the latent equalizer does not survive marginalisation to
   the observable — was measured across an AR(2) family. R2 excludes AR(2) log-
   scales as a matter of class, not of approximation.

Both predictions are about whether the *premises* of those two negative results
survive the swap. Neither is a claim that the measurements were wrong.

**Honest caveat.** Shrinking a class to rescue a minimax theorem is exactly the
move `optimality-proof` called tautological. The defence here is that the shrink
is *forced by an independently motivated axiom about sampling*, and that the
axiom is **checkable on the record** (C6). Whether that defence holds is the
thing this thread has to establish, and it is the single place it is most likely
to fail.

### C4 — the shape adversary is reassigned, not bounded. *(medium)*

Theorem B: an i.i.d. scale mixture `eps = sqrt(u) z` adds `Var(log u)` to
`gamma_0` and leaves `gamma_1` untouched — i.e. it injects a **white** component
into the log-scale, one with zero correlation time at *every* sampling rate.

Under R2 that component is not in the class. It has not left the world; it has
been **reassigned to layer 1**, where Theorem A is exact: at fixed variance the
Kalman filter is minimax over all mean-zero shapes and the Gaussian is least
favourable.

> **R2 is the cut between the two layers, and it puts the cut at the sampling
> rate.** Slower than `Δ` is regime, handled by the walk. Faster than `Δ` is
> shape, handled by Theorem A. The old framing let shape leak into layer 2 as a
> relocation along a `gamma_1` level set; the new one forbids the leak by
> construction.

**Prediction.** A `t_5` sensor under the new class is read as Gaussian with an
inflated `R` — the KL projection — and Theorem A says that projection is minimax.
**Test:** re-run `optimality-proof/0013` and `0022` (which already measured where
`fit()` lands under a `t_5` shape, and which shape functional drives it) against
a `kappa = 0`, `sigma^2`-only class.

**Caveat, sharp.** `0003`/`0004` measured adversary leverage on the *nonlinear*
filter growing like `s_M^2` (spread 0.0016 → 1.23 as `s_M` goes 0 → 2). "Free" is
a minimax statement about the linear layer; the filter is not the linear layer.
The cost of the reassignment has to be measured, and **kurtosis < 3**
(light-tailed sensors — bounded, quantised, saturating) remains outside the
representable cone either way.

> *Symbols.* `kappa` here is the OU mean-reversion rate, not the kurtosis that
> `optimality-proof` writes as `kappa`. The collision is unfortunate and this
> workstream keeps `kappa` for the rate throughout.

### C5 — the box endpoints become derived, closing AUD-2. *(medium-high, cheap to test)*

The Fisher information a single scalar innovation carries about a log-scale is
elementary. For `y ~ N(0, v)` with `v = e^lambda`, `∂_lambda log p = -1/2 +
(y^2/v)/2`, so `I(lambda) = 1/2` — a one-event blur width of `sqrt(2)` nats. When
the channel carries only a fraction `f` of the innovation variance
(`f = (∂S/∂lambda)/S`), the information is `f^2/2` and the blur is `sqrt(2)/f`.

That single number sets both ends of the box:

- **ceiling** — the per-step increment SD must satisfy `sigma sqrt(Δ) <= sqrt(2)/f`.
  Above it the log-scale moves further between observations than one observation
  can locate it, which is R2 failing *on the record itself*.
- **floor** — over an effective memory of `N` steps, `N sigma^2 Δ >= 2/f^2`, or the
  record contains no evidence that anything changed at all.
- **spacing** — the Sparrow rule on that blur width, which is the repo's own
  `_GAP_FACTOR` (still a proxy; AUD-1 is untouched by this thread).
- **rung count** — `≈ log N / gap`.

At the shipped `forget = 0.999` (`N = 1000`) and `f = 1`: admissible per-step
increment SD in `[0.0447, 1.4142]` — a dynamic range of 31.6 — with
`log(1000)/1.5 = 4.61` → **5 rungs, geometric, ratio `e^{1.5} = 4.48` in
`sigma^2`.**

The shipped `_SS = (0.20, 0.40, 0.80, 1.60, 3.20)` is **5 rungs, geometric, ratio
4.0 in `s^2`** (log-ratio 1.386 against the derived 1.5). The derivation retrodicts the convention rather than overturning it,
which is the outcome most worth having: the numbers stay, the `AUDIT[measured]`
grade does not.

**The one place they disagree, and it is a real prediction.** `s` is the
*stationary* SD; the per-step increment SD is `s sqrt(1 - phi^2)`. Across the
shipped 15-member box that runs 0.062 → 2.285, so **2 of 15 members sit above the
`sqrt(2)` resolution ceiling**: `s = 3.20` at `phi = 0.70` (2.285) and at
`phi = 0.85` (1.686).  (Arithmetic and a numerical confirmation of
`I(lambda) = 1/2` — 0.49935 on 2e6 samples — in `0001_box_ends.py`.)

> **Prediction:** those members are inert or harmful, and pruning them costs
> nothing measurable. **Test:** re-run the scalar hero gate and the arm rig with
> them removed; the acceptance numbers should not move.

### C6 — resolvability is checkable on the record, with no oracle. *(high)*

The obvious objection to R2 is that it is an assumption about the world and the
world does not have to comply. The answer is that, unlike stationarity, **it is
falsifiable from the record alone.**

`wall-correspondence/0036` already built the instrument and did not label it this
way. Its criterion: *the continuum limit exists iff the prequential code length
per unit physical time converges to a **nontrivial** limit under refinement* —
and its measured lesson is that the failure mode is **triviality, not blow-up**
(both refinements converged; they converged to different limits, 0.888
nat/observation apart, the bad one to white noise).

That is exactly the failure mode of "the data is not meaningful". Sample a regime
too slowly and the inferred log-scale process does not diverge — it goes white,
and the code length per unit physical time drops to the trivial branch.

**So the test of R2 is a decimation check the filter can run on itself:** filter
the record at `Δ` and at `2Δ`, compare the inferred `lambda` paths on the common
grid and the prequential code length per unit *physical* time. Agreement means
the record resolves the regime. A gap means it does not, and names the channel.

**This is the thread's most concrete product.** It gives the README's standing
open — *"attribution degrades with relative degree"*, where a jerk disturbance
reaching a rate gyro through `dt^2/2` is blamed on the sensors at 103× the oracle
— a home and a diagnosis: that rig **violates R2 on that channel**, and a filter
that can say so is better than one that silently mis-attributes. Per-channel `f`
is exactly the relative-degree factor.

### C7 — `forget` loses its stated job, and may gain a derived one. *(open)*

Its doc defines it as the escape for stationarity being violated. Remove
stationarity and there is nothing to escape. Two candidate fates:

1. **It disappears.** The bank's weights stop needing to be un-frozen because the
   class no longer asserts a marginal the world can leave.
2. **It survives with a derived meaning:** the effective record length `N` that
   sets the box floor in C5.

Fate 2 creates the sharpest tension this reframing produces, and it must be
stated rather than smoothed over. The current design rule is absolute: *"Nothing
structural may read it: no floor, box end, or class bound derives from `forget`,
and every construction in this filter must remain valid at `forget = 1`."* C5's
floor reads exactly that number. At `forget = 1` the floor becomes
`2/(f^2 N_record)` — well defined, but it means **the box grows as the record
lengthens**, which is new behaviour and has to be shown harmless (it should be:
the new rungs are slower hypotheses, which a longer record genuinely can support).

## 4. What this costs

Stated plainly, because it is not free.

- **R2 is strictly stronger than stationarity in one direction.** The old
  `(phi, s)` box hedged against fast scale variation by including members at
  `phi = 0.70`; R2 forbids sub-`Δ` regime structure outright. A filter that
  assumes resolvability and is wrong is **confidently** wrong, where the old one
  was vaguely right. C6 is the mitigation, not the refutation.
- **The minimax repair may be circular.** See C3's caveat. Shrinking a class to
  make a theorem true is the move the parent workstream already rejected once.
- **`kappa = 0` has no restoring force**, so in a long quiet stretch the posterior
  on `lambda` widens without bound. Candidate resolution worth testing: the
  caller-supplied `process=` / `measurement=` bases stop being rough starting
  guesses and become the **normalisation that makes the problem well posed** —
  which is what `E[e^lambda] < ∞` (the constraint `0024` found silently assumed)
  needs, and which fits the give-what-you-know philosophy exactly. Unverified.
- **AUD-1 is untouched.** Sparrow spacing and the `_SPAN_S = 3.0` support
  convention stay proxies. `_SPAN_S` arguably gets *worse* before it gets better:
  "±3σ of the class prior" presumes the class prior has a σ, and at `kappa = 0`
  it does not — the span has to come from the resolution instead.

## 5. Next, in order

1. **C2's coordinate check** — the cheapest decisive step. Recompute the
   identification ridge in `(kappa, sigma^2)` on the scalar rig. If the ridge is
   the `kappa` axis, the `phi` box is removable and everything downstream gets
   simpler.
2. **C5's pruning test** — remove the 2 over-ceiling members, re-run the hero
   gate and the arm rig. Cheap, and it is the first real number this thread can
   produce.
3. **C6's decimation diagnostic** — implement the self-check on the scalar rig
   and on the rate-gyro rig from `multivariate-statfilter/0054`, where R2 is
   expected to *fail* and the diagnostic should say so.
4. **C3's proof** — Theorem C on the one-moment increment class. Only worth the
   effort once 1 and 2 have said the reframing survives contact.
5. **C1's `Q(a)`** — the continuous spectral density. Independently wanted; the
   axiom makes it mandatory.
