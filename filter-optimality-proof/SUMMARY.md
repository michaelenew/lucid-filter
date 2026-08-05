# Current state

An attempt to prove that the adaptive filter in
[`../adaptive-random-walk-filter/`](../adaptive-random-walk-filter/SUMMARY.md) is
optimal, and to locate precisely where the proof fails.

**One layer is proved. One is measured but not proved. One is open. The class of
processes turned out to be the hard part, and defining it correctly is the main
result so far.**

## The question, made precise

Four things have to be named before "optimal" means anything: a class of
processes, a class of procedures, a loss, and an optimality notion. Three are
forced. Procedures: all causal measurable estimators. Notion: minimax, since the
premise is that no prior over the class is available. Class: forced by a
degeneracy argument (below). The loss: **the seam is now removed — both layers
read under code length.** Layer 1 transfers to log-loss verbatim
(**Theorem A′**, [`output/03`](output/03-logloss-shape-minimaxity.md),
verified in `exploration/0035`): the Kalman code's cross-entropy depends on
the shape only through the same quadratic form as the MSE risk, so the same
three lines give the same saddle under code length, and the loss the theorems
use is the loss `fit()` optimises. The old measurements stand as the seam's
epitaph: the two criteria agreed on parameter choice to $+0.23\%\pm0.21$
(`exploration/0012`), and committing to MSE instead was never viable
(`0027`: PEM leaves variance-only directions unidentified). What `0017`/
`0023` showed about class geometry is unchanged but now single-loss: the
open wounds are marginalisation and class size, not a loss mismatch
(`exploration/0036`).

## The class

The proposed class — *all unbiased random walks whose update distribution may
change in shape and magnitude over time, nothing else given* — has no
non-trivial optimal procedure. Proposition 1 (`exploration/0001` §2): if the
noise variances can move unpredictably, then at any step the "level jumped" and
"sensor glitched" hypotheses are **identically distributed**, so the competitive
ratio against a filter that knows the variance path is unbounded for *every*
causal estimator. The class must constrain how fast the scales move, and — by
scale equivariance, since the class cannot depend on whether $x$ is in metres or
feet — the constraint must live on the log scale. Two numbers per channel:
magnitude $s_c$ and persistence $\varphi_c$. **The filter's scale parameters are
the definition of the class, not parameters within it.**

The measured class (`exploration/0005` §5):

$$\theta_t=\theta_{t-1}+w_t,\quad x_t=\theta_t+v_t,\quad w_t\sim N(0,Qe^{\lambda^P_t}),\ v_t\sim N(0,\sigma^2e^{\lambda^M_t})$$
$$\lambda^c\ \text{stationary},\quad \gamma^c_0=s_c^2,\quad \gamma^c_1=\varphi_cs_c^2,\quad
\mathbb E[e^{\lambda^c}]<\infty,\quad c\in\{P,M\}$$

**That last constraint is new and was silently assumed throughout**
(`exploration/0024` §3). Prescribing only $\gamma_0,\gamma_1$ admits a
stationary $\lambda$ with finite variance and polynomial tails, for which
$\mathbb E[e^\lambda]=\infty$ — the *actual* noise variance then has infinite
mean while every stated constraint holds. Under squared error that is harmless
(a huge $R_t$ just means ignore $x_t$), but under log-loss
$\sup_{p}\mathbb E_p[-\log m^\ast]=\infty$ and the minimax problem is vacuous.
The two losses do not even agree on whether the problem is well-posed.

Equivalently: increments are Gaussian scale mixtures, with the mixing scale
constrained in magnitude and persistence only. Every such mixture has
$\kappa\ge3$, but **the converse is false** — not every $\kappa\ge3$ law is a
scale mixture, so the class is strictly smaller than "all shapes with
$\kappa\ge3$." (An earlier version of this line stated the equivalence; it is
one-way.)

**Theorem B** (`exploration/0015` §2) makes the shape adversary exact and
elementary. An i.i.d. mixture $\varepsilon=\sqrt u\,z$, $\mathbb Eu=1$, shifts
the log-scale by $\log u$, so it adds $\operatorname{Var}(\log u)$ to $\gamma_0$
and **leaves $\gamma_1$ untouched**:

$$\tilde s^2=s^2+\operatorname{Var}(\log u),\qquad \tilde\varphi=\varphi s^2/\tilde s^2,
\qquad \tilde s^2\tilde\varphi=\gamma_1\ \text{invariant.}$$

The shape adversary moves along a level set of $\gamma_1$, with the Gaussian at
the minimal-$\gamma_0$ endpoint. This is "shape and scale are the same
coordinate, $\varphi$ is the dial" made exact. Two consequences worth carrying:
**the sufficient statistic is $\operatorname{Var}(\log u)$, not kurtosis**
($\log(\kappa/3)=\log\mathbb E[u^2]$ is a different functional, and they coincide
only for lognormal mixing); and a heavy tail always lands on the
$\rho_2>\rho_1^2$ side, relocating *within* the class but *off* the AR(1)
submanifold the filter models.

## Layer 1 — shape. **Proved, under both losses.**

(Squared error below, from `output/01`; code length in
[`output/03`](output/03-logloss-shape-minimaxity.md) by the same three lines
— both equalizers exist because both losses reach the shape only through
$\mathbb E_p[e_t^2]$.)

[`output/01-shape-minimaxity.md`](output/01-shape-minimaxity.md). Given the
variance path, the Kalman filter is *exactly* minimax over all mean-zero noise
shapes, the Gaussian is *exactly* least favourable, and the value is the Riccati
error. Three lines: the KF is linear so its risk is identical for every shape at
fixed variances (an equalizer); at the Gaussian it is the exact conditional
mean; weak duality closes the gap.

This replaces the central-limit-theorem defence of conditional Gaussianity with
something stronger — no limit is taken, no approximation is made, and it holds
finite-sample for increment laws that look nothing like Gaussian.

## Layer 2 — the log-scale dynamics. **Proved under log-loss. False under MSE.**

[`output/02-logloss-least-favourable.md`](output/02-logloss-least-favourable.md),
Theorem C (standard: max-entropy = minimax redundancy, specialised). Among *all*
processes with prescribed $\gamma_0,\gamma_1$ — not necessarily Gaussian, Markov
or mixing — the Gaussian AR(1) is an **exact equalizer** under code length,
because $-\log p^\ast$ is affine in precisely the statistics the class fixes.
Equalizer + Bayes-at-one-member + weak duality, the same three lines as
Theorem A. That is the filter's log-scale model exactly, including the
$\nu=s^2(1-\varphi^2)$ identity in `core.py`.

**Two gaps, and they are all that remains of layer 2.**

1. **The loss is wrong, and this one is now known to be fatal as stated.**
   Measured across the full admissible range of $\gamma_2$ (`exploration/0017`),
   the max-entropy member is **not** least favourable under squared error: risk
   is monotone in $\gamma_2$, in *opposite directions* in two regimes, with the
   max-entropy member interior both times and a spread up to 4.6%. Squared-error
   risk is not affine in $(\gamma_0,\gamma_1)$, so no equalizer exists and
   Theorem A's argument cannot transfer. This must become a bound on the
   discrepancy, not an equality.
2. **The path is latent, and the equalizer does not survive it.** Theorem C
   scores code length on $\lambda$; `fit()` scores likelihood of $x$. Measured
   (`exploration/0023`): across an AR(2) family on which Theorem C guarantees the
   *latent* code length is identical, the *observable* code length varies
   monotonically by 1.3%, at $|t|=3$–4.5. So gap 2 **closes negatively** — it is
   not "not established", it is false.

**Neither reframing rescues layer 2.** Under MSE the max-entropy member is not
least favourable (`0017`); under observable log-loss it is not either — members
below it in $\gamma_2$ sit above it at $t=4.5$ (`0023`). Committing to one loss
leaves the filter's model interior to the class under both.

## Layer 3 — the six numbers. Open, but the sign is favourable.

`fit()` maximises over them. Minimaxity of a Bayes rule for a fixed prior says
nothing about a rule that first estimates the prior from the same data.

**`fit()` estimates the *relocated* class parameters, and estimates them well.**
Under a shape adversary the process moves to $(\tilde s,\tilde\varphi)$ by
Theorem B, and `fit()` lands there: for $t_5$ the prediction is $(0.890,0.355)$
and the measurement $(0.907\pm0.065,\ 0.488\pm0.089)$, with the predicted point
the best of every point tested and `fit()` $+0.07\%\pm0.04$ from it
(`exploration/0013`). **An earlier claim here — that `fit()` returns a
KL-projection 25–30% away from the class parameters — was an arithmetic error and
is withdrawn** (`exploration/0015` §1).

What survives is narrower and still true. The relocated log-scale is an
ARMA(1,1) while the filter's family is AR(1), so ML genuinely is a quasi-MLE
projecting onto a family that does not contain the truth, and White's sandwich
theory is the right asymptotic tool. The projection simply costs nothing
measurable. Its signature is visible in `exploration/0022`: fitted $s_M$
overshoots Theorem B exactly on the rows whose induced log-scale marginal is
bimodal, and fits cleanest on the lognormal row where it is Gaussian.

Two measured facts constrain how `fit()` should be judged. Running at the
**true, un-relocated** $(s_M,\varphi_M)$ costs $+5.98\%\pm0.94$ ($t=6.4$). And
even on **well-specified** data the true parameters are not MSE-optimal
($+0.65\%$, $t=4.2$) — a signature of the GPB1 collapse. "Recover the true
parameters" is the wrong success criterion in both directions.

## What the measurements settled

`exploration/0003`, `0004`. The worst leak — that the filter is nonlinear, so
Theorem A does not transfer to it — is real, has exactly **one** dimension, and
closing it makes the class and the model coincide.

- Adversary leverage is zero at $s_M=0$ (where Theorem A is exact) and grows
  like $s_M^2$: spread across shapes 0.0016 → 1.23 as $s_M$ goes 0 → 2.
- Leverage is monotone in a shape functional, and **that functional is
  $\operatorname{Var}(\log u)$, not kurtosis** (`exploration/0022`). Holding
  $\kappa=9$ exactly while sweeping $\operatorname{Var}(\log u)$ over a 36x
  range moves fitted $s_M$ from 0.693 to 1.267, more than ten se, where the
  kurtosis account predicts a single value 1.184 throughout. `0004`'s "monotone
  in kurtosis alone" held only because the two functionals correlate across the
  shapes it tested — they are equal by construction for lognormal mixing.
- A Gaussian scale mixture has $\kappa\ge3$ with equality iff degenerate, so
  **the Gaussian is the least favourable shape within the family the filter's
  own model generates.**
- The reason is structural, not numerical: the class specifies the log-scale
  only through $\gamma_0,\gamma_1$, and an i.i.d. shape contributes exactly
  $\operatorname{Var}(\log u)$ to $\gamma_0$ and nothing to $\gamma_1$
  (Theorem B).
- The reason: i.i.d. scale variation *is* excess kurtosis, persistent scale
  variation *is* heteroscedasticity, and $\varphi$ is the dial between them. A
  heavy-tailed adversary is inside the model at $\varphi=0$ — which is why $t_5$
  noise makes the filter *beat* the linear path oracle. This is
  `theory/06`'s persistence axis arriving from an unrelated direction.
- **Directly confirmed** (`exploration/0007`): feeding the same variance path
  through a $t_5$ shape moves the fitted $s_M$ from 0.55 to 0.907 and the fitted
  $\varphi_M$ from 0.93 to 0.488. The filter reads a heavy tail as *impulsive*
  scale variation, and follows the relocation. Fitted $s_M$ is monotone in
  kurtosis across all four shapes: 0.224, 0.389, 0.515, 0.907.
- **And `fit()` lands on the predicted relocation.** With the correct statistic
  $\operatorname{Var}(\log u)=\psi'(5/2)=0.4904$ for $t_5$, Theorem B predicts
  $(0.890,0.355)$ and `fit()` measured $(0.907\pm0.065,\ 0.488\pm0.089)$ — 0.26
  and 1.5 se. Paired over 40 seeds the corrected point is the best of every
  point tested, with `fit()` $+0.07\%\pm0.04$ from it, against $+1.53\%$ for the
  old miscalculated point and $+5.60\%$ for the un-relocated truth
  (`exploration/0013`). **An earlier claim here that ML lands 25–30% short and
  targets a KL-projection was an arithmetic error** ($\log(\kappa/3)$ for
  $\operatorname{Var}(\log u)$), now withdrawn. The residual true statement: the
  relocated log-scale is ARMA(1,1) while the filter's family is AR(1), so ML does
  project — it just costs nothing measurable.
- **Limitation, sharp:** $\kappa<3$ is outside the representable cone. At
  $s_M=1.5$ the filter loses 24% against its Gaussian row on two-point noise.
  Bounded sensors, quantised readings and saturating instruments are all
  light-tailed. The filter **under-reads** the scale there (fitted $s_M$ 0.224
  and 0.389 against a true 0.55) rather than missing it — $\gamma_1$ stays
  observable even when $\gamma_0$ is suppressed.

## A cousin family, unexpected

`exploration/0002` probe B, 40M samples per cell. The three-way amplitude
conservation law of `theory/07` — $\mathbb E[a\mid a+b+c]=(V_a/V_e)e$, weights
summing to 1 — is **not** Gaussian-specific. It holds across the symmetric
$\alpha$-stable family with the dispersion $c^\alpha$ in the role of the
variance, verified linear (constant across quantile bins) and at the predicted
level for $\alpha=2.0,1.8,1.5,1.2$. So there are two independent one-parameter
deformations of the filter, with it at the boundary of both:

| axis | deformation | filter recovered at |
|---|---|---|
| $p$ (Burg order) | how much temporal structure the log-scale asserts | $p=1$ |
| $\alpha$ (stability index) | tail index of the increments | $\alpha=2$ |

For $\alpha<2$ the variance is infinite, so squared error is the wrong loss and
layer 1 would need redoing under $\mathbb E\lvert\cdot\rvert^r$, $r<\alpha$.

## The gap ledger

| gap | status |
|---|---|
| Leak 1 — shape adversary | **closed as a separate leak.** Theorem B: an exact relocation along a $\gamma_1$ level set, and `fit()` finds it to $+0.07\%$ |
| Leak 2 — two losses | **closed as a loss seam** (Theorem A′, `output/03`): both layers now read under code length, verified `0035`. What its probes found about class geometry (`0017`/`0023`) survives single-loss, folded into the marginalisation row |
| Leak 3 — parameters estimated | quasi-MLE under misspecification; measured sign favourable, no theory |
| Leak 4 — GPB1 collapse | **directly measured against exact PF** (`exploration/0034`): $0.006\%$ at $s{=}0.2$; $1.35\%$ at $s{=}0.55$ persistent; **$19.9\%$ at $s{=}1.2$ persistent**. No quadrature order closes it |
| marginalisation ($\lambda\to x$) | untouched; blocks Theorem C from applying to `fit()` |
| $\kappa<3$ (light tails) | genuine, outside the model, unchanged |

**The obstruction is not the choice of loss — it is that the class is too big**
(`exploration/0024`). Fixing two moments of $\lambda$ leaves the risk varying
across members under *either* loss, with the filter's model interior rather than
at a maximum. An equalizer exists only when the loss is affine in what the class
fixes: true for Theorem A, true for Theorem C on the latent path, false in every
other combination tried. Reaching minimaxity would mean shrinking the class to
the AR(1) family, which makes the theorem nearly tautological.

**The filter is fine; the ambition was wrong.** Every measured gap is between
0.2% and 5%. What fails is an exact minimax characterisation over a two-moment
class, not the filter.

**Committing to MSE in practice does not work via prediction error**
(`exploration/0027`). PEM looked uniformly at-least-as-good on the
$(s_M,\varphi_M)$ slice, but with all six parameters free it recovers $\sigma^2$
wrong by up to a factor of nine and inflates $s_M$ to 1.54 on data with no scale
variation at all. The squared innovation depends on the parameters only through
the predicted mean, so any direction moving the predictive variance without
moving the gain is nearly free under it — and $\sigma^2$ and $s_M$ inflating
together is exactly such a direction. `fit()`'s default stays on
log-likelihood; PEM is exposed as an option with the caveat documented.

## Rate of approach — how close is the filter to Bayes-for-its-own-model?

The regret against the model's own Bayes rule decomposes:

$$ R_{\text{filter}} = \underbrace{R_{\text{quad}}(n)}_{\text{Gauss-Hermite order}}
   \;+\; \underbrace{R_{\text{collapse}}}_{\text{GPB1 collapse}} $$

**The quadrature term is geometric in the order** (`exploration/0029`). Fixing
parameters at truth and comparing GH orders 3–31 gives successive θ-MSE excess
ratios of $3.5, 4.2, 5.1$ per +2 orders in the strong regime — the standard
$O(\rho^{-2n})$ rate for analytic integrands. The default of order 5 loses
$6.4\%$ θ-MSE and $4.3\times10^{-2}$ nat/pt against order 31 at $s{=}1.2$;
essentially nothing at $s\le 0.55$.

**The collapse term is nonlinear in $s$ and is structural** (`exploration/0034`).
Measured directly against a Rao–Blackwellized particle filter (exact for the
model at large $N$; validated against plain-Kalman to machine precision), the
GH-31 filter loses **$19.9\%$ θ-MSE at $s{=}1.2$ persistent**, $1.35\%$ at
$s{=}0.55$ persistent, and $0.006\%$ at $s{=}0.2$. Order 5, 9 and 31 all sit
within a percent of each other in the strong regime while all sitting ~17%
above the PF reference — no quadrature order closes the collapse gap.

**Score bias translates to parameter bias, not to estimate bias**
(`exploration/0033`). Fitting the strong regime at orders 5, 9, 13 shifts
$\varphi_M$ from 0.845 to 0.890 to 0.906 (truth 0.930; t-statistics $-14.3, -6.8,
-3.4$). But θ-MSE across orders is flat within noise (0.365, 0.354, 0.371), the
same "parameters move, estimate does not" signature as PEM in `0027`. Bump the
order to 9 when the fitted parameters themselves are what a caller reads;
default order 5 is honest for tracking.

**Implication for the regret target.** At $s\le 0.55$ the filter is within
$\sim 1\%$ θ-MSE of Bayes at the default. At $s{=}1.2$ it is about 25% off,
almost entirely from the collapse; the natural next filter class is GPB2 or a
particle filter, not more quadrature.

## Next, in order

1. **Bound the loss discrepancy at layer 2.** Theorem C is exact under log-loss;
   `0017` measures up to 4.6% divergence under MSE across the admissible
   $\gamma_2$ range. The equality is unavailable, so the target is an inequality:
   how far from minimax can modelling the max-entropy member be? A bound in
   terms of $\gamma_2$'s admissible width would finish layer 2 in the only form
   still open to it.
2. **A GPB2 filter or a compute-budget curve for the PF.** `0034` shows the
   collapse leaves ~20% θ-MSE unclosed at $s{=}1.2$, and no quadrature order
   fixes it. GPB2 (one Gaussian per grid state, no collapse across states) or a
   modest particle count would close most of that gap and give the "secondary
   mode for exactness" the four-point message floated a concrete form.
3. **Rate of approach in $s$.** `0034` has four points ($s{=}0, 0.2, 0.55,
   1.2$) on the collapse cost, growing faster than $s^2$. A denser sweep and a
   fit would say what functional of $s$ (and $\varphi$) actually governs it —
   candidate ansatz: something like $\varphi \cdot (e^{s^2}-1)$.
4. **Push Theorem C through the marginalisation** to the observable
   (`exploration/0005` §6). Would give a genuine minimax statement for `fit()`
   under log-loss — worth having even with gap 1 open.
5. **The I-MMSE weld** (Guo–Shamai–Verdú): mutual information is the integral of
   MMSE over SNR. The only route that would RELATE the two losses rather than
   choosing between them.
6. **The $\alpha$-stable family** under $L^r$ loss.

## Notes for the parent workstream

Not applied there, since it is that workstream's deliverable.

**A crash bug in `fit()`, worth fixing regardless of anything here.**
`_expit(z) = 1/(1+math.exp(-z))` in `statfilter/core.py` raises `OverflowError`
once $z<-709$, and `fit()`'s inner `ll()` catches only `ValueError`, so the
exception escapes and kills the fit. `_logit` clamps to $|{\rm logit}|\le20.7$ so
starts are safe, but stage 1 is an unconstrained Nelder–Mead search and on
sufficiently impulsive data it walks out there — reproduced in
`exploration/0018`, diagnosed and worked around in `0021`. The fix is two lines
and algebraically identical:

```python
def _expit(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)
```

plus widening `ll()`'s `except ValueError` to `except (ValueError, OverflowError)`.
The failure mode is a hard crash rather than a degraded estimate, so any user
fitting impulsive data can hit it.

**Quadrature order — now measured across three axes.**

* *Filter accuracy at truth* (`exploration/0029`): geometric convergence in the
  order. Default order 5 loses $6.4\%$ θ-MSE at $s{=}1.2$; negligible below.
* *Fit accuracy* (`exploration/0033`): the order-5 SCORE is biased enough to
  shift the argmax. Fitted $\varphi_M$ moves 0.845 → 0.890 → 0.906 as order
  goes 5 → 9 → 13 (truth 0.930) with t $=-14.3, -6.8, -3.4$. Tracking θ-MSE
  is flat across orders — the parameters move but the estimate does not.
* *Loglik direction* (`exploration/0029`): low-order GH reports systematically
  OPTIMISTIC log-likelihood in impulsive regimes (order 3 higher than order 31
  by $2.6\times10^{-3}$ nat/pt at $s{=}0.55$ impulsive). The score can lie in
  the direction that matters for fitting.

Recommendation: bump to order 9 when the fitted volatility parameters are the
deliverable; default order 5 is honest when tracking is.

## Layout

- `output/` — results that stand on their own. Theorem A (shape minimaxity,
  squared error), Theorem C (log-scale minimaxity, log-loss), and Theorem A′
  (shape minimaxity under code length — the seam's removal).
- `exploration/` — numbered, later is more recent. Start at `0015` (Theorem B
  and the correction to the moment formula), `0017` (why max-entropy does not
  transfer to squared error), and `0036` (the seam removed, and the bet it
  settled); those carry the current argument. `0005` is
  still the best statement of the class itself, with its §3 superseded by
  `0015` §1. `0009`'s headline is withdrawn — see `0015` §1.
