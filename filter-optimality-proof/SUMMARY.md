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
degeneracy argument (below). The loss is not forced — the layer-1 argument is a
squared-error argument and the layer-2 argument is a log-loss argument, and they
are not yet one theorem. **Measured, that seam is smaller than it looks: where
`fit()` lands is $+0.23\%\pm0.21$ from the MSE-optimal parameters, $t=1.1$**
(`exploration/0012`). Real in principle, below a quarter of a percent in
practice.

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
$$\lambda^c\ \text{stationary},\quad \gamma^c_0=s_c^2,\quad \gamma^c_1=\varphi_cs_c^2,\quad c\in\{P,M\}$$

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

## Layer 1 — shape. **Proved.**

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
2. **The path is latent.** Theorem C scores code length on $\lambda$; `fit()`
   scores likelihood of $x$, with $\lambda$ integrated out, and the affineness
   that gives the equalizer does not survive the marginalisation
   (`exploration/0005` §6). Untouched.

The seam between the two losses, measured at $+0.23\%\pm0.21$ for *which
parameters to use* (`exploration/0012`), is therefore **not uniformly small**:
for *which member is worst* the two losses disagree outright. The honest
statement is that they agree where measured on parameter choice and diverge on
class geometry.

## Layer 3 — the six numbers. Open, but the sign is favourable.

`fit()` maximises over them. Minimaxity of a Bayes rule for a fixed prior says
nothing about a rule that first estimates the prior from the same data.

Sharpened by measurement (`exploration/0009`, `0012`): **`fit()` does not
estimate the process's $(s,\varphi)$ at all.** When the truth is outside the
family — e.g. $t_5$ noise, whose log-scale is skewed rather than Gaussian — ML
returns the parameters of the KL-nearest *representable* model. That is ordinary
quasi-MLE under misspecification, so White's sandwich theory is the right tool
rather than standard ML asymptotics.

It is also the right thing to do. Running the filter at the **true**
$(s_M,\varphi_M)$ costs $+5.98\%\pm0.94$ MSE against the point `fit()` finds
($t=6.4$): the truth describes a model the filter cannot run, the KL-projection
describes the best one it can. And even on **well-specified** data the true
parameters are not MSE-optimal ($+0.65\%$, $t=4.2$) — a measured signature of the
GPB1 collapse. "Recover the true parameters" is the wrong way to judge `fit()`.

## What the measurements settled

`exploration/0003`, `0004`. The worst leak — that the filter is nonlinear, so
Theorem A does not transfer to it — is real, has exactly **one** dimension, and
closing it makes the class and the model coincide.

- Adversary leverage is zero at $s_M=0$ (where Theorem A is exact) and grows
  like $s_M^2$: spread across shapes 0.0016 → 1.23 as $s_M$ goes 0 → 2.
- Leverage is monotone in **kurtosis alone**, over $\kappa\in[1,15]$. Two
  structurally unrelated shapes matched at $\kappa=5$ agree to 0.5 se.
  **In tension with Theorem B**, which says the sufficient statistic is
  $\operatorname{Var}(\log u)$, and two laws can match in $\kappa$ while
  differing in it. Most likely the shapes tested happened to be close in both
  (the functionals correlate across the usual families), but that is a guess.
  Read this bullet as "monotone in a shape functional that kurtosis tracks in
  the cases tested" until the discriminating test below is run.
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
| Leak 2 — two losses | **the live one.** Agree on parameter choice ($+0.23\%\pm0.21$); **disagree outright on class geometry** — max-entropy is least favourable under log-loss, not under MSE |
| Leak 3 — parameters estimated | quasi-MLE under misspecification; measured sign favourable, no theory |
| Leak 4 — GPB1 collapse | measured signature: $+0.65\%$ even when correctly specified |
| marginalisation ($\lambda\to x$) | untouched; blocks Theorem C from applying to `fit()` |
| $\kappa<3$ (light tails) | genuine, outside the model, unchanged |

The shape of the problem has changed. Leak 1 is gone and Leak 2 is no longer a
formality — it is the reason layer 2 cannot be finished, and `exploration/0017`
shows it is not a small discrepancy that a sharper argument will absorb.

## Next, in order

1. **Bound the loss discrepancy at layer 2.** Theorem C is exact under log-loss;
   `0017` measures up to 4.6% divergence under MSE across the admissible
   $\gamma_2$ range. The equality is unavailable, so the target is an inequality:
   how far from minimax can modelling the max-entropy member be? A bound in
   terms of $\gamma_2$'s admissible width would finish layer 2 in the only form
   still open to it.
2. **The $\kappa$ vs $\operatorname{Var}(\log u)$ discriminating test.**
   Construct two scale mixtures matched in kurtosis and deliberately split in
   $\operatorname{Var}(\log u)$. Decides whether `exploration/0004`'s
   kurtosis-sufficiency claim needs restating. Cheap, and it settles a live
   inconsistency in this file.
3. **Push Theorem C through the marginalisation** to the observable
   (`exploration/0005` §6). Would give a genuine minimax statement for `fit()`
   under log-loss — worth having even with gap 1 open.
4. **The I-MMSE weld** (Guo–Shamai–Verdú): mutual information is the integral of
   MMSE over SNR. This is now the most interesting item rather than a tidy-up,
   because it is the only route that would relate the two losses rather than
   choosing between them.
5. **The $\alpha$-stable family** under $L^r$ loss.

## A note for the parent workstream

Not applied there, since it is that workstream's deliverable. On *well-specified*
data, `fit()`'s $\varphi_M$ is biased low at the default quadrature order and
converges as the grid refines — 0.847, 0.873, 0.889, 0.904 at orders 5, 7, 9, 13
against a true 0.930 — while $s_M$ is flat and correct throughout
(`exploration/0009`). `theory/07` §D verified order 5 against 9 and 15 for
$s_P$; this is the same check for $\varphi_M$, and unlike $s_P$ it trends. If a
fitted $\varphi_M$ is to be read as a number rather than a direction, fit at
order 9 or 13. Tracking MSE is unaffected.

## Layout

- `output/` — results that stand on their own. Theorem A (shape minimaxity,
  squared error) and Theorem C (log-scale minimaxity, log-loss).
- `exploration/` — numbered, later is more recent. Start at `0015` (Theorem B
  and the correction to the moment formula) and `0017` (why max-entropy does not
  transfer to squared error); those two carry the current argument. `0005` is
  still the best statement of the class itself, with its §3 superseded by
  `0015` §1. `0009`'s headline is withdrawn — see `0015` §1.
