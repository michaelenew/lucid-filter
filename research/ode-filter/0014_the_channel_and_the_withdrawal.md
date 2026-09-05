# 0014 — The dynamics channel's two ends, and why the invariance argument fails

Two probes: [`0012`](0012_persistence_of_the_dynamics.py) builds the trust/belief
split for the dynamics, [`0013`](0013_minimax_over_directions.py) replaces
[`0011`](0011_the_drift_shape.md) §3's worst-case-over-three-scenarios with a
worst case over directions — and **withdraws its conclusion.**

---

## 1. The impulsive end of the dynamics channel is multiplicative noise

> **⚖️ ATTRIBUTION —** _Writing $\alpha_t=\bar\alpha+\delta_t$ so a white $\delta$ gives signal-proportional (multiplicative) noise and a persistent $\delta$ gives a coefficient regime change is the standard random-coefficient / time-varying-parameter model, with persistence as an AR(1) hyperparameter._ Prior art: random-coefficient autoregression (Nicholls & Quinn 1982); time-varying-parameter models (Harvey). That white $\delta$ moves only the predictive variance, not the mean, is a direct conditional-expectation calculation. Status: RECOMBINATION.

Worth stating before measuring, because it is the point. Write
$\alpha_t = \bar\alpha + \delta_t$. Then

$$x_t = \bar\alpha^{\top}z_{t-1} + \big(\underbrace{\delta_t^{\top}z_{t-1}}_{\text{scales with the state}} + w_t\big)$$

- $\varphi_A \to 0$: $\delta$ is white, and the extra noise has variance
  $z^{\top}\Sigma_\delta z$ — **process noise proportional to signal power.
  Multiplicative, not additive.** This is [`0004`](0004_dynamics_uncertainty_is_process_noise.py)'s
  identity read as a generative model rather than as a propagation rule.
- $\varphi_A \to 1$: $\delta$ persists — **the ODE coefficients changed.**

One coordinate, two named ends: the parent's structure exactly, one level up.
And the impulsive end is something the parent's model *cannot* express — its
log-scale modulates the variance by an exogenous process, never in proportion to
the signal. The channel carries the same centre / magnitude / persistence triple
as each of the parent's noise channels: $(\bar\alpha, s_A, \varphi_A)$.

**$\varphi_A$ identifies.** Generating from each end at $p=1$ with
$\bar\alpha = 0.80$, $s_A = 0.12$, $n=1200$, 12 seeds, fitting all three numbers
by marginal likelihood (fig09):

| generated | $\hat{\bar\alpha}$ | $\hat s_A$ | $\hat\varphi_A$ (mean) | (median) |
|---|---|---|---|---|
| static ($s_A=0$) | $0.805\pm0.005$ | $0.047\pm0.014$ | $0.363\pm0.097$ | 0.363 |
| impulsive ($\varphi_A=0$) | $0.842\pm0.023$ | $0.119\pm0.026$ | $0.306\pm0.099$ | **0.189** |
| persistent ($\varphi_A=0.99$) | $0.870\pm0.032$ | $0.147\pm0.015$ | $\mathbf{0.972\pm0.010}$ | 0.989 |

The persistent end is recovered almost exactly and is cleanly separated from the
impulsive one. $s_A$ is recovered to $0.119$ against a truth of $0.12$ at the
impulsive end. On static data $\hat s_A$ is $0.047$ — small but $3.4$ se from
zero, and $\hat\varphi_A$ is meaningless there. **That is the parent's own
caveat reappearing verbatim** ("$\varphi$ is only meaningful where the
corresponding $s$ is above zero"), which is mild evidence the structure really
is the same object.

**But the dial does not move the point forecast, and provably cannot.** With
$\delta_t$ white and independent of $z_{t-1}$,

$$\mathbb E[x_t\mid z_{t-1}] = \bar\alpha^{\top}z_{t-1},\qquad
\mathrm{Var}(x_t\mid z_{t-1}) = z_{t-1}^{\top}\Sigma_\delta z_{t-1} + Q$$

so at $\varphi_A = 0$ the channel contributes to the predictive **variance** and
nothing at all to the predictive **mean**. Measured, the forecast MSE ratios
against a pure random walk in $\alpha$ are $0.995$–$1.005$ at $h = 1, 5, 20$
across all three generating regimes — nothing, as the algebra requires. The
log-likelihood does move, consistently: $+0.0036 / +0.0040 / +0.0026$ nat/point
on static / impulsive / persistent, i.e. 3–5 nats over 1200 points for two extra
parameters. (In-sample with two extra parameters; an out-of-sample check is the
honest follow-up and has not been run.)

This completes a three-way split that is worth carrying:

| object | visible in |
|---|---|
| the state $z_t$ | tracking error (and barely in anything else) |
| the level of $\alpha$ | $h$-step forecast **mean** error — [`0006`](0006_alpha_is_a_forecasting_parameter.py) |
| the persistence of $\alpha$ | predictive **variance** / calibration — this probe |

Each has its own loss, and each is nearly invisible in the other two. Scoring
the wrong one is how a real effect looks like nothing.

## 2. The minimax claim is withdrawn

> **⚖️ ATTRIBUTION —** _Judging a drift law by its worst case over the unknown direction of parameter movement (rather than an average over hand-chosen scenarios) is the minimax / least-favorable-prior criterion._ Prior art: minimax decision theory (Wald 1950); least-favorable priors (Huber 1964 robust estimation). This is a self-correcting NEGATIVE-RESULT: the earlier worst-case claim is withdrawn under a proper direction sweep. Status: NEGATIVE-RESULT.

[`0011`](0011_the_drift_shape.md) §3 rested the case for the invariant drift law
on a worst case over three scenarios I chose. Three points with a `min()` on
them is not a worst case. [`0013`](0013_minimax_over_directions.py) sweeps the
*direction* of the shift — $\alpha_1 = \alpha_0 + r(\cos\psi, \sin\psi)$ over 12
angles — at two base points, one interior to the stationarity triangle and one
near its boundary (0008's own base point, in case the effect was a near-boundary
phenomenon). Fraction of the static-to-oracle gap closed, $h=5$ (fig10):

| base | anisotropy $\vert\gamma_1\vert/\gamma_0$ | law | best | **median** | **worst** |
|---|---|---|---|---|---|
| interior | 0.866 | `iso` | 0.932 | **0.716** | −0.071 |
| interior | 0.866 | `fisher-shape` | 0.883 | 0.398 | −0.069 |
| boundary | 0.938 | `iso` | 0.787 | **0.648** | **0.000** |
| boundary | 0.938 | `fisher-shape` | **0.929** | 0.107 | −0.014 |

**The isotropic law wins on the median at both base points and ties or wins on
the worst case.** The near-boundary hypothesis is refuted along with the
original claim. `0011` §3's conclusion — that invariance buys the worst case —
is withdrawn; it was an artifact of one of the three scenarios happening to be a
direction where `iso` chose $\nu^\ast=0$.

What replaces it is more interesting than a hedge. Look at how the two laws
spend their advantage. At the boundary base, `fisher-shape` posts the single
best angle of anything measured (0.929 at $\psi=0.52$, against `iso`'s 0.76
there) and then sits at $\approx0$ for five of the twelve angles. `iso` is
between 0.55 and 0.79 at seven angles. And `iso` chose $\nu^\ast = 0.010$ at
almost every angle, while `fisher-shape`'s $\nu^\ast$ jumped between 0.004 and
0.025 — a single scalar magnitude cannot serve a kernel whose effective step
varies that much with direction.

> **The Fisher shape concentrates the drift into a narrow cone.** When the truth
> moves inside that cone it is the best thing measured; otherwise it is nearly
> static. Concentrating without knowing the direction is precisely what a
> minimax criterion penalises.

## 3. Why the parent's invariance argument worked and this one does not

This is the part worth keeping, and it took a refutation to see.

The parent forced its drift law with **scale equivariance**: the class cannot
know whether $x$ is in metres or feet. That is a symmetry **of the world**.
Nature genuinely is indifferent to the choice of unit, so a law that respects it
is not merely well-formed — it is *true*.

Čencov invariance is a symmetry **of the notation**. It says a drift law should
not depend on whether we write the recurrence in lag coefficients, in roots, or
in damping-and-frequency. That is a real constraint on well-formedness, and
$\Sigma_{\text{drift}} \propto Q\Gamma^{-1}$ really is the unique law satisfying
it. But nature is *not* indifferent to how we parameterise our model. Physical
systems have preferred dynamical coordinates — a damping coefficient drifts for
reasons that have nothing to do with a frequency, and the Fisher metric knows
nothing about which is which.

$$\text{scale equivariance} : \text{a symmetry of the world}
\qquad\ne\qquad
\text{reparameterisation invariance} : \text{a symmetry of our notation}$$

Both are invariance arguments; only the first carries information about how the
parameter moves. **Being the unique well-formed answer is not the same as being
the right answer, and the parent's success made that easy to conflate.**

Note also that `iso` is not the winner because it is *right* — it is not
invariant either, and it is only "isotropic" relative to the lag-coefficient
coordinates, which are as arbitrary as any. It wins because spreading the drift
uniformly is what "no information about the direction" actually implies, and
because *some* metric has to be picked to say "uniformly". The honest statement
is that **the choice of metric on the dynamics is an open modelling degree of
freedom that no invariance principle closes.**

## 4. What follows: the shape has to be learned

If no principle supplies the drift shape, it goes where every other unknown in
this repository goes — into the likelihood. For $p=2$ the shape is a $2\times2$
covariance up to scale: an anisotropy ratio and an orientation, two numbers. For
$p=3$, five. Learned by marginal likelihood alongside everything else, so still
no *free* parameters, just more *learned* ones.

The cost is identifiability, and the parent workstream is the warning: it
measured $s_P$ at $0.0017$ nats/point of evidence and had to tell callers not to
read it. Whether a drift-shape matrix carries enough evidence to be worth
estimating is a question the parent already knows how to ask — profile the
likelihood along the shape coordinates and count nats. That is the next probe,
and it is the one that decides whether the dynamics channel is a two-number
object like the parent's or a $\tfrac{p(p+1)}2$-number one.

## Next, in order

1. **Profile the drift shape.** How many nats per point does the data carry
   about the anisotropy and orientation of $\Sigma_{\text{drift}}$? If the
   answer is like $s_P$'s, the shape is not estimable and `iso` is the right
   default by default. Directly answerable with the parent's own method.
2. **Score the persistence dial on calibration**, out of sample. §1 shows point
   forecasts cannot see $\varphi_A$ and the algebra says they never will; the
   claim that it helps rests on 3–5 in-sample nats and needs a real test.
3. **$p=3$: the full target class**, offset plus oscillator. The architecture
   extends directly; the grid is the compute budget.
4. **Learn $Q$ and $\sigma^2$ jointly.** Every probe so far holds them at truth.
5. **Order selection**, the way the offset root became testable in
   [`0011`](0011_the_drift_shape.md) §2.
