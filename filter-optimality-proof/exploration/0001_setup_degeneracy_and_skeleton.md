# 0001 — What "optimal" can mean here: the class, a degeneracy theorem, and a skeleton

> **Superseded in two places by [`0005`](0005_leak1_result_and_the_honest_class.md).**
> §3's class definition ("shapes otherwise arbitrary") is wrong and is replaced
> by `0005` §5. §5's Burg argument overstates what has been established —
> max-entropy on the latent log-scale path does not transfer to a minimax
> property on the observable without a further step; see `0005` §6. Everything
> else here stands.

First pass. Goal was to write down a rough proof and find where it breaks. It
breaks in four identifiable places, one of which is more interesting than
expected and is the main thing to read (§6, §7 Leak 2).

Script: `0002_saddle_and_allocation_probes.py` (three probes, all exact or
40M-sample Monte Carlo; every number below is from it).

---

## 1. The object, and what has to be chosen before "optimal" means anything

The filter (`adaptive-random-walk-filter/output/statfilter/core.py`) is:

$$\theta_t=\theta_{t-1}+w_t,\quad w_t\sim N(0,\ Q e^{\lambda^P_t});\qquad
x_t=\theta_t+v_t,\quad v_t\sim N(0,\ \sigma^2 e^{\lambda^M_t})$$
$$\lambda^c_t=\varphi_c\lambda^c_{t-1}+\sqrt{\nu_c}\,z_t,\qquad c\in\{P,M\}$$

with $(Q,\sigma^2,\varphi_P,\varphi_M,s_P,s_M)$ fitted by maximum marginal
likelihood, inference by exact forward recursion over a quadrature grid on
$(\lambda^P,\lambda^M)$, and the level posterior collapsed to one Gaussian per
step (GPB1).

An optimality claim needs four things named. Stating them is most of the work,
because three of the four turn out to be forced and the fourth is where the
proof splits in two.

| | choice | status |
|---|---|---|
| **class of processes** $\mathcal C$ | see §2–§3 | *forced*, by a degeneracy argument |
| **class of procedures** $\mathcal D$ | all causal measurable estimators | free, take the largest |
| **loss** | squared error, or cumulative log-loss | **the seam — see §6** |
| **optimality notion** | minimax over $\mathcal C$ | forced: no prior on $\mathcal C$ is available, that being the premise |

The fourth line deserves a note, because it dissolves the worry about
$\lim_{x\to\infty}\mathrm{Unif}(0,x)$. "No information about the noise scale" does
not have to be expressed as an improper prior. It can be expressed as
*equivariance* under the group that acts on scales, and by the Hunt–Stein
theorem — the group $(\mathbb R_+,\times)$ is abelian, hence amenable, so the
theorem applies — the best equivariant procedure is minimax. The improper limit
is a way of *computing* the equivariant answer, not something the argument needs
to be well defined. That is also the first place the log scale appears, and it
appears for a reason rather than for convenience: Haar measure on
$(\mathbb R_+,\times)$ is $d\sigma^2/\sigma^2$, i.e. Lebesgue in $\lambda=\log\sigma^2$.

**Consequence, worth stating separately:** any constraint the class places on how
the scales move must be a constraint on $\lambda$, not on $\sigma^2$, or the
class is not invariant to the units of $x$ and "optimal" would depend on whether
the series is measured in metres or feet.

---

## 2. The unconstrained class is empty of content

The user's proposed class — *all unbiased random walks where the update
distribution may change in shape and magnitude over time, and nothing else is
given* — has no non-trivial optimal procedure. Not "hard to analyse": empty.

> **Proposition 1.** Let $\mathcal C_\infty$ be all processes
> $\theta_t=\theta_{t-1}+w_t$, $x_t=\theta_t+v_t$ with independent mean-zero
> $w_t,v_t$ of *arbitrary* variances $Q_t,R_t\in(0,\infty)$, unknown to the
> filter. Let $\delta^{\mathrm{orc}}$ be the Kalman filter that knows
> $\{Q_t,R_t\}$. Then for every causal $\delta$,
> $$\sup_{p\in\mathcal C_\infty}\ \frac{\mathbb E_p(\theta_t-\delta_t)^2}{\mathbb E_p(\theta_t-\delta^{\mathrm{orc}}_t)^2}\ =\ \infty .$$

*Proof.* Work at a single step $t_0$; let the state be known exactly up to
$t_0-1$, so the prior variance entering $t_0$ is $P>0$. Consider two members of
$\mathcal C_\infty$, both with all other variances equal:

* $p_{\mathrm{jump}}$: $Q_{t_0}=A$, $R_{t_0}=1$ — the level really moved;
* $p_{\mathrm{out}}$: $Q_{t_0}=1$, $R_{t_0}=A$ — the sensor glitched.

Under both, $x_{t_0}-m_{t_0-1}$ has the same law, mean zero and variance
$P+A+1$: **the data cannot distinguish them at all at step $t_0$.** Any causal
$\delta$ therefore applies the same (possibly randomised, possibly nonlinear)
map to the same distribution and produces the same conditional law for
$\delta_{t_0}$.

The oracle's risks differ: $\rho^{\mathrm{jump}}\to1$ and
$\rho^{\mathrm{out}}\to P$ as $A\to\infty$, both $O(1)$. But $\theta_{t_0}$ sits
$\Theta(\sqrt A)$ from $m_{t_0-1}$ under $p_{\mathrm{jump}}$ and exactly at it
under $p_{\mathrm{out}}$, and those two targets are $\Theta(\sqrt A)$ apart while
$\delta_{t_0}$ has one law. So
$\max\{\mathbb E_{\mathrm{jump}}(\theta-\delta)^2,\ \mathbb E_{\mathrm{out}}(\theta-\delta)^2\}\ \ge\ \tfrac14\,\Theta(A)$,
and the ratio is $\Omega(A)\to\infty$. $\blacksquare$

Three things follow, and they are the useful part.

1. **The class must be constrained, and the binding constraint is temporal.**
   The construction uses only that $(Q_{t_0},R_{t_0})$ is unpredictable from
   $(Q_{t_0-1},R_{t_0-1})$. If the scales are constrained to be predictable from
   their own past, the adversary loses the coin flip — not at $t_0$, where the
   data is genuinely uninformative, but at $t_0+1,t_0+2,\dots$, where the two
   hypotheses diverge. **This is exactly the PA-vs-MA confusion of
   `theory/06`, and Proposition 1 says it is not a defect of that analysis but
   the defining difficulty of the class.**

2. **The one-step singularity is a theorem, not a conservatism.** `theory/04`
   §4 reported "the one-point delay is not a design conservatism; it is the
   information bound". Proposition 1 is the strong form: at one point the two
   channels are not merely hard to separate, they are *identically
   distributed*, so the information is zero rather than small.

3. **The minimal repair is a bound on the conditional spread of
   $\lambda_t$ given $\lambda_{t-1}$** — one number per channel for "how much"
   and one for "how predictable". That is $(s_c,\varphi_c)$. So the filter's two
   scale parameters per channel are *the definition of the class*, not
   parameters within it. (Minimality is asserted from the structure of the
   proof, not proved. Flagged.)

---

## 3. The class, stated

$$\mathcal C(s,\varphi)=\Big\{\ \theta_t=\theta_{t-1}+w_t,\ x_t=\theta_t+v_t\ \Big|\
\begin{array}{l}w_t,v_t \text{ independent, mean zero},\\
\operatorname{Var}w_t=Qe^{\lambda^P_t},\ \operatorname{Var}v_t=\sigma^2e^{\lambda^M_t},\\
\lambda^c \text{ stationary, } \gamma_0^c=s_c^2,\ \gamma_1^c=\varphi_c s_c^2,\\
\text{shapes of } w_t,v_t \text{ otherwise arbitrary}\end{array}\Big\}$$

Two moments per channel on the log-scale path, and *nothing at all* about
shape. This is the tractable formalisation the exploration was looking for: it
never mentions a distribution over noise magnitudes, only two autocovariances of
the log-scale, which is the least that Proposition 1 permits.

---

## 4. Layer 1 — shape. **Proved.**

Promoted to [`../output/01-shape-minimaxity.md`](../output/01-shape-minimaxity.md).

Given the variance path, the Kalman filter is *exactly* minimax over all
shapes, the Gaussian is *exactly* least favourable, and the value is the Riccati
error. Proof is three lines: the KF is linear so its risk is the same for every
shape at fixed variances; at the Gaussian it is the exact conditional mean;
weak duality closes the gap.

This replaces the central-limit-theorem defence of conditional Gaussianity with
something stronger and finite-sample. No limit is taken and no approximation is
made, and the result holds for increment laws that look nothing like Gaussian.
Probe A confirms it numerically: Gaussian hits $\rho$ exactly at 1.0000, uniform
0.9760, $t_5$ 0.9589, two-point 0.8992 — every alternative strictly easier, in
both tail directions.

**The light-tail worry is answered.** Uniform increments are *easier* than
Gaussian, so a Gaussian-designed filter is conservative there, never wrong. The
class does not need to exclude them and the theory does not need to cover them
specially.

---

## 5. Layer 2 — the log-scale dynamics. Burg, and it is an exact match.

Given only $\gamma_0=s^2$ and $\gamma_1=\varphi s^2$ for the path $\lambda^c$,
which member of $\mathcal C(s,\varphi)$ should the filter be built for?

> **Burg's maximum entropy theorem** (Cover & Thomas ch. 12). Among all
> stationary processes with prescribed autocovariances $\gamma_0,\dots,\gamma_p$,
> the one of maximum entropy rate is the Gaussian AR($p$) process fitted to
> them.

At $p=1$ that is $\lambda_t=\varphi\lambda_{t-1}+\sqrt\nu z_t$ with
$\nu=s^2(1-\varphi^2)$ — **the filter's log-scale model, exactly, including the
stationary-variance identity `core.py` uses.** Not a family containing it; it.

This is the strongest single piece of evidence for the "right tool for the job"
intuition, and it has a sharp form: the constraint set has two moments per
channel and the max-entropy law has two parameters per channel. The model
carries no degrees of freedom the constraint did not pay for, and none of the
constraint's degrees of freedom go unrepresented.

**It also generates the cousins immediately.** Prescribe $\gamma_0..\gamma_p$
and Burg returns the Gaussian AR($p$) log-scale; the filter is the $p=1$ member
of a ladder, and $p$ counts how much temporal structure the class asserts. That
is a concrete answer to "what cousins of the filter are optimal for other
assumptions", at least along this axis.

---

## 6. The seam: entropy and difficulty point the same way at layer 1 and *opposite* ways at layer 2

This is the finding I did not expect, and it is where the single-logic story
develops a genuine joint.

Layer 1 works because, at fixed variance, **more entropy = harder**: the
Gaussian both maximises entropy and maximises MMSE, so max-entropy and
least-favourable are the same element. That coincidence is what makes Theorem A
free.

At layer 2 it reverses. Probe C: exact one-step MMSE for
$\theta\sim N(0,P)$ observed through a log-scale mixture of *fixed total
variance* $R$, as a function of the log-scale SD $s$ ($P=R=1$, so
$\rho=0.5$):

| $s$ | 0 | 0.25 | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|---|
| MMSE | 0.5000 | 0.4998 | 0.4973 | 0.4694 | 0.4066 | 0.3246 | 0.1667 |
| /$\rho$ | 1.0000 | 0.9996 | 0.9945 | 0.9388 | 0.8132 | 0.6492 | **0.3335** |

Monotone **down**. A more volatile scale, at the same average variance, makes
the level *easier* to estimate — because the filter can tell which observations
were taken through a quiet channel and weight them up. So along the $s$
direction, more entropy in the $\lambda$ path means *less* difficulty, and
max-entropy is **not** least-favourable for squared error.

Three consequences, and they matter for how the proof is written.

**(a) The filter's $s>0$ is a claim about the world, not a robustness hedge.**
Over a class with total variance fixed, the least-favourable member is the plain
homoscedastic Gaussian random walk and the ordinary Kalman filter is exactly
minimax over it. Everything the adaptive filter adds is an attempt to *exploit*
structure, not to *survive* it. That is consistent with the measured battery and
explains its shape: on stationary diffusions, where there is no structure to
exploit, the ratios are 1.001–1.005, i.e. the small price of carrying machinery
that finds nothing.

**(b) Burg cannot be justified by the layer-1 argument.** It has to be justified
for a loss where entropy and difficulty *are* aligned, and there is exactly one:
**cumulative log-loss.** Higher entropy rate is by definition harder to code, so
over a convex class the max-entropy element is least favourable and the
predictive distribution matched to it is minimax-regret (Topsøe's robustness
theorem / the redundancy–capacity theorem). Under log-loss, Burg gives minimaxity
directly.

**(c) The criterion under which layer 2 works is exactly the criterion `fit()`
optimises.** `fit_()` maximises the exact marginal log-likelihood, which is
cumulative log-loss to the sign. So the correct theorem statement for the
existing implementation is a log-loss theorem, and the MSE results in the
battery are a separate empirical fact rather than the thing proved. This is
either a pleasing alignment or a warning that the two halves of the project are
optimising two different things and have not been reconciled. **It is currently
the latter.**

The candidate weld is the **I-MMSE identity** (Guo–Shamai–Verdú 2005):
$\frac{d}{d\,\mathrm{snr}}\,I(X;\sqrt{\mathrm{snr}}\,X+N)=\tfrac12\mathrm{MMSE}(\mathrm{snr})$.
Mutual information is the integral of MMSE over SNR, so a minimax statement in
information is a minimax statement in *SNR-integrated* MSE. That would convert
layer 2 into an MSE claim of a weaker but honest kind. Untested; it is the most
promising single next move.

---

## 7. Where the proof breaks

Ranked by how much they cost, worst first.

**Leak 1 — minimaxity does not compose across the layers.** Theorem A neutralises
the shape adversary *against a filter that already knows the variance path*,
because such a filter is linear and its risk depends on second moments alone.
The adaptive filter is nonlinear: it reads the magnitude of its own innovations
to infer $\lambda_t$. A shape adversary can therefore corrupt the scale
inference while leaving every variance — and hence the oracle's risk —
untouched. Concretely, two-point noise at variance $R$ never produces a large
innovation, so the filter under-reads the scale; $t_5$ noise at the same
variance produces occasional large ones and it over-reads. Both are in
$\mathcal C(s,\varphi)$; both leave the oracle unmoved. **This is directly
measurable and should be measured next** — run the fitted filter against
fixed-variance/varying-shape adversaries and record regret against the path
oracle. If the degradation is small the leak is a technical gap; if it is large,
the class must restrict shape after all, and the honest class is Gaussian scale
mixtures rather than "arbitrary".

**Leak 2 — two losses (§6).** Layer 1 is MSE, layer 2 is log-loss. Not yet one
theorem. This is the seam in what otherwise really is a single self-contained
logic, and it should be reported as such rather than papered over.

**Leak 3 — the six numbers are estimated, not marginalised.** Everything above
is conditional on $(Q,\sigma^2,\varphi_c,s_c)$. `fit()` maximises over them, which
is empirical Bayes; minimaxity of the Bayes rule for a fixed prior says nothing
about the rule that first estimates the prior from the same data. `theory/07` §E
already flags this from the other direction ($s_P$ unidentified, ML the wrong
estimator for a flexibility parameter). A fully marginalised version would close
both at once, and would make the object a genuine Bayes rule for a hyperprior —
at which point admissibility arguments become available, which minimax arguments
alone never give.

**Leak 4 — GPB1.** The filter is not the exact Bayes rule even for its own
model. I briefly hoped the collapse might be exact at the saddle point (the
adversary plays Gaussian, so the posterior is Gaussian, so nothing is lost).
It is not: the posterior over $\theta_t$ is a mixture over the $\lambda$ grid
regardless of the shape of the noise, and the adversary's preferred $s$ is
$0$ (§6a) precisely where GPB1 *is* exact — so the approximation error is worst
where the filter is furthest from the minimax configuration. Awkward but not
fatal; it is a bounded numerical cost, and `theory/07` §E already flags the
pure-step probe as the case to measure.

---

## 8. The allocation identity is not Gaussian-specific — probe B

`theory/07` §B's amplitude conservation law,

$$e_t=\frac{P}{S}e_t+\frac{Q_t}{S}e_t+\frac{R_t}{S}e_t,\qquad S=P+Q_t+R_t,$$

is the statement that for independent $a,b,c$ summing to $e$,
$\mathbb E[a\mid e]=(V_a/V_e)\,e$ — linear, with weights the variance shares. The
three coefficients sum to $1$ because the three sources sum to $e$. Exact
algebra, and the reason "allocation without detection" is possible at all: the
attribution is a conditional expectation, so it exists at every step whether or
not anything happened.

The natural question is whether linearity characterises the Gaussian, which
would make the conservation law and the Gaussian assumption *the same
assumption* — tidy, and it would explain why the theory feels seamless. It does
not, and the failure is more interesting than the success would have been.

**Probe B**, 40M samples per cell, independent symmetric $\alpha$-stable
$X\sim S(\alpha,c_X{=}1)$, $Y\sim S(\alpha,c_Y{=}2)$, binwise
$\mathbb E[X\mid X{+}Y{=}z]/z$ across six quantile bins of $z$:

| $\alpha$ | predicted $c_X^\alpha/(c_X^\alpha+c_Y^\alpha)$ | binwise ratios |
|---|---|---|
| 2.0 | 0.2000 | 0.2000, 0.2002, 0.2007, 0.2005, 0.1995, 0.1999 |
| 1.8 | 0.2231 | 0.2228, 0.2234, 0.2250, 0.2237, 0.2231, 0.2228 |
| 1.5 | 0.2612 | 0.2598, 0.2613, 0.2631, 0.2615, 0.2611, 0.2607 |
| 1.2 | 0.3033 | 0.2899, 0.3034, 0.3031, 0.3031, 0.3032, 0.3492 |

Constant across bins — that constancy *is* the linearity — and at the predicted
level. (The two outer cells at $\alpha=1.2$ are tail sampling noise; the mean of
a stable law with $\alpha=1.2$ converges slowly.)

So **the three-way conservation law holds for the whole symmetric $\alpha$-stable
family, with the dispersion $c^\alpha$ in the role of the variance.** Both
ingredients survive: $c^\alpha$ adds across independent sources exactly as
variance does, and the allocation weights are its shares. The Gaussian filter is
the $\alpha=2$ member of a one-parameter family, and the natural class for
$\alpha<2$ is $\alpha$-stable Lévy motion observed in $\alpha$-stable noise —
a real and well-studied process class, not a contrivance.

This is a second cousin-generating axis, independent of the $p$ of §5:

| axis | deformation | filter recovered at |
|---|---|---|
| $p$ (Burg order) | how much temporal structure the log-scale asserts | $p=1$ |
| $\alpha$ (stability index) | tail index of the increments | $\alpha=2$ |

Two caveats before this gets used. For $\alpha<2$ the variance is infinite, so
squared error is the wrong loss and Theorem A does not transfer — the natural
loss is $\mathbb E|\cdot|^r$ for $r<\alpha$, and the whole layer-1 argument
would need redoing. And $\mathbb E[X\mid X+Y]$ requires $\alpha>1$. The family is
$1<\alpha\le2$.

*(Status: probe B is a numerical result, strongly supported but not a proof. The
linear-regression property of stable vectors is standard — Samorodnitsky &
Taqqu ch. 4 — but I have not checked the reference and am not asserting the
attribution.)*

---

## 9. Next moves, in order

1. **Measure Leak 1.** Fit the filter, then run it against fixed-variance-path
   adversaries with two-point / uniform / $t_5$ / mixture shapes, and record
   regret against the path-oracle Kalman filter. This is the one leak that could
   invalidate the class definition rather than just leave a gap in the argument,
   and it is cheap to test. **Do this first.**
2. **Write layer 2 properly as a log-loss theorem.** Burg + Topsøe, stated over
   $\mathcal C(s,\varphi)$, with the convexity and existence conditions actually
   checked rather than gestured at. This is the piece most likely to become a
   clean standalone result.
3. **Try the I-MMSE weld** (§6c). If it works, layers 1 and 2 are one theorem in
   an integrated-MSE sense and the seam closes.
4. **The $\alpha$ family.** Redo layer 1 for $\alpha$-stable under $L^r$ loss and
   see whether a saddle point survives. If it does, the "$\sqrt{\text{nats}}$"
   exponent of `theory/04` should deform with $\alpha$, which would be a sharp
   testable prediction.

## 10. One claim from the prior work I would not carry forward unexamined

`theory/04` §2, "influence is the square root of information",
$a_k\propto\sqrt{\Delta^{\text{nats}}_k}$. For a linear estimator with weights
$a_k$ and independent observations, the Fisher information contributed by the
$k$-th is proportional to $a_k^2$ by direct computation, so the relation is an
identity of the linear-Gaussian setting rather than a discovery about it. It is
still operationally right and the practical content — *gate influence on
$\sqrt{\text{nats}}$, not nats* — is correct and non-obvious. But it should be
stated as an identity, and `theory/05` is right that its extension past the
location channel is a conjecture. Under §8 it becomes a sharp one: if the
exponent is really $1/2$ because information is quadratic in amplitude, then in
the $\alpha$-stable family it should be $1/\alpha$·(something), and checking
that is a real test of whether the relation has content.
