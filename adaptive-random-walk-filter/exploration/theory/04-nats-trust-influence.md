# 04 — Nats → trust → influence, and a definition of *trustworthy*

Script: `scripts/THEORY-004-nats-to-influence.py` · Figures: fig11, fig12, fig13

## 1. Nats → trust: the exponential confirmation loop is literal (fig12)

Evidence adds in log-odds. With $\Lambda_m=\sum_{t\le m}\text{LLR}_t$,

$$\text{trust}_m = \sigma(\Lambda_m),\qquad 1-\text{trust}_m = \sigma(-\Lambda_m)\ \approx\ e^{-\Lambda_m}$$

So the "rapid exponential approach to 100% trust" is not an analogy or a design
choice — **residual doubt decays by a factor of $e$ per nat**, and that is all
that "nats" means operationally. 2.2 nats = 90%, 4.6 = 99%, 6.9 = 99.9%.

### The noisy-OR form is a different rule, and it isn't the calibrated one

The proposed combination
$\text{relevance}=1-\prod_i(1-p_i)$
is a noisy-OR: correct for "at least one of several independent causes fired",
wrong for "several independent observations bear on one hypothesis". The
calibrated rule is log-odds addition. Both approach 1 exponentially, at different
rates and from different places (fig12, right). The distinction matters because
the noisy-OR has no way to express *disconfirming* evidence — every term can only
push toward 1 — whereas a log-odds sum moves both ways, which is exactly what is
needed when the second measurement says "actually that was an outlier".

## 2. Nats → influence: a square root, not a proportion (fig13, right)

From [02](02-relevance-decay-and-the-tail.md), verified exactly across
$q$ = 0.005 / 0.05 / 0.5:

$$\Delta^{\text{nats}}_k\propto(1-K)^{2k},\qquad a_k\propto(1-K)^{k}
\qquad\Longrightarrow\qquad \boxed{\ a_k \propto \sqrt{\Delta^{\text{nats}}_k}\ }$$

**Information is an energy; influence is an amplitude.** The artifact whose
trustworthiness we track is the *estimator's sensitivity* $\partial\hat\mu/\partial x$,
and sensitivities compose like amplitudes while informations compose like
variances. Concretely at $K$=0.2, lag 10: nats share 1.2%, correct influence
share 10.7% — allocating by nats directly would under-weight by 9×.

This also answers the "it needn't be the data point that gets the trust" caution.
The object with a conserved budget is the **weight vector** $\{a_k\}$ with
$\sum a_k=1$; any artifact $g(x_0,x_1,\dots)$ inherits its influence through
$\partial g/\partial x_k$. The conservation law $\sum_k a_k=1$ is what makes
"% effect" well defined, and it holds for any unbiased estimator of a level.

## 3. Information → *trustworthy* information

Nats are never absolute. $\text{LLR}$ is evidence **for** one hypothesis **against**
another; quoting a nat count without naming the alternative is the same category
error as quoting a probability without a sample space. Every gate this project
built and abandoned quoted a one-sided score.

**Proposed definition.** The trustworthy evidence for mode $h$ after $m$ points is

$$\Lambda^{\text{robust}}_h(m) \;=\; \min_{h'\in\mathcal H\setminus\{h\}} \mathrm{KL}\!\left(P_h^{(m)}\,\|\,P_{h'}^{(m)}\right),
\qquad \mathcal H=\{H_0,\text{PA},\text{MA},\text{PR},\text{MR}\}$$

the row-minimum of the pairwise LLR matrix, $H_0$ included. This is the e-value /
GRO reading: evidence that survives the most favourable competing explanation.
It has the properties we want — it is in nats, it converts to trust by
$\sigma(\cdot)$, it is zero when the mode is unidentifiable (so it is
automatically zero at $m=1$, where both planes are singular), and it needs no
scale factor because a log-likelihood ratio's coefficient is fixed at 1 by Bayes.

### The confirmation ledger (fig11)

Points needed for $\Lambda^{\text{robust}}\ge\log 99$ (99:1), $q$=0.05:

| mode | 2 SD / 1.5× | 3 SD / 2× | 4 SD / 3× | 6 SD / 6× |
|---|---|---|---|---|
| **PA** jump | never | **3** | **2** | **2** |
| **MA** outlier | never | **8** | **2** | **2** |
| **PR** $Q$ change | never | never | never | **45** |
| **MR** $\sigma^2$ change | never | **41** | **15** | **5** |

Three things this table says:

- **Small events never become trustworthy, and that is correct.** A 2-SD
  excursion is what $H_0$ produces routinely. It should be absorbed as process
  noise, not detected. The framework declines to detect it without needing a
  threshold to be chosen.
- **Location events are cheap, scale events are expensive, and $Q$-changes are
  in a class of their own.** A 6× change in $Q$ needs 45 confirmations; a 3×
  change never clears the bar. This is the honest reason the project needed a
  2000-point variogram — not the estimator, the *regime*.
- **The asymmetry is a design instruction.** Jump handling can be fast and
  local. $Q$ tracking cannot be, ever. Building one mechanism to do both is what
  made every previous gate trade a factor of 2–4 on jumps for a factor of 8–11
  in the wrong regime.

## 4. Influence allocation after a jump (fig13, left)

Bayesian model averaging over {shift at $t_0$, no shift} gives the share of the
posterior mean carried by post-jump data:

$$\alpha_m = \pi_m + (1-\pi_m)\big(1-(1-K)^m\big),\qquad \pi_m=\sigma(\Lambda_m)$$

For a 4-SD jump at $q$=0.05: $\alpha$ = 0.61 at $m$=1, **0.99 at $m$=2**, versus
the plain Kalman filter needing $m$=11 to reach 0.90. The oracle's stated target
— "immediately incorporate ~99% of trust as a direct location update, leaving the
noise parameters untouched" — is achieved at $m=2$ and is *not* achievable at
$m=1$, because at $m=1$ the location plane is singular and the shift cannot be
told from an outlier. **The one-point delay is not a design conservatism; it is
the information bound.**

Note that the noise parameters are untouched by construction here, not by
stipulation: the location and scale blocks are orthogonal, so evidence about a
jump carries exactly zero information about $Q$ or $\sigma^2$.

## 5. Two honest costs on top of the ledger

**Occam.** The oracle knows $\delta$. A real filter must marginalise it. With
$\delta\sim N(0,\tau^2)$, Sherman–Morrison gives in closed form

$$E[\log \mathrm{BF}] = \tfrac12\Big[\tfrac{tc}{1+tc} + \tfrac{\delta_0^2tc^2}{1+tc} - \log(1+tc)\Big],\quad t=\tau^2,\ c=u^\top\Sigma^{-1}u$$

At $\tau=4\times$increment-SD, $m$=2:

| jump | oracle | $\delta$ marginalised | Occam cost |
|---|---|---|---|
| 2 SD | 2.62 | 1.44 | 1.19 |
| 3 SD | 5.91 | 4.57 | 1.34 |
| 4 SD | 10.50 | 8.95 | 1.55 |
| 6 SD | 23.62 | 21.48 | 2.14 |

**The cost of not being the oracle is 1.2–2.1 nats, and it grows only
logarithmically in the event size.** That is a bounded, cheap, and — importantly
— *derived* penalty. It shifts the 3-SD row of the ledger and leaves the 4-SD row
alone. This is the first quantity in the project that plays the role $c$, $a_j$
and the 6.0 kept trying to play, and unlike them it has a closed form.

**Fluctuation.** $\Lambda$ is a random variable. For a Gaussian mean-shift LLR,
$\mathrm{Var}(\Lambda)=2E[\Lambda]$, so at the 4.6-nat threshold the standard
deviation is 3.0 (shaded bands in fig11). A single-shot threshold crossing is
therefore not reliable — realised evidence at a true 4.6 nats runs anywhere from
1.6 to 7.6 at $\pm1$SD. **Any usable rule must compare accumulated evidence to a
boundary, not test a single value against a threshold**, which is the sequential-
testing (SPRT / e-process) form rather than the point-test form. That is the
concrete next construction.
