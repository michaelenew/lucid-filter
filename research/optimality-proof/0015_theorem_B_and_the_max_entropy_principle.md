# 0015 — Theorem B, and the possibility that max-entropy is the whole argument

Two things here. A **correction** that reverses much of `0009`, and a **theorem**
that makes the shape adversary exactly computable — after which the whole
structure looks like one principle applied twice rather than two arguments.

## 1. The correction: `Var(log u)`, not `log(κ/3)`

`0005` §3 asserted that an i.i.d. shape with kurtosis $\kappa$ adds
$\log(\kappa/3)$ to the log-scale variance $\gamma_0$. **That is right only for
lognormal mixing.** For a Gaussian scale mixture $\varepsilon=\sqrt u\,z$ with
$\mathbb Eu=1$, the shape shifts the log-scale by $\eta=\log u$, so the correct
increment is $\operatorname{Var}(\log u)$, whereas

$$\log(\kappa/3)=\log\mathbb E[u^2]\ \ne\ \operatorname{Var}(\log u).$$

For Student-$t_5$ the mixing law is $u=3/\chi^2_5$, giving
$\operatorname{Var}(\log u)=\psi'(5/2)=0.4904$ against the $\log 3=1.0986$ the
wrong formula supplied. So the predicted relocation is $(0.890, 0.355)$, not
$(1.184,0.201)$ — and `0007` measured $\mathrm{fit}()$ landing at
$(0.907\pm0.065,\ 0.488\pm0.089)$, i.e. **0.26 se and 1.5 se from the corrected
prediction.**

Re-measured directly (`0013`, 40 seeds, paired, $t_5$ data):

| parameters | $(s,\varphi)$ | MSE | vs best | se | $t$ |
|---|---|---|---|---|---|
| **corrected moment point** | (0.89, 0.35) | 0.19582 | — | — | — |
| where `fit()` lands | (0.90, 0.45) | 0.19595 | +0.07% | 0.04 | **1.6** |
| `0005`'s moment point | (1.20, 0.15) | 0.19882 | +1.53% | 0.29 | 5.2 |
| the truth | (0.55, 0.93) | 0.20680 | +5.60% | 0.84 | 6.7 |

The corrected point is the best of the four, and `fit()` is tied with it.

**So `0009`'s headline — "`fit()` does not estimate the class parameters, it
returns the KL-projection" — was an artifact of my arithmetic.** `fit()` lands
on the moment-matched relocation to within 0.07% of MSE. The narrower true
statement survives: the relocated process is an ARMA(1,1) in log-scale while the
filter's family is AR(1), so ML must still project onto the family — but the
projection is evidently almost exactly the moment point, so it costs nothing
measurable. Leak 3's quasi-MLE framing stands; the claimed 25–30% discrepancy
does not.

What is *not* affected: `0012`'s finding that the true parameters cost +5.98%
(it is the same finding — the relocated parameters are the right ones), and that
even well-specified, the true parameters are not MSE-optimal (+0.65%, GPB1).

## 2. Theorem B — the shape adversary is a relocation along a $\gamma_1$ level set

> **Theorem B.** Let $\lambda=(\lambda_t)$ be stationary with $\gamma_0=s^2$,
> $\gamma_1=\varphi s^2$. Let $(u_t)$ be i.i.d., $u_t>0$, $\mathbb Eu_t=1$,
> $\operatorname{Var}(\log u_t)<\infty$, independent of $\lambda$ and of
> $(z_t)\overset{iid}\sim N(0,1)$. Put
> $$v_t=\sqrt{\sigma^2e^{\lambda_t}}\cdot\sqrt{u_t}\,z_t .$$
> Then with $\eta_t=\log u_t$, $m=\mathbb E\eta_t$,
> $\tilde\lambda_t=\lambda_t+\eta_t-m$ and $\tilde\sigma^2=\sigma^2e^{m}$,
> $$v_t=\sqrt{\tilde\sigma^2e^{\tilde\lambda_t}}\;z_t,$$
> where $\tilde\lambda$ is stationary with
> $$\tilde\gamma_0=s^2+\operatorname{Var}(\log u),\qquad \tilde\gamma_1=\gamma_1 .$$

*Proof.* $\tilde\sigma^2e^{\tilde\lambda_t}=\sigma^2e^{m}e^{\lambda_t+\eta_t-m}
=\sigma^2e^{\lambda_t}u_t$, which is the first display. $\eta$ is i.i.d. and
independent of $\lambda$, so $\tilde\gamma_0=\operatorname{Var}\lambda+
\operatorname{Var}\eta$ and, for $k\ge1$,
$\tilde\gamma_k=\operatorname{Cov}(\lambda_t+\eta_t,\lambda_{t+k}+\eta_{t+k})
=\gamma_k$. $\blacksquare$

Elementary, and exact — no approximation, no limit. Three consequences.

**(a) The adversary moves along a level set of $\gamma_1$.** In class
coordinates, $\tilde s^2=s^2+\operatorname{Var}(\log u)$ and
$\tilde\varphi=\varphi s^2/\tilde s^2$, so

$$\tilde s^2\tilde\varphi=s^2\varphi=\gamma_1\quad\text{is invariant.}$$

An i.i.d. shape can add $\gamma_0$ and can never touch $\gamma_1$. **The
Gaussian shape ($u\equiv1$) is the endpoint of minimal $\gamma_0$ on that
curve.** This is the exact form of the informal claim in `0005` §4 that shape
and scale are the same coordinate with $\varphi$ as the dial.

**(b) Kurtosis is *not* the sufficient statistic — $\operatorname{Var}(\log u)$
is.** Two mixing laws with the same $\kappa=3\mathbb E[u^2]$ can have different
$\operatorname{Var}(\log u)$ and therefore relocate to different points. This
**contradicts `0004`'s empirical finding** that leverage is monotone in kurtosis
alone, with two structurally unrelated shapes matched at $\kappa=5$ agreeing to
0.5 se. Both cannot be right in general. The likely resolution is that the two
shapes `0004` used happened to have similar $\operatorname{Var}(\log u)$ — the
two functionals are strongly correlated across the usual families — but that is
a guess and it is now the sharpest available discriminating test:
**construct two scale mixtures matched in $\kappa$ but deliberately split in
$\operatorname{Var}(\log u)$, and see which functional predicts the risk.**
Until that is run, the SUMMARY's "monotone in kurtosis alone" should be read as
"monotone in a shape functional that kurtosis tracks in the cases tested".

**(c) The relocated process is never at the max-entropy point.** If $\lambda$ is
AR(1) then $\tilde\lambda$ is AR(1) + i.i.d. = ARMA(1,1), with

$$\frac{\tilde\rho_2}{\tilde\rho_1^{\,2}}=\frac{\tilde\gamma_0}{\gamma_0}>1 .$$

So a heavy tail always lands strictly on the $\rho_2>\rho_1^2$ side of the
class. It relocates *within* the class but *off* the AR(1) submanifold the
filter models. This is the precise sense in which Leak 1 reduces to, but does
not vanish into, Leak 3.

**Scope caveat.** Theorem B covers Gaussian scale mixtures. Every such mixture
has $\kappa\ge3$, but **the converse is false** — not every $\kappa\ge3$ law is
a scale mixture. The SUMMARY's gloss "Gaussian scale mixtures — every shape with
kurtosis $\ge3$" overstates this and should be corrected to the one-way
implication.

## 3. The principle: max entropy, twice

With Theorem B in place the architecture reads as one idea applied at two
levels:

> **Withdrawn in part — see [`0017`](0017_max_entropy_is_not_least_favourable_under_MSE.md).**
> The conjecture below is **false under squared error** and **true under
> log-loss** ([`output/02`](proofs/02-logloss-least-favourable.md), Theorem C).
> The section is kept because the diagnosis of *why* it splits is the useful
> part: both layers are equalizer arguments, and an equalizer exists exactly
> when the loss is affine in what the class fixes. Read §3 with that correction.

| | constraint | max-entropy law | least favourable? |
|---|---|---|---|
| **Layer 1** (`output/01`) | variance $R$ | Gaussian | **yes — proved, exactly** |
| **Layer 2** | $\gamma_0,\gamma_1$ of $\log$-scale | Gaussian AR(1) (Burg) | **under log-loss yes; under MSE no** |

The filter models the Gaussian at layer 1 and the Gaussian AR(1) at layer 2. If
the right-hand column is "yes" both times, the design rule is a single sentence
— *model the maximum-entropy member of whatever you are constrained to, because
it is the worst case and therefore cannot surprise you* — and the two layers are
one theorem instantiated twice rather than two arguments joined.

That is worth stating plainly because it is the claim that the whole structure
is self-contained rather than assembled. It is **not established.** Layer 1 is
proved; layer 2 is a conjecture with a clean test.

**The mechanism that would make it true**, and the reason to doubt it: the
filter's update step reads $|{\rm innovation}_t|$ *contemporaneously*, so it
exploits scale variation whether or not it modelled the dynamics that produced
it. Any member with more exploitable structure than the AR(1) is therefore
easier, even to a filter not built for it. The doubt is that this argues for
one side only — a member with *less* predictable log-scale ($\rho_2<\rho_1^2$,
oscillatory) might be genuinely harder, which would put the AR(1) in the
interior of a monotone family rather than at a maximum, and break the saddle.

## 4. The test, and what it decides

The construction: hold $(\gamma_0,\gamma_1)$ fixed, realise class members as
AR(2) log-scales spanning $\rho_2$ on both sides of the max-entropy value
$\rho_2=\rho_1^2$, and hold the filter fixed at the AR(1) model — i.e. at the
Bayes rule for the max-entropy member.

`0014` ran this but **its grid was too narrow to decide anything**: it swept
$\rho_2$ over $\pm12\%$ of $\rho_1^2$, which at $\rho_1=0.5$ is $[0.175,0.28]$
out of an admissible $(2\rho_1^2-1,\,1)=(-0.5,1)$. Its readings — a shallow
monotone rise in the moderate regime, noise in the persistent one — cover too
little of the class to locate a maximum. `0016` redoes it across the full
admissible range at 80 seeds and supersedes it.

The prediction is an **interior maximum** at $\rho_2=\rho_1^2$, with risk
falling on both sides. That signature matters: no monotone confound can produce
it, so a two-sided fall is much stronger evidence than a one-sided one.

- **Both sides fall** → max-entropy is least favourable at layer 2, Theorem A's
  three-line saddle argument transfers verbatim, and layer 2 closes up to the
  GPB1 gap already measured at +0.65% (`0012`).
- **Monotone in $\rho_2$** → it does not, the Burg step is decorative rather
  than load-bearing, and the honest statement is that the filter's log-scale
  model is *a* member of the class chosen for tractability, not the worst one.

Either outcome is worth having, and the second would be the more important
finding.

## Next, after `0014`

1. The $\kappa$ vs $\operatorname{Var}(\log u)$ discriminating test from (b) —
   it decides whether the SUMMARY's kurtosis claim needs restating.
2. If `0014` falls both sides: write layer 2 as a theorem conditional on the
   max-entropy-least-favourable lemma, and attempt that lemma directly.
3. The marginalisation gap (`0005` §6) is untouched by any of this and remains
   the deepest hole in layer 2.
