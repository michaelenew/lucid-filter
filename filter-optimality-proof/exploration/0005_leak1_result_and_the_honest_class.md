# 0005 — Leak 1 measured: the adversary's only lever is kurtosis, and that closes the class

Scripts: `0003_leak1_shape_adversary.py`, `0004_is_kurtosis_the_sufficient_statistic.py`

This was meant to be a damage report on the worst leak in `0001` §7. It is
instead the most useful result so far, because the leak turns out to have
exactly one dimension, and closing it makes the class definition and the
filter's own model coincide.

---

## 1. What was at risk

Theorem A ([`../output/01-shape-minimaxity.md`](../output/01-shape-minimaxity.md))
removes the noise shape from the problem *for a filter that knows the variance
path*, because such a filter is linear and its risk is a function of second
moments alone. The adaptive filter is not linear: it reads the magnitude of its
own innovations to infer $\lambda_t$. So an adversary might move its risk while
leaving every variance — and therefore the path oracle's risk — untouched, and
the "shape otherwise arbitrary" clause in the class would have to go.

**Design.** One log-AR(1) measurement-variance path. Shapes standardised to
variance 1 and driven through their own inverse CDFs from the *same* uniform
draws, so the series are paired seed for seed and the path oracle's MSE is held
essentially fixed by construction. Ratio reported is adaptive MSE / path-oracle
MSE. Any spread across shapes is the leak, and nothing else is.

---

## 2. The leak is real, it vanishes exactly where the theorem says it must, and it grows like $s_M^2$

`0003`, true parameters supplied, $n=1200$, 6 seeds, $Q=0.05$, $\varphi_M=0.93$:

| $s_M$ | gaussian | two-point | uniform | $t_5$ | spread |
|---|---|---|---|---|---|
| 0.00 | 0.9996 | 1.0007 | 1.0001 | 0.9990 | **0.0016** |
| 0.35 | 1.0164 | 1.0326 | 1.0295 | 0.9773 | 0.055 |
| 0.55 | 1.0396 | 1.0747 | 1.0615 | 0.9861 | 0.089 |
| 1.00 | 1.1183 | 1.2478 | 1.1668 | 1.0539 | 0.194 |
| 1.50 | 1.2697 | 1.7900 | 1.4507 | 1.1644 | 0.626 |
| 2.00 | 1.4583 | 2.5210 | 1.8156 | 1.2896 | 1.232 |

Three things.

**The $s_M=0$ row is a validation of the experiment, not a result.** At
$s_M=0$ the filter *is* linear, Theorem A applies exactly, and the spread must
be zero. It is 0.0016. The apparatus is measuring what it claims to measure.

**Spread grows roughly like $s_M^2$** (spread$/s_M^2$ = 0.45, 0.29, 0.19, 0.28,
0.31 across the last five rows). $s_M^2$ is the variance of the log-scale, i.e.
exactly how much of the filter's behaviour is driven by reading innovation
magnitudes. The adversary's leverage is proportional to the filter's reliance on
the nonlinear channel. That is the leak stated quantitatively.

**The ordering never changes: two-point worst, then uniform, then Gaussian, then
$t_5$** — and $t_5$ sits *below 1*, i.e. the adaptive filter beats the linear
path oracle outright. Both facts are Theorem A read forwards: the oracle is
locked at LMMSE, and a nonlinear filter can beat it whenever the shape is
non-Gaussian, which this one does when the non-Gaussianity is of a kind it can
represent.

That ordering is kurtosis: 1, 1.8, 3, 9.

---

## 3. Kurtosis is the whole lever

`0004`, $s_M=1.5$, 16 seeds:

| shape | kurtosis | ratio | se | realised kurt at $n{=}1200$ |
|---|---|---|---|---|
| two-point | 1.0 | 1.6944 | 0.068 | 1.00 |
| uniform | 1.8 | 1.4894 | 0.038 | 1.80 |
| gaussian | 3.0 | 1.3633 | 0.038 | 3.00 |
| Student-$t_7$ | 5.0 | 1.2924 | 0.042 | 4.93 |
| Student-$t_5$ | 9.0 | 1.2493 | 0.047 | 7.90 |
| lognormal scale mixture | 9.0 | 1.1742 | 0.033 | 8.92 |
| Student-$t_{4.5}$ | 15.0 | 1.2265 | 0.049 | 10.40 |

Monotone decreasing in kurtosis across the whole range. The first matched-pair
test ($t_5$ vs lognormal mixture, both kurtosis 9) is inconclusive as designed —
they differ by 1.3 se — but the *realised* kurtoses differ too (7.90 vs 8.92),
because a $t_5$'s fourth moment is carried by rare events and has not converged
at $n=1200$. Interpolating on realised kurtosis reconciles them to about 1 se.
That confound is worth noting on its own: **the statistic the filter is
calibrated to is one that is itself hard to estimate**, which is the same
identifiability theme as the rest of the project.

Redone with shapes whose fourth moment does converge — matched at kurtosis 5,
one a two-component Gaussian scale mixture, one a generalised normal
($\exp(-|x/a|^\beta)$, $\beta=1.148$):

| shape | ratio | se | realised kurt |
|---|---|---|---|
| two-component scale mixture | 1.2195 | 0.035 | 5.09 |
| generalised normal | 1.2467 | 0.043 | 4.98 |

Apart by 0.5 se. **Kurtosis is the sufficient statistic, to the precision
available.**

One caveat on this pair, stated because it limits the claim: both of these
shapes happen to be Gaussian scale mixtures, so the pair alone does not
discriminate "kurtosis matters" from "distance from the scale-mixture family
matters". The light-tailed rows do: two-point and uniform have kurtosis below 3
and are therefore *provably not* Gaussian scale mixtures, and they sit on the
same monotone curve as everything else. Taken together the reading is kurtosis.

### Why kurtosis, and not something else

This is not a numerical coincidence, and the reason is short enough to be worth
stating exactly.

$\mathcal C(s,\varphi)$ specifies the log-scale path through $\gamma_0$ and
$\gamma_1$ **and nothing else** — that was forced in `0001` §2, since anything
weaker is degenerate and anything stronger is unjustified. Now let
$v_t=\sqrt{\sigma^2e^{\lambda_t}}\,u_t$ with $u_t$ i.i.d. of unit variance and
kurtosis $\kappa$. The marginal kurtosis of $v$ is $\kappa e^{s^2}$; a Gaussian
scale mixture with total log-scale variance $s_{\mathrm{tot}}^2$ has kurtosis
$3e^{s_{\mathrm{tot}}^2}$. Equating,

$$s_{\mathrm{tot}}^2=s^2+\log\frac\kappa3,\qquad
\varphi_{\mathrm{eff}}=\varphi\,\frac{s^2}{s_{\mathrm{tot}}^2}$$

the second because an i.i.d. $u$ contributes to $\gamma_0$ and nothing to
$\gamma_1$. **So a shape enters the class through exactly one number,
$\log(\kappa/3)$, added to $\gamma_0$.** Kurtosis is sufficient because
$\gamma_0$ and $\gamma_1$ are all the class has, and kurtosis is the only thing
a shape contributes to them.

Two consequences, both sharp:

- **The shape adversary is a reparameterisation, not an attack.** A $\kappa>3$
  shape relocates the process to a different $(s,\varphi)$ *within* the class —
  higher $s$, lower $\varphi$, by the formulas above. The filter is not being
  hit from outside its model; it is being moved inside it. That is why $t_5$
  noise makes it *beat* the linear oracle. If this is right, **Leak 1 collapses
  into Leak 3** — it is not a new gap but the already-acknowledged one about
  estimating the six parameters. `0006` tests the prediction directly.
- **The light-tail limitation is one inequality.** $\kappa<3$ gives
  $s_{\mathrm{tot}}^2 < s^2 + \log 1 = s^2$ and, once $\log(3/\kappa)>s^2$,
  gives $s_{\mathrm{tot}}^2<0$: **no representation exists at all.** At
  $s=0.55$ that threshold is $\kappa<3e^{-0.3025}=2.22$, which two-point (1.0)
  and uniform (1.8) are both below. The filter cannot express them, and the
  measured penalty is what that costs.

The sufficiency is exact for *placing a shape inside the class*. It is not a
proof that the filter's **risk** depends on the shape only through $\kappa$,
since the filter's per-step likelihood is a nonlinear function of $e_t$ and
therefore sees more than four moments. The matched pair above bounds that
residual dependence at 0.5 se.

---

## 4. Why that closes rather than breaks the class

A Gaussian scale mixture always has kurtosis $\ge 3$, with equality iff the
scale is degenerate — immediately from $\kappa = 3\,\mathbb E[\sigma^4]/(\mathbb E[\sigma^2])^2$
and Jensen. So:

> Among Gaussian scale mixtures, **the Gaussian is the minimum-kurtosis member**,
> and the measurements say minimum kurtosis is the worst case for the filter.
> The Gaussian is therefore the least favourable shape *within the family the
> filter's own model generates.*

That is a genuine closure and I did not expect it. The class over which the
filter is (conjecturally) minimax is exactly the class it can represent —
neither larger nor smaller. Nothing is left over and nothing is missing, which
is the "extent of the theory equals the extent of the problem" feeling, with a
reason attached.

**And the reason is that shape and scale are the same coordinate.** An i.i.d.
scale variation *is* excess kurtosis; a persistent scale variation *is*
heteroscedasticity. They are the same object read at two time constants, and
$\varphi$ is the dial between them — $\varphi\to0$ is shape, $\varphi\to1$ is
scale. So a heavy-tailed shape adversary is not outside the model at all: it is
inside it at $\varphi=0$, which is why $t_5$ *helps* the filter rather than
hurting it. This is `theory/06`'s persistence axis arriving from a completely
different direction, and it is the strongest independent support the $(a,\varphi)$
square has had.

The corresponding limitation is sharp and honest: **$\kappa<3$ is outside the
representable cone**, and there the filter is not minimax and loses real
accuracy — at $s_M=1.5$, 24% against the Gaussian row and 70% against the path
oracle. Bounded sensors, quantised readings and saturating instruments are all
light-tailed, so this is a practical caveat, not only a theoretical one. The
filter attributes a large innovation to a raised scale because under its model
that is the only thing a large innovation can be.

## 5. The revised class

$$\mathcal C(s,\varphi)=\Big\{\ \begin{array}{l}\theta_t=\theta_{t-1}+w_t,\quad x_t=\theta_t+v_t,\quad w_t,v_t \text{ independent, mean zero},\\
\text{each a Gaussian scale mixture: } w_t\sim N(0,Qe^{\lambda^P_t}),\ v_t\sim N(0,\sigma^2e^{\lambda^M_t}),\\
\lambda^c \text{ stationary with } \gamma_0^c=s_c^2,\ \gamma_1^c=\varphi_cs_c^2,\ c\in\{P,M\}\end{array}\Big\}$$

Compared with `0001` §3 the words "shapes otherwise arbitrary" are gone, and
what replaces them is not a restriction bolted on but the observation that the
scale-mixture representation *already* covers every shape with $\kappa\ge3$ —
including all the ones the battery probes (heavy tails, outliers,
heteroscedasticity). The class lost nothing it was using.

---

## 6. Correction to `0001` §5–§6

`0001` §5 says Burg's theorem gives the max-entropy law of the $\lambda$ path
and treats that as delivering a minimax property for the filter. **The
max-entropy step is right; the transfer to the observable is not established and
`0001` overstates it.** Burg constrains and maximises entropy over the law of
$\lambda$, which is latent. The log-loss the filter actually incurs is on $x$,
and $H(x)=H(x\mid\lambda)+I(x;\lambda)$ is not an affine functional of the
$\lambda$-law's constrained moments — so the equalizer property that makes
max-entropy laws minimax does not obviously push forward through the
marginalisation. Options: restate the constraints directly on the observable's
law; or work in the complete-data $(x,\lambda)$ space where the equalizer does
hold and treat the marginalisation as a separate step; or accept the Jaynes
reading (max-entropy as *not injecting unwarranted structure*) which is a
modelling principle rather than a theorem. Open.

Worth recording alongside it, because it is the encouraging half: **layers 1 and
2 have the same proof shape.** Theorem A works because the linear filter is an
*equalizer* across shapes (constant risk) and is exactly Bayes at one member;
weak duality then closes it in one line. The max-entropy argument works the same
way — $-\log p^*$ is affine in the constraint functionals, so $\mathbb E_p[-\log p^*]$
is constant across the class, and $p^*$ is exactly Bayes at itself. Equalizer
plus one exact member plus weak duality, twice. The two layers use different
losses but not different ideas, which is a partial answer to whether the theory
is one thing or two glued together: the *mechanism* is one thing; the loss is
not yet.

---

## 7. Next

1. **A minimum-kurtosis analogue of Theorem A.** Conjecture: over increments
   with variance and kurtosis prescribed, the filter's risk is monotone in
   kurtosis, so the least favourable member of $\{\kappa\ge3\}$ is the Gaussian.
   §3 is evidence; a proof would likely come from expanding the filter's
   posterior weight over the $\lambda$ grid to second order in the standardised
   fourth cumulant. This is the single most tractable open item and it would
   promote the main claim from measured to proved.
2. **Push the $\lambda$-layer argument to the observable** (§6). Harder, and the
   piece the whole layer-2 story rests on.
3. Unchanged from `0001` §9: the I-MMSE weld, and the $\alpha$-stable family.
