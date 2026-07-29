# 0024 — The seam is structural in the class, benign in the parameters, and the class is not well-posed under log-loss

Scripts: `0019_is_the_seam_structural.py`, `0023_does_the_equalizer_survive_marginalisation.py`.
Together these answer the question of whether the filter's practical success rests
on the two losses happening to agree in the tested regimes.

**Short answer: no, but not because they agree.** They agree closely on *which
parameters to pick* and disagree *in sign* on *which class member is worst*. The
filter's success survives because every individual gap is small; the proof does
not survive, because no single minimax statement covers both losses.

## 1. In parameter space, the seam is benign — and PEM is uniformly better

Eleven regimes spanning weak/strong scale variation, impulsive/persistent
log-scale and slow/fast level. $Q,\sigma^2$ at truth, scan over
$(s_M,\varphi_M)$, 12 seeds, penalties paired.

| regime | ML | pen | PEM | pen | BEST |
|---|---|---|---|---|---|
| weak / impulsive | (0.10,0.00) | +0.38% | (0.30,0.60) | +0.02% | (0.30,0.30) |
| weak / mid | (0.10,0.60) | +0.32% | (0.30,0.60) | 0.00% | (0.30,0.60) |
| weak / persistent | (0.10,0.85) | +0.23% | (0.30,0.60) | 0.00% | (0.30,0.60) |
| mid / impulsive | (0.55,0.00) | +0.27% | (0.80,0.60) | +0.14% | (0.80,0.30) |
| mid / mid | (0.55,0.60) | +0.07% | (0.80,0.60) | 0.00% | (0.80,0.60) |
| mid / persistent | (0.55,0.85) | 0.00% | (0.55,0.85) | 0.00% | (0.55,0.85) |
| strong / impulsive | (1.10,0.00) | +0.29% | (1.50,0.30) | +0.17% | (1.50,0.00) |
| strong / mid | (1.10,0.60) | 0.00% | (1.10,0.60) | 0.00% | (1.10,0.60) |
| strong / persistent | (1.10,0.85) | +0.27% | (0.80,0.85) | 0.00% | (0.80,0.85) |
| mid/pers, slow level | (0.55,0.85) | 0.00% | (0.55,0.85) | 0.00% | (0.55,0.85) |
| mid/pers, fast level | (0.55,0.85) | 0.00% | (0.55,0.85) | 0.00% | (0.55,0.85) |

**The ML penalty never exceeds 0.38%**, and in five regimes it is exactly zero.
There is no corner where the hybrid construction costs anything material. So the
filter's practical success is not a lucky coincidence of regime.

**But PEM — minimising squared one-step innovation, an entirely observable
criterion — is never worse than ML and sometimes better** (max penalty 0.17% vs
0.38%; it matches BEST exactly in eight of eleven regimes). So the
"commit fully to MSE" option is implementable and marginally superior. The gain
is real and systematic but small enough that it is a refinement, not a redesign.

Caveats: the $(s_M,\varphi_M)$ grid is coarse, so single-grid-step differences
are at the resolution limit; several penalties are comparable to their standard
errors; and $Q,\sigma^2$ were pinned, so this is a two-parameter slice of a
six-parameter question.

## 2. In class geometry, the two losses rank the class in *opposite order*

This is the finding that matters. `0023` sweeps the same AR(2) family as `0017`,
in the same regime ($s=0.8$, $\varphi=0.5$, max-entropy at $\rho_2=0.25$), and
records observable **code length** rather than MSE.

| $\rho_2$ | code length vs $p^\ast$ | $t$ | | raw MSE vs $p^\ast$ (`0017`) |
|---|---|---|---|---|
| 0.05 | **+0.164%** | 4.5 | | — |
| 0.15 | **+0.084%** | 4.4 | | — |
| **0.25** | 0.000% | — | | 0.00% |
| 0.40 | −0.141% | −4.2 | | — |
| 0.65 | −0.459% | −3.8 | | +2.16% |
| 0.90 | −1.142% | −3.1 | | +4.58% |
| −0.44 | — | | | −1.62% |

**Code length falls monotonically in $\rho_2$; raw MSE rises monotonically in
$\rho_2$.** Same class, same regime, same members. Under log-loss the worst
member is at *low* $\rho_2$; under squared error it is at *high* $\rho_2$. All
the code-length contrasts resolve at $|t|=3$–4.5.

Two consequences:

- **The equalizer does not survive marginalisation.** Theorem C guarantees the
  *latent* code length is identical across every one of these rows. The
  *observable* code length varies by 1.3% and monotonically. So `output/02`'s
  gap 2 **closes negatively**: Theorem C is a theorem about coding $\lambda$ and
  it does not reach $x$.
- **No reframing rescues it.** `0017` already showed max-entropy is not least
  favourable under MSE. `0023` now shows it is not least favourable under
  observable log-loss either — $\rho_2=0.05$ sits *above* $p^\ast$ at $t=4.5$.
  Committing to either loss leaves the max-entropy member interior.

Family A (same autocovariances, different $\lambda$ *marginal*) points the same
way but resolves weakly: two-point +0.084% ($t$=0.3), uniform +0.044% ($t$=0.2),
$t_5$ +3.1% ($t$=1.1), $t_3$ +463% ($t$=1.8). Every row sits at or above
$p^\ast$, as the convexity argument predicts — $H(x\mid\lambda)$ involves
$\log(a+\sigma^2e^\lambda)$, which is convex in $\lambda$, so heavier $\lambda$
marginals cost more at fixed variance. Directionally consistent with theory,
individually inconclusive; the variance is enormous precisely because the effect
is driven by rare huge variances.

## 3. The class is not well-posed under log-loss

The $t_3$ row is not just noisy — it is a symptom. The class constrains
$\operatorname{Var}(\lambda)=\gamma_0$ and nothing about $\lambda$'s marginal
shape. A stationary $\lambda$ with finite variance but polynomial tails has

$$\mathbb E[e^{\lambda}]=\infty,$$

so the **actual noise variance $\sigma^2e^\lambda$ has infinite mean** while
every constraint defining the class is satisfied. Such a member is admissible.

The two losses then differ on whether the minimax problem is even well-posed:

- Under **squared error**, this is harmless. A huge $R_t$ means the filter
  ignores $x_t$, and the loss is bounded by the prior variance. $\sup_p$ stays
  finite.
- Under **log-loss**, it is fatal. $-\log m^\ast(x)$ grows like $x^2$, and
  $\mathbb E[x^2]=\infty$, so $\sup_{p\in\mathcal C}\mathbb E_p[-\log m^\ast]=\infty$.
  The minimax value is infinite and the problem is vacuous.

So the class as written in `SUMMARY.md` — stationary $\lambda$ with prescribed
$\gamma_0,\gamma_1$, nothing else — **admits members with infinite expected noise
variance**, and needs a further constraint (finite $\mathbb E[e^\lambda]$, or a
tail bound on $\lambda$) that does not follow from "magnitude and persistence".
That constraint has been implicitly assumed throughout and never stated.

## 4. What this says about the whole programme

The obstruction is **not** the choice of loss. It is that the class is too big.

Fixing two moments of $\lambda$ leaves the risk — under *either* loss — varying
across members with the filter's model in the interior rather than at a maximum.
An equalizer exists only when the loss is affine in what the class fixes, which
is true for Theorem A (variance, squared error, linear rule) and for Theorem C
on the latent path (code length, affine in $\gamma_0,\gamma_1$), and false in
every other combination tried.

To get a minimax theorem for the actual filter one would have to **shrink the
class** — constrain $\gamma_2$ and beyond, e.g. to log-scales that are genuinely
Markov — which makes the class the AR(1) family and the theorem nearly
tautological. That is the tension flagged at the end of `0017` and it now looks
like the central obstacle rather than a loose end.

**The filter is fine.** Every measured gap is between 0.2% and 5%. What fails is
the ambition of an exact minimax characterisation over a two-moment class.

## Next

1. State the missing $\mathbb E[e^\lambda]<\infty$ constraint in the class
   definition and check what else it silently repairs.
2. If minimaxity is unreachable, the honest target is a **regret bound**: how far
   from the best member-specific filter can the AR(1) model be, over the class,
   under each loss? `0017` and `0023` already bound it empirically at ~5% and
   ~1.3% in the regimes measured. A proof of any such bound would be worth more
   than a minimax statement that requires assuming the answer.
3. PEM is worth a proper look as a *practical* matter (§1), independent of theory.
