# 0012 — The log-loss / squared-error seam does not measure, and the KL-projection beats the truth

Scripts: `0010_is_the_KL_projection_MSE_optimal.py` (surface scan),
`0011_seam_size_paired.py` (the contrasts, paired, 30 seeds).

This closes the question opened in `0001` §6 and sharpened in
[`0009`](0009_fit_targets_the_KL_projection_not_the_class.md): `fit()` targets
log-loss, the filter is used for squared error, and those are different criteria.
**Empirically they are the same criterion to within the noise.**

> **⚖️ ATTRIBUTION —** _Measured result: the log-loss (ML) parameter choice is within $+0.23\%$ MSE of the squared-error optimum, and the KL-projection beats the true parameters under misspecification._ No new theory — it is a paired empirical comparison of two estimation criteria (ML vs MSE-optimal) on the specific SV rig, plus a measured GPB1 signature ($+0.65\%$ even when well-specified). The rig-specific numbers are the contribution. Status: NEGATIVE-RESULT (a measured "the seam does not bite here" finding).

## The contrasts

$Q$ and $\sigma^2$ held at truth, $(s_M,\varphi_M)$ varied, 30 seeds, $n=1200$.
All differences are **paired** — the same series filtered with different
parameters — so the standard error of the difference is much smaller than that
of either MSE. Truth is $(0.55,0.93)$.

**Gaussian data** (well specified — the truth is in the family):

| parameters | MSE | vs best | se | $t$ |
|---|---|---|---|---|
| MSE argmin $(0.45,0.88)$ | 0.21487 | — | — | — |
| loglik argmax $(0.60,0.88)$ | 0.21512 | +0.12% | 0.12 | **1.0** |
| the truth $(0.55,0.93)$ | 0.21626 | +0.65% | 0.15 | 4.2 |
| homoscedastic $(s_M=0)$ | 0.21933 | +2.08% | 0.40 | 5.3 |

**Student-$t_5$ data** (misspecified — the truth's log-scale is not Gaussian):

| parameters | MSE | vs best | se | $t$ |
|---|---|---|---|---|
| MSE argmin $(0.75,0.15)$ | 0.19605 | — | — | — |
| **ML / KL-projection** $(0.90,0.45)$ | 0.19651 | **+0.23%** | 0.21 | **1.1** |
| loglik argmax $(0.90,0.75)$ | 0.19741 | +0.69% | 0.38 | 1.8 |
| moment-matched $(1.20,0.15)$ | 0.19944 | +1.73% | 0.50 | 3.5 |
| **the truth** $(0.55,0.93)$ | 0.20778 | **+5.98%** | 0.94 | 6.4 |
| homoscedastic $(s_M=0)$ | 0.21910 | +11.76% | 1.26 | 9.3 |

## Four readings

**1. The seam does not measure.** Where `fit()` actually lands — the
KL-projection — is $+0.23\%\pm0.21$ from the MSE-optimal point, $t=1.1$:
indistinguishable. On well-specified data the same contrast is
$+0.12\%\pm0.12$. **Maximising likelihood costs the filter nothing detectable,
under misspecification as well as without it.** `0001` §6 treated the two losses
as the seam in an otherwise single logic; the seam is real in principle and
below a quarter of a percent in practice.

This is the strongest available support for the intuition that the theory is one
self-contained thing rather than two ideas glued. It is evidence, not proof —
one process class, one misspecification direction, one sample size — but it is
the relevant evidence and it points one way.

**2. The KL-projection beats the true parameters, decisively.** Running the
filter at the *correct* $(s_M,\varphi_M)$ costs $+5.98\%\pm0.94$, $t=6.4$,
against the point `fit()` finds. So the "shortfall" of
[`0007`](0007_relocation_confirmed_but_ML_undershoots.md) was **the filter doing
the right thing**, not a defect: the truth's parameters describe a model the
filter cannot run, and the KL-projection describes the best model it can. Under
misspecification, landing away from the truth is correct.

That retires the framing of `0007` §"What is not confirmed" and much of
`0009`'s worry. Leak 3 remains open as a *theoretical* matter — quasi-MLE
asymptotics are not standard ML asymptotics — but its practical sign is
favourable rather than adverse.

**3. The moment-matched prediction is genuinely worse, not merely imprecise.**
$(1.20,0.15)$ costs $+1.73\%\pm0.50$, $t=3.5$. The formula in `0005` §3 did not
just miss the ML point; it pointed at an inferior one. Worth recording because
the formula is intuitive enough to be tempting — matching $\gamma_0$ and
$\gamma_1$ is the natural thing to want, and it is not what you want.

**4. Even well specified, the true parameters are not MSE-optimal**
($+0.65\%$, $t=4.2$). Small, but clearly resolved at 30 seeds. This is a
measured signature of Leak 4, the GPB1 collapse: because the level posterior is
forced to a single Gaussian each step, the best *filter* parameters are not the
generating ones, and a slightly lower $s_M$ compensates. **"Recover the true
parameters" is not the right target even in the well-specified case**, which
undercuts a natural way of judging `fit()` and is worth knowing before anyone
tunes against it.

**5. The adaptive machinery earns its keep**, as a by-product: pinning
$s_M=0$ costs 2.08% on heteroscedastic Gaussian data and 11.76% on $t_5$. The
larger figure is the point of `0005` §4 restated — a heavy tail *is* scale
variation, so the machinery that handles one handles the other.

## Where the argument now stands

| gap | status |
|---|---|
| Leak 1 — shape adversary | reduced to a relocation within the class; the residual is Leak 3 |
| Leak 2 — two losses | **empirically benign**, $+0.23\%\pm0.21$; theoretical transfer still unproved |
| Leak 3 — parameters estimated | reframed as quasi-MLE under misspecification; practical sign favourable |
| Leak 4 — GPB1 | now has a measured signature: +0.65% even when correctly specified |
| $\kappa<3$ | genuine, outside the model, unchanged |

The remaining *theoretical* work is unchanged and is now better isolated:
pushing the layer-2 max-entropy argument through the marginalisation to the
observable (`0005` §6), and a minimum-kurtosis analogue of Theorem A. Both are
about proving things that the measurements already indicate.

## Caveats on these numbers

- Grid resolution. The argmins come from `0010`'s coarse grid, so "the MSE
  argmin" is the best *grid point*, not the true optimum; the differences quoted
  against it are therefore slight over-estimates of everyone's penalty.
- One misspecification direction ($t_5$, i.e. inverse-gamma mixing) and one
  sample size. The seam could be larger elsewhere; nothing here bounds it.
- $Q$ and $\sigma^2$ were held at truth to isolate $(s_M,\varphi_M)$. A full
  six-parameter version could behave differently, though `0003`'s fitted rows
  suggest not.
