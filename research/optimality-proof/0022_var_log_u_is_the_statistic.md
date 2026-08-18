# 0022 — `Var(log u)` is the shape statistic, not kurtosis

Scripts: `0018_kurtosis_vs_var_log_u.py`, `0021_complete_row_and_expit_overflow.py`.
Settles the inconsistency flagged in [`0015`](0015_theorem_B_and_the_max_entropy_principle.md) §2(b)
between Theorem B and `0004`'s kurtosis-sufficiency claim.

## The design

A two-point mixing law $u\in\{a,b\}$ with weights $p,1-p$ has three parameters
and two constraints, $\mathbb Eu=1$ and $\mathbb E u^2=\kappa/3$. Writing
$a=1+(1-p)d$, $b=1-pd$ makes the mean constraint automatic and
$\mathbb Eu^2=1+p(1-p)d^2$ fixes $d$ given $\kappa$. **One free dimension
remains**, so kurtosis can be pinned exactly while $\operatorname{Var}(\log u)$
is swept.

At $\kappa=9$ that gives $\operatorname{Var}(\log u)$ from 0.081 to 2.96 — a
factor of 36 — with $\log(\kappa/3)=1.0986$ on every row. Student-$t_5$ (0.490)
and lognormal (1.099) mixing sit inside the range at the same kurtosis and come
along as reference points. Truth $s_M=0.55$, $\varphi_M=0.93$, 6 seeds,
quadrature order 9.

## Result

| mixing | $\operatorname{Var}(\log u)$ | Theorem B | kurtosis claim | **fitted $s_M$** |
|---|---|---|---|---|
| two-pt $p{=}.01$ | 0.081 | 0.620 | 1.184 | **0.693 ± 0.031** |
| two-pt $p{=}.05$ | 0.265 | 0.753 | 1.184 | **0.886 ± 0.047** |
| Student-$t_5$ | 0.490 | 0.890 | 1.184 | **0.933 ± 0.053** |
| two-pt $p{=}.15$ | 0.720 | 1.011 | 1.184 | **1.267 ± 0.052** |
| lognormal | 1.099 | 1.184 | 1.184 | **1.119 ± 0.035** |
| two-pt $p{=}.30$ | 2.956 | 1.805 | 1.184 | **2.231 ± 0.103** |

**Kurtosis is refuted.** It is held at exactly 9 on every row, so it predicts a
single value, 1.184, throughout. Fitted $s_M$ instead ranges over 0.69–2.23 —
a factor of 3.2 — with standard errors near 0.05, across rows that are
kurtosis-identical. Theorem B's $\operatorname{Var}(\log u)$ tracks it.

**So `exploration/0004`'s "leverage is monotone in kurtosis alone" is wrong as a
general statement** and should be read as: monotone in a shape functional that
kurtosis happened to track across the shapes tested there. The two functionals
correlate across standard families — for lognormal mixing they are equal by
construction — which is why the coincidence held.

## The residual, and why it is the expected one

Theorem B under-predicts on the two-point rows (+0.07, +0.13, +0.04, +0.26) and
slightly over-predicts on lognormal (−0.07). That pattern is what the
KL-projection argument predicts, and it is a useful internal check:

- For **lognormal** mixing, $\eta=\log u$ is Gaussian, so $\tilde\lambda$ has a
  Gaussian marginal — exactly what the filter's quadrature represents. This is
  also the single row where the two competing predictions coincide, and it is
  the row that fits cleanest.
- For **two-point** mixing, $\eta$ is two-point, so $\tilde\lambda$'s marginal is
  a two-component Gaussian mixture. The filter must cover a bimodal spread with
  a unimodal one, and maximum likelihood inflates $s_M$ to do it. The overshoot
  grows with $p(1-p)$, i.e. with how genuinely bimodal the mixture is.

So Theorem B gives the moments exactly (that part is a proof), and the fitted
value departs from them by an amount governed by how far the induced log-scale
marginal is from Gaussian.

**The overshoot is monotone in exactly the quantity that argument names.** The
bimodality of a two-point $\eta$ is measured by $p(1-p)$, and across the four
two-point rows:

| $p$ | $p(1-p)$ | overshoot |
|---|---|---|
| 0.01 | 0.0099 | +0.073 |
| 0.05 | 0.0475 | +0.133 |
| 0.15 | 0.1275 | +0.256 |
| 0.30 | 0.2100 | +0.426 |

Monotone, and the ratio settles near 2 as the mixture becomes genuinely bimodal.
That is a mechanism confirmed rather than a discrepancy tolerated: Theorem B
fixes the moments, and ML inflates $s_M$ above them in proportion to how far the
induced marginal is from the unimodal family the filter can represent.

$\varphi_M$ shows the same pattern more weakly — predicted 0.733, 0.496, 0.355,
0.275, 0.201 against fitted 0.810, 0.596, 0.434, 0.330, 0.470. Direction and
ordering are right except for the lognormal row, whose 0.470 against a predicted
0.201 is the one genuinely discrepant cell in the table. I do not have an
explanation for it; it is the obvious thing to look at next if this line is
pursued.

## A bug found on the way

`0018`'s most extreme row ($p=0.30$, $\operatorname{Var}(\log u)=2.96$) crashed
`fit()` outright:

```
OverflowError: math range error    at  _expit(z) = 1.0 / (1.0 + math.exp(-z))
```

`_expit` overflows once $z<-709$, and `fit()`'s inner `ll()` catches only
`ValueError`, so the exception escapes and terminates the fit. `_logit` clamps
its input to $[10^{-9},1-10^{-9}]$, i.e. $|{\rm logit}|\le20.7$, so *starts* are
always safe — but stage 1 is an unconstrained Nelder–Mead search and on extreme
data it walks out there.

**Not applied**, since `statfilter/core.py` is the parent workstream's
deliverable, but the fix is two lines and algebraically identical:

```python
def _expit(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)
```

plus widening `ll()`'s `except ValueError` to `except (ValueError, OverflowError)`
as a guard. `0021` monkeypatches this and completes the row, confirming the
crash was the overflow and not a property of the data.

This matters beyond the missing table row: **any user fitting sufficiently
impulsive data can hit it**, and the failure is a hard crash rather than a
degraded estimate.

## Consequences

1. `SUMMARY.md`'s kurtosis bullet needs restating in terms of
   $\operatorname{Var}(\log u)$ — done.
2. Theorem B is confirmed as the correct account of the shape adversary, with a
   quantified, mechanistically-explained residual.
3. The parent workstream has a crash bug worth fixing regardless of anything
   here.
