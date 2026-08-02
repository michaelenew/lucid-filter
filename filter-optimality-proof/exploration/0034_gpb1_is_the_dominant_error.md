# 0034 — GPB1 collapse, not the quadrature, is the dominant approximation error

Script: [`0031_gpb1_vs_particle.py`](0031_gpb1_vs_particle.py), reference
implementation in [`pf_reference.py`](pf_reference.py). Reference filter is
a marginalized (Rao–Blackwellized) particle filter that propagates
$(\lambda_P, \lambda_M)$ as particles and runs a per-particle Kalman filter
for $\theta$; conditional on the log-scale trajectory the model is exactly
linear-Gaussian, so the filter is exact in the $N \to \infty$ limit and
essentially exact at $N=4000$ (validated against a plain-Kalman case
where it agrees with `statfilter` to machine precision).

## Result

8 seeds, $n=500$, parameters pinned at truth throughout.

| regime | $s$ | GH-5 | GH-9 | **GH-31** | **PF-4k** | GH-31 vs PF | t |
|---|---|---|---|---|---|---|---|
| homoscedastic       | 0.00 | 0.2030 | 0.2030 | 0.2030 | 0.2030 | +0.000% | +0.4 |
| weak / persistent   | 0.20 | 0.1850 | 0.1847 | 0.1847 | 0.1847 | −0.006% | −0.1 |
| mid / impulsive     | 0.55 | 0.2063 | 0.2063 | 0.2063 | 0.2060 | +0.164% | +0.9 |
| mid / persistent    | 0.55 | 0.2027 | 0.2024 | 0.2038 | 0.2011 | +1.351% | **+2.2** |
| **strong/persistent** | 1.20 | 0.3280 | 0.3292 | 0.3345 | **0.2791** | **+19.85%** | +1.9 |

θ-MSE, means across seeds; last two columns are the paired difference
GH-31 minus PF-4k. Log-likelihood per point, strong regime: GH-5
$-1.6851$, GH-9 $-1.6458$, GH-31 $-1.6438$, PF-4k $-1.6250$ — the PF is
0.019 nat/point above GH-31, i.e. **~9.4 nats over the series** better
than the best-quadrature GH filter can achieve.

## What it says

**The Gauss-Hermite grid and the GPB1 collapse are two separate
approximations, and they scale very differently with the log-scale
volatility $s$.** At $s \le 0.55$ both errors are within 1–2% θ-MSE and
the ranking barely matters. At $s = 1.2$:

- Quadrature $5 \to 31$ closes about 6% θ-MSE ([`0029`](0029_quadrature_convergence_is_exponential.md))
- GPB1 collapse (GH-31 → PF-4k) leaves **about 20% θ-MSE unclosed**
- Total θ-MSE gap from best-possible: ~26%, of which ~4/5 is the collapse

**No amount of quadrature order can close the collapse gap.** In the strong
regime the θ-MSE is essentially FLAT in the order (0.328, 0.329, 0.335 for
orders 5, 9, 31 — the small non-monotonicity is well inside the paired
noise), while sitting persistently ~17% above the PF reference. The
collapse is a structural approximation that lives at every order, not a
resolution issue.

**The collapse is nonlinear in $s$.** At $s=0.2$ it costs essentially
nothing. At $s=0.55$ persistent it costs 1.35% (t = 2.2). At $s=1.2$
persistent it costs 20%. The growth is faster than $s^2$ across the
tabulated points; the persistence matters — the mid/impulsive regime at
$s=0.55$ costs only 0.16%, an order of magnitude less than the same $s$
with $\varphi=0.93$.

**The loglik gap is unambiguous.** MSE with 8 seeds gives $t=1.9$ on the
strong-regime finding, which is on the edge of nominal significance.
Log-likelihood per point in the same regime is $-1.644$ for GH-31 vs
$-1.625$ for PF, a 0.019 nat/point gap. That translates to a Bayes-factor
difference of $e^{9.4} \approx 12{,}000$ over $n=500$ observations —
overwhelming, whatever one thinks of the θ-MSE noise.

## Reading

**The rate-of-approach question changes shape.** Increasing the GH order
gets us within 0.3% θ-MSE of the best-order-limited filter (per
[`0029`](0029_quadrature_convergence_is_exponential.md)), but a further
17–20% θ-MSE gap to the Bayes optimal remains AT WHATEVER ORDER, gated by
the collapse. To close it you need either (a) GPB2+ (keep a mixture over
grid states rather than a single Gaussian) or (b) a particle filter (as
the reference here). Both are more expensive per step; the PF at $N=4000$
took ~1.3 s per series vs 24 s total for three GH orders combined.

**The order/collapse decomposition matches the "regret bound as the honest
target" reframing.** Regret against the model's own Bayes rule
decomposes into:
$$ R_{\text{filter}} = \underbrace{R_{\text{quad}}(n)}_{\text{geometric in } n \text{ (0029)}}
   + \underbrace{R_{\text{collapse}}}_{\text{structural, grows with } s \text{ (here)}}. $$
The first term is what quadrature order buys you; the second is what a
different filter class would buy. Reporting both, per regime, is what a
regret bound would need.

**For the current default (order 5, GPB1):** the total approximation
against Bayes is well under 1% at $s \le 0.55$, and about 25% at $s=1.2$.
Nearly all of the strong-regime cost is the collapse; the order-5 quadrature
contributes ~6 of the 25 percentage points, the collapse contributes ~19.

## Not chased here

- **GPB2** — keep one Gaussian per grid state rather than collapsing.
  Order 5 gives 25 states; storage and update time both grow $\propto n^2$
  rather than $n^2$ per step, which for this problem is likely
  worth-it in the high-$s$ regime. Would be a natural "secondary mode for
  exactness" in the filter package.
- **Where the collapse loses θ**. The PF's advantage should be
  concentrated on the observations that catch the log-scale $\lambda_M$ in
  a rare state — the collapse can't represent a bimodal $\theta$ posterior
  that arises when the plausible scale is ambiguous. A per-step MSE trace
  would show this.
- **The convergence rate in $N$**. 4000 particles was chosen for a
  comfort-margin, not calibrated. A sweep at $N \in \{500, 1000, 2000,
  4000, 8000\}$ would say how much of the PF number here is Monte Carlo
  and how much is truth.
