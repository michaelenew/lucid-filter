# 0020 — The orientation test was impossible; the orientation is readable anyway

Three probes: [`0017`](0017_orientation_at_an_isotropic_metric.py) (the test
[`0016`](0016_bias_variance_and_the_shape_profile.md) §"Next" asked for),
[`0018`](0018_headroom_check.py) (the two controls `0017` was missing),
[`0019`](0019_alignment_law.py) (a direction sweep).

**Net: `0016`'s conclusion that the drift orientation is unreadable is
withdrawn. It is readable. The probe that suggested otherwise was measuring a
$\tau$ artifact, and the follow-up I designed to settle it could not have
worked.**

---

## 1. The test I designed cannot exist, and the reason is structural

For $p=2$ the information metric is
$\tilde\Gamma = \begin{psmallmatrix}\gamma_0&\gamma_1\\\gamma_1&\gamma_0\end{psmallmatrix}$,
with eigenvalues $\gamma_0\pm\gamma_1$ along $(1,\pm1)/\sqrt2$ and condition
number

$$\frac{1+\rho_1}{1-\rho_1},\qquad \rho_1 = \frac{\gamma_1}{\gamma_0} = \frac{\alpha_1}{1-\alpha_2}$$

**The metric's anisotropy is the process's lag-1 autocorrelation** — the same
$\rho_1$ that sets the differencing cost $(1-\rho_1)$ in
[`0011`](0011_the_drift_shape.md) §1. So an isotropic metric requires
$\rho_1=0$, which for a second-order system requires $\alpha_1=0$ exactly: poles
at $\pm i\rho$, four samples per period. **There is no such thing as an
isotropic information metric on a smooth process.**

That was stated as a caveat when the probe was written. It turned out to be
fatal. A controlled pair sharing $\rho$ and $\alpha_2$, differing only in
$\alpha_1$:

| | $\alpha$ | $\rho_1$ | metric cond. | $\gamma_0$ | top eigenvalue |
|---|---|---|---|---|---|
| A | $(0,\ -0.5625)$ | 0.000 | 1.00 | 1.46 | 1.46 |
| B | $(1.238,\ -0.5625)$ | 0.792 | 8.63 | 3.93 | 7.04 |

`0017` found no shape evidence at A (0.00 and 0.21 millinats/pt, at or below
`0015`'s 0.38 null floor) and read it as an identifiability result. It is not.
`0018` supplies the control `0017` omitted — a **static** baseline, which
`0017` never ran:

| condition | isotropic drift over static |
|---|---|
| A, $\psi_{\rm true}=0.5$ | $-0.57$ millinats/pt |
| A, $\psi_{\rm true}=1.5$ | $-0.43$ |
| A, isotropic control | $-0.25$ |
| **B**, $\psi_{\rm true}=1.5$ | $\mathbf{+3.58}$ |

**At base A, with the same kernels, allowing drift at all is worse than not
allowing it.** There is no headroom for the shape to improve on, so `0017`
measured nothing. Base A carries $4.8\times$ less information about its own
dynamics than base B (top eigenvalue 1.46 against 7.04), and the reason is the
same $\rho_1$: information about the dynamics comes from the state exploring,
and a rough process explores less per unit of variance.

> **A process must be smooth for its dynamics to be learnable, and smoothness is
> exactly what makes the information metric anisotropic.** The isotropic-metric
> limit is not a clean laboratory — it is the limit in which there is nothing to
> learn.

## 2. The orientation is readable; `0016`'s conclusion is withdrawn

[`0015`](0015_is_the_drift_shape_estimable.py) profiled at the argmax
$\hat\tau=8$ and returned $\hat\psi=1.571$ against a generating $0.9$, which
`0016` read as "orientation not recovered". Profiling $\psi$ at $\tau=4$
instead, over seven generating orientations at base B
([`0019`](0019_alignment_law.py), fig14):

| $\psi_{\rm true}$ | 0.262 | 0.785 | **0.900** | 1.309 | 1.833 | 2.356 | 2.880 |
|---|---|---|---|---|---|---|---|
| $\hat\psi$ | 0.524 | 0.524 | **1.047** | 1.571 | 2.094 | 2.618 | 2.618 |

**Every cell returns one of the two nearest kernel nodes**, and the one cell
that is a clean test — $\psi_{\rm true}=0.900$ — returns the nearest node
exactly. That is the cell `0015` got wrong, so **`0015`'s miss was an artifact
of profiling at $\tau=8$, not a failure of orientation.**

**A design flaw, stated because it bounds the strength of this result.** Six of
the seven test angles are odd multiples of $\pi/12$ against a $\pi/6$ kernel
grid, i.e. **exact midpoints between nodes**, so for those six there is no
"correct" node and the most that can be asked is that the argmax be adjacent
rather than distant. All six are adjacent; under a uniform argmax over six nodes
that has probability $(1/3)^6\approx0.0014$. Real evidence, but a rerun with
truths placed on or near the nodes would be worth more than the arithmetic. That
is the first thing to redo.

## 3. The alignment law I proposed is refuted

`0018`'s numbers suggested that drift is worth modelling in proportion to its
alignment with the metric's high-information axis ($\psi=0.785$ here). `0019`
tests it. Headroom (isotropic drift over static, millinats/point):

| $\psi_{\rm true}$ | 0.262 | 0.785 | 0.900 | 1.309 | 1.833 | 2.356 | 2.880 |
|---|---|---|---|---|---|---|---|
| angle to hi-info axis | 0.523 | **0.000** | 0.115 | **0.524** | **1.048** | 1.571 | **1.047** |
| headroom | $+1.89$ | $+4.76$ | $+13.00$ | $+17.65$ | $+1.80$ | $-1.31$ | $+0.37$ |

Direction matters enormously — a $14\times$ range with a sign change. But it is
**not** a function of the angle to the principal axis: the pair at angle
$0.523$ / $0.524$ gives $+1.89$ against $+17.65$, a $9.3\times$ difference at
the same angle, and the pair at $1.047$ / $1.048$ gives $+0.37$ against $+1.80$.
The overall correlation is $-0.599$, carried entirely by the trough. The law is
refuted; only the trough survives — the minimum is at $\psi=2.356$, which is
exactly the low-information axis, and it is the one direction where tracking the
drift is worse than not tracking it.

**The likely reason, and it is a flaw in the experiment rather than a finding.**
Each condition holds the *Euclidean* drift magnitude $\lVert\Delta\alpha\rVert$
fixed across directions. But moving $\alpha_2$ changes $\rho$ and therefore
$\gamma_0$ sharply, while moving $\alpha_1$ changes $\theta$ at fixed $\rho$ —
so equal Euclidean steps are wildly unequal changes to the process, and the
conditions differ in signal variance and in fitted $\sigma^2$ as well as in
direction. Some unknown part of the $14\times$ spread is that.

Which points at the fix, and it closes a loop:

> $\lVert\Delta\alpha\rVert$ is not a meaningful measure of "how much the
> dynamics changed". The natural measure is the **information distance**, i.e.
> $\Delta\alpha^{\top}\tilde\Gamma\,\Delta\alpha$. The direction sweep has to
> hold the *Fisher* length constant, not the Euclidean one.

So the Fisher metric returns in a third role. [`0014`](0014_the_channel_and_the_withdrawal.md)
refuted it as a law for how $\alpha$ *moves*. Here it is not that either. It is
the right way to *measure* how far $\alpha$ has moved — which is the one thing a
metric is actually for, and the one use of it that no experiment here has
contradicted.

## Where the shape question now stands

| | status |
|---|---|
| anisotropy magnitude $\tau$ readable | **yes** — 6.69 mnats/pt against a 0.38 floor ([`0015`](0015_is_the_drift_shape_estimable.py)) |
| orientation $\psi$ readable | **yes** — `0016`'s withdrawal; 7/7 land adjacent, the one clean cell exact |
| $\hat\tau$ unbiased | **no** — 8 against a truth of 4, at the top of the tested range |
| worth learning operationally | **not shown** — forecast-MSE ratios 0.994–1.003, i.e. nothing at $h=5$ |
| drift value depends on direction | **yes, strongly** — $14\times$ and a sign change |
| ...explained by metric alignment | **no** — refuted, and partly confounded by design |

The interim position from `0016` — learn $\tau$, hold $\psi$ isotropic — no
longer follows, since $\psi$ is readable. But nor does "learn both", because
neither has yet shown an operational gain. **The defensible statement is that
the drift shape is estimable and not yet shown to be worth estimating**, and the
next probes are about the second half of that, not the first.

## Next, in order

1. **Redo the direction sweep at constant Fisher length**, with the generating
   orientations on the kernel nodes rather than between them. This fixes both
   §2's design flaw and §3's confound, and it is the same script with two lines
   changed.
2. **Score the shape on prequential log-loss out of sample.** Every shape number
   here is in-sample with more members to maximise over; the 0.38 null floor
   bounds that but does not remove it. Per
   [`0016`](0016_bias_variance_and_the_shape_profile.md) §1 this is also the
   only loss that can see a variance-side gain, and the forecast-MSE ratios
   near 1.000 are exactly what a variance-side gain looks like under MSE.
3. **$p=3$: the full target class.** Unblocked by any of this; the architecture
   extends directly and the grid is the compute budget.
4. **Learn $Q$ and $\sigma^2$ jointly with $\alpha$.** Still held at truth
   everywhere.
5. **Order selection**, the way the offset root became testable in
   [`0011`](0011_the_drift_shape.md) §2.
