# 0011 — Levels not differences, a testable offset, and the drift shape

Three probes since [`0007`](0007_what_the_probes_settle.md):
[`0008`](0008_anisotropy_at_p2.py) (the anisotropy test at $p=2$),
[`0009`](0009_instrument_the_differences.py) (differenced instruments, and
whether the offset root is real), [`0010`](0010_iso_profile_check.py) (a
robustness check on the sharpest number in `0008`).

One of my own proposals is withdrawn, one question is closed positively, and the
Čencov proposal survives — but only under the optimality notion this repository
had already committed to.

---

## 1. Differencing costs a factor $(1-\rho_1)$ in SNR, and that is why levels win

[`0007`](0007_what_the_probes_settle.md) §2 explained IV's degradation under a
unit root as weak instruments — lagged levels of an integrated series being
dominated by the common trend — and proposed imposing the unit root and
instrumenting the *differenced* series. **Tested, it is worse at every noise
level** ([`0009`](0009_instrument_the_differences.py) part A, fig07), recovering
the oscillator modulus $\rho$ (truth 0.9489):

| $\kappa$ | IV on levels | IV on differences |
|---|---|---|
| 0.10 | 0.947 | 0.933 |
| 0.25 | 0.940 | 0.865 |
| 0.50 | 0.926 | 0.713 |
| 1.00 | 0.859 | 0.438 (23/40 fits lose the complex pair) |

The explanation and the repair are both withdrawn. What is true instead is
elementary and exact. For any stationary $x$ with lag-1 autocorrelation
$\rho_1$, observed as $y = x+v$ with $v$ white:

$$\mathrm{Var}(\Delta x) = 2\gamma_0(1-\rho_1),\qquad \mathrm{Var}(\Delta v)=2\sigma^2
\qquad\Longrightarrow\qquad
\boxed{\ \mathrm{SNR}_{\Delta} = \mathrm{SNR}_{\text{level}}\cdot(1-\rho_1)\ }$$

**Differencing always costs a factor $(1-\rho_1)$ in signal-to-noise.** For the
oscillator here $\rho_1 = \alpha_1/(1-\alpha_2) = 0.9394$, so the cost is
$16.5\times$ — verified against simulation, $\mathrm{Var}(\Delta x)/2\gamma_0 =
0.06070$ against a predicted $1-\rho_1 = 0.06057$.

The contraction worth keeping: **the parent workstream's "work in increments"
and this workstream's "work in levels" are the same principle at different
smoothness.** Differencing pays only when it removes a nonstationarity that
would otherwise dominate, and it always charges $(1-\rho_1)$. For a random walk
both sides diverge together — $\gamma_0\to\infty$ and $\rho_1\to1$ — and the
trade is exactly neutral, which is why the parent could difference for free. A
second-order ODE with a lightly damped mode is smooth, $\rho_1$ is near 1, and
the charge is real. **Smoothness is what makes the two workstreams differ, and
this is where it first bites.**

## 2. "Is the offset constant?" is answerable, by the machinery already present

Exact Gaussian ML with the offset root free (3 parameters) against pinned at
$z=1$ (2 parameters, via $\alpha_1 = 1+\beta_1$, $\alpha_2=\beta_2-\beta_1$,
$\alpha_3=-\beta_2$), $n=2000$, $\kappa=0.5$, 20 seeds
([`0009`](0009_instrument_the_differences.py) part B):

| true $z_0$ | $2\cdot$LLR | $\hat z_0$ (free) | $\lVert\Delta\alpha\rVert$ free | pinned |
|---|---|---|---|---|
| 1.00 | $1.36 \pm 0.44$ | $0.9992 \pm 0.0005$ | 0.0214 | 0.0210 |
| 0.98 | $19.56 \pm 1.13$ | $0.9797 \pm 0.0011$ | 0.0187 | **0.0377** |

One extra parameter, so $\chi^2_1$ with mean 1.0 and 95th percentile 3.84. At a
true unit root the free fit buys $1.36$ — exactly the null expectation, and it
recovers $\hat z_0 = 0.9992$. At $z_0=0.98$ it buys $19.56$ and pins the root to
$0.9797\pm0.0011$. Pinning costs nothing when it is right and doubles the
coefficient error when it is wrong.

So the modelling commitment "the offset is exactly constant" is **not** an
assumption the filter has to make. It is a hypothesis the same marginal
likelihood can test, with no new machinery and no threshold — read the LLR, do
not compare it to a critical value.

## 3. The anisotropy is load-bearing, and it flips sign

[`0008`](0008_anisotropy_at_p2.py) is the test [`0007`](0007_what_the_probes_settle.md)
§6 said was needed: $p=2$, a damped oscillator, the smallest case in which the
Fisher metric has an anisotropy at all. Three drift laws, each with its own
$\nu$ chosen by marginal likelihood, on a 4356-node grid over the whole
stationarity triangle:

| | shape | volume |
|---|---|---|
| `iso` | $\Sigma \propto I$ | — |
| `fisher-shape` | $\Sigma\propto\tilde\Gamma^{-1}/\sqrt{\det}$ | matched to `iso` |
| `fisher-full` | $\Sigma\propto\tilde\Gamma^{-1}$ | warped |

Determinant-matching is what makes `iso` and `fisher-shape` differ **only** in
shape and orientation. $h=5$ forecast MSE, ratio to a static-$\alpha$ filter,
with the fraction of the static-to-oracle gap closed:

| scenario | `iso` | `fisher-shape` | `fisher-full` | oracle |
|---|---|---|---|---|
| no shift | 1.000 (—) | 1.000 (—) | 1.000 (—) | 0.992 |
| damping $0.95\to0.85$ | **1.000 (0%)** | **0.925 (70%)** | 0.975 (25%) | 0.893 |
| frequency $0.35\to0.55$ | **0.448 (89%)** | 0.487 (83%) | 0.462 (86%) | 0.380 |

Paired, `fisher-shape` against `iso` at $h=5$: $0.932\pm0.018$ ($t=-3.69$) on the
damping shift, $1.116\pm0.014$ ($t=+8.55$) on the frequency shift,
$1.0001$ ($t=+1.65$) with no shift.

Three readings.

**Unlike the $p=1$ warp, the anisotropy does something.** $\pm10\%$ at $|t|$ up
to 8.6, where the warp moved nothing at $|t|<0.7$. The load-bearing half of the
proposal is real. (At $h=20$ the same signs hold but the frequency cell weakens
to $t=+1.27$; $h=5$ is where it is clean.)

**It is not uniformly better.** Geometric mean of the two paired ratios is
$1.020$ — on average, a wash slightly favouring `iso`.

**~~But the worst case is decisive, and worst case is this repository's stated
notion.~~ WITHDRAWN — see [`0014`](0014_the_channel_and_the_withdrawal.md) §2.**
`iso` closes 0% of the gap on the damping shift and 89% on the frequency shift;
`fisher-shape` closes 70% and 83%, so the worst case over these three scenarios
is 70% against 0%. But three scenarios I chose are not a worst case.
[`0013`](0013_minimax_over_directions.py) sweeps 12 *directions* at two base
points and finds `iso` ahead on the median at both (0.716 vs 0.398 interior,
0.648 vs 0.107 near the boundary) and level or ahead on the worst case. The
70%-against-0% figure came from one scenario that happened to be a direction
where `iso` chose $\nu^\ast=0$. The paragraph below stands as written but its
conclusion does not.
[`optimality-proof`](../optimality-proof/SUMMARY.md) already
fixes the optimality notion as minimax, "since the premise is that no prior over
the class is available" — and a scenario-averaged comparison is exactly a prior
over the class. *Which direction the truth moves in* is precisely the
information the class refuses to supply, so the average over the three scenarios
I happened to choose is not evidence; the worst case is.

Note also that `fisher-full` sits between `iso` and `fisher-shape` on both
shifts. The volume warp — the part refuted at $p=1$ — dilutes the shape effect
rather than adding to it. That is consistent across the two probes: **the
anisotropy is the content, the warp is not.**

## 4. The 0% is real, not a scan artifact

`iso` choosing $\nu^\ast=0$ on the damping shift carries §3's whole minimax
reading, and `0008`'s $\nu$ scan had six points. Profiled on 17 values over an
independent data batch ([`0010`](0010_iso_profile_check.py), fig08):

- **`iso` has no interior optimum at all.** Its log-likelihood is flat to
  $\nu=0.007$ and monotone decreasing after; the argmax is $\nu=0$. Even at its
  best *forecast* $\nu$ (0.010) it gains $0.2\%$.
- **`fisher-shape` has a clear interior optimum at $\nu = 0.0072$**, worth
  $4.8\%$, and the log-likelihood-optimal $\nu$ **coincides with the
  forecast-optimal $\nu$**.

That last point matters beyond this comparison. The parent workstream measured a
real seam between log-loss and squared error, and `optimality-proof` found
PEM unusable for a six-parameter fit. Here the two criteria agree on the same
$\nu$ to the resolution of the profile — one cell, but the right sign.

So: with an isotropic drift the damping direction is not adaptable at *any*
magnitude. This is a qualitative difference between the two laws, not a
percentage.

## 5. Where the Čencov proposal now stands

$\Sigma_{\text{drift}} \propto Q\,\Gamma^{-1}$ is the **unique**
reparameterisation-invariant drift law — under a linear reparameterisation
$\alpha\to M\alpha$ a covariance transforms as $\Sigma\to M\Sigma M^{\top}$, and
$I^{-1}$ is the only natural object that does. `iso` is not invariant: it is an
artifact of writing the recurrence in lag-coefficient coordinates.

| claim | status |
|---|---|
| reproduces the parent's log-scale law | **yes**, analytically ($I(\log\sigma^2)=\tfrac12$) |
| the volume warp helps | **no** — null at $p=1$, dilutive at $p=2$ |
| the anisotropy matters | **yes** — $\pm10\%$, $\vert t\vert$ to 8.6 |
| uniformly better than isotropic | **no** — sign flips with shift direction |
| better in the worst case | ~~yes~~ **no** — withdrawn, [`0014`](0014_the_channel_and_the_withdrawal.md) §2 |
| adaptivity free when unneeded | **yes** at $p=2$ as at $p=1$ (ratio 1.000) |

The honest summary: **invariance does not by itself produce a dominant drift
law, and no drift law can be dominant, because dominance would require knowing
which direction the dynamics move — which is the one thing the class does not
say.** That much stands. The claim that invariance nevertheless buys the worst
case did not survive a proper sweep over directions — see
[`0014`](0014_the_channel_and_the_withdrawal.md) §2 for the measurement and §3
for why the parent's invariance argument works where this one does not.

## Next, in order

1. **Persistence for the dynamics channel.** Still the biggest hole. Everything
   so far is a pure random walk in $\alpha$; the parent's second number per
   channel — impulsive versus persistent — has no analogue, and it is exactly
   the trust/belief split the previous construction lacked. Concretely: does
   "the dynamics wobbled for one step" separate from "the dynamics changed"?
2. **A minimax statement to match the minimax claim.** §3 leans on a worst case
   over three hand-chosen scenarios. The proper object is the worst case over a
   *direction* of parameter movement — a one-parameter sweep of the shift
   direction in $\alpha$-space, with the drift laws compared at each angle.
   That is a small extension of `0008` and would turn a suggestive ordering into
   a measured minimax.
3. **$p=3$: the full target class**, offset plus oscillator, with the drift law
   §3 selects. The grid cost is the compute budget; `0008`'s architecture
   extends directly.
4. **Learn $Q$ and $\sigma^2$ jointly with $\alpha$.** Every probe so far has
   held them at truth. The parent's `fit()` shows the six-parameter search is
   the hard part, and adding $\alpha$ and $\nu$ makes eight.
5. **Order selection** — whether $p$ itself can come from the same marginal
   likelihood, as the offset root now does (§2).
