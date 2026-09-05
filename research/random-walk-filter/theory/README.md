# The information-domain foundation

Started as exploration rather than a filter; it produced one anyway, in
[`statfilter/`](../../../lucid/statfilter/README.md). Files 01–06 are exact throughout
(closed form or exact linear algebra on the model's own covariance) — no
simulation, no fitted constants. 07 is where measurement begins.

Model, unchanged from the whole thread:

$$\theta_t = \theta_{t-1} + w_t,\ w\sim N(0,Q); \qquad x_t = \theta_t + v_t,\ v\sim N(0,\sigma^2); \qquad q = Q/\sigma^2$$

Working in increments $d_t = x_t - x_{t-1} = w_t + v_t - v_{t-1}$, which annihilates
the unknown level exactly. Under no change these are stationary MA(1):
$\gamma_0 = Q+2\sigma^2$, $\gamma_1 = -\sigma^2$, $\gamma_{k\ge2}=0$.

> **⚖️ ATTRIBUTION —** _Differencing the local-level model to annihilate the unknown level gives a stationary MA(1) on increments; this is the REML/marginal likelihood for the state-space model, and the increment autocovariances are the classic result._ Prior art: local-level/random-walk-plus-noise model has MA(1) first differences (Harvey 1989, *Forecasting, Structural Time Series*); REML/marginal likelihood (Patterson & Thompson 1971). Status: REPRODUCTION.

| file | question |
|---|---|
| [01-information-accounting.md](01-information-accounting.md) | how many nats does point $n$ carry about $(Q,\sigma^2)$? how good can 5/10/20 points be? |
| [02-relevance-decay-and-the-tail.md](02-relevance-decay-and-the-tail.md) | how fast does a past point stop mattering, and what actually sets the tail length? |
| [03-four-deviation-modes.md](03-four-deviation-modes.md) | the 4-mode geometry; what separates the modes and how fast |
| [04-nats-trust-influence.md](04-nats-trust-influence.md) | nats → trust → influence; a definition of *trustworthy* information |
| [05-open-questions.md](05-open-questions.md) | what this does not settle |
| [06-gradient-allocation.md](06-gradient-allocation.md) | **the reframe**: the modes as one smooth square, allocation without detection |
| [07-the-finished-computation.md](07-the-finished-computation.md) | **the result**: the scan cost closed, the allocator built and measured |

Scripts: `scripts/THEORY-0{01..10}-*.py`. Figures: `figures/fig*.png` (24) plus
raw numbers in `figures/theory00*.json`. The filter that came out of it is
[`statfilter/`](../../../lucid/statfilter/README.md).

---

## The nine results

**1. Per-point information about the noise parameters decays as exactly $1/n$.**
$\Delta_n = \tfrac12\log\det I_n - \tfrac12\log\det I_{n-1} \to d/2n = 1/n$ nats.
Hyperbolic, so the sum diverges logarithmically — there is no $n$ at which the
next point stops being worth having.

> **⚖️ ATTRIBUTION —** _Per-point Fisher information about a fixed d-dimensional parameter decays as d/2n, so the entropy-reduction sum diverges logarithmically._ Prior art: standard asymptotics of Fisher information accumulation for i.i.d.-like data (Cramér–Rao / Fisher; textbook). Status: REPRODUCTION.

**2. The level channel never decays.** It contributes $\tfrac12\log\frac1{1-K}$
nats *per point, forever*. The two channels have incompatible memory laws, which
is why one truncation length can't serve both. (fig01, fig05)

> **⚖️ ATTRIBUTION —** _At Kalman steady state each observation reduces the level posterior variance by factor (1-K), contributing ½log(1/(1-K)) nats forever; contrasting this constant rate with the 1/n parameter-info decay is a clean framing of a standard fact._ Prior art: steady-state Kalman filter and the algebraic Riccati equation (Kalman 1960; Anderson & Moore, *Optimal Filtering* 1979). Status: REPRODUCTION.

**3. "95% in 20 points" is false for $Q$, and it is a Cramér–Rao fact, not an
engineering failure.** At $n=20$ the best possible relative SD of $\hat Q$ is
**92%** at $q$=0.5 and **846%** at $q$=0.005. $\sigma^2$ is much easier (35–52%).
No estimator does better. (fig02)

> **⚖️ ATTRIBUTION —** _The "95% accuracy in 20 points" target is impossible for Q as a Cramér–Rao lower bound, not an estimator failure — a correct application of the CRLB to this model's Fisher information._ Prior art: Cramér–Rao bound (Cramér 1946; Rao 1945). Status: REPRODUCTION.

**4. But the *decision* is far cheaper than the parameter.** Excess tracking MSE
from not knowing the noises falls below 5% at $n\approx$ **100 / 150 / 500** for
$q$ = 0.5 / 0.05 / 0.005. So the 2000-point tail is 4–20× more than the tracking
task needs — the instinct that it's oversized is right, the number is ~100–500,
not 20. (fig03)

> **⚖️ ATTRIBUTION —** _The decision (gain K, hence tracking MSE) is far cheaper to learn than the parameters, because MSE is quadratically flat near the optimal gain — parameter error propagates weakly into a flat objective._ Prior art: plug-in gain robustness / quadratic-loss flatness is standard adaptive-filtering lore; the specific excess-MSE crossover numbers (n≈100/150/500) are the measured original content. Status: RECOMBINATION.

**5. Influence is the square root of information.** Verified exactly: the
incremental nats a point at lag $k$ contributes decay as $(1-K)^{2k}$, while its
optimal influence on the mean decays as $(1-K)^{k}$.
$$\boxed{\ \text{influence}_k \ \propto\ \sqrt{\text{nats}_k}\ }$$
Information is an energy, influence is an amplitude. Allocating influence in
proportion to nats is wrong by a square. (fig04, fig13)

> **⚖️ ATTRIBUTION —** _The BLUE weight of a lag-k observation decays as (1-K)^k while its incremental Fisher information decays as (1-K)^{2k}, so influence ∝ √(nats) — a re-expression relating the exponentially-weighted Kalman smoother weights to the information they carry._ Prior art: the (1-K)^k weights are the unrolled exponentially-weighted Kalman/steady-state form (standard); the "influence = amplitude, information = energy / square-root" packaging is a nice framing but follows directly from those two known decay rates. Status: RECOMBINATION.

**6. Nothing makes a finite tail optimal unless the parameters themselves drift.**
If $(\log Q,\log\sigma^2)$ random-walk with per-step SD $\omega$, the
information-optimal window is
$$L^* = \sqrt{\frac{3d}{\omega^2\,\mathrm{tr}\,I_1}}\ \ \propto\ \frac1\omega,\qquad d=2$$
With $\omega=0$, $L^*=\infty$ and the long tail is *correct*. The current
2000-point tail is the optimum for $\omega\approx1.7\times10^{-3}$. Getting
$L^*=20$ needs $\omega\approx0.18$ — parameters changing 18% per step, at which
point they aren't estimable at all. **The tail length is a statement about
regime volatility, not about estimation efficiency.** (fig06)

> **⚖️ ATTRIBUTION —** _An optimal memory/window length L* ∝ 1/ω from minimising estimation variance (d/2L) plus staleness (drift ω² accumulated over the window) — the classic bias-variance window/forgetting-factor optimisation for tracking a drifting parameter._ Prior art: forgetting-factor / sliding-window trade-off in recursive estimation (Ljung & Söderström 1983); optimal memory length under parameter drift is standard adaptive-estimation. Status: RECOMBINATION.

**7. The four modes are the corners of one smooth square, and the confusable
axis is persistence, not channel.** Two continuous coordinates: $a\in[0,1]$
(which noise channel) and $\varphi\in[0,1]$ (impulse → step). The named modes are
the four corners; everything between them is a real deviation with a real
posterior. The Fisher correlations at $m$=2 are **0.79 within the process channel
across persistence** (PA↔PR) and **0.26 across channels** (PA↔MA) — so what is
hard is telling a spike from a shift, not telling process from measurement. (fig14)

> **⚖️ ATTRIBUTION —** _Parameterising the four "modes" as one continuous (channel a, persistence φ) square and reading Fisher correlations between the corners; the confusable axis is persistence not channel._ Prior art: Fisher-information / score geometry between competing model perturbations is standard (Rao score test; Fisher-metric geometry, Amari 1998); the specific 2-coordinate embedding and its correlation numbers are the original assembly. Status: RECOMBINATION.

> This revises the previous headline. Parameterising the anomalies as *mean*
> shifts made the location and scale blocks exactly orthogonal; that was a
> property of the oracle framing, not of the process. See [06](06-gradient-allocation.md).

**8. Detection is strictly more expensive than not detecting, and the gap grows
without bound.** Asking *where* a change occurred pays a null penalty that rises
like $\log n$ (3.92 nats at $n$=10, 9.24 at $n$=3,000). Asking *how large the
deviation is at each $t$* — one variance component over the whole series — pays
the boundary-LRT constant **1.353 nats, with no $n$ in it**. The scan penalty is
the price of a location estimate an allocator never uses. This is the
information-theoretic argument for the whole reframe. (fig19)

> **⚖️ ATTRIBUTION —** _Locating a change (scanning over t₀) pays a multiple-comparisons null penalty that grows like log n, whereas estimating a single variance component over the whole series pays only a fixed boundary-LRT constant (½χ²₁ at the 90th percentile = 1.353 nats)._ Prior art: the maximum of n dependent LLRs / scan-statistic penalty ~ log n is classic changepoint theory (GLR scan, Willsky & Jones 1976; Siegmund); the boundary-parameter LRT with ½χ²₀+½χ²₁ null is Chernoff 1954 / Self & Liang 1987. Status: REPRODUCTION.

**9. The filter that falls out beats a hindsight-tuned constant gain, with
nothing supplied.** Four seeds, nine probes: **geometric mean 0.678, worst case
1.017.** On stationary diffusions — where a constant gain is genuinely optimal —
the ratio is 1.001–1.005, so adaptivity is close to free when it isn't needed.
Six parameters, all learned; no $\epsilon$, no window, no gate, no tail.
(fig20–fig23, [07](07-the-finished-computation.md))

> **⚖️ ATTRIBUTION —** _Measured battery result: the fully-learned SV Kalman filter beats a per-series hindsight-tuned constant-gain Kalman (geo mean 0.678, worst 1.017) and is near-free on stationary diffusions._ Prior art: an adaptive filter beating a fixed-gain baseline off-stationarity is expected (Mehra 1970); the specific numbers on this rig are the original content. Status: RECOMBINATION.

---

## The confirmation ledger

> **Superseded as an operational object by [06](06-gradient-allocation.md).**
> These are oracle numbers (event size and event time known). Non-oracle, the
> same 4-SD jump earns 0.79 nats of worst-case *attribution* evidence at $m$=2,
> not 10.5, and reaches 99:1 at $m\approx9$. The table stands as an upper bound.

Points needed for **99:1 worst-case** evidence — evidence that survives the most
favourable competing explanation, including "nothing happened" ($q$=0.05):

| mode | small event | medium | large |
|---|---|---|---|
| **PA** level jump | 2 SD: never | 3 SD: **3** | 4 SD: **2** |
| **MA** outlier | 2 SD: never | 3 SD: **8** | 4 SD: **2** |
| **MR** $\sigma^2$ change | 1.5×: never | 2×: **41** | 3×: **15** |
| **PR** $Q$ change | 2×: never | 3×: never | 6×: **45** |

> **⚖️ ATTRIBUTION —** _Points-to-99:1-evidence for each mode via the expected LLR = KL divergence between competing hypotheses; location events are cheap, Q-changes brutally expensive because Q is a tiny fraction of the increment variance._ Prior art: Chernoff–Stein / expected-LLR = KL as the rate of evidence accumulation (Chernoff 1952; Kullback 1959); sample-size-for-target-odds is standard detection theory. The specific per-mode numbers are the measured content. Status: REPRODUCTION.

Two things to take from this table. First, the user's instinct about jumps is
exactly right and now has a number: a 4-SD jump earns 99:1 in **two** points, and
the oracle-optimal influence allocation to post-jump data goes 0.61 → 0.99 over
those two points, versus 11 points for the plain Kalman filter. Second, the
$Q$ channel is brutally expensive — at $q$=0.05 a *tripling* of $Q$ never becomes
confidently identifiable, because $Q$ is only 2.4% of the increment variance.
That asymmetry, not any engineering choice, is what the 2000-point variogram was
paying for.

## What "trustworthy" means, provisionally

Nats are never absolute — they are always *for* one hypothesis *against* another.
The proposal here: **trustworthy information is the evidence that survives
minimisation over the alternative set** (the row-minimum of the pairwise LLR
matrix, including $H_0$). That is what the ledger above reports, and it is what
should gate influence.

> **⚖️ ATTRIBUTION —** _Defining "trustworthy" evidence as the LLR that survives minimisation over the competing hypothesis set (row-minimum KL, H₀ included)._ Prior art: this is the least-favourable-alternative / minimax-LLR and the e-value / GRO (growth-rate-optimal) reading of evidence (Grünwald, de Heide & Koolen 2019/2024; Vovk & Wang e-values); robust/minimax testing (Huber–Strassen). The packaging is standard, applied here to the four-mode set. Status: REPRODUCTION. Two known costs on top of it are quantified in
[04](04-nats-trust-influence.md): the Occam penalty for not knowing the event
size (1.2–2.1 nats, bounded), and the fact that realised evidence has
SD $\approx\sqrt{2\Lambda}$ around its expectation — at 4.6 nats the standard
deviation is 3.0, so a single-shot threshold test is not reliable.
