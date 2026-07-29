# The information-domain foundation

Started as exploration rather than a filter; it produced one anyway, in
[`statfilter/`](../statfilter/README.md). Files 01–06 are exact throughout
(closed form or exact linear algebra on the model's own covariance) — no
simulation, no fitted constants. 07 is where measurement begins.

Model, unchanged from the whole thread:

$$\theta_t = \theta_{t-1} + w_t,\ w\sim N(0,Q); \qquad x_t = \theta_t + v_t,\ v\sim N(0,\sigma^2); \qquad q = Q/\sigma^2$$

Working in increments $d_t = x_t - x_{t-1} = w_t + v_t - v_{t-1}$, which annihilates
the unknown level exactly. Under no change these are stationary MA(1):
$\gamma_0 = Q+2\sigma^2$, $\gamma_1 = -\sigma^2$, $\gamma_{k\ge2}=0$.

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
[`statfilter/`](../statfilter/README.md).

---

## The nine results

**1. Per-point information about the noise parameters decays as exactly $1/n$.**
$\Delta_n = \tfrac12\log\det I_n - \tfrac12\log\det I_{n-1} \to d/2n = 1/n$ nats.
Hyperbolic, so the sum diverges logarithmically — there is no $n$ at which the
next point stops being worth having.

**2. The level channel never decays.** It contributes $\tfrac12\log\frac1{1-K}$
nats *per point, forever*. The two channels have incompatible memory laws, which
is why one truncation length can't serve both. (fig01, fig05)

**3. "95% in 20 points" is false for $Q$, and it is a Cramér–Rao fact, not an
engineering failure.** At $n=20$ the best possible relative SD of $\hat Q$ is
**92%** at $q$=0.5 and **846%** at $q$=0.005. $\sigma^2$ is much easier (35–52%).
No estimator does better. (fig02)

**4. But the *decision* is far cheaper than the parameter.** Excess tracking MSE
from not knowing the noises falls below 5% at $n\approx$ **100 / 150 / 500** for
$q$ = 0.5 / 0.05 / 0.005. So the 2000-point tail is 4–20× more than the tracking
task needs — the instinct that it's oversized is right, the number is ~100–500,
not 20. (fig03)

**5. Influence is the square root of information.** Verified exactly: the
incremental nats a point at lag $k$ contributes decay as $(1-K)^{2k}$, while its
optimal influence on the mean decays as $(1-K)^{k}$.
$$\boxed{\ \text{influence}_k \ \propto\ \sqrt{\text{nats}_k}\ }$$
Information is an energy, influence is an amplitude. Allocating influence in
proportion to nats is wrong by a square. (fig04, fig13)

**6. Nothing makes a finite tail optimal unless the parameters themselves drift.**
If $(\log Q,\log\sigma^2)$ random-walk with per-step SD $\omega$, the
information-optimal window is
$$L^* = \sqrt{\frac{3d}{\omega^2\,\mathrm{tr}\,I_1}}\ \ \propto\ \frac1\omega,\qquad d=2$$
With $\omega=0$, $L^*=\infty$ and the long tail is *correct*. The current
2000-point tail is the optimum for $\omega\approx1.7\times10^{-3}$. Getting
$L^*=20$ needs $\omega\approx0.18$ — parameters changing 18% per step, at which
point they aren't estimable at all. **The tail length is a statement about
regime volatility, not about estimation efficiency.** (fig06)

**7. The four modes are the corners of one smooth square, and the confusable
axis is persistence, not channel.** Two continuous coordinates: $a\in[0,1]$
(which noise channel) and $\varphi\in[0,1]$ (impulse → step). The named modes are
the four corners; everything between them is a real deviation with a real
posterior. The Fisher correlations at $m$=2 are **0.79 within the process channel
across persistence** (PA↔PR) and **0.26 across channels** (PA↔MA) — so what is
hard is telling a spike from a shift, not telling process from measurement. (fig14)

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

**9. The filter that falls out beats a hindsight-tuned constant gain, with
nothing supplied.** Four seeds, nine probes: **geometric mean 0.678, worst case
1.017.** On stationary diffusions — where a constant gain is genuinely optimal —
the ratio is 1.001–1.005, so adaptivity is close to free when it isn't needed.
Six parameters, all learned; no $\epsilon$, no window, no gate, no tail.
(fig20–fig23, [07](07-the-finished-computation.md))

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
should gate influence. Two known costs on top of it are quantified in
[04](04-nats-trust-influence.md): the Occam penalty for not knowing the event
size (1.2–2.1 nats, bounded), and the fact that realised evidence has
SD $\approx\sqrt{2\Lambda}$ around its expectation — at 4.6 nats the standard
deviation is 3.0, so a single-shot threshold test is not reliable.
