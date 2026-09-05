# 02 — Relevance decay, redundancy, and what actually sets the tail

Script: `scripts/THEORY-002-relevance-decay.py` · Figures: fig04, fig05, fig06

The stated goal was "a process that tracks the decay of relevance of each
successive measurement based on their distance into the past **and** correlation
to more recent points." Both halves are in closed form.

## A. Three different decay laws, and they are not the same law (fig04)

For the level $\theta_t$:

**Marginal relevance** — what $x_{t-k}$ says on its own. $\theta_t\mid x_{t-k}$
has variance $\sigma^2+kQ$, so

$$I^{\text{marg}}_k = \frac1{\sigma^2+kQ}\qquad\text{hyperbolic; }\textstyle\sum_k I^{\text{marg}}_k \text{ diverges}$$

**Incremental relevance** — what $x_{t-k}$ adds *given every newer point*. Because
the random walk is time-reversible, $\mathrm{Var}(\theta_t\mid x_{t-k:t})$ is just
the ordinary Riccati recursion run $k{+}1$ steps from a diffuse prior. Its
convergence rate is the Riccati derivative $\sigma^4/(P^-+\sigma^2)^2=(1-K)^2$, so

$$\Delta^{\text{nats}}_k \ \propto\ (1-K)^{2k}\qquad\text{geometric}$$

**Optimal influence** — the BLUE weight of $x_{t-k}$ in the estimate:

$$a_k = K(1-K)^k,\qquad \textstyle\sum_k a_k = 1\qquad\text{geometric, half the rate}$$

Verified at machine precision for $q$ = 0.005 / 0.05 / 0.5:

| $q$ | $K$ | incremental-nats ratio | $(1-K)^2$ | influence ratio | $\sqrt{\text{nats ratio}}$ |
|---|---|---|---|---|---|
| 0.005 | 0.0683 | 0.86815 | 0.86815 | 0.93175 | 0.93175 |
| 0.05 | 0.2000 | 0.64004 | 0.64000 | 0.80000 | 0.80002 |
| 0.5 | 0.5000 | 0.25000 | 0.25000 | 0.50000 | 0.50000 |

> **⚖️ ATTRIBUTION —** _Three distinct decay laws for a past observation's relevance to the current level: marginal 1/(σ²+kQ) (hyperbolic), incremental Fisher (1-K)^{2k} (geometric, via the Riccati derivative and random-walk time-reversibility), and BLUE influence weight K(1-K)^k (geometric, half-rate)._ Prior art: exponentially-weighted Kalman/steady-state weights and Riccati convergence rate (1-K)² are standard (Anderson & Moore 1979); reversibility of the random walk is textbook. Status: REPRODUCTION.

### The redundancy is the gap, and the conversion law is a square root

The marginal law is hyperbolic and its sum diverges; the joint information is
finite ($1/P$). The entire difference is redundancy — a point far in the past
tells you a lot about $\theta_t$ *in isolation* and almost nothing *given the
newer points that already encode it*. Conditioning on newer data is precisely
"correlation to more recent points", and it converts hyperbolic into geometric.

The second, sharper result:

$$\boxed{\ a_k \ \propto\ \sqrt{\Delta^{\text{nats}}_k}\ }$$

**Influence is an amplitude; information is an energy.** This is the natural
nats→influence conversion, and it is *not* proportionality. Allocating "% effect"
in proportion to nats would be wrong by a square — at lag 10 with $K=0.2$ the
nats share is $0.64^{10}=0.012$ but the correct influence share is
$0.8^{10}=0.107$, a factor of nine. (fig13, right panel, collapses all three $q$
onto the identity line.)

> **⚖️ ATTRIBUTION —** _The gap between the divergent marginal information and the finite joint information is redundancy from correlation with newer points; the nats→influence conversion is a square root (aₖ ∝ √(Δnatsₖ))._ Prior art: information redundancy under correlation is standard information theory; the √ relation is an algebraic consequence of the two known geometric rates (1-K)^k and (1-K)^{2k} — a re-expression, not a new law. Status: RECOMBINATION.

## B. The two channels cannot share a window (fig05)

| | law | horizon |
|---|---|---|
| level, $\theta_t$ | $(1-K)^{2k}$ | $\sim1/K$ points, then nothing |
| noise, $(Q,\sigma^2)$ | $1/n$ | never; the sum diverges logarithmically |

At $q$=0.05 the level channel is down 100× by lag 12 and below double precision
by lag 60. The parameter channel at $n$=1000 still delivers $10^{-3}$ nats per
point and will keep doing so forever. Any single truncation length is either
throwing away parameter information or keeping level information that has been
zero for decades of samples.

**This is the structural argument for the ladder** — but it says the rungs should
be indexed by *channel*, not by lag. The level channel wants a horizon of $1/K$.
The parameter channel wants everything.

> **⚖️ ATTRIBUTION —** _The level channel forgets geometrically (horizon ~1/K) while the parameter channel's information accrues forever (1/n, divergent sum), so no single truncation length serves both._ Prior art: restates the two decay laws above (steady-state Kalman forgetting vs Fisher-information accumulation); standard. Status: REPRODUCTION.

## C. So what makes a finite tail optimal? Only hyper-drift. (fig06)

Under a stationary $(Q,\sigma^2)$ the answer is: **nothing**. $L^*=\infty$. Every
finite buffer is strictly suboptimal, and the $\epsilon$-leverage rule is a
compute concession, not an information-theoretic one.

A finite tail becomes optimal exactly when the parameters themselves drift. Let
$\phi=(\log Q,\log\sigma^2)$ random-walk with per-step SD $\omega$. Using a
rectangular window of the last $L$ points costs, in nats:

$$\mathcal L(L)=\underbrace{\frac{d}{2L}}_{\text{estimation}} \;+\; \underbrace{\frac12\mathrm{tr}(I_1)\,\omega^2\frac{(L-1)(2L-1)}{6L}}_{\text{staleness}}$$

(the staleness term is exact: $\mathrm{Var}(\phi_t-\bar\phi_{\text{window}})=\omega^2\sum_{j<L}((L-j)/L)^2$.)
Minimising:

$$\boxed{\ L^* = \sqrt{\frac{3d}{\omega^2\,\mathrm{tr}\,I_1}}\ \propto\ \frac1\omega\ }$$

> **⚖️ ATTRIBUTION —** _A finite window is optimal only when the parameters themselves drift: minimising estimation variance (d/2L) plus staleness (drift ω² over the window) gives L* = √(3d/(ω²·tr I₁)) ∝ 1/ω._ Prior art: the bias-variance / forgetting-factor window optimisation under a random-walk parameter is classic recursive estimation (Ljung & Söderström 1983; Benveniste, Métivier & Priouret 1990); the closed form and the mapping "tail length ↔ regime volatility ω" is a clean re-derivation. Status: RECOMBINATION.

Numerically confirmed against a grid search over $L\in[3,10^5]$ across four
decades of $\omega$. With $\mathrm{tr}\,I_1 \approx 0.48/0.45/0.35$ for
$q$ = 0.005/0.05/0.5:

| $\omega$ | $L^*$ ($q$=0.05) | interpretation |
|---|---|---|
| $10^{-4}$ | 36,700 | parameters essentially frozen |
| $1.7\times10^{-3}$ | **2,000** | ← what the current filter's tail assumes |
| $10^{-2}$ | 367 | noticeably non-stationary |
| $0.18$ | 20 | a new regime every few samples |

### What this means for the project

The buffer length was never an estimation question. **$L$ encodes a belief about
how fast the world's noise parameters move**, and that belief has been implicit
and unstated. Making it explicit does three things:

1. It replaces $\epsilon$ (a memory-vs-accuracy dial with no meaning) with
   $\omega$ (a physical rate with units, estimable from data, and *the same kind
   of object as $Q$* — a drift rate for a parameter).
2. It makes the tail *adaptive by construction*: $\hat\omega$ rises after a
   regime change, $L^*$ contracts, old data is discarded because it is stale, not
   because a leverage threshold fired.
3. It exposes the regress honestly. $\omega$ is a hyper-parameter drift rate, and
   asking for *its* drift rate opens the next level. The difference from previous
   rounds of this regress is that $\omega$ has a **finite reach**: $L^*\propto1/\omega$
   means an error of a factor $f$ in $\omega$ costs $\mathcal L$ a factor
   $\tfrac12(f+1/f)$ — at $f=2$, an 25% excess. The loss surface in $L$ is flat
   enough (fig06, right) that $\omega$ does not need to be known well. That is a
   materially better position than $c$, $\nu$, or $a_j$ were in.

The rung weights already in use, $w_k\propto1/(V(k)^2k)$, are the inverse-variance
weights for the *stationary* problem — i.e. the $\omega=0$ case. The hyper-drift
version multiplies them by a staleness discount, which is the derivation the
ladder never had.
