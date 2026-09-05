# 0043 — What s₀ is, and why gravity is exponentially sensitive to it

> **AI-generated, not peer-reviewed.** Code:
> `0043_the_casimir_coupling.py`.

Their 0139 found the coupling is not derived: 0074's construction
gives a **family** of profiles indexed by a bin scale s₀, and across
it ξ/a moves by 10¹². A second gap: the derivation lives on Spin(4),
the simulation on a single SU(2) — worth 4×10²¹ more.

That volatility is not noise. It is pointing at a single scalar.

> **⚖️ ATTRIBUTION —** _"Casimir" here is the Casimir OPERATOR of a Lie algebra (eigenvalue j(j+1)), not the Casimir 1948 vacuum-force effect — the borrowed name is loose. The genuine facts are standard representation theory / harmonic analysis: κ=(8/3)⟨j(j+1)⟩ is a dimension-weighted mean Casimir eigenvalue, and the heat kernel on SU(2) decays a character as exp(−τj(j+1)), so the "coupling" is a mean forgetting rate. Spin(4)=SU(2)×SU(2) fusion raising κ is rep-theory bookkeeping._ Prior art: Casimir operator / eigenvalue j(j+1); heat kernel on compact groups; Peter–Weyl. Status: REPRODUCTION (rep theory) + SPECULATIVE (the "record precision = coupling" correspondence).

## 1. The coupling is a mean Casimir, exactly

For any counting amplitude A = Σ c_n χ_n,

> **κ = (2/3)·Σ c_n n(n²−1) / Σ c_n n = (8/3)·⟨j(j+1)⟩**, weights
> c_n·d_n.

| profile | numeric | Casimir formula |
|---|---|---|
| flat 1…6 | 13.33725 | 13.33333 |
| peaked | 10.40281 | 10.40000 |
| rising | 16.01411 | 16.00000 |
| falling | 8.99955 | 9.00000 |
| random | 15.01283 | 15.00000 |

Exact — the residual is the polynomial fit, not the formula.

> **The entire coupling is one scalar: the dimension-weighted mean
> Casimir.** The profile's shape, peak and width are all invisible
> to κ. That is why s₀ moved the hierarchy so violently — s₀ does not
> perturb the amplitude, it **moves its mean Casimir**.

## 2. And the Casimir is a forgetting rate

On this side j(j+1) is not a group-theory label. It is the rate at
which sector j is forgotten under diffusion — the heat kernel decays
a character as exp(−τ j(j+1)). Measured by simulating the walk and
fitting each sector's decay: the ratio of measured rate to j(j+1) is
flat across sectors to 25%.

> So κ — a mean Casimir — is a **mean forgetting rate**, i.e. a
> **record precision.** That closes a loop: 0032 derived
> G = 1/(4πp) with p a record precision, and the lattice coupling
> turns out to be one too.

## 3. Spin(4) vs SU(2) is one record versus two

Spin(4) carries a self-dual *and* an anti-self-dual frame record.
Fusing them gives χ_n² = χ₁+χ₃+…+χ_{2n−1}, which pushes weight to
higher spin — so the mean Casimir rises.

| profile | SU(2) κ | Spin(4) κ | ratio |
|---|---|---|---|
| flat 1…6 | 13.333 | 32.000 | 2.40× |
| peaked | 10.400 | 24.175 | 2.32× |
| falling | 9.000 | 24.000 | 2.67× |

> **The number of independent records fused into one amplitude is an
> exponentially consequential choice.** That is why their unrecorded
> Spin(4) → SU(2) step was worth 4×10²¹. It is not bookkeeping — it
> is a statement about how many records the world writes per event.

## 4. So s₀ is a resolution, and the filter knows what sets those

s₀ is the bin width on the bivector magnitude: the resolution at
which the record distinguishes two frame pairs by their wedge area.
Coarse binning piles weight into low spin (small Casimir, weak
coupling, small hierarchy); fine binning spreads it to high spin.

0041 established this side's rule for exactly that kind of quantity:
**a resolvable count is exp(channel capacity)**. So s₀ is not free —
it is set by the frame-magnitude channel's capacity.

**And that is the same quantity three times:**

| | |
|---|---|
| N, the level | = exp(phase-channel capacity) — 0041 |
| s₀, the bin width | = range / exp(frame-magnitude capacity) — here |
| their requirement (D) | = "why this record precision" — 0041 §5 |

> **Three roads, one question.** The program's last free parameter,
> its last underived constant, and the volatility that exposed both
> are the same open problem wearing three costumes: **what sets the
> precision of the world's record?**

That is a better place to be than three separate unknowns, and it is
a filter question rather than a geometry one. It does not answer it.
