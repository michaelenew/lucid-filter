# 0048 — The item-2 blocker is an estimator problem, not a physics one

> **AI-generated, not peer-reviewed.** Code:
> `0048_the_estimator_problem.py`.

Their criticality item 2 — the graviton propagator — sits at the
statistical floor. The last diagnosis said **throughput**: not enough
configurations. A C kernel was built, delivered ~30× more sweeps, and
**the result did not move**. So the diagnosis was wrong, and the
question comes back here in the form the filter can actually answer:

**what is the estimator doing with the samples it already has?**

> **⚖️ ATTRIBUTION —** _Not physics — a textbook estimation fix, explicitly named in the code as "multihit / link integration": Rao–Blackwellisation, replacing a sampled factor by its conditional mean (Var(E[X|Z])=Var(X)−E[Var(X|Z)]), gives up to 120× variance reduction at fixed samples. The correct ratio-of-product-variances law (vs a naive (1+v)^j) and the finding that the gain persists for stiff links are genuinely useful, correctly-derived engineering._ Prior art: Rao–Blackwell theorem (Rao 1945; Blackwell 1947); multihit / link integration in lattice MC (Parisi–Petronzio–Rapuano 1983). Status: REPRODUCTION (Rao–Blackwell), a sound and useful application.

## s1 — The error grows with the operator, the signal does not

Take a correlation between two products of `k` noisy positive
factors, where the signal lives in exactly one factor on each side and
the other `k−1` are pure nuisance:

| k | estimate | s.e. | |
|---|---|---|---|
| 1 | −0.019… | 0.0028 | |
| 4 | −0.019… | 0.0079 | |
| 8 | −0.019… | 0.0219 | ← 7.8× the error, same signal |

The **signal is identical at every k**. Only the error grows, because
each extra factor contributes its own fluctuation to the estimator and
**none of that fluctuation carries any signal**.

That is their situation exactly: the site operator is six plaquettes
of four links each, and the long-distance signal is a small correlated
piece riding on top of ~24 links' worth of local noise.

## s2 — Rao-Blackwell, and why it beats √n

Replace a sampled factor by its **conditional mean** given the rest.
`Var(E[X|Z]) = Var(X) − E[Var(X|Z)]`: the estimate is unchanged and
the variance can only fall. It is exact, not an approximation.

| factors integrated out | s.e. | variance reduction | predicted |
|---|---|---|---|
| 0 | 0.022243 | 1.00× | 1.00× |
| 2 | 0.013600 | 2.67× | 3.39× |
| 4 | 0.007101 | 9.81× | 12.03× |
| 6 | 0.003445 | 41.68× | 52.00× |
| 7 | 0.002028 | **120.28×** | 146.76× |

Estimates at j = 0, 4, 7: −0.032466, +0.002650, +0.000252 — the same
signal throughout.

The law is a **ratio of product variances**,

    reduction(j) = [ (1+v/m²)^{2k} − 1 ] / [ (1+v/m²)^{2(k−j)} − 1 ] ,

not the naive `(1+v/m²)^j`. A first pass here used the naive form and
**under-predicted by ~15×**; the ratio form tracks the measurement.

120× in variance is 120× the samples by brute force. The C kernel
bought 30×. **The estimator is worth more than the kernel was.**

## s3 — And it pays even when the links are stiff

I expected this to evaporate for nearly deterministic factors, and was
about to tell them to measure their conditional variance first in case
the fix did not pay. **It does not evaporate**, and the ratio law says
why: the baseline compounds over all 2k factors while the reduced
estimator keeps only 2(k−j), so the ratio stays large as v → 0.

| v/m² | reduction at j = 7 | predicted |
|---|---|---|
| 0.350 | 162.32× | 146.76× |
| 0.050 | 11.32× | 11.54× |
| 0.005 | **7.33×** | 8.29× |

> **The gain comes from the LENGTH of the operator, not the noisiness
> of each link.** Their links are stiff at κ ≈ 17 and it still pays
> ~8× in variance, because their operator is long.

## s4 — The port spec

For each link `U` in a measured plaquette, replace `U` by its
conditional mean given everything else. For a class-function weight
and staple `S` that is `Ū = c(|S|)·Ŝ†`, a one-dimensional integral
computed once per link per measurement. For the Spin(4) weight the two
factors couple through the 2-D table, so integrate one factor at a
time holding the other — still 1-D quadrature. Where the weight is a
table and the six plaquettes touching a link do **not** combine into
one effective staple, estimate the same conditional mean the original
way: extra local Metropolis hits on that link alone, averaged.

Do **not** gate the decision on link stiffness (s3).

> **The whole move: stop sampling what can be integrated.** It is not
> a physics idea, it is an estimator idea, and the filter has been
> making it since its first stone — a posterior mean is always a
> better estimator than a draw.

## What this is worth

An honest caveat travels with the port and is theirs to measure, not
mine to assert: their six plaquettes **share links**, so substituting
a conditional mean is exact for a single plaquette and approximate for
their product. The variance reduction above is real; whether it
arrives **unbiased** in their operator is a measurement on their side.
