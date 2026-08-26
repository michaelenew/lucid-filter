# 0029 — reprofiling the discounted opens against the extended domain

Premise (user): many "that doesn't matter much" verdicts were filed on **simple** domains
(scalar, `H=I`, quiet, orthogonal). The domain has since exploded (5-DOF IMU fusion, phased
noise, mixing `H`, collinear modes, coupled dynamics). The tell — the robotics case gives
~1.05× in quiet/idealized regimes even with the suboptimalities in place, but gets much worse
once realistic — suggests a dormant suboptimality could be load-bearing when realistic. So:
re-measure each open's cost (mis-specified filter RMSE / an oracle without the suboptimality) in
an **idealized** regime vs a **realistic** one.

## Result: the discounting mostly HOLDS — but the *critical regime* was mis-framed

**Opens that do NOT flip (validated across the extended domain — deprioritize):**

| open | idealized cost | realistic cost | verdict |
|---|---|---|---|
| **Fixed `V`** (joints coupled 0.5, so the true process eigenvectors are *not* the filter's block-diagonal `V`) | 0.95× | 1.19× — *same as baseline* | HOLDS. The coupling adds ≈0 over the baseline lag; the fixed-`V` mis-spec is invisible to position RMSE. |
| **Diagonal `R`** (accelerometers common-mode correlated 0.6) | 0.95× | 1.13× — *same as baseline* | HOLDS. Correlation is a second-order effect on the state estimate. |
| **Large-`n` / dimension** | — | 1.39× at `n_dof=1`, 1.39× at `n_dof=8` | HOLDS. The cost is **flat with dimension** — a *scalar* arm already shows the full bursty cost, so low-dim domains hid nothing. |
| **Static drift** on strong axes | 0.37 nats wander, **0.94× state cost** in quiet | — | HOLDS for state (no cost); a minor false-alarm floor for the *diagnostic* only. |
| **Collinear confound**, for *state* | — | **1.14×** (accel-hot burst) | HOLDS — the "state cost of collinearity is small" claim (0027) is right; collinear modes only need their *total*, which the filter gets. |

So the extended domain did **not** expose a mis-discounted modeling open. Fixed `V`, diagonal
`R`, dimension, and static drift were all correctly triaged.

## What actually flips — and it isn't what we thought

The realistic-regime degradation is entirely the **adaptation dynamics under a noise burst**
(dormant at ~1.0× quiet, 1.3–1.9× bursty), and the expensive facet is **not the collinear
confound** we had been chasing. It is **adaptation lag when the *absolute* (position) sensor
degrades**:

| burst | adaptive / oracle |
|---|---|
| accelerometer hot (collinear with process) | **1.14×** |
| **potentiometer hot (orthogonal, the absolute reference)** | **1.86×** |

When the bad pot gets worse, **position observability collapses** — the accelerometer only
gives acceleration, which integrates to a drifting position, so the absolute reference is
load-bearing. The oracle down-weights the pot instantly; the adaptive **lags** in raising the
pot's noise scale and briefly trusts a sensor it should be discarding, and position estimation
is unforgiving there. This is the regime that "gets much worse once realistic," and it was
**mis-attributed** to the collinear confound (which is actually the *cheap* one for state).

## Reprioritized backlog

1. **PROMOTE — adaptation lag on a degrading absolute sensor** (observability-fragile). The
   dominant realistic-regime cost (1.86×). Fix direction: faster / observability-weighted
   adaptation of the absolute sensor's noise scale, so the filter sheds a failing position
   reference quickly. This is the concrete, worth-doing piece of the "adaptation lag" open —
   sharper than the generic Mehra target (0028).
2. **KEEP as-is (diagnostic-only)** — collinear confound (per-sensor gate residual) and static
   drift: they cost the *diagnostic*, not the state. Matter only for the health-monitor use.
3. **DEPRIORITIZE — learn `V`, full `R`, large-`n` truncation**: measured to not matter for
   state even in coupled / correlated / high-dimensional realistic regimes.

Code: `0029_reprofile.py`.
