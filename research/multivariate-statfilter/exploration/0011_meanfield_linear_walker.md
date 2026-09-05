# 0011 — the scaling attack: linear mean-field is cheap but not faithful

> **⚖️ ATTRIBUTION —** _A mean-field (factored, one window per axis) posterior is linear-cost but double-counts the ambiguous process/measurement variance, so it cannot recover the coupling the exact grid does — a clean negative result; also restates the known "state tracking is nearly blind to the noise parameters (forecasting is not)" law._ Prior art: mean-field / variational factorization (standard); noise-adaptation matters little for the filtered state — adaptive-KF folklore, e.g. Mehra 1972 / Bar-Shalom. Status: NEGATIVE-RESULT.

The shipped `WalkingVectorFilter` runs the state KF over the tensor-product scale grid
— `nodes^(#active axes)`, exponential, the robotics-practicality blocker. Goal: linear
in the active axes (settle for quadratic). 0003 measured the scale-Fisher as nearly
block-diagonal, motivating a **mean-field / factored** posterior (one 1-D window per
axis) at **linear** cost `D·nodes`.

## Result: linear cost, but the coupling breaks faithfulness

Measured against the exact grid (n=2, m=2, mixing H). Cost is linear as designed —
`3·5 = 15` evals/step vs the grid's `5³ = 125`, and it scales `D·nodes` not `nodes^D`.
Direction tracks (corr > 0.9). **But accuracy degrades on the coupling:**

| case | exact grid | mean-field (1 sweep) | + STAR-GPB1 state | + 4 sweeps |
|---|---|---|---|---|
| xi2 hot → eta2 leak | 0.31 | 0.98 | 0.97 | 0.84 |
| static eta2 drift | ~0 | 0.44 | 0.43 | 0.41 |

- The **STAR-GPB1 state** (collapse the state over the union-of-windows star, still
  linear) is *identical* to a point state — so the leak is **not** the state collapse.
- It is the **mean-field scale update itself**: when process and measurement can both
  explain an innovation, independent per-axis updates **double-count** the ambiguous
  variance — both inflate. Iterating the coordinate updates to their joint fixed point
  (4 sweeps) only partly relieves it (0.98 → 0.84); the mean-field fixed point is not
  the true joint posterior for a coupled block.

## Conclusion and path

**Pure linear (factored) is ruled out on faithfulness** — the one process↔measurement
coupling block genuinely needs joint treatment (the grid's whole value). The cost is:

- **Within-block is safe to factor** (0003: process eigenmodes decouple, sensors
  decouple) — mean-field there is fine and linear.
- **The process↔measurement block must be joint.** This is exactly the structure the
  shipped scalar `VectorFilter` already handles faithfully with its two channels
  (global process scale × global measurement scale, `order²` grid).

So the **quadratic** (sub-exponential) target: a **2-D joint grid over the block
coupling** (global process × global measurement scale, `order²` constant) **×
mean-field per-axis deviations within each block** (linear) — `order² + D·nodes`, no
exponential. Or the general **pairwise/Bethe** representation (`D²·nodes²`) if the
block structure proves too rigid. Next probe: the block-joint hybrid, benchmarked for
faithfulness (must match the grid's 0.31, not 0.84) at sub-exponential cost.

A **coupling-aware quadratic** (each axis gridded 2-D against the *opposite block's*
global scale, `D·nodes² = 75` vs grid 125) was also tried — it *still* leaks
(eta2 0.81 vs grid 0.31). So the coupling attribution resists even pairwise factoring;
only the full tensor grid recovers it. Two live paths remain for faithful
sub-exponential: an **adaptive/sparse grid** (instantiate only high-weight nodes — the
scale posterior concentrates once converged, so this is the same joint grid pruned to
its occupied blob, sub-exponential *in practice* while staying exact), or accepting the
attribution leak.

## The decision-relevant measurement: state tracking barely uses the adaptation

For robotics the deliverable is **state tracking**, not the noise-attribution
diagnostic. Measured (4 seeds): the shipped per-component walker's state RMSE is
**barely better than a non-adaptive KF** at the base covariances —

| regime | shipped walker | non-adaptive KF |
|---|---|---|
| process mode hot | 0.769 | 0.774 |
| sensor hot | 0.866 | 0.895 |
| static | 0.754 | 0.746 |

i.e. <3% on a moderate (×e^1.4) shift — the scalar filter's known law ("tracking is
nearly blind to the noise parameters; **forecasting** is not"). So the coupling
attribution that resists factoring is load-bearing for the **diagnostic** (which
sensor/mode is hot — the fault-detection/animation use) and for **forecasting**, *not*
the state estimate. (Caveat: the adaptation's state value grows with how far the noise
strays from the base — the scalar headline result was a *large* out-of-fit regime
shift, RMSE 1.0 vs 3.6; this test's shift is moderate.)

## Corrected upshot: per-component is load-bearing for state, and scaling matters

The "adaptation barely helps state" reading held only for *moderate* shifts. Across
magnitudes (hot band, seeds):

| shift | proc-hot walker/non-adapt | sensor-hot walker/non-adapt |
|---|---|---|
| ×e^1.4 | 0.76 / 0.79 (×1.04) | 0.96 / 1.02 (×1.06) |
| ×e^5 | 0.76 / 2.45 (**×3.2**) | 2.21 / 5.25 (**×2.4**) |

And **per-component vs the cheap 2-channel** (global process × global measurement),
one sensor hot at ×e^5: **2.38 vs 6.24 — per-component tracks the state 2.6× better**,
because the global channel over-inflates *every* sensor when one degrades, over-trusting
the failing sensor onto the clean ones. (Process-mode hot: ×1.15 — process is more
shared, so the gap is smaller there.)

So per-component **is** load-bearing for the core robotics deliverable (state tracking
under a heterogeneous sensor failure — the motivating case), not just the diagnostic.
The scaling problem is real and must be solved *faithfully*. Since factoring loses the
coupling (above), the path is the **adaptive / sparse grid**: instantiate only the
high-weight nodes of the *same joint grid* — exact where it matters, and sub-exponential
in practice because the scale posterior concentrates on a small blob once converged (it
only broadens during a transition). That is the make-it-fast target.
