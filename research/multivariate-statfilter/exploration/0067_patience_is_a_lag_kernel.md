# 0067 — Patience is a lag kernel, not a slower exponential

> **AI-generated, not peer-reviewed.** Script: [`0067_lag_kernel_patch.py`](0067_lag_kernel_patch.py) (stage A:
> the walk on a kernel-weighted trailing buffer of scores, one member per kernel mode, on `main`'s source).
> Per-copy traces with [`0065_per_copy_trace.py`](0065_per_copy_trace.py); scalar A/B 12 seeds. Measurements
> in §3 as they land.

## 1. The idea (the user's, restated in the filter's terms)

Every construction so far weights past evidence by lag `k = t − s` with `w(k) = f^k`: exponential, maximal at the
current step. The memory ladder (0064) varies `f`; the attribution copies (0059) vary how much of the *current*
score the walk takes. In all of them the most recent measurement is the heaviest and "patience" only ever meant
a slower decay — a family of EMAs.

That is the wrong shape for the decision the copies exist to make. A white sensor spike and a process jump are
**indistinguishable at the step they arrive**: the same innovation, the same score on both axes. They separate on
the *next* innovation — a sensor spike reverses (the state was pulled toward it and snaps back), a process jump
persists. A kernel peaked at lag 0 therefore scores every hypothesis on the one step that carries no information
and rewards the wider prior (0059's theorem, seen from this side); a kernel peaked at lag 1–2 scores them on the
step that carries the answer. So patience is a **kernel whose mode is at a lag ≥ 1**, with a tail beyond, and
eagerness is the mode at 0. The ladder runs over kernel shapes.

Implementation: keep a trailing buffer of the last `N` per-step values, `N` where the kernel's tail mass past
the cutoff is below the aliasing tolerance `3.1e-4` (the same `ε₀` the grids use), and take the kernel-weighted
average. The natural family is the gamma in the lag, `w(l) ∝ l^{a−1} e^{−l/b}`, mode `(a−1)b`; chi-squared is
the scale-2 member (`ν` degrees of freedom: shape `ν/2`, mode `ν − 2`). The exponential is shape 1, and a
gamma of integer shape `a` is exactly the `a`-fold cascade of EMAs — which is the one-line statement of what
the current form is missing: the mode of an EMA cascade sits at `(a − 1)b`, and every construction to date
is `a = 1`.

Two sites read the kernel, and in the unified form they read the same one: the **walk** (the step on a
kernel-weighted score and information over the member's own trailing scores) and the **weights** (each rung's
evidence as the kernel-weighted sum of its members' log-likelihoods). Stage A below is the walk alone, on
`main`, with the weights as `main` has them; stage B is the weights.

## 2. Stage A: the construction

One member per class cell per kernel mode `m ∈ MODES`. Each step's innovation on an axis implies a **scale
reading** — the current scale plus that step's own clipped Newton increment, `μ_t + clip(grad/info)` — and a
mode-`m` member moves toward the kernel-weighted average of its trailing readings, `μ ← μ + K_μ (Σ_l w_m(l) r_{t−l} − μ)`,
clipped to the budget as always (the buffer is per axis; until it reaches the kernel's support the member reads
the latest only). Mode 0 is `main`'s walk exactly (verified bit-identical on the scalar rig).

*Not* this: the first form buffered the raw scores and stepped on their kernel average. That re-applies a
spike's gradient, computed at the old scale, for the whole length of the kernel after the scale has already
moved — twenty steps at the full budget — and the scales ran off to a singular innovation covariance on both
arm seeds. A kernel over lags must weight *readings* (evidence about the scale), not stale gradients; the
reading form has the fixed point the gradient form lacks (once the scale sits at the readings' average the
step is zero). The copies compete in the bank's joint posterior as
the rate copies did; `patience` is the weight on the modes above 0. Kernels used (chi-squared, `b = 2`):

| mode | `w(0..5)` | buffer `N` |
|---|---|---|
| 0 | 1 | 1 |
| 1 | 0, .26, .23, .17, .12, .08 | 20 |
| 2 | 0, .16, .19, .17, .14, .11 | 22 |
| 4 | 0, .04, .09, .13, .14, .13 | 26 |

## 3. Measurements

(pending)
