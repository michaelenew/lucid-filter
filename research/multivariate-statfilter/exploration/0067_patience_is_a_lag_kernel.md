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

## 3. Stage A, measured: the mode copies never take weight on the arm, and the reason is structural

Arm seed 1 (the seed with the grid's metre excursion), per-copy traces, chi-squared kernels:

| window | mixture | eager copy (mode 0): weight, own error | patient copies (modes 1 / 2 / 4): weight, own error |
|---|---|---|---|
| SENSOR onset (40 steps) | 13.96× (worst 0.091 m) | 1.00, 13.96× | 0.00, 27.9× (worst 0.224 m) — identical for modes 1, 2, 4 |
| SENSOR | 5.95× | 1.00, 5.95× | 0.00, 57× |
| PROCESS / POTFAIL / BOTH | 1.40 / 1.80 / 2.59× | 1.00 | 0.00; own errors 137× / 148× / 17× |

Seed 0 diverged to NaN under every mode set (a numerical failure of the patch on that seed, not traced). Scalar
hero, 12 seeds: jump 1.7474 → 1.8367 ({0,1}) / 1.8820 ({0,1,2,4}), steady 0.3833 → 0.3821, C unchanged — the
patient copies hold ~half the weight in calm and lose it at the jump, which they delay.

Note first what the eager copy does here against `main` on the same seed: 13.96× / 5.95× in SENSOR against
`main`'s 2.95× / 3.48×, with no metre excursion (worst 0.091 m). So on this seed the mode-0 copy is *not*
`main`'s filter even though its walk is bit-identical to `main`'s on the scalar rig — the same coupling
0065 §5 found with the rate copies (something the copies share moves the eager one), now with copies that
never win. That coupling is prior to any kernel question and is still not isolated.

Why the patient copies cannot win, from the step trace ([`diag`](0067_lag_kernel_patch.py) of one cell,
joint 0's accelerometer axis, through the onset): the walk on the arm is *rate-limited*, `K_μ × budget ≈
0.075 × 0.3 = 0.02` nats per step, so the sensor scale needs ~250 steps to climb the 5.4 nats a ×15 burst
asks for; the eager member climbs at that rate and the patient member at ~70% of it (its target averages
readings taken at older, lower scales). A sustained burst is not a transient: at every lag the reading says
"higher", the kernel has nothing to discriminate, and the patient member is simply the slower walker on
the one quantity that *is* identifiable per step — the total innovation variance. It loses the predictive
likelihood at every step and never holds weight. The kernel was applied to the magnitude of the walk; the
attribution question is about its *direction* (which of a confounded pair moves), and 0068 says what the
kernel on that direction is.

**Stage A is rejected**, for a reason that is derivable rather than measured: a lag kernel on the walk slows
the identifiable sum; the object it belongs to is the split.
