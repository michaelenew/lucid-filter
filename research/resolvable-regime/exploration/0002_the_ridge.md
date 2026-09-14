# 0002 — C2 is false, and the falsification is worth more than the prediction was

> **AI-generated, not peer-reviewed.** Code
> [`0002_the_ridge_coordinates.py`](0002_the_ridge_coordinates.py), full output
> [`0002_output.txt`](0002_output.txt). 8000-step records, 5 seeds, GPB1 grid
> filter on 241 nodes, central-difference observed information.

## What was predicted

[`0001`](0001_the_axiom.md) C2: the identification ridge that `adaptive-grid`
findings 13–16 measured in `(phi, s)` is the **mean-reversion** direction — per
step the likelihood sees the log-scale's increment variance `nu`, and `kappa`
enters only over many steps, so the flat direction should move `kappa` and hold
`nu`. Predicted signature along the flat eigenvector: `|dlog nu| ~ 0`,
`|dlog kappa| ~ 1`.

## What was measured

Smallest-eigenvalue eigenvector of the observed information, read off in four
charts. `|dlog .|` per unit length along each direction:

| truth | | `nu` | `gamma_0 = s^2` | `gamma_1` | `kappa` |
|---|---|---|---|---|---|
| `phi=0.95, s=0.55` | flat | 0.787 | **0.238** | 0.209 | 0.578 |
| | stiff | 0.619 | **1.394** | 1.435 | 0.816 |
| `phi=0.85, s=0.80` | flat | 0.681 | **0.150** | 0.048 | 0.628 |
| | stiff | 0.748 | **1.406** | 1.532 | 0.778 |
| `phi=0.70, s=0.40` | flat | 0.523 | **0.067** | 0.181 | 0.676 |
| | stiff | 0.908 | **1.412** | 1.675 | 0.736 |
| `phi=0.70, s=1.60` | flat | 0.546 | **0.096** | 0.139 | 0.658 |
| | stiff | 0.895 | **1.411** | 1.680 | 0.753 |

Information eigenvalue ratios 12–52. The Hessian is not lying: at a step of 0.25
the log-likelihood drops 0.02–1.8 along the flat direction against 1.0–22.3 along
the stiff one, in every cell.

**C2 is false.** The flat direction moves `nu` by 0.52–0.79 — it does not hold it
— and moves `kappa` by only 0.52–0.68, not 1.

## What is true instead, and it is sharp

Read the `gamma_0` column. The flat direction holds `gamma_0 = s^2` to
0.067–0.238, and the stiff direction moves it by **1.394–1.412**. In this chart
`dlog gamma_0` has gradient `(-1, +1)` in `(log kappa, log sigma^2)`, of norm
`sqrt(2) = 1.4142` — so 1.394–1.412 is between **99.86% and 98.6% of the maximum
possible**. The stiff eigenvector is aligned with `∇ log gamma_0` to within
**3.2°–9.7°**.

> **The data's stiff coordinate is the stationary variance `s^2`. The flat
> coordinate is persistence at fixed `s^2` — which is exactly the `phi` axis of
> the shipped `(phi, s)` chart.**

So the shipped chart is already correctly oriented, and the proposed
`(kappa, sigma^2)` chart is worse for this purpose: it is rotated ~45° off the
stiff/flat split, which is why the eigenvectors come out mixed in it.

## Where the reasoning went wrong

C2 confused two different questions.

- *What one step sees* is the increment. That governs the **walk** — how fast the
  window centre can chase a moving scale — and C2's argument is correct about it.
- *What the record sees* is the whole autocovariance function, and over a record
  the sample variance of `lambda` is pinned much harder than any increment. That
  governs the **bank**, which is the thing whose box was under discussion.

`nu = gamma_0 (1 - phi^2)` is a small difference of two well-determined numbers
once `phi` is near 1 — the classic badly-conditioned combination. It is the
*derived* quantity, not the primitive one. C2 assumed the reverse.

## What this buys anyway

The negative result lands on a graded open. `_PHIS` / `_SS` carry
`AUDIT[measured]` — the ledger's worst grade, *"ridge flatness does not clear the
bar"* (AUD-2). This measurement does not derive the box's **ends**, but it does
derive its **orientation and shape**:

- the box's two axes are the information matrix's own eigenvectors, to within
  3–10°, so the product grid is not an arbitrary chart — it is the diagonalising
  one;
- `s` is the stiff axis and `phi` the flat one, so an allocation that spends
  **more nodes on `s` (5) than on `phi` (3)** is the right way round.

That was previously asserted by flatness measurement. It is now a computation.

## What it costs the parent thread

C2 was [`0001`](0001_the_axiom.md)'s cheap motivation — "better coordinates, and
the bank drops 15 → 5". **That motivation is gone.** `(kappa, sigma^2)` is not a
better chart, and nothing here shrinks the bank.

Worse, the measurement raises a direct objection to the axiom itself. If
`gamma_0 = sigma^2 / (2 kappa)` is the sharply-identified coordinate, then the
`kappa = 0` member the new axiom wants to admit has `gamma_0 = ∞` — it sits at
infinity in the one direction the data determines well. Either the data will
never sit there, or `gamma_0` stops being the right coordinate off the stationary
manifold. This has to be settled by measurement, not by argument.

**That is [`0003`](0003_what_stationarity_costs.py): generate from `kappa = 0`
and ask what a stationary class costs on it.** It is now the thread's decisive
probe, and the honest position until it reports is that this workstream has one
falsified prediction and no established motivation.
