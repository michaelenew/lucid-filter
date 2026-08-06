# 0009 — The sharper likelihood moves the fit toward truth, and the boundary stays ill-posed: the case for marginalising s_P

Probes: [`0006`](0006_the_fit_on_the_imm_likelihood.py) ·
[`0007`](0007_decomposing_the_remaining_gap.py) ·
[`0008`](0008_the_kick_control.py) ·
numbers: `figures/gap0006.json`, `gap0007.json`, `gap0008.json`

## The gap, fully decomposed (`0007`)

On `0038` B's exact data, every owner of the static-to-oracle span
(0.1029 nats/pt), separated by switching each independently:

| owner | share | note |
|---|---|---|
| captured by the shipped forced channel | 80.0% | `0039`'s number, reproduced |
| **the GPB1 collapse** | **9.5%** | repairable: IMM, no new parameters |
| the channel model (AR(1) log-scale vs true regime chain) | 6.8% | a model commitment, kept |
| detection lag | 3.7% | irreducible for any causal filter |

The causal ceiling — per-node covariances on the *true* two-state grid with
hazard-matched transitions — reaches **96.3%** of the oracle. Two side
results: GPB1 handed that perfect grid still loses 12.6 points to its
collapse; and a no-mixing bank ($T=I$) closes **0.0%** — after 400 baseline
steps the regime node carries so much accumulated prejudice it never
recovers. The kernel's forgetting is essential, not overhead. Resolution is
not the gap: order 9 is mildly *worse* than order 5 here.

## The fit comes most of the way home (`0006` B)

The staged fit, identical in every pass except the likelihood it sees, on
three draws of `0029`-style data (truth $Q=1$, $s_P=0.8$, $\varphi_P=0.9$):

| seed | GPB1 endpoint $(Q, s_P, \varphi_P)$ | IMM endpoint |
|---|---|---|
| 19 | (0.39, 1.44, 0.70) | (0.68, 1.07, 0.83) |
| 43 | (0.11, 2.14, 0.74) | (0.49, 1.28, 0.85) |
| 44 | (0.57, 1.42, **0.00**) | (0.89, 1.08, 0.65) |

GPB1's endpoints scatter along the ridge — $Q_{\text{eff}}$ pinned near 1.1,
everything else loose, $\varphi_P$ anywhere from 0 to 0.74. The IMM fit
moves **every coordinate toward the truth on every seed**. It does not land
exactly: $\hat s_P$ still overshoots (~1.1 vs 0.8) with $\hat Q$ low. The
ridge is tilted, not abolished, at $n = 900$.

## And the boundary stays ill-posed (`0006` C, `0008`)

At `0032`'s cached fitted parameters the IMM recursion is **0.0031 nats/pt
better** — the literal do-no-harm gate passes. But the IMM *fit* on that
window puts $\hat s_P = 0.87$ where `0039` had declared the fitted zero
correct. The suspicion that it was detecting the window's three 6-SD kicks
(one-off process disturbances — exactly what an impulsive channel
represents) is **refuted by the control**: same window, same draws, kicks
removed —

| window | GPB1 $\hat s_P$ | IMM $\hat s_P$ |
|---|---|---|
| with kicks, seed 20260801 | 1.47 | 1.19 |
| **without kicks** | 0.00 | **0.85** |
| with kicks, seed 7 | **0.00** | 0.41 |
| without kicks | 0.00 | 0.32 |

IMM reads structure into kick-free homoscedastic windows; GPB1, for its
part, stares at three 6-SD kicks on seed 7 and reports a dead-flat model
(and on the other seed reports $s_P = 1.47$ — the endpoint inconsistency
again, now within one probe). The kicks do move $\hat s_P$ up under IMM
(1.19 > 0.85, 0.41 > 0.32), so the sharper likelihood is *seeing* them; it
just has no stable zero to stand on.

**The reading**: near $s_P = 0$ the Fisher information in $s_P$ vanishes —
a spread parameter's likelihood is flat to second order at zero spread — so
a *point estimate* there is ill-posed under any likelihood. GPB1's total
flatness collapses $\hat s_P$ to 0 or lets it slide to 2; IMM's restored
curvature stops the slide at large $s$ but still scatters small-positive
estimates over [0.3, 0.9] where the truth is 0. Neither is a defect of the
search. **It is the plug-in that is wrong**, which is `0039`'s item 0a
arrived at from the opposite direction — and the premium ledger (`0006` D:
IMM +0.0055 ± 0.0009 for an unnecessary channel, against 0.0921 of
exposure) says the *cost* of the ill-posed boundary is 16× smaller than the
cost of the boundary zero it replaces.

## Where this leaves the patch

Two changes are now earned, and they are one design:

1. **Per-node covariances (IMM) in `core.py`** — closes 9.5 points of the
   gap at fixed parameters, brings fitted endpoints most of the way home,
   passes the literal 0032 gate at fitted parameters, costs ~1.4× in the
   reference implementation, and is bit-identical to the shipped recursion
   at $s = 0$.
2. **Marginalise $s_P$ instead of plugging it in** — a small grid over
   $(\varphi_P, s_P)$ carried the way every other nuisance in this filter
   already is. Under GPB1 this was pointless (a flat likelihood integrates
   to indifference); under IMM the posterior has something to say. The
   ill-posed point estimate then stops being anyone's problem: $s_P = 0$
   and $s_P = 0.3$ are both members with weight, and the 35× loss asymmetry
   is priced by the mixture instead of gambled on a boundary.

Not established: the marginalised design's cost and its score on the full
`0032` series; the batched IMM's fit-time cost inside `fit_` (the reference
fits ran ~4 minutes at $n = 900$ against ~15 s shipped — the polish, not
the evaluator, is where the time went); crypto's real basins under IMM.
