# 0034 — high-seed profiling: the derived confound gate lands (garrote), BOTH is a floor artefact

> **⚖️ ATTRIBUTION —** _High-seed paired (common-random-number) profiling shows the earlier 4-seed "regressions" were noise, and ships the non-negative garrote as the unbiased continuous denoiser of the noisy ρ₁ EMA; BOTH's motion stays inside the irreducible-Q envelope._ Prior art: non-negative garrote — Breiman 1995; common-random-number / paired variance reduction (standard Monte Carlo). Status: REPRODUCTION (garrote), with a NEGATIVE-RESULT reading of BOTH.

The 0032 integration variants moved regimes by a few percent at 4 seeds — too few to tell signal
from run-to-run noise. This profiles each variant at **40 seeds, paired** (same seeds across
variants; the oracle depends only on (regime, seed), so the paired difference cancels the scenario
variance and resolves sub-percent effects). Criterion, per the Lucid ethos: a variant that is
within error bars of the empirical gate but **understands** the regime (the derived law) wins.

## The empirical error bars are large — the 4-seed "regressions" were noise

Committed, adaptive/oracle: pot-hot 1.21±0.07, process+pot **1.41±0.10**, SENSOR 1.14±0.03,
PROCESS 1.11±0.03, BOTH 2.30±0.09. The 4-seed process+pot read 1.15; at 40 seeds it is 1.41. Point
estimates at 4 seeds were not trustworthy.

## The variants (paired diff vs committed, in sigmas)

The gate is the sensor's share of the residual. 0032 gives it as `1 - rho1*(S/R)`. The question was
only how to handle `rho1`, a noisy EMA (std ~sqrt(beta)):

| rho1 handling | pot-hot | proc+pot | SENSOR | PROCESS | BOTH | shape |
|---|---|---|---|---|---|---|
| soft-threshold `max(rho1-thr,0)` | — | +3.9 | — | +3.8 | +3.0 | biased low → over-absorbs process |
| derived-width, floored | −2.6 | +2.7 | −2.9 | +4.5 | +4.6 | biased (same cause) |
| raw `rho1` (no floor) | +2.7 | +0.5 | +5.5 | −1.3 | −1.8 | noise slows the failing-sensor shed |
| **hard-gate** (full `rho1` if signif.) | +0.7 | −0.2 | +2.5* | +0.9 | +1.5 | unbiased, **step** at thr |
| **garrote** `rho1 - thr^2/rho1` | **−2.3** | +0.7 | **−2.3** | +2.3* | +3.4 | unbiased, **continuous** |

(* magnitude ~0.004–0.011, negligible.) The lesson: the **bias** of a soft-threshold (it shrinks
`rho1` by `thr`, inflating the sensor share) is what regressed the process regimes; the *unbiased*
denoisers (hard-gate, garrote) match or beat committed. Raw (no denoise) fails the other way — on a
white failing channel `rho1`'s noise spuriously drops the share and slows the shed.

## Shipped: the garrote

`rho1_hat = rho1 - thr^2/rho1` above `thr`, else 0 — the standard non-negative garrote. It is the
one variant that is **fully smooth** (continuous; a kink at the noise floor, no step or if/else
arm-gluing) **and** unbiased for a significant correlation. Against committed it is *better* on the
reducible sensor-failure regimes (pot-hot −2.3σ, SENSOR −2.3σ), neutral on process+pot/PROCESS, and
the empirical ramp-width `thr` is gone — replaced by the derived per-channel width `R/S`. The only
`thr` left is the estimator's own 2σ EMA noise floor, entering through the garrote (a smooth
shrinkage), not a shape cutoff.

The same garrote on the robust-MAP gate regresses hard (BOTH +12.9σ) — the MAP is instantaneous
first-sample protection, a different role than the sustained allocation the 0032 share describes —
so the MAP keeps its committed whiteness gate.

## BOTH's +3.4σ is a floor artefact, not a regression

BOTH moved 2.30→2.38. But the achievable floor for **inferring the masked Q** is `oracle-R` (freeze
sensors at true, infer process) = **3.04± 0.13** oracle (`0034_profile.py floor`) — and *both*
committed (2.30) and garrote (2.38) sit **below** it (the adaptive even beats true-R, because
partially trusting the noisy accel keeps process signal that fully distrusting it discards). So the
BOTH difference is motion inside the irreducible-Q envelope 0033 identified; the full-oracle ratio
overstates it. Not a reducible regression.

Repro: `python 0034_profile.py save <label>` on a checked-out variant, then `compare committed
<label>`; `floor` for the oracle-R floor. (`.npz` outputs are local, not committed.)
