# 0035 — deriving the empirical parameters (the "no tuned params" pass)

> **⚖️ ATTRIBUTION —** _An audit replacing "magic" constants with derived ones: the 2σ χ²₁ outlier point 1+2√2, and the process-walk gain as a Newton whitening rate K*/b_k built from the closed-loop Lyapunov error covariance and the Mehra lag-1 innovation autocovariance C₁=HAMHᵀ−HFKR._ Prior art: Mehra 1970 lag-1 innovation autocovariance; Lyapunov steady-state covariance (standard); χ² innovation gating (standard). Status: RECOMBINATION.

An audit of the filter's constants, each classified derived / labeled-budget / measured, and the
measured ones fixed where a derivation exists. Profiled at 40 seeds, paired (`0034_profile.py`).

## Fixed

**The outlier-shed floor `nis - 4` → `nis - (1 + 2 sqrt2)`.** `nis = e^2/S` is `chi^2_1` (mean 1,
std sqrt2), so its 2-sigma significant-outlier point is `1 + 2 sqrt2 ~ 3.83` — the "4" was that,
rounded. Exact substitution, neutral (paired diff ~0).

**The process-walk gain `_Q_DRIVE = 0.2` → the derived Newton whitening rate `K*/b_k`.** The process
mode whitens its lag-1 innovation autocorrelation `sig_k`; the right relaxation is Newton at the
*sensor walk's own* rate `K*`: `mu += (K*/b_k) sign(sig)(|sig|-thr)`, where `b_k = -d sig_k/d mu_k`
is the steady-state whitening sensitivity. `b_k` is a construction-time quantity: with the gain from
the *assumed* scale but the *true* process cov `Q0`, the actual a-priori error cov solves the
closed-loop Lyapunov `M = A M A^T + F K R K^T F^T + Q0` (`A = F(I-KH)`), the lag-1 innovation
autocovariance is the Mehra `C1 = H A M H^T - H F K R` (zero at the optimum), and `b_k` is its
mode-direction slope by finite difference (`_whiten_gain`). For this rig `K*/b_k = 0.227` on the
jerk modes — i.e. **`_Q_DRIVE = 0.2` was already the Newton gain**; the derivation replaces the
magic number with the per-mode formula (the correct generalisation to any `F, H, Q0`), capped at the
SOR limit `K*/(4K*) = 1/4` so a barely-whitening mode is not over-relaxed. Neutral except BOTH
+0.06 (the Newton rate does not over-relax; see below).

## Measured, but validated as the derived value — kept

The sensitivity sweep (0034, 2x each) showed the process gain has an *over-relaxation* optimum at
~0.8 (BOTH 2.38→1.97) that breaks at 1.6 (process+pot). That is SOR acceleration, **not** the
derived rate — the Newton gain is `K*/b ~ 0.2` across operating points (nominal and disturbance).
The derived Newton rate is the principled choice; the 0.8 over-relaxation is a heuristic that only
helps the irreducible BOTH regime and is left out.

## Confound-coupled — the observability build

- **`_SHED` (shed speed), `_WHITE_MIN` (shed whiteness floor).** Sensitive (2x moves regimes several
  sigma) — a genuine balance: faster shedding helps the sensor-failure regimes, hurts the process
  regimes, because at a process *onset* the whiteness gate `wg` lags (its `rho1` EMA has not built),
  so a fast shed misfires on process. The gentle `_SHED` is the onset-safety-vs-speed balance; it is
  not derivable without distinguishing process from sensor *without* the temporal lag — the
  cross-sensor **observability-weighted** build.
- **`_Q_REVERT` (process-scale forgetting).** A labeled timescale: slower is better for a sustained
  disturbance (BOTH), faster forgets a transient. Tie-able to the adaptation timescale; left as a
  budget for now.

## Already derived / labeled budgets (unchanged)

`K* = (1-phi)/4`, `q_mu`, the spectral floor, the Sparrow grid `1.5 s`, the sensor share
`1 - rho1 (S/R)`, `thr = 2 sqrt(beta)` (the EMA noise floor). `_BETA`, `_SPAN_S` are labeled budgets.
The **robust-MAP whiteness gate** `clip(1 - (rho1-thr)/thr)` is built entirely from the derived
`thr` (a detection ramp at the noise scale) — no measured constant; a garrote/`S-R` variant regresses
(the MAP is instantaneous first-sample protection, a different role than the sustained allocation).
