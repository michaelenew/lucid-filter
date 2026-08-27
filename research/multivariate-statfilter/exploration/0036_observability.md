# 0036 — the observability/decoupling shed: retiring the shed's empiricism where it's derivable

The fast shed had two measured constants: `_SHED` (raise-rate boost) and `_WHITE_MIN` (whiteness
floor). They exist because the shed must fire fast on a failing sensor but not misfire on a process
burst, and the whiteness gate that tells them apart has an EMA that **lags a burst onset**. This
build retires them for the sensors where it is possible, and pins down why the rest is irreducible.

## The weight — process-decoupling, not observability criticality

First idea: gate the shed by observability criticality (how much losing a sensor hurts the state).
It FAILS — it is high for any irreplaceable sensor, including a process-coupled one (a lone
position sensor with process near position: the single-sensor crusher test, which this broke).

The right static quantity is the channel's **process-coupling**: the lag-1 innovation
autocorrelation it picks up under a process disturbance (gain from the assumed nominal scale, actual
error cov from the true elevated cov via the closed-loop Lyapunov, Mehra lag-1 `C1 = HAM Hᵀ − HFKR`;
it saturates at the channel's intrinsic closed-loop decay, a parameter-free limit). **Decoupling =
1 − coupling** is bimodal and clean:

| sensor | coupling | decoupling |
|---|---|---|
| 5-DOF pot (position, jerk 3 integrations away) | ~0.00 | ~1.0 |
| 5-DOF accel (jerk-driven) | ~0.74 | ~0.26 |
| 2-state lone position sensor (process near position) | ~0.94 | ~0.06 |

## The hybrid shed (`_decouple_weight`)

- **Decoupled absolute reference** (decouple > 0.5): a channel that stays white under process can
  never confuse a failure with a burst, so an outlier is *always* a failure → shed **fast and
  statically**, `rate = min(1, K*(1+surprise))`, no whiteness gate, no `_SHED`/`_WHITE_MIN`. This is
  the derived part, and it also made the failing-absolute regimes better.
- **Process-coupled channel** (decouple ≤ 0.5): its outlier could be its own failure OR a process
  burst — a call only the **dynamic** whiteness can make — so it keeps the gentle whiteness-gated
  shed (`_SHED`, `_WHITE_MIN`).

## Result (paired, 40 seeds, vs the last clean baseline PR#20)

pot-hot 1.19→**1.05**, process+pot 1.42→**1.31**, SENSOR 1.13→1.13, PROCESS 1.12→1.12,
BOTH 2.38→**2.35** — neutral-or-better on **every** regime, with the failing-absolute regimes
clearly better. (This commit also fixes a leaked `_Q_REVERT = 0.016` from PR#21 back to 0.008.)

## Why the coupled shed is irreducible — both dead ends recorded

- A **static** weight cannot replace the dynamic gate: gating the accel's shed by decoupling alone
  regressed SENSOR (+10σ) — a genuine white failure on a coupled channel then never sheds, because
  the static weight cannot tell it from a process burst.
- **Removing `_WHITE_MIN`** from the coupled branch regressed the process regimes (+5–7σ): without
  the floor, an onset misfire runs away (raising `R` dilutes the channel's ρ1 → it looks whiter →
  more shed). The gentle slope + whiteness floor together bound the onset-lag misfire.

So the coupled-channel shed is the one confound-coupled corner that stays empirical; the derivable
part (decoupled references) is derived, and the residual is now confined and explained.
