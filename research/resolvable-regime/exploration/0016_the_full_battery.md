# 0016 — The full battery: span 6 wins the scalar rig, loses the async rig, so it does not ship

> **AI-generated, not peer-reviewed.** The shipped filter modified on this branch
> and measured against its own acceptance rigs. `lucid.py` ends this probe with
> `_SPAN_S = 3.0` unchanged and two numerical guards added.

## What was tried, and why

[`0012`](0012_diagonalise.md)–[`0014`](0013_the_arbiter.md) diagonalised the
window-geometry plane on the scalar hero rig and found: `gap = 1.5` pinned by code
length; `span` a code-length **plateau** over 1.5–12; and within it `span = 6` the
jump-window RMSE preference. The hypothesis to test on the real filter: *span 6 is
a genuine improvement, and the tiny code-length signal is the truth that RMSE's
noise obscured.*

So `_SPAN_S` was set to 6.0 in `lucid.py` and the full acceptance battery run.

## What the battery found

### 1. The scalar hero rig — span 6 wins, cleanly (96 paired seeds, [`0015`](0015_span6_analysis.py))

| metric | shipped (span 3) | span 6 | t | change |
|---|---|---|---|---|
| jump-window RMSE | 3.323 | 2.783 | **−12.1** | **−16.3%** |
| regime-C RMSE | 0.823 | 0.776 | −3.5 | −5.8% |
| steady RMSE | 0.1341 | 0.1342 | 1.8 | +0.06% |
| total code length | 1.91202 | 1.91039 | **−3.9** | −0.09% |
| code length, block C | 2.61722 | 2.61219 | −4.2 | −0.19% |
| settle steps | 27.7 | 24.8 | −1.8 | −10.4% |

The user's read was right *here*: the jump improves 16% at `t = −12`, every
code-length block improves, steady state is flat, and the total code-length
improvement is now **resolved** (`t = −3.9`, where at 24 seeds it was `t = −1.46`).
On the scalar rig span 6 is close to a straight win.

### 2. The arm rig — neutral (1 matched seed)

| regime | span 3 | span 6 |
|---|---|---|
| calm | 1.14 | 1.16 |
| SENSOR | 2.70 | 2.45 |
| PROCESS | 1.16 | 1.08 |
| POTFAIL | 1.11 | 1.01 |
| BOTH | 1.06 | 1.10 |

Mixed and small — span 6 slightly better on three regimes, slightly worse on two.
Not where the decision turns.

### 3. The asynchronous multi-rate rig — span 6 is a catastrophe

The pointwise rig (5 Hz absolute, 100 Hz rate, 12 Hz jittered, sensor 2 degrades
×10), driven entirely through `observe(sensor, value, t=)`:

| ratio to oracle | shipped (span 3) | span 6 |
|---|---|---|
| whole run | 1.162 | **3.772** |
| calm | 1.027 | 1.027 |
| sensor 2 hot | 1.438 | **7.695** |

Span 6 makes the filter **worse than the fixed-noise filter** (2.103) on the very
rig that shows off adaptation. This is not noise and not corruption — it
reproduces on a clean re-run.

## Why: `_SPAN_S` is not a pure reach knob

`_SPAN_S` enters three places, not one:

```
node count : K = ceil(_SPAN_S / _GAP_FACTOR)      -> reach          (the intended axis)
walk cap   : _Pmu_cap = (_SPAN_S * s)^2           -> GROWS AS span^2
walk floor : _Ifloor  = (1-phi)/(4 (_SPAN_S s)^2) -> SHRINKS AS 1/span^2
```

At `phi=0.95, s=0.8`, span 3→6 takes `_Pmu_cap` from 5.76 to **23.04** (4×) and
`_Ifloor` from 2.17e-3 to 5.43e-4. So doubling the span quadruples how far the
scale **walk mean** is allowed to wander and quarters its drift floor.

On a **synchronous** rig every step is a full observation, so the walk is
corrected every step and the looser cap is just extra headroom — pure reach, which
is why the scalar rig sees only the upside. On the **asynchronous** rig a slow
sensor (12 Hz on a 100 Hz clock) reports once per ~8 steps, and its scale walk
runs with **partial-event, low per-event identifiability** in between. There the
4× cap is not headroom — it is room to run away, and during the ×10 degradation
the walk overshoots and the estimate pays for it. The scalar sweep could not see
this because the scalar rig has no partial events.

> The scalar-rig code-length **plateau in span is rig-local.** It holds where
> every event is complete and fails where events are partial, and the mechanism is
> the `span²` coupling into the walk's own freedom, not the node count.

## Decision

`_SPAN_S` stays at **3.0** — the value that holds every rig. The diagonalisation
is a real improvement on synchronous problems and a real regression on
asynchronous ones, so it is not a global default. (A future per-axis or
per-event-completeness span would keep the scalar win without the async loss;
that is a bigger change than a constant and is not attempted here.)

## What the experiment leaves in the filter — two genuine bug fixes

Running span 6 exposed a hard crash that also reaches the **shipped** span 3:

- `_inv_sym` / the log-determinant raised `LinAlgError` / returned `NaN` on a
  numerically singular innovation covariance. At an extreme scale node the process
  variance dwarfs every sensor variance, so two sensors of one state have
  innovation correlation `1 − eps`: `S` is finite but numerically rank-deficient
  (measured cond 1.9e15).
- **Reachable at the shipped span 3 by a user**, not only at span 6: a
  give-what-you-know call with a wide scale box on a two-sensor fault rig —
  `LucidFilter(faults=True, ss=(0.5, 2.0, 8.0, 16.0), H=[[1],[1]], ...)` — crashes
  without the guard, at span 3.

Two guards, both `AUDIT[budget]` (bit-identical on the non-singular path, which the
kernel pins enforce; a singular node's likelihood limit is 0, so the guard makes
it contribute nothing rather than crash):

- `_inv_sym` falls back to a per-block pseudo-inverse when `np.linalg.inv` refuses;
- `_logdet_sym` maps a non-finite / non-positive `slogdet` to a large finite
  penalty (`_LOGDET_SINGULAR`), so an all-singular softmax window stays NaN-free.

Full suite after the change (span 3 restored, guards in): **55 passed, 16
skipped** — including the bit-for-bit kernel pins, so the guards change no shipping
number.

## Limits

- Arm span-6 is 1 matched seed, not the 3-seed acceptance run; enough to show it
  is not the deciding rig, not enough for a table.
- The `span²` mechanism is argued from the constants and the async/sync contrast,
  not isolated by an ablation that holds `_Pmu_cap` fixed while moving node count.
  That ablation would confirm it and is the obvious next step if span is revisited.
