# 0044 — the derived reach is always-safe: it activates only where a structural witness exists

Tested whether the derived eligibility degrades gracefully when the pot+accel structure (one integrated
+ one direct-readout sensor per joint) is absent. It does, exactly.

## Eligibility is sane across sensor configs (derived from H, Q0, rho)

| per-joint sensors | derived eligibility |
|---|---|
| `('pos',)` | pos=0.00 |
| `('acc',)` | acc=0.00 |
| `('pos','acc')` | pos=1.00, acc=0.00 |
| `('pos','vel','acc')` | pos=1.00, vel=0.91, acc=0.00 |
| `('pos','pos')` | pos=0.50, pos=0.50 |

A single sensor per joint has no coupled witness -> elig=0 (reach off). Redundant sensors witness each
other -> 0.50 each. With pos/vel/acc, position is most decoupled from the direct process footprint
(elig 1.00), acceleration least (0.00), velocity between (0.91) -- the ordering the physics demands.

## No-witness rigs: derived == floor to the digit (never misfires)

| config | regime | floor | derived | diff |
|---|---|---|---|---|
| `('pos',)` | pot-hot | 1.174 | 1.174 | +0.000 |
| `('pos',)` | process | 4.249 | 4.249 | +0.000 |
| `('acc',)` | pot-hot | 1.050 | 1.050 | +0.000 |
| `('acc',)` | process | 1.048 | 1.048 | +0.000 |

With no witnessing partner the reach is identically the floor -- including the pos-only process regime
(4.249, intrinsically hard: no accel to catch the process, the pot drifts freely), where a naive reach
would misfire hardest. The reach does nothing there because it CAN do nothing safely. This is the
always-safe property: the reach only turns on where the known (F, H, Q0) structure provides a witness
that separates process from sensor per-step, and stays at the floor everywhere else.

## Remaining

Still to do: the principled magnitude (0045 -- walk to the derived target at a q-free spatially-gated
rate, `rate = K* + elig*discount*(1-K*)`, the explicit saturated limit) and the AR(1)-family / higher
seed regression check. Nothing merged; hook off-by-default.
