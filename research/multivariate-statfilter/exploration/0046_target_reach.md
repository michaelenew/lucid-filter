# 0046 — robust-MAP target REJECTED; the magnitude keeps one not-delicate constant on a flat plateau

> **⚖️ ATTRIBUTION —** _Negative result: a robust-MAP walk target is worse (it reads the instantaneous e² and is more reactive, not gentler), and the BOTH-safety comes from a finite quadratic soft-threshold surcharge (a bounded per-step move); q on a flat plateau in ~[1,8] is benign but not derived._ Prior art: soft-thresholding / graduated robustness (standard); Student-t gain inflation — Agamennoni et al. 2012. Status: NEGATIVE-RESULT.

Tried to erase 0045's small BOTH regression two ways.

## Robust-MAP target makes it worse (rejected)

| regime | floor | etar target | 0045 q-free | 0045 q=4 |
|---|---|---|---|---|
| pot-hot | 1.522 | 1.159 | 1.056 | 1.099 |
| SENSOR | 1.230 | 1.300 | 1.247 | 1.233 |
| PROCESS | 1.089 | 1.159 | 1.096 | 1.096 |
| BOTH | 2.144 | 2.391 | 2.251 | 2.165 |

Walking toward the robust-MAP eta_r (0031) is WORSE across the board (BOTH 2.391, SENSOR/PROCESS also
regress). The reason: eta_r reads the INSTANTANEOUS e_i^2, and in BOTH the pot's e^2 is process-
elevated -- so eta_r is a MORE reactive target, not a gentler one. The earlier intuition (eta_r
"bounded/outlier-aware" -> gentler) was wrong; the boundedness is in the DOWN-weight direction, not the
reach direction. Rejected.

## The BOTH regression is full SATURATION, not the target -- a finite surcharge is BOTH-safe

The cause is the jump SIZE. q-free sets gain->1 and jumps the entire remaining distance in one step, so
a single spurious un-discounted step (instantaneous-discount chi^2 noise) moves mu a lot. The 0045 q=4
QUADRATIC surcharge `K* q (wg step)^2/s^2` moves mu by a BOUNDED amount per step, so rare spurious
steps don't accumulate -- BOTH stays +0.021 while pot-hot is 1.099. The 0042 sweep already showed the
losses are q-FLAT over q=0.5..8 (BOTH ~2.16); only q->inf overshoots. So:

- **best config**: the derived spatial gate (elig*discount, 0043) x a FINITE quadratic surcharge,
  q ~ 4. Strong gains (pot-hot 1.52->1.10, process+pot 1.68->1.33 below floor), BOTH-safe (+0.02).
- **q is a residual, not-delicate constant**: any q in ~[1, 8] gives BOTH-safe strong reach (flat
  plateau); the exact value is immaterial. The CONFOUND crack (the hard part) is fully derived and
  parameter-free; only the reach SPEED carries this one benign constant.

## Honest status vs the "no tuning parameters" bar

The confound separation, eligibility, coupling, discount, and safety are all derived from (F, H, Q0)
with no constant. The reach MAGNITUDE retains q -- benign (order-of-magnitude, flat plateau) but not
derived. Attempts to remove it (q-free saturation -> BOTH cost; robust-MAP target -> worse) did not
land. So under a strict "no constant in the model" reading, the reach is not yet admissible; under a
"no DELICATE/tuned constant" reading, q on a flat plateau qualifies. This is the decision to put to the
user. The parameter-free FLOOR (no reach) remains the safe default either way; the reach is a strict
improvement gated on accepting q. Nothing merged; hook off-by-default.

## Why the quadratic surcharge is BOTH-safe and q-free is not (both clip to `gap`)

Both forms clip the per-step move to `gap`, so the clip is not the difference. The difference is SHAPE
in the small/moderate-surprise regime. q-free moves `~gate*(1-K*)*step` -- LINEAR in the surprise, so
it reaches on EVERY gate-open step, including the moderate process-elevated steps of BOTH. The
quadratic surcharge `K* q (wg step)^2/s^2` is a SOFT THRESHOLD: negligible for moderate steps, engaging
only on LARGE surprises (a real sensor burst). So it ignores BOTH's moderate process elevation and
reaches only genuine bursts -- that is what keeps BOTH safe. The reach SHAPE (quadratic soft-threshold)
is the load-bearing part; q is only its scale.

## Open thread if we keep hunting for a derived q

The quadratic surcharge is exactly the Student-t E-step gain inflation `q_eff = q(1 + delta^2/nu)`,
delta = step/s, with q ~ 1/nu (0039). So a derived q is a derived heavy-tail DOF nu for the log-scale
process. The optimality-proof (Theorem B: a heavy scale tail adds to gamma0) may pin nu from the
well-posedness constraint E[e^lambda] < infinity (0024) -- the maximal tail that keeps the scale
integrable. Whether that yields a specific nu (hence q) is untested and the next lead for a fully
parameter-free reach.
