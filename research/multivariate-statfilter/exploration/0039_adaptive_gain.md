# 0039 — the adaptive-gain walk: reach via a heavy-tail on the log-scale, kept confound-safe by C1

The no-shed walk under-reaches a burst because its gain is the fixed steady `K* = (1-phi)/4`. Rather
than a sigma-point posterior (0038, which discarded the C1 confound machinery and blew up on
process), keep the existing C1-based walk and only make its GAIN adaptive: a heavy tail on the
log-scale process noise (the Student-t E-step `q_eff = q(1 + delta^2/nu)`, `delta = step/s`) inflates
the gain on a surprising step, so the walk reaches. The step is still multiplied by the C1-derived
share `wg` (0032), so a process disturbance is attributed away, not run up — the confound stays with
the built lag-correlation machinery, **no new gate**.

`rate = min(1, K*(1 + q * (wg*step)^2 / s^2))`, `q = _QREACH`.

## The surprise must be the C1-ATTRIBUTED one

With the RAW `step` in the surprise the reach blows up process (process+pot 6.67x at q=8): a process
onset has a large raw `step` and `wg` has not yet dropped. Using the attributed `(wg*step)` halves
the damage (process+pot 3.20x at q=8) and reaches identically.

## Result (adaptive/oracle, 4 seeds, phi=0.9, s=0.5), attributed surprise

| regime | q=0 | q=0.5 | q=2 | q=8 | shed | no-shed |
|---|---|---|---|---|---|---|
| CALM | 1.05 | 1.05 | 1.04 | 1.04 | — | 1.02 |
| SENSOR | 1.39 | 1.13 | 1.10 | 1.09 | — | 1.13 |
| pot-hot | 1.13 | **1.03** | 1.01 | **0.99** | 1.03 | 1.13 |
| PROCESS | 1.08 | 1.26 | 1.65 | 2.15 | — | 1.12 |
| BOTH | 2.45 | 3.08 | 3.37 | 3.67 | — | 2.35 |
| process+pot | 1.18 | 1.31 | 1.98 | 3.20 | — | 1.18 |

The reach WORKS: pot-hot 1.13 -> 0.99 (beats the shed) and SENSOR 1.39 -> 1.09, monotone in q.
But it TRADES against the process regimes (BOTH, process+pot grow with q), because at a process
**onset** the whiteness `wg`'s EMA lags: for ~1/beta steps the channel looks white and the inflated
gain opens on the process before C1 confirms it is correlated. This is the responsiveness-vs-confound
limit, now living in `q`: reach faster than C1 can confirm and you misfire on process onsets.

## Open: the derived q

`q` is bounded by the C1 (lag-correlation) confirmation rate `~1/beta` -- you cannot reach faster
than the machinery can tell process from measurement. Whether there is a principled convergent /
minimax / asymptotically-stable q is the subject of the parallel s-interior-optimum study. `_QREACH`
is exposed and defaulted OFF (=0, exactly the safe no-shed walk) until that q is derived.
