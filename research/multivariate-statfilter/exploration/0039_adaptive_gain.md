# 0039 — the adaptive-gain walk: reach via a heavy-tail on the log-scale, kept confound-safe by C1

> **⚖️ ATTRIBUTION —** _An adaptive-gain reach via a Student-t heavy-tail E-step (q_eff = q(1+δ²/ν)) gated by the derived process share; works on a single sensor but trades net-negative multivariate because the whiteness EMA lags a process onset by ~1/β. The q-study classifies q as a convergent floor (=q_μ) plus an un-derivable minimax reach surcharge._ Prior art: Student-t / variational-Bayes heavy-tailed filtering — Agamennoni et al. 2012; stochastic-approximation gain — Robbins–Monro 1951. Status: NEGATIVE-RESULT.

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

## Resolved: q is a convergent floor + a minimax reach surcharge (the q-study)

A parallel study (scratch_*.py; fast scalar rig, faithful to WalkingFilter -- K* exact, q_mu form
matched) quantified the burst-optimal swing / q across burst magnitude B, duration L, ridge phi, and
AR(1) data families. It reproduces the 5-DOF interior optimum (phi=0.85, B=5, L=150 -> s*=0.80) and
cleanly decomposes it:

- **Steady (jitter) optimum is B-INDEPENDENT** -- s_steady* = 0.15 for every B, i.e. q_steady* sits on
  the finding-18 floor q_mu = K*^2/(I_char(1-K*)) (~0.011 at phi=0.85). Sustained tracking wants the
  SMALLEST q that still tracks. This piece is **convergent / derivable**, and it is the K* gain.
- **Transient (reach) optimum GROWS with B**: s_transient* ~ 0.10 B, and in the windowed walker
  q_reach ~ B^2/(c*tau) (c~3 from the sigma-span sqrt(3 Sig) reaching B in ~tau steps). Sweeping the
  latency tau, q* increases monotonically with B at every tau and with 1/tau at every B, with **NO
  fixed point** -- a **minimax** quantity set by the worst-case (B_max, tau) envelope, not intrinsic.

Classification: not asymptotically-drifting (stable across n/seeds); convergent in the sustained
limit; minimax in the reach requirement. A single fixed q cannot serve both. On the AR(1) *family*
(genuine stationary wandering) there is only a flat self-consistent ridge (s* ~ s_data, ~1% deep) --
confirming the sharp interior optimum is a BURST/reach phenomenon, not a stationary-family one.

**Conclusion.** The convergent floor q_mu is derivable and is what the filter already uses (K*). The
reach surcharge is not a derivable constant -- it is a burst-envelope commitment (a tuning
parameter). On the multivariate rig it is additionally confound-bounded (tau >= 1/beta, the C1
confirmation rate) and, as 0039 measured, trades net-negative because the process onset-lag misfire
outweighs the sensor-reach gain. So the **parameter-free filter holds the floor q = q_mu** (the
current no-shed walk, pot-hot ~1.13). The shed's extra ~0.1x was an implicit burst-envelope
commitment -- real, but not parameter-free. (The study's own architectural suggestion, a q_mu floor
transiently opened on a white surprise, is exactly 0039's adaptive gain; it works on a single sensor
but the confound onset-lag defeats it multivariate -- so it is not net-positive here.)
