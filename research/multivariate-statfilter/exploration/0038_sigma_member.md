# 0038 — the within-member scale posterior: reach works, but per-step likelihood reintroduces the confound

Debugging the sigma-point windowed-GPB1 scale posterior (the parent's cure for the single-point
walk's under-reach, 0008) on the 5-DOF rig.

## The bug was the stationary under-reach (fixed)

The first port used the *stationary* AR(1) prediction (`mu_p = phi mu`, `Sig_p = phi^2 Sig + nu`).
Instrumented, `mu[pot]` stalls at 0.9 while the true burst scale is 5.42 — the phi-reversion pulls
back `0.1*mu` each step and the moment-matched window is capped at `s^2`, so it can neither open nor
walk far enough. The failing pot stays trusted, the state blows up (pot-hot 4.5x). CALM is perfect
(0.99), confirming it is a reach limit, not a code error.

The **unbounded** prediction (`mu_p = mu`, `Sig_p = Sig + q`, the window grows) reaches: `mu[pot]`
hits 5.6 = the truth, pot-hot drops to ~1.3-1.5, CALM stays ~1.0. So the scale posterior *is*
responsive on a burst and smooth on calm — the trade-off resolution a fixed `(phi,s)` cannot give.

## But it blows up on every process regime (the real obstacle)

Full sweep (unbounded, phi=0.9, s=0.5, 4 seeds):

| regime | q=0.3 | q=1.0 | q=3.0 | no-shed filter |
|---|---|---|---|---|
| CALM | 0.99 | 0.99 | 1.01 | 1.02 |
| SENSOR | 1.30 | 1.24 | 1.16 | 1.13 |
| pot-hot | 1.96 | 1.54 | 1.30 | 1.13 |
| **PROCESS** | **33.9** | **61.4** | **75.2** | 1.12 |
| **BOTH** | **22.5** | **40.5** | **50.0** | 2.35 |
| **process+pot** | **47.8** | **68.7** | **70.5** | 1.18 |

The posterior runs the scale away on process disturbances. This is the 0024 confound runaway, and
its cause is exact: the reweighting uses the **per-step** KF likelihood, which by optimality-proof
**Prop 1 cannot separate Q from R** — a process disturbance is equally well explained by inflating a
sensor's R, so the posterior inflates the wrong axis and coasts/diverges. The raw 0008 walker (which
got corr 0.99 on an n=2,m=2 rig with no strong collinear confound) discards the whiteness/Mehra
machinery (0032/0035) that the production filter uses to separate them from the innovation
*sequence*. The production filter does not blow up (PROCESS 1.12) precisely because it keeps that.

## Conclusion / path

A scale posterior is **not a drop-in** for the walk: it recovers responsiveness but reintroduces the
confound. The reach and the Q-vs-R separation both have to live in the same update. The targeted fix
is a **whiteness-gated adaptive scale posterior**: the window opens on a *white* surprise (a genuine
sensor failure, via the derived 0032 share that already subtracts the process-correlated part
`rho1*C0`) but stays closed on a *correlated* process burst. That is what the shed approximated with
a tuned rate; the principled version is the posterior window opening driven by the whiteness-share
residual, not the raw innovation. `q` (the open window-growth rate / finding-18 analogue) is then the
one remaining constant.

(The parallel "learn the shape online / KDE" thread is a seed with no prototype, and Theorem B shows
a heavy tail is just the AR(1) at phi->0 — so the adaptive posterior, not a KDE, is the lever; but it
still needs the whiteness separation.)
