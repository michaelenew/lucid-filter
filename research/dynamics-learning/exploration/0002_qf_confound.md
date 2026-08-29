# 0002 — the Q↔F confound splits through per-member means at a derived rate; the operative mask is burst-THEN-fault

Second rung (`0002_qf_confound.py`): the 0001 rig with a live noise axis.  Members on
the anchored 2×2 grid {a 0.9, 0.6} × {q, 4q}, each a full KF with its own gain,
hazard-mixed with the product kernel (rho = 1/T per axis).  The ×4 burst is chosen to
be maximally confusable on variance alone: it inflates innovation variance ×1.76 where
the fault inflates it ×1.84.  Scenarios CALM / DYN / NOISE / BOTH (all at t*),
plus STAGGER (burst at 1200, fault at 1800).  150 seeds each.  The pairwise KL
machinery from 0001 generalizes: member gains come from their own assumed q, truth
covariances from the joint recursion, so every delay below has a derived rate next to it.

## 1. The split works, with no whiteness statistic anywhere

Settled marginal posteriors (window [t*+400, T)):

| scenario | truth dyn/noise | bank4 P(dyn) | bank4 P(noise) |
|---|---|---|---|
| CALM  | 0/0 | 0.004 | 0.013 |
| DYN   | 1/0 | 0.997 | 0.014 |
| NOISE | 0/1 | 0.009 | 0.983 |
| BOTH  | 1/1 | 0.994 | 0.980 |

The 0053 §1 mechanism — per-hypothesis filters carrying sequence evidence through
their **means** — is sufficient to split wrong-F from elevated-Q in the enumerated
setting: a wrong-F member mispredicts *correlated with the state*, an elevated-Q member
mispredicts only in scale, and the joint likelihood tells them apart at the pairwise KL
rate.  The hypotheses here are stable anchors (0053 §3's requirement); nothing walks.

State cost (settled RMSE/oracle): bank4 reads **1.001 / 1.002 / 1.000 / 1.000** across
CALM/DYN/NOISE/BOTH — at the oracle everywhere, including under simultaneous stress
(the 0052 process+pot analogue).  Transition windows peak at 1.09.

## 2. What each missing machinery costs (the SUMMARY's warning, quantified)

- **bankD (dynamics only) under NOISE**: false "dynamics fault" crossings in **100%**
  of seeds (settled mean P(dyn) stays low, ~0.04, but every seed spikes past 0.5).
  A dynamics-only probe measured without the noise machinery would still *look* fine on
  state RMSE (1.185 vs frozen 1.191 — a wrong-dynamics member acts like a higher-gain
  filter), which is exactly why the warning matters: the failure is in the *flag*, not
  the track, and a fault flag that fires on every noise burst is useless.
- **bankN (noise only) under DYN**: it eats the fault — settled P(noise) = **0.977**
  under a pure dynamics change, state cost 1.393.  Masking compensates part of the
  track (1.393 < frozen's 1.672: believing "more noise" raises the gain) at the price
  of a confidently wrong attribution.
- **bank4** transient wrong-axis spikes exist (37%/29% of seeds cross 0.5 briefly)
  but settle correctly and cost ~nothing in state (1.000/1.002) — the hedge economics
  of 0001 §2 again.

## 3. Detection and attribution are different speeds, and the theory prices both

Pairwise KL rates under each truth (nats/step), with the binding attribution frontier:

| truth | vs nearest member | rate | rho-frontier |
|---|---|---|---|
| DYN   | 'both'  | 0.082 | 98 steps |
| NOISE | 'nom'   | 0.163 | 49 steps |
| BOTH  | 'dyn'   | 0.143 | 56 steps |

Detecting *that something changed* is fast (marginals cross at 14.7 ± 0.8 DYN,
33.4 ± 2.0 NOISE — the noise axis is intrinsically slower, its evidence is
second-order/variance-level exactly as the SUMMARY says).  Pinning *which member*
is the slow part — the settled posteriors above take the ~50–100 step frontier times
to purge the confounded neighbor.  Both speeds are set by KL rates computed before
running anything.

## 4. Simultaneous BOTH barely masks; burst-then-fault is the real mask

A negative result worth filing: the prediction "the burst masks the fault" is nearly
invisible in the simultaneous-BOTH race (dyn delay 15.9 ± 1.1 vs 14.7 unmasked),
because the *joint* member fights 'nom' at 0.55 nats/step — the joint evidence rescues
the marginal.  The mask binds when the bank has already settled on 'noise' and the
fault arrives later, needing to win the slow duel 'both' vs 'noise' at
KL = **0.2006** nats/step (vs 0.4886 in calm noise — the burst more than doubles the
fault-detection time, structurally: same mean signal divided by a larger S):

- STAGGER measured dyn delay from the fault: **25.7 ± 1.0** vs ~15 unmasked.
- Accounting: launch log-odds at the fault −7.27 (above the −8.01 floor) →
  Wald-from-launch 36.2 at the stationary duel rate; measured sits below via the same
  mixing-bonus and post-change-transient effects audited in 0001 §1 (not re-audited).
- bankD "detects" in 13.2 steps in STAGGER — but it had already false-fired at the
  burst in 93% of seeds, so its fast number is meaningless.  Detection speed claims
  require the noise machinery to be live; 0001's bank numbers survive because its rig
  had honest calm noise.
- bank4 settles STAGGER at RMSE/oracle **1.001**, attribution 0.994/0.982.

## Carried forward

- The enumerated grid gets the split for free; mechanisms (b)/(c) (continuous
  departure channels) must show the same split when the dynamics axis is a *walk*, not
  an anchor — 0053 §2's regression is the cautionary precedent.  Test at the drone rig
  (0004) with the noise walk live.
- The noise axis here is a 2-point jump grid, not the shipped AR(1) scale walk; wiring
  the real scale machinery beside the dynamics bank is 0006 integration work.
- Excitation dependence of the *confound* (does low excitation collapse the split back
  toward 0041's scalar bound?) → 0003.
