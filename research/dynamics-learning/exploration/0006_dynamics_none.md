# 0006 — dynamics=None from nothing: oracle-grade within ~150 steps, the frequency change re-learned, and the anchorless case is more forgiving than feared

The cell's own semantics (`0006_dynamics_none.py`): n=2 rotation-decay truth
`x_t = λR(φ)x_{t-1} + Bu_t + w`, both components observed noisily, and the filter is
told NOTHING — prior F = I (the parent's random walk), B = 0, parameter covariance at
the class cap (cold start = honest ignorance).  Mid-run the rotation *frequency*
doubles (φ 0.15 → 0.30) — the odefilter's recorded limit verbatim ("`g` is one scalar
along one direction … it cannot express a change of frequency").  Machine: parent
hedge (F=I, B=0) + augmented KF walker on (x, vecF, B) + 4 time-anchored spawn slots
(walker clones re-anchored every 100 steps with P_θ = cap and weight ρ·100 — the
pruned run-length/BOCPD realization of the jump class, anchors in TIME where 0004/0005
had anchors in parameter space).  30 seeds, two class hazards.

## Measured

RMSE / supplied-dynamics-oracle (per-seed ± se), ρ = 1/50000:

| phase | window | parent | surrogate (walker alone) | lucid (bank) |
|---|---|---|---|---|
| from scratch | [0,100)   | 3.36 ± 0.11 | 1.214 ± 0.015 | 1.214 ± 0.015 |
|              | [100,400) | 3.74 | 1.065 ± 0.004 | 1.065 ± 0.004 |
|              | settled   | 3.80 | 1.066 ± 0.001 | 1.066 ± 0.001 |
| freq change  | [0,100)   | 5.25 | 1.302 ± 0.019 | **1.247 ± 0.013** |
|              | [100,400) | 5.59 | 1.057 | 1.058 |
| no-change control | settled | — | 1.066 | 1.066 |

- **Told nothing, the filter reaches within 7% of a supplied-dynamics oracle in a few
  hundred steps**, from an identity prior, while never sitting above the told-nothing
  parent in any window (hedge ratios 0.19–0.37 — the bank is 3–5× *better* than the
  parent everywhere, and can only degrade to it, never below it).
- **The frequency change is caught and re-learned in ~100–200 steps** by the full-F
  walker — the odefilter's "obvious next axis" is closed by the multivariate lift, as
  the SUMMARY intended.
- **The hazard sets the calm floor**: ρ 1/7000 → floor 1.148; ρ 1/50000 → 1.066.  The
  floor is the price of 6 walked parameters at drift q_θ = cap·ρ (same phenomenology
  as 0004's settled 1.075).

## The honest surprise: anchorless re-learning is NOT slow here

The probe was built expecting the anchorless surrogate to crawl after the change (its
0001 detection latency was 3.6× the frontier) and the time-anchored spawns to rescue
it.  Measured: the surrogate recovers in ~100–200 steps at *both* hazards, and the
spawns only shave the first-window transient (1.302 → 1.247); after 100 steps they add
nothing, and in calm they cost nothing (identical floors).  The reason is structural:
this rig observes the full state, so parameter evidence has relative degree 0 and even
a small steady parameter gain drains the error quickly — *recovery* was never the
walk's weak phase (0001 showed that too); *detection latency and post-jump
calibration* were, and nothing here consumes a detection.  The feared coupling
(one q_θ tying calm floor to reaction speed) is real but mild when evidence is
immediate.

Where time-anchors should actually bind, recorded as the open: partial observation
(relative degree > 0, evidence through P_x,θ — the 0004 drone with `dynamics=None`),
low hazard AND a latency consumer (a safety monitor reading the fault posterior — the
0005 setting without nameable sides).  The spawn machinery is cheap, correct, and
dormant until then; keep it, but do not credit it yet.

## Design summary for the `dynamics=None` cell (what 0001–0006 settled)

1. **One class, one prior**: dynamics faults are a jump process with hazard ρ (the
   labeled commitment, e.g. 1/mission) and class scale (cap).  Every gain, drift,
   floor, restart width, and spawn mass derives from (ρ, cap).  Zero tuning constants.
2. **Detect by anchored bank** — anchors in parameter space when faults are nameable
  (0004/0005), anchors in time when not (this probe) — **refine by augmented-EKF
   walker** (cross-covariance mandatory under partial observation, 0004), **restart
   variance to cap on detection as a shared event** (0003), **keep the nominal/parent
   member forever** (the hedge that makes the aggressive frontier end affordable,
   0001).
3. **Noise machinery lives inside the same bank** (0002); the detection frontier is
   priced against the best wrong member including the noise members (0004's rule).
4. **u must be measurable from the filter's information set** (0004's closed-loop
   bias); parameterize in the linearizing (inverse-inertia) coordinates the physics
   offers.
