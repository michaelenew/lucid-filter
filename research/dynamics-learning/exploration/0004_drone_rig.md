# 0004 — the drone rig passes acceptance: on the (masking-corrected) frontier, θ recovered exactly, calm and gust free — after two bugs worth their own findings

The acceptance target (`0004_drone_rig.py`): planar quadrotor (n=6), PD waypoint
autopilot, mocap sensing (x, z, φ only — deliberately no IMU, so parameter effects
reach the measurements only through integration), payload attaches at t*: m ×1.30,
I ×1.15.  The machine is the 0001–0003 stack assembled: members
{nominal, anchor, walker} × {q, 4q}, hazard-mixed (ρ = 1/T), full per-member filters;
the **anchor** is the nameable class guess (m ×1.25, I ×1.0 — deliberately not the
truth); the **walker** is an augmented EKF on (state, 1/m, 1/I) with jump-class drift,
cap, and the 0003 variance-restart on the detection edge.  All dynamics, Jacobians,
and parameter sensitivities enter as callables of (state, u, θ).  20 seeds × 4
scenarios; error bars are per-seed ± se.

## Acceptance scorecard

| criterion | target | measured | |
|---|---|---|---|
| detection delay | on the frontier | **28.9 ± 1.7** vs D* = 29.6 | pass |
| recovery vs refit-oracle | ≤ 1.2× within memory | 1.131 ± 0.014 at [50,200); 1.075 ± 0.004 settled | pass |
| calm | ≈ 1.00 | **1.0003** | pass |
| never worse than frozen-F0 | all regimes | worst case CALM 1.0003 vs 1.0000; GUST 1.002 vs 1.072; FAULT 1.6 vs 2.3 | pass |
| recovered parameters | — | m̂ 1.303 (true 1.300), Î 0.0233 (true 0.0230) | — |
| tuning constants | zero | ρ = 1/T, class scale 0.5·θ0, caps = class prior; nothing else | pass |
| embedded cost | 0053 §5 budget | ~1 ms/step numpy for all 6 members (0053's clustered single member: 3.0) | pass |

GUST (process ×4, no fault): no persistent false fault (20% of seeds cross the
detection readout transiently; state cost 1.0017 — the hedge economics of 0001 hold
multivariate).

## The frontier correction: masking lives INSIDE the bank

The measured KL ceiling (true-θ member vs nominal) is 1.13 nats/step (D_min ≈ 8), and
the anchor's edge over the plain nominal is 1.09 (≈ 8 steps).  But the anchor must
beat its **best** wrong rival, and that is the nominal×4q member — which partially
explains fault innovations as noise (0002's masking, now appearing in the frontier
itself): the anchor's edge over it is only **0.294 nats/step → D* = 29.6**.  Measured
delay 28.9 ± 1.7 sits exactly on it.  Rule for the ladder: *the detection frontier of
a bank is the llr differential against the best wrong member, not against the
nominal* — carrying the noise machinery (which 0002 proved mandatory for attribution)
costs a ~3.7× detection factor here, and that cost is information-theoretic, not
implementational.

## Two findings that came out of failures (first iterations measured, then fixed)

1. **Closed-loop truth-feedback biases identification.**  With the autopilot flying on
   the *true* state, Î settled 50% high (0.035 vs 0.023): the torque command
   correlates with process noise the filters cannot see (u = g(true state) ⊃ noise),
   the classic closed-loop identification bias.  Flying the autopilot on
   *measurements* (α-β observer on y, so u is measurable from the filters'
   information) removes it completely: Î 0.0233 vs true 0.0230.  Constraint for 0006
   and for any real deployment: **u must be measurable from the filter's information
   set**; if the vehicle's controller uses state the filter doesn't see, the departure
   channel inherits a bias.
2. **Estimate inverse inertias.**  Newton's equations are *linear* in (1/m, 1/I), so
   the walker parameterizes θ = (1/m, 1/I): the augmented-EKF parameter block then has
   no linearization error and the sensitivity callable `∂f/∂θ` does not depend on the
   current estimate.  (The 0033 linearizing-coordinate move, supplied free by the
   physics.)  On its own it did **not** remove the Î bias — the closed loop did — but
   it removes a needless approximation and is kept.

Also confirmed at n=6: with position-only sensing, an *instantaneous*
innovation-regression has a zero regressor (one step of θ moves velocities, not
positions) — the walker must be the augmented EKF whose P_x,θ carries the multi-step
sensitivity.  0001's "the cross-covariance is worth nothing" was a relative-degree-0
artifact; the scalar rig observed the state the parameter acts on directly.

## The honesty scenario dissolved (a good negative result)

HOVER (fault lands mid-hover, maneuvers resume 1500 steps later) was built to show Î
standing at honest width until excitation arrives.  Measured: Î converges *during*
hover (0.0231 ± reported 0.0031) — because a stabilized quadrotor is never
torque-quiet: the autopilot's response to sensor noise dithers τ continuously, and
that dither is persistent excitation.  Detection at hover is also fast (29.9 ± 1.3 —
mass talks through the thrust/gravity channel regardless of maneuvering).  The true
zero-excitation honesty case (0003 Part B) does not occur on this vehicle; the
restart-to-cap machinery is still the right behavior for vehicles/axes where it does
(a fixed-wing's unexcited lateral axes, a wheeled vehicle at rest).

## The analyzed residual: the settled 1.075

Settled FAULT tracking is 1.075 ± 0.004 — inside acceptance, but not 1.00, and the
walker member itself (weight 0.93) carries it.  Localization: the gap lives in the
z/vz channels (vz 1.29×, z 1.09×; x/φ/ω ≈ 1.01×) — the thrust·(1/m) product is the
high-leverage term (T ≈ 12.7, so a 0.4% persistent im error is a 0.05 m/s² thrust
bias that the z-loop must keep re-absorbing).  The obvious suspect — the q_θ
random-walk hold floor — was tested and **refuted**: q_θ/100 leaves the ratio at
1.071.  Remaining candidate: the cost of *carrying* P_θ in the gain (the augmented
filter's z-gain is detuned by parameter uncertainty regardless of how right θ̂ is).
Open for 0006: whether the exact jump-hold class (uncertainty ~0 between events,
cap on detected events — a two-state θ-prior instead of the diffusion surrogate)
recovers the last 7%.

## Carried to 0005/0006

- Frontier rule: bank frontier = llr edge over the best wrong member.
- Closed-loop constraint: u measurable from the filter's information set.
- Inverse-inertia (linearizing) parameterization for mechanism (b).
- The 1.075 residual and the jump-hold θ-prior open.
