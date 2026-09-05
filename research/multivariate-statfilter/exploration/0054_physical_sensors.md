# 0054 — sensors the arm could actually carry: no constant `H` exists, and freezing the linearisation costs 15–195×

> **⚖️ ATTRIBUTION —** _A physical-sensor rig showing no constant measurement matrix exists (accelerometers double as inclinometers), so H must be relinearised each step (EKF), with the Jacobian by complex-step differentiation; freezing the linearisation costs 15–195× the oracle — a measured ablation — plus a relative-degree finding that the score cannot price the disturbance-to-sensor lag._ Prior art: extended Kalman filter relinearisation (standard); complex-step Jacobian — Squire & Trapp 1998, Martins et al. 2003. Status: NEGATIVE-RESULT.

`0052`'s rig gave every joint an "accelerometer" reading that joint's own angular acceleration
through a constant, diagonal `H`, on a chain of four coplanar pitch joints. Neither half is
real: no sensor reads a joint coordinate, and nobody builds that arm. This probe replaced the
rig in three recorded steps — two of which corrected *this probe's own first attempts* — and
the record keeps all three, because each one measured something.

## The rig as it stands (`../scripts/arm5dof.py`)

**The chain** is the one most 5-DOF arms share: two orthogonal DOFs at the base (yaw about
the vertical, shoulder pitch), an elbow carrying two more (elbow pitch, then a roll about the
forearm), and a short wrist holding the effector on the one remaining flex axis. The roll is
what makes the wrist interesting — J5's flex axis is carried by the forearm roll, so where it
points is a function of the configuration.

**The sensors**: a potentiometer per joint (σ 0.06 rad ≈ 3.4°, the bad absolute sensor), and
a MEMS accelerometer near each link's distal end, 6 cm off the link axis, with its **two
lateral axes** recorded (σ 0.03 m/s² each; the part is 3-axis, and the along-link axis adds
little here). Each axis reads proper acceleration

    y = s · R_j(θ)ᵀ ( p̈_j(θ, ω, α) + g e_z ),        s ∈ {x, y}

— a configuration-dependent lever-arm map on the joint accelerations, centripetal/Coriolis
terms quadratic in the rates, and **gravity resolved in a link frame that moves with every
joint below it**, which is what makes an accelerometer an inclinometer. There is no constant
`H` here at all, so the map reaches the shipped filter as `LucidFilter(H=callable)` and is
linearised at every step, exactly as a moving `F` is. The Jacobian is by **complex-step
differentiation** — `h` is written complex-safe and batched, so one evaluation yields the
value and all 15 columns — and agrees with central differences to **1.3e-9**.

The servo flies on an alpha-beta-gamma tracker of the potentiometers, never on the true state
(0004's closed-loop bias, one level down), at a bandwidth the potentiometer can support: a
triple pole at s = −8 on a 0.06 rad sensor injects 1.57 rad/s³ of jerk noise — 2.6× the
process noise the filter is told about, so an "oracle told the true schedule" would not have
been told the truth — while s = −4 injects 0.18.

## Headline: the physical rig, and the frozen-linearisation ablation (3 seeds)

Tip RMSE over the bursts, metres:

| | |
|---|---|
| raw potentiometer | 0.2273 ± 0.0032 |
| fixed noise | 0.0309 ± 0.0020 |
| **oracle** (told the true noise schedule) | **0.0087 ± 0.0006** |
| **lucid** | **0.0190 ± 0.0040** |
| frozen-H | 0.4684 ± 0.0579 |

Ratio to the oracle, per regime. **`frozen-H` is the same lucid filter with the measurement
map linearised once, at the home pose, and never again** — it is even spotted the true
gravity offset `h(0)`, so its handicap is purely the frozen Jacobian:

| regime | lucid | fixed | frozen-H |
|---|---|---|---|
| calm | 0.94 | 2.89 | 89.5 |
| accelerometers ×15 | 2.77 | 4.99 | 15.7 |
| vibration ×20 | 1.11 | 7.28 | 130.5 |
| one potentiometer ×15 | 1.07 | 2.10 | 195.2 |
| vibration + accelerometers | 2.10 | 2.73 | 14.8 |

Linearising per step is load-bearing, not decoration: freeze it and the filter is 15–195×
the oracle *while the adaptive noise machinery works perfectly* — the walk faithfully books
the linearisation error as sensor noise, which is exactly the wrong thing to trust a reading
by. The lucid filter is never worse than the fixed one, and the fixed one now pays 2.1–7.3×
because on this rig the accelerometers double as inclinometers, so knowing *when* to trust
them is worth more than it was on any earlier cut.

**The one elevated cell, decomposed.** SENSOR at 2.77 is two effects, measured by fifths of
the window on seed 0 (ratio per 50 steps after onset): 9.4, 2.7, 2.1, 1.7, 3.1. The first
fifth is the **reach transient** — ten accelerometer scales travelling to the settled value,
which lands on 2 ln 15 = 5.42 to two decimals — the same transient-attribution open recorded
in `sequence-demix/0005` and on the drone's MULTIPATH window. The settled tail at ~2× is a
different thing: on a rig where the accelerometers carry nearly all the information (gravity
reads the angle at an effective σ ≈ 0.004 rad against the potentiometer's 0.06), an oracle
told "accels ×15, process calm" switches cleanly to a pots-plus-prior solution, while the
bank — told nothing — keeps paying for the hypotheses that have to stay live for it to notice
the burst *end*. BOTH runs at ~1.0 in every fifth on the same seed: the same accel burst with
a process burst underneath inflates the oracle's own error, and the gap closes. Filed as a
measured residual with its cause, not tuned away.

## The record of getting here — two instructive wrong turns

**First cut: an angular link sensor, on the old coplanar chain.** A link-mounted angular
sensor reads the chain beneath it, `Σ_{i≤j} (a_j·a_i) α_i` — state-dependent in general. But
axis-dot couplings are *constant* wherever axes are parallel or orthogonal, and on the old
chain that was every coupling but one (link 5 ← joint 1, sweeping −0.72 to −0.06). So while
the diagonal model was badly wrong there (model error 7–13× the sensor σ; run on physical
data it cost **10–154× the oracle**), the honest headline was "diagonal where it should have
been a constant *sum*" — a reactive `H` was motivated by one coefficient, not forced.
Reviewer pressure on exactly this point is what pushed the rig to linear accelerometers,
where no constant `H` exists even approximately.

**Second cut: linear accelerometers, one sensitive axis.** Recording only the x axis (along
the mount offset) measured at 3.0–4.7× the oracle — a real regression, and the diagnosis was
this probe's own §4 pathology sitting *inside* the hero rig: a roll about the link a sensor
is bolted to produces purely **tangential** acceleration, invisible to the radial axis, so
the forearm-roll α column of `H` was 0.00 on every sensor. A ×20 roll-jerk burst was then
invisible per step and got blamed on the J4 potentiometer (pot scale driven to 2.2 while the
mode reached 1.4). Base yaw had the same hole. Recording the second lateral axis gives roll
and yaw their tangential lever arms (6 cm on their own links, 0.14–0.4 m through the chain),
and PROCESS fell 4.67 → 1.11. The lesson generalises: **a rig audit is not done until every
disturbance axis has a sensor at relative degree ≤ its neighbours** — one recorded axis per
IMU is an economy real designs don't make, and the filter cannot rescue information the
sensor set does not carry.

## Why the dynamic sensor is an accelerometer, not a rate gyro — the limit, quarantined

On a diagonal control rig where only the read derivative differs:

| | calm | SENSOR | PROCESS | POTFAIL | BOTH |
|---|---|---|---|---|---|
| angular accel (reads α) | 1.07 | 1.09 | 1.29 | 1.29 | 0.91 |
| rate gyro (reads ω) | 1.07 | 1.96 | **103.1** | 2.50 | 1.71 |

A jerk disturbance reaches a rate channel through dt²/2 and an acceleration channel through
dt — 200× weaker per step at 100 Hz — and the scale walk scores per step, so on a rate-gyro
rig a process burst is nearly invisible in the one-step score while the sensor axis is fully
visible, and the burst is blamed on the sensor. Same shape as 0004's relative-degree finding
on the dynamics channel, recorded here for the noise channel. Open: nothing in the walk's
construction prices in the relative degree between a disturbance and the sensors that see it.

## What is still not physical

The plant is five independent triple integrators in joint space — no mass matrix, so no
inertial coupling, no Coriolis torques, no gravity load; the truth is the recursion the
filter models; the disturbance is a white jerk on the command channel. None of that flatters
the measurement map, which is what this probe exists to get right.
