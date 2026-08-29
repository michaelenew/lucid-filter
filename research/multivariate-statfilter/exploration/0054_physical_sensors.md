# 0054 — sensors the arm could actually carry: the diagonal `H` was worth 10–154× the oracle

`0052`'s rig gave every joint an "accelerometer" that read that joint's own angular
acceleration, `alpha_j`, through a constant diagonal `H`. **No sensor does that.** An inertial
sensor is bolted to a *link*, and what it reads is the motion of the whole chain beneath it,
resolved on an axis that rotates with the arm. On this arm the consequence is not subtle:

| | what the rig assumed | what a link sensor reads |
|---|---|---|
| link 1 | `al_1` | `al_1` |
| link 2 | `al_2` | `al_2` |
| link 3 | `al_3` | **`al_2 + al_3`** |
| link 4 | `al_4` | **`al_2 + al_3 + al_4`** |
| link 5 | `al_5` | **`al_5 − 0.15 al_1`**, and the `0.15` sweeps to `0.53` |

Joints 2, 3 and 4 all rotate about their local *y*, so their axes are exactly parallel and the
coupling is exactly 1; the joint 1 ↔ joint 5 coupling is not constant at all. Measured against
0052's own sensor sigma, the model error is **7–13× sigma** on joints 3, 4 and 5 — a filter
handed the diagonal `H` is being told something false and loud.

`../scripts/arm5dof.py` replaces it with the physical map,

    y_j = sum_{i<=j} (a_j . a_i)(theta) al_i  +  sum_{i<=j} sum_{l<i} om_l om_i (a_j.(a_l x a_i))

— coupled, state-dependent, and with a rate-quadratic term from the rotating axes, so `h(x)`
is **not** `H(x) x`. It reaches the shipped filter as `LucidFilter(H=callable)`, returning the
Jacobian *and* the predicted measurement. The analytic Jacobian agrees with central
differences to **1.8e-10**. Two other things were fixed at the same time: the demo's servo now
flies on an alpha-beta-gamma tracker of the potentiometers rather than on the true state (which
handed every contender a noiseless linear functional of the state — 0004's closed-loop bias,
one level down), at a bandwidth the potentiometer can support (a triple pole at s = −8 on a
0.06 rad sensor injects 1.57 rad/s³ of jerk noise, **2.6× the process noise the filter is told
about**, so the "oracle told the true schedule" would not have been told the truth; s = −4
injects 0.18).

## What the rig is worth now (3 seeds, tip RMSE, metres)

| | over the bursts |
|---|---|
| raw potentiometer | 0.2961 ± 0.0105 |
| fixed noise | 0.0267 ± 0.0047 |
| **oracle** (told the true noise schedule) | **0.0090 ± 0.0008** |
| **lucid** | **0.0096 ± 0.0006** |
| the retired diagonal `H` | 0.4653 ± 0.0815 |

Ratio to that oracle, per regime. `diag-H` is the **retired model run on this same physical
data** — the price of the shortcut, and the reason the feature had to exist:

| regime | lucid | fixed | diag-H |
|---|---|---|---|
| calm | 0.92 | 2.48 | 22.55 |
| accelerometers ×15 | 0.87 | 1.98 | 10.28 |
| vibration ×20 | 1.32 | 4.36 | 89.07 |
| one potentiometer ×15 | 1.26 | 3.62 | 153.91 |
| vibration + accelerometers | 1.06 | 3.44 | 31.14 |

The filter is 0.87–1.32× the oracle on a *harder and fairer* rig than the one it was measured
on before (0052 reported 0.98–1.22 on the diagonal one), and the fixed filter now pays
2.0–4.4× rather than 1.0–5.5×. Ratios below 1.00 are not a paradox: an oracle-tuned filter is
optimal in expectation, not per realisation. The coupled map is *more* informative than the
diagonal one — each sensor sees several joints — which is why the lucid tip error over the
bursts falls from 0.0166 m on the old rig to 0.0096 m here.

Cost is unchanged: 15 members, `G = 101` star nodes, 25 active axes, no split ladder (every
process eigenmode is still read by more than one sensor), ~36 ms/step against the old 40.

## Why the dynamic sensor is an accelerometer and not a rate gyro — a real limit, isolated

The first attempt at this rig used a **rate gyro** on each link, which is the more common piece
of hardware. It failed, and the way it failed is worth keeping.

| diagonal rig, only the read derivative differs | calm | SENSOR | PROCESS | POTFAIL | BOTH |
|---|---|---|---|---|---|
| accelerometer (reads `alpha`) | 1.05 | 0.71 | **0.96** | 1.26 | 1.06 |
| rate gyro (reads `omega`) | 1.08 | 1.61 | **99.56** | 2.00 | 3.02 |

Under a process burst the gyro rig blames the *sensors*: within a few steps all five gyro
log-scales run to ≈ 6 (e⁶ ≈ 400× variance) and the filter throws away its good sensor, then
recovers about a second later. It is a **relative-degree** effect, and it is derivable rather
than mysterious. The disturbance is a jerk. It reaches an angular-acceleration channel through
`dt` and a rate channel only through `dt²/2` — 200× weaker per step at 100 Hz. The scale walk
scores *per step*, so on a rate sensor a process burst is nearly invisible in the one-step
score while the sensor axis is fully visible, and the cheapest one-step explanation of large
innovations is "the sensor got noisy". This is the same shape as 0004's finding on the dynamics
channel ("with position-only sensing a parameter's effect reaches the measurements only through
integration, and an instantaneous innovation-regression has a zero regressor") — but for the
**noise** channel, where it had not been recorded.

Filed as an open: the walk's attribution of a process burst degrades with the relative degree
between the disturbance and the sensor that sees it, and nothing in the current construction
prices that in. The coupled accelerometer rig does not suffer from it, so the demo is honest;
a rig whose only dynamic sensor is a rate gyro is not currently a rig this filter handles well,
and saying so is cheaper than discovering it on hardware.

## What is still not physical here

Stated so it can be argued with. The plant is five independent triple integrators in joint
space: no mass matrix, so no inertial coupling, no Coriolis, no gravity torque — the arm is a
kinematic benchmark, and a real manipulator's joints are dynamically coupled as well as
kinematically. The disturbance is a white *jerk*, which is smoother than a real vibration (a
disturbance torque is an acceleration, and would enter one derivative lower); jerk is used
because it is the input channel, and because it keeps every process eigenmode visible to two
sensors rather than one. And the truth is the same recursion the filter models, so the rig
isolates the estimation question rather than the discretisation one. None of those flatter the
measurement map, which is the thing this probe exists to get right.
