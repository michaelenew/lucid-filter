# 0053 — the de-mix mechanism is per-node MEANS; the walking-window realization regresses (reverted)

> **⚖️ ATTRIBUTION —** _Isolates that the de-mix requires per-hypothesis MEANS (each hypothesis runs its own filter and mispredicts its own innovations — the no-EMA form of whiteness discrimination), i.e. full multiple-model estimation rather than a GPB1 collapse; the walking-window realization of it regressed and was reverted (recorded open)._ Prior art: multiple-model adaptive estimation with per-model filters (MMAE/GPB2) — Magill 1965, Ackerson & Fu 1970, Bar-Shalom; innovation-sequence Q/R discrimination — Mehra 1970. Status: NEGATIVE-RESULT.

The 0052 residual: on the collinear accel↔jerk pair, attribution leaks both ways (an accel burst
lifts the process scale +6.2 and vice versa) while state tracking sits at the floor.  This probe
asked what the de-mix *requires*, found it, built it into the engine, and measured the build
regressing 4 of 6 regimes — so the engine is **reverted** to the 0052 structural caltrop and the
finding is recorded here for the next realization.

## 1. The mechanism, isolated (static single-joint 2-D grid, `0053_pernode_demix.py`)

One joint (n=3 kinematic, m=2 pot+accel), a static 25-node grid over (ξ = jerk scale, η = accel
scale), exact Bayes weights, three state treatments:

| regime (truth) | shared m, shared P | shared m, per-node P | **full per-node KFs** |
|---|---|---|---|
| SENSOR (0, 5.4) | (+3.84, +5.10) | (+3.22, +5.14) | **(+1.50, +5.55)** |
| PROCESS (6.0, 0) | (+4.63, +1.63) | (+4.67, +1.61) | **(+5.51, +0.46)** |

- **Per-node P alone does NOT de-mix** (also measured in-engine: leak unchanged).  With a shared
  mean every hypothesis sees the same innovation sequence; the pot-channel variance veto a high-Q
  hypothesis's own P implies is orders too weak at dt = 0.01.
- **Per-node MEANS de-mix** (leak ÷2.5–3.5): each hypothesis runs its own filter, so a high-Q node
  *chases* white sensor noise and mispredicts its own innovation statistics — the lag-1 sequence
  evidence enters through the means, with no EMA and no whiteness statistic.  This is the no-EMA
  form of the 0025/0027/0032 whiteness discrimination, and the concrete content of 0050/0051's
  "per-node P propagation" suggestion (which needed the means, not just P).

## 2. The realization that failed (walking windows + pairs + per-node KFs)

Axial windows alone cannot use the evidence — the walk slides *along* the confound ridge because
no node represents the joint alternative (no corners; exactly 0013's caveat).  So the engine was
rebuilt: each process eigenmode paired with its best-reading sensor (argmax structural Fisher
weight `(H v_k)_i^2 / rho_i` — 0043/0051's "the confound is local"), the pair sharing a joint 2-D
window; per-node full KFs everywhere; IMM mixing through the AR(1) kernel; still linear in D.

Seed-0 spots looked excellent (SENSOR 1.004, PROCESS 1.000, procpot 1.101 — at the oracle).  The
4-seed profile did not hold up:

| regime | structural (shipped) | pair+per-node | paired diff |
|---|---|---|---|
| CALM | 0.983 | 0.983 | +0.4σ |
| SENSOR | 1.135 | 1.112 | −0.3σ |
| pot-hot | 1.201 | 1.652 ± 0.415 | +1.4σ |
| PROCESS | 1.085 | 1.235 ± 0.110 | +1.9σ |
| BOTH | 1.108 | 2.470 ± 0.815 | +1.8σ |
| process+pot | 1.214 | 2.347 ± 0.714 | +1.7σ |

The regressions come with large seed variance and severe *velocity* degradation (vel/oracle 3.0
in PROCESS, 5.4 in BOTH) — occasional runaways, not a uniform cost.  The suspect is the
interaction of per-node means with the *walking* window: hypotheses are slots relative to a moving
centre, so a node's accumulated filter state belongs to yesterday's hypothesis; under IMM mixing a
wandering far-node mean re-enters the collapse.  Attribution improved only partially anyway
(SENSOR proc leak +6.8 → +3.9): the window posterior's memory is the class's ~1/(1−φ) steps, which
bounds how much sequence evidence it can hold (the 0041 Lorden frontier, showing up as predicted).
Cost was also ~6× (per-node Riccati + means for every node).

**Decision: reverted.**  `lucid.py` stays the 0052 structural caltrap engine (state at/near the
floor everywhere, attribution honest-but-coupled on the collinear pair).

## 3. What a working realization needs (the open, sharpened)

- The evidence carrier is a per-hypothesis FILTER (means), per §1 — keep.
- The hypotheses must be *stable anchors*, not slots on a walking centre: candidates are the 0051
  per-cluster static grids (windows wide enough not to walk, per-joint so they stay small), or
  anchored hypothesis pairs at derived offsets (e.g. ±class-swing on each collinear pair) whose
  votes drive the walk without carrying it.
- The posterior memory that holds the sequence evidence should be the *bank weight* timescale
  (`forget`, ~1000 steps), not the scale kernel's 1/(1−φ) — the two timescales are currently tied
  together in the window posterior.
- Blowout guard: any per-node-mean scheme needs a bound tying node means to the collapse (the IMM
  mixing was not enough at the far nodes).

## 4. Accelerometer-only layout (the other 0052 question, `save <label> <seeds> acc`)

| regime | lucid/oracle | fixed/oracle | vel ratio | oracle angle RMSE |
|---|---|---|---|---|
| CALM | 0.995 | 1.000 | 1.00 | 0.019 (potacc: 0.006) |
| SENSOR | 1.228 | 1.281 | 1.24 | 0.038 |
| PROCESS | 1.140 | 0.988 | 1.87 | 0.019 |
| BOTH | 1.403 | 1.014 | 2.35 | 0.046 |

Two findings.  (1) Position is unobservable through accelerometers alone: even the oracle drifts
(3× worse absolute in calm, unbounded in T); the filter matches the equally-blind oracle (0.995) —
it loses nothing beyond physics.  (2) **Noise adaptation loses most of its value without
redundancy**: with pots the adaptive beat fixed 2.4–5.5× on bursts by shifting trust between
sensors; accel-only has nowhere to shift — SENSOR adaptive ≈ fixed, and in PROCESS/BOTH adaptive
is *worse* than fixed (1.14 vs 0.99, 1.40 vs 1.01): the accel⊕jerk pair is the entire observation
and fully collinear, so moving scales along that ridge only adds variance (0033's masked-Q with no
absolute witness).  The fusion IS the story.

## 5. Cost anatomy (embedded viability)

Measured per step, numpy, 5-DOF rig (during this probe's engine; the shipped engine is ~6× less):

| configuration | ms/step |
|---|---|
| 15-member bank × coupled n=15 model × per-node KFs | 258.6 |
| single member, same model | 18.4 |
| per-joint clusters (5 × n=3, m=2), single member | **3.0** |
| per-joint clusters, full bank | 44.0 |

Three multiplicative levers: the bank (15×; the ridge is flat for tracking, 0037 — 1–3 members
lose nothing on state), the coupled n³ state (block-diagonal robotics models factor into per-joint
sub-filters — 0051's clustering — at no approximation when the blocks are exact), and numpy→C
(10–50×).  A clustered single-member step is a handful of 3×3 ops — tens of µs in C.  Caveat for
the factorization: when `F`/`B` are supplied as *callables* (linearized nonlinear dynamics), block
structure cannot be detected statically — it must be declared by the caller or probed numerically
at linearization points, and it can change with the operating point.  See the SUMMARY opens.
