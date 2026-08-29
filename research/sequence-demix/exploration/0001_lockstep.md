# 0001 — the lockstep is the scale-Fisher's null direction, and it is exactly singular

Replicates the scalar lockstep on the hero rig and locates it in the geometry, so that the rest
of the workstream knows precisely which directions a per-step score can carry and which one it
cannot.  Script: `0001_lockstep.py`.

## 1. The lockstep, measured

`LucidFilter()` on the README-004 rig, all 15 bank members instrumented:

    max over t and members of |mu_xi - mu_eta| = 3.7e-15

Not "close": the two walks are the same trajectory to machine precision, in every member.  The
consequence, in learned quantities:

| regime | learned Q | learned R | learned total | truth total | learned ratio | truth ratio |
|---|---|---|---|---|---|---|
| A | 0.525 | 0.525 | 1.050 | 1.020 | **1.00000** | 0.02000 |
| C | 5.369 | 5.369 | 10.74 | 9.020 | **1.00000** | 0.00222 |

The total is learned well in both regimes.  The ratio is not learned at all — it is the supplied
base, to five decimal places, in a regime where the truth moved by a factor of nine.

## 2. Why: the innovation enters both scores through one common factor

In one channel every matrix in the score is a scalar, so

    score_k = 0.5 * dS_k * (e^2/S^2 - 1/S),     dS_xi = Q,  dS_eta = R.

Measured over a grid of (xi, eta, P, e): `max |score_xi/score_eta - Q/R| = 7.1e-15`.  The
innovation sets the STEP SIZE and never the DIRECTION.  This is stronger than "the two walks
happen to coincide at the default base": no reweighting of that per-step score — by Fisher, by
window posterior, by anything measurable at time t — can move the split, because the split is not
in the score at all.  SUMMARY §6's claim is exact.

## 3. The geometry: the scale-Fisher is exactly rank 1, and its null direction is the split

The full per-step scale-Fisher `I_ab = 0.5 tr(Si dS_a Si dS_b)` at the steady state:

| class | eigenvalues | lam_min/lam_max | null direction |
|---|---|---|---|
| scalar (n=1, m=1) | (0, 0.1459) | **0** | (-0.707, +0.707) = xi - eta |
| at (xi,eta) = (-3.9, 0) | — | 0 | (-0.9998, +0.0202) = (R, -Q)/‖·‖ |
| at (xi,eta) = (+2.0, -1.0) | — | 0 | (+0.0497, -0.9988) = (R, -Q)/‖·‖ |

The null direction is `(R, -Q)` up to scale at every operating point.  Integrating that direction
field gives `dQ = -dR`: **the null manifold is the level set of the total**, which is just the
statement that the one-step likelihood sees `S = P + Q + R` and nothing else.  So the two
coordinates of the problem are

* the **total** — per-step identifiable, and what the walk is good at;
* the **split** — in the exact null space of the per-step Fisher, invisible to any per-step
  score, and therefore something a *bank* has to carry, not a walk.

That is Proposition 1 written in coordinates, and it is what candidate A is built on.

## 4. The same measurement on the 5-DOF arm — the activation rule any build must reproduce

Correlation-form spectrum of the arm's scale-Fisher (D = 25): five eigenvalues at 3.0, five at
2.0, five at 2.2e-5, and ten at ~1e-16.  Read naively that is fifteen "confounded" directions and
a ladder on each — unaffordable, and wrong.  Reading the RAW Fisher instead:

    diag(I) ranges 1.07e-27 .. 0.492;  the only axes whose own information is non-negligible are
    the ten SENSORS.  Among those ten there is no degenerate direction at all
    (eigenvalues 0.275 x5, 0.492 x5).

The ten near-zero directions pair a **process eigenmode carrying essentially no base variance**
(the 1e-12 ridge in `Q0`) with a pot — half of a confound is not a confound, and scaling a mode
with `lam = 1e-12` against a real sensor variance is not a hypothesis worth a rung.  The five at
2.2e-5 are the genuine accel↔jerk pairs (0027's |C| = 1), and they are *near*-degenerate, not
degenerate: `H v_k` has a nonzero pot component (`dt^3/6` against `dt`), so `dS_xi` is not
proportional to `dS_eta` as a matrix and the block is full rank.

So the structural test that gives the right answer on both rigs, with no magnitude threshold, is:

> a group is a process eigenmode read by **exactly one** sensor, both axes carrying
> non-negligible information; then, and only then, the 2x2 scale-Fisher block is exactly rank 1.

On the hero rig that finds `(xi_0, eta_0)`.  On the arm it finds nothing — which is what gate 2
needs, and it is a fact about the arm's structure rather than a switch anyone set.
