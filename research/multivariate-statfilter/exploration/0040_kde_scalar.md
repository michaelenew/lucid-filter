# 0040 — first KDE probe: the scalar rig is the wrong rig; the obstacle is multivariate

Tested "learn the log-scale marginal (gamma0) instead of committing to a fixed s" on a scalar
local-level rig with one bursting sensor. Two honest negatives, both informative.

## The KDE beats fixed-s on the grid, but the grid loses to the plain walk

| filter (8 seeds) | calm/orc | burst/orc | all/orc |
|---|---|---|---|
| fixed s=0.3 (grid) | 1.36 | 1.80 | 1.47 |
| fixed s=2.5 (grid) | 1.32 | 1.92 | 1.45 |
| **KDE (learned)** | 1.35 | **1.73** | **1.44** |
| single-POINT walk (production) | 1.22 | 1.52 | **1.28** |

- The fixed-s grid is nearly **s-flat** (no interior optimum) -- a full posterior over eta reaches by
  likelihood regardless of the prior width. So the strong interior optimum we saw at 5-DOF is a
  **single-point / coupled-member artifact**, not intrinsic to scale tracking.
- The KDE is the best of the grid variants (best burst, learns gamma0, no committed s -- the purity
  win). But my grid transition (10% resample from the marginal) is crude and adds noise, so the
  whole grid family LOSES to the simple single-point walk (all 1.28). This probe's grid is underbuilt.

## The deeper reason it is inconclusive: wrong rig

The scalar rig has ONE sensor. When it bursts there is no backup, so the level just coasts on the
tiny process noise -- and the ORACLE coasts too -- so the reach barely changes the ratio and the
plain walk is already fine (burst 1.52). The 5-DOF interior optimum needs what this rig lacks: (a) a
SECOND sensor to lean on when one fails (pot fails -> use the accel), and (b) the process/measurement
CONFOUND (a sensor reading the process-driven state). s mattered at 5-DOF through the robust-MAP
(sigma^2 = s^2) and the cap (SPAN_S*s)^2, not through the scalar walk gain -- neither is in this rig.

## Where this leaves the KDE

The KDE is still the right PURITY target (learn gamma0, no committed s) and it wins among the grid
variants here. But its practical value has to be tested where the obstacle actually is -- the
MULTIVARIATE confound -- and there the reach is bounded by the C1 confirmation rate ~1/beta (0039,
Prop 1): a per-axis learned marginal still reaches by per-step likelihood, which cannot separate Q
from R, so it needs the C1 machinery and inherits the 1/beta bound. Learning gamma0 does not move
that bound. So: KDE = purity + a modest scalar reach gain; the multivariate reach ceiling is the
confound, unchanged. A faithful next test is a per-axis learned marginal ON the multivariate rig with
C1 attribution, measured against the no-shed floor -- not another scalar grid.
