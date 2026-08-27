# 0014 — pairwise two-hot arms for the diagnostic de-mix (right idea, wrong weighting)

0013's axial caltrop tracks state at linear cost but leaks on the *scale attribution*
(the process↔measurement de-mixing lives in the corners the axial cross skips). Two
proposals tested: (a) two-hot arms — the `r_p·r_m` process×measurement 2-D pairs (only
that block couples, 0003) — a **quadratic** de-mix; (b) drop the arm extent to 3
nodes/axis.

## Results — neither tweak landed as implemented

- **3-node extent degrades** (static drift −0.21, over-reach) — the ±1 arm is too short
  to estimate the score cleanly. (User's own caveat: "don't degrade if the intuition
  doesn't hold." It didn't; keep 5.)
- **The pairwise de-mix made the leak WORSE**, not better (eta2 0.73 vs the axial
  caltrop's 0.42), even after fixing a double-update (accumulate each axis's score
  across its pairs, update once). Root cause: the 2-D arms carry the **stationary prior
  `w0×w0`**, which suppresses the far *process* offset, so the joint mass slides to the
  *measurement* corner of the coupling ridge — the de-mix inherits the same prior-shrink
  that made the profile-mean under-reach in 0013. Weighting the 2-D *score* by that
  prior-distorted profile biases the sensor's score upward.

## What this says

The corners **do** carry the de-mixing (the idea is right), but the de-mix must be read
from a signal that is **not** shrunk by the stationary prior — a likelihood-ridge / MLE
reading of the 2-D arm, or a prior-free score contrast between the "process explains it"
and "measurement explains it" directions. The naive prior-weighted average is the wrong
estimator. This is a genuine open — the diagnostic de-mix at quadratic cost is
*plausible* but not yet achieved; getting the 2-D estimator's prior handling right is
the crux.

## Bottom line

The **axial caltrop (0013) stands as the validated win**: linear cost, state tracking
matching the exact grid (×3.15/×3.44 over non-adaptive), with a bounded diagnostic leak
(~0.4). The pairwise diagnostic de-mix is the right direction but needs the prior-free
2-D estimator; it is not a blocker for productionising the state-tracking win.

Code: `0014_pairwise_caltrop.py`.
