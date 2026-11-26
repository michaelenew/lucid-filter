# 0009 — predictions, recorded before the run

> **AI-generated, not peer-reviewed.** Written before
> [`0009_state_moves_with_weight.py`](0009_state_moves_with_weight.py) was first executed.

`0008` found: during each spike on arm seed 2 the star's weight is on one wide class
(`s = 1.6`, then `s = 3.2`) whose own state estimate is 30–200× worse than the others';
the spikes do not shrink when the rates are scaled down 30×; and the shipped filter keeps
all of its weight on a single class (`s = 0.8`) throughout the record.

**The fix under test.** When the chain moves weight from member `i` to `j`, `j`'s state is
replaced by the mixture of where its weight came from (the interacting-multiple-model
step). Variants: carry the state mean and covariance (`m`), the scale-walk centres
(`mu`), or both.

- **H1.** Carrying both removes the spikes on seed 2: calm within 2× of shipped's 0.563.
- **H2.** `m` alone is not enough — the wide member's damage is in its scale walk — so
  `mu` alone does better than `m` alone.
- **H3.** The SENSOR-window gain survives (below 2.0× the oracle on seed 2; shipped 2.14).
- **H4.** Log-likelihood stays above shipped's.

Low confidence in H1: once the wide member holds nearly all the weight it is no longer
being refreshed by anyone, so if it goes wrong *while leading* this fix cannot help.
