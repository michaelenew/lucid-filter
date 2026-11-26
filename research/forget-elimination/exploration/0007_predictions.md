# 0007 — predictions, recorded before the run

> **AI-generated, not peer-reviewed.** Written before
> [`0007_the_star_in_the_filter.py`](0007_the_star_in_the_filter.py) was first executed.
> One scalar record was run as a smoke test while building the patch (star: RMSE 0.639,
> log-likelihood −1699.7; shipped: 0.713, −1714.6); the predictions below were written
> after seeing that single record and before any multi-seed or multi-rig run.

Arms: **A** shipped (`forget = 0.999`); **B** `forget = 1` (pure Bayes, the nominal filter);
**S** the derived star on the `(phi, s)` cells with `forget = 1` everywhere else.

**F1. Scalar hero rig (12 seeds).** S beats A in total log-likelihood (the filter's own
loss) with a paired `t` above 2. RMSE: better in the jump window, within noise in the
steady window and in the loud window.

**F2.** B is no better than A on any window (the documented reason `forget` exists).

**F3. Asynchronous rig (5 seeds).** S within ±5% of A on both the whole-record and the
hot-window ratio.

**F4. Arm (seed 0).** S does not NaN and is within ±10% of A in every window. This is the
rig that broke the last two grid changes; the star adds no member and no node to any
member, so the mechanism of those failures is absent — but the bank's weights enter the
arm's output, so a difference is possible.

**F5.** The star's rate ends well below the ceiling on the hero rig's 900 steps only if the
record is long enough to walk down; expect it to stay within a factor of a few of the
ceiling (one or two moves).
