# 0016 — predictions, recorded before the run

> **AI-generated, not peer-reviewed.** Written before
> [`0016_rungs_at_their_own_rate.py`](0016_rungs_at_their_own_rate.py) was first executed.

`0015` gave the hazard rungs a chain scaled by the class star's fraction. It closed about
half the gap (flagging delay after a long calm 1.55× → 1.33× the default's). The suspected
reason: how fast the *class* drifts and how fast the *hazard* drifts are different
unknowns, and in a calm world the class star walks down to a small fraction and takes the
rungs' chain with it.

**Variant under test:** the rungs' chain runs at its own ceiling, not scaled by the class
star — the hazard is assumed to drift as fast as it can be resolved. This is `0002`'s
stance, for this one axis, and `0002` found such a stance pays a standing cost in a calm
world of about half its hop rate (here ~1e-3 nats/step).

- **X1.** Flagging delay within 10% of the default's after both calms.
- **X2.** Reported hazard before the fault within a factor 1.5 of the default's, either side.
- **X3.** False alarms in calm at or below the default's.
- **X4.** State RMSE after the fault within 1% of the default's.
- **X5.** Log-likelihood over the calm lower than the star-scaled version's, by about
  1e-3 nats/step (the standing cost), and not lower than the default's.
