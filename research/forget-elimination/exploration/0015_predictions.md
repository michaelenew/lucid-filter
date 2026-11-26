# 0015 — predictions, recorded before the run

> **AI-generated, not peer-reviewed.** Written before `0014_the_fault_rungs.py` was re-run
> with a chain along the hazard rungs in `_ClassDrift` (rate between neighbouring rungs
> from the ladder's own Whittle metric, about `2/mem` per step, scaled by the star's
> fraction; walker state carried between rungs).

- **W1.** After 4,000 calm steps the reported hazard under `forget=None` is within a
  factor 2 of the default's 3.3e-4 (it was 4.7× lower).
- **W2.** The fault read-out crosses ½ within 15% of the default's delay after both calms
  (it was 1.4× and 1.55× later).
- **W3.** The state RMSE after the fault stays within 1% of the default's.
- **W4.** False alarms in the calm do not exceed the default's rate.

Low confidence in W1/W2: the star's fraction is small in a calm world, and the rungs'
chain is scaled by it, so in exactly the case that matters — a long calm — the chain may be
too slow to matter.
