# 0008 — predictions, recorded before the run

> **AI-generated, not peer-reviewed.** Written before
> [`0008_the_bad_seed.py`](0008_the_bad_seed.py) was first executed.

The question: on arm seed 2 the star's tip error spikes in steps 500–600 and 1400–1500
(`0007`). Why?

**Step 1 — where the weight is and who is wrong.** Record, every step, the bank's weight on
each `(phi, s)` class and the tip error of each class's own state estimate.

- **G1.** During the spikes the star holds substantially more weight on the wide classes
  (`s ≥ 1.6`) than shipped does.
- **G2.** Those classes' own estimates are the bad ones (tip error ≥ 10× the narrow
  classes'), so the spike is weight on bad members, not every member being wrong.

**Step 2 — dose.** Re-run with the ceiling rates scaled by 0.3, 0.1, 0.03.

- **G3.** The spike shrinks monotonically with the scale and is gone (calm within 2× of
  shipped) by 0.1.
- **G4.** The SENSOR-window gain survives at 0.1 (still below 2.0× the oracle on this seed).

If G1/G2 fail, the mechanism in `0007` (weight moved onto members with poor state) is
wrong and step 2 is uninformative about it.
