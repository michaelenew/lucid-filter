# 0014 — predictions, recorded before the run

> **AI-generated, not peer-reviewed.** Written before
> [`0014_the_fault_rungs.py`](0014_the_fault_rungs.py) was first executed.

**The concern.** With `faults=True` the bank carries 16 hazard rungs — hypotheses about how
fault-prone the world is, from 0.42 per step down to 2.9e-7. Under the default, `forget`
acts on the rungs' weights like every other weight. Under `forget=None` nothing does: the
class chain covers `(phi, s)` and the split, not the rungs, so the posterior over the
hazard is pure Bayes and should concentrate on "quiet" during a long calm and then be slow
to admit a fault.

**The test.** AR(1) dynamics switch 0.9 → 0.3 after a calm of 600 or 4,000 steps. Arms:
default; `forget = 1`; `forget=None`.

- **V1.** After 4,000 calm steps the reported hazard is lower under `forget = 1` and
  `forget=None` than under the default.
- **V2.** Detection (fault readout above ½) after the long calm is slower under
  `forget=None` than under the default, but by less than 1.5×: the per-step evidence
  separating two low rungs in calm is about the difference of their rates, ~1e-3 nats, so
  4,000 steps accumulate only a few nats.
- **V3.** After the short calm the three arms detect within 20% of each other.
- **V4.** `forget=None` and `forget = 1` behave alike on the rungs (the drift adds nothing
  there), so any difference between them is the class drift acting elsewhere.
