# 0013 — predictions, recorded before the run

> **AI-generated, not peer-reviewed.** Written before
> [`0013_profile_the_ramp.py`](0013_profile_the_ramp.py) was first executed, after `0012`
> showed `forget=None` 9% worse than the default on a slow ramp of sensor noise (paired
> `t` = +9.95) and worse in log-likelihood too.

Candidates for the loss, and how each is separated:

- **the state step** (`_carry`): run the drift with it switched off.
- **the standing weight on neighbours**: run a single-class filter at the class the default
  settles on; if the default is effectively one class and that class alone matches it, the
  drift's loss is what it pays for not committing.
- **lag in the reported noise level**: compare the reported measurement scale with the
  truth on the rising and falling flanks.

**R1.** Switching the state step off recovers less than a third of the gap — on a scalar
level with a known model, members' states differ little.
**R2.** The loss is on the flanks (rising and falling), not on the flat stretches.
**R3.** The default sits on one narrow, high-memory class throughout; the drift spreads
across several and keeps a wide one alive, and its reported scale is noisier.
**R4.** `forget = 1` is as good as the default here (nothing to re-select on a smooth ramp).
