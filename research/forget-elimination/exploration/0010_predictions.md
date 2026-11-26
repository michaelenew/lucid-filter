# 0010 — predictions, recorded before the run

> **AI-generated, not peer-reviewed.** Written before
> [`0010_the_battery.py`](0010_the_battery.py) and
> [`0010b_suite_under_drift.py`](0010b_suite_under_drift.py) were first executed, after the
> class drift was moved into `lucid.py` as `forget=None` and one hero record had been run
> as a smoke test (RMSE 0.655, log-likelihood −1702.0; default 0.713, −1714.6).

**B1. The filter's own path reproduces the patch.** On the arm, `forget=None` gives the
numbers of `0009`'s "star + m" to within 1% in every window on every seed (the state step
was re-implemented per split rung).

**B2. Hero rig with state carried** (it was weights-only in `0007`): the log-likelihood
gain survives at more than +10 nats with paired `t` > 3; loud-window RMSE still better
than shipped; jump and steady within noise.

**B3. Async rig:** RMSE within noise of shipped on whole and hot; log-likelihood higher.

**B4. The shipped test suite with `forget=None` as the default:** at most three of the 55
behavioural tests fail, none of them with a non-finite output. The most likely failures
are tests whose thresholds were set on the escape's behaviour (fault detection timing,
the offset channel's inertness) rather than tests of structure.
