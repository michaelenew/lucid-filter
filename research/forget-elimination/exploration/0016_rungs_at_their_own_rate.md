# 0015 / 0016 — fixing the hazard rungs

> **AI-generated, not peer-reviewed.** Code [`0014_the_fault_rungs.py`](0014_the_fault_rungs.py)
> (the rig), [`0016_rungs_at_their_own_rate.py`](0016_rungs_at_their_own_rate.py); outputs
> [`0014_output.txt`](0014_output.txt) (no chain on the rungs),
> [`0015_output.txt`](0015_output.txt) (a chain scaled by the class star),
> [`0016_output.txt`](0016_output.txt) (the comparison),
> [`0016b_output_in_filter.txt`](0016b_output_in_filter.txt) (the filter as it now stands);
> predictions written first in [`0015_predictions.md`](0015_predictions.md) and
> [`0016_predictions.md`](0016_predictions.md). 4 seeds.

`0014` measured the defect: with nothing moving weight between the hazard rungs, a long
calm teaches the bank that faults do not happen, and a fault is then flagged 1.55× later
than under the default.

## Two attempts

**`0015` — the rungs get the class's chain.** Neighbouring rungs may swap no faster than
they can be resolved; the rate is read from the Whittle arclength the ladder is built in
(the same grid as the split ladder, so about `2/mem` per step between neighbours), and
scaled by the star's fraction like the other two chains. Walker state is carried between
rungs.

**`0016` — the rungs drift at their own ceiling**, not scaled by the star.

| calm | arm | hazard believed before the fault | false alarms in calm | steps to read-out > ½ (median) | … held 50 | RMSE, 300 steps after | calm log-lik/step − default, ×1000 |
|---|---|---|---|---|---|---|---|
| 600 | default | 1.1e-3 | 0.3% | 60 | 92 | 0.2928 | 0 |
| 600 | no chain (`0014`) | 8.0e-4 | 0.1% | 84 | 104 | 0.2933 | |
| 600 | chain × star (`0015`) | 9.3e-4 | 0.1% | 69 | 104 | 0.2923 | −1.30 |
| 600 | **chain at own ceiling (`0016`)** | 9.5e-4 | 0.1% | 68 | 104 | 0.2922 | −1.31 |
| 4,000 | default | 3.3e-4 | 0.4% | 82 | 94 | 0.2728 | 0 |
| 4,000 | no chain | 7.1e-5 | 0.1% | 127 | 135 | 0.2738 | |
| 4,000 | chain × star | 1.5e-4 | 0.1% | 109 | 126 | 0.2732 | −0.62 |
| 4,000 | **chain at own ceiling** | **3.9e-4** | 0.2% | **92** | **98** | 0.2726 | −0.73 |

Per seed after the long calm, first crossing: default 107 / 68 / 66 / 97; own ceiling
112 / 73 / 68 / 207. Three seeds within 8%; on the fourth the default's early crossing did
not hold (its held read-out is at 210, the drift's at 207).

## Predictions against the runs

| | prediction | verdict |
|---|---|---|
| W1 | chain × star: hazard within 2× of the default's after the long calm | **marginally FALSE** — 2.3× lower (was 4.7×) |
| W2 | chain × star: flagging within 15% of the default's | **FALSE after the long calm** — 1.33× (was 1.55×); 1.15× after the short one |
| W3, W4 | state RMSE within 1%; false alarms not above the default's | confirmed |
| — | the stated doubt: the star's fraction is small in calm and takes the rungs with it | **that is what happened** |
| X1 | own ceiling: flagging within 10% of the default's | **marginally FALSE** — 1.12× and 1.13× on first crossing; 1.04× and 1.13× held |
| X2 | hazard within 1.5× of the default's | confirmed — 1.2× higher / 1.2× lower |
| X3, X4 | false alarms at or below the default's; RMSE within 1% | confirmed |
| X5 | own ceiling costs ~1e-3 nats/step of calm over the star-scaled chain | **much less** — 0.12e-3 |

## Reading

1. **Why one fraction for everything was wrong.** The star measures how fast the *class*
   `(phi, s)` drifts. How fault-prone the world is, is a different quantity with its own
   rate of change, and a calm record carries no evidence about that rate at all — nothing
   distinguishes "the hazard is steady and low" from "the hazard is free to move" until a
   fault arrives. Tying the rungs to the class star froze them exactly when the class was
   calm, which is exactly when the belief about faults goes stale.

2. **At their own ceiling the rungs match the default's report** to within what four seeds
   can resolve, with fewer false alarms and the same state error, for 0.12e-3 nats per
   step of calm. This is what is now in the filter (`_ClassDrift._P`), and its numbers
   reproduce the probe's exactly.

3. **It is a stance, not a measurement of the rate.** The hazard is assumed to drift as
   fast as it can be resolved and nothing walks it down. That is `0002`'s position for
   this one axis; `0002` found it expensive for the class and the cost here is small
   because the ladder's ceiling is itself slow (a rung every ~500 steps). A walk of its
   own on this axis would be the consistent completion; with no evidence in calm to move
   it, it would sit where it starts, so it has not been built. Graded *convention* in the
   audit: a shape assumption, backed by the derivation in point 1 and the measurement
   above, whose consequences reach only the fault report.

4. **The same question now applies to the split.** It too is scaled by the class star
   (`0013`). No test has shown that to be wrong — the ramp, stale and gap modes are all
   won — but it rests on the same simplification.

5. **A residual the rungs do not explain.** After the short calm the drift flags ~13%
   later than the default with or without the fix (and the star is still at its ceiling in
   a 600-step record). Not investigated; state error is unaffected there too.
