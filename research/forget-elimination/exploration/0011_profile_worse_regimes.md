# 0011 — the arm windows where the drift loses on a seed

> **AI-generated, not peer-reviewed.** Code
> [`0011_profile_worse_regimes.py`](0011_profile_worse_regimes.py), output
> [`0011_output.txt`](0011_output.txt). Traces of the default and of `forget=None` on the
> same records: per step, the tip error, the bank's weight on each `(phi, s)` class, each
> class's own tip error, and the drift's rate.

On six seeds the drift is not worse than the default in any arm window on average
(`0010`). On individual seeds it is: PROCESS on seeds 0 and 5, BOTH on seeds 3, 4 and 5.
Three of those were traced.

## What the traces show

**In every block of every trace the default has all of its weight on one class, `s = 0.8`.**
It does not select on the arm (`0008` found this on seed 2; it holds on seeds 3 and 5).

| seed, window | default → drift | where the loss is | where the drift's weight is |
|---|---|---|---|
| 5, PROCESS | 1.169 → 1.331 | **diffuse**: 10–20% in every 50-step block; the worst 5% of steps carry 13% of the excess | split between `s = 0.4` and `s = 0.8` (0.2–0.5 / 0.4–0.7) |
| 5, BOTH | 1.025 → 1.139 | **at the onset**: ×1.9 and ×1.45 in the first 60 steps, then 10–13% *better*; the worst 5% of steps carry 61% | 0.94–0.96 on `s = 0.4` at the onset |
| 3, BOTH | 1.341 → 1.505 | **at the onset**: ×1.56 in steps 1500–1550, then 2–22% *better*; the worst 5% of steps carry 94% | 0.86–0.96 on `s = 0.4` |

Under the drift every class's own estimate is within a few percent of the bank's (the
state step keeps them together), so there is no better member being passed over. The
difference from the default is the class the bank is *in*.

## Reading

1. **The loss is the price of being right in the calm.** In a calm stretch the narrower
   class `s = 0.4` is the better description, the drift selects it, and the arm's calm
   window improves on five seeds of six (`t` = −3.3). A narrower class moves its noise
   level more slowly, so when a burst starts the drift is a class behind for 50–60 steps —
   it must first resolve that the class has changed. The default never left the wider
   `s = 0.8` and so is already there.
2. **The default's advantage at an onset is not a decision it made.** It is pinned to one
   mid-width class by the evidence of the whole record; that class is a compromise that
   costs it the calm window and the SENSOR window and happens to suit the first steps of a
   process burst.
3. **After the onset the drift is ahead** in both BOTH traces (10–22% better in the later
   blocks), and on the six-seed mean BOTH is 1.14 against 1.60.
4. **PROCESS on seed 5 is different**: a diffuse 10–20% with the weight split across two
   classes, not an onset. Not explained beyond that; the six-seed mean is 1.23 against 1.25.

## What would remove the onset loss

Faster recognition that the class has widened. The chain only moves between neighbouring
classes, one resolution time per move; a burst is a jump of the noise level, and the bank
has a separate jump channel (the hazard ladder) that the class chain does not consult.
Letting a detected jump reprice the class as it already reprices a walker is the direction
this points; it has not been tried.
