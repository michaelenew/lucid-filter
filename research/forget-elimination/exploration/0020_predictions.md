# 0020 — predictions, recorded before the run

> **AI-generated, not peer-reviewed.** Written before
> [`0020_a_star_per_axis.py`](0020_a_star_per_axis.py) was first executed.

**The construction** ([`drift3.py`](drift3.py), a probe-side replacement for `_ClassDrift`):

1. *Grid-free ladders.* The split and hazard beliefs diffuse in their own arclength with
   hop rate `D/Δt²` between neighbouring rungs (`0019`).
2. *The interval as yardstick.* A ladder lives on a bounded range of length `L` with
   information `I = 1` per step. The ceiling is the diffusion at which the steady blur
   equals the spread of knowing nothing: `√(D/I) = L²/12`, so `D_max = L⁴/144` —
   0.042 for the split (`L = π/2`), 0.0084 for the hazard (`L = π/3`). No grid, no memory.
3. *A star per axis.* Three unknown fractions — class, split, hazard — each beneath its own
   ceiling, on a caltrop star: a centre and two arms either way along each axis (13 node
   copies of the bank's weights). At most one axis recentres per step. All start at their
   ceilings.

Today's behaviour corresponds to fractions of about `2e-4` (split) and `1e-3` (hazard) of
these ceilings, so each ladder star has seven to nine e-folds to walk down.

- **H1.** Grid independence holds: the ramp result at `_LADDER_MEM` 1000 and 4000 agrees
  within 2%.
- **H2.** On the ramp it is within 7% of the filter as it stands (1.033 × the oracle), and
  still better than the default (1.118).
- **H3.** By the end of the ramp record the split star sits within two e-folds of `2e-4`.
- **H4.** Fault rig: flagging within 25% of the default's after both calms; believed hazard
  within a factor 3. Early in the record, with the hazard belief still near uniform, false
  alarms exceed the default's.
- **H5.** Hero rig (900 steps — too short for a nine-e-fold descent): log-likelihood lands
  between the default's and the filter-as-it-stands'; the loud-window gain is partly lost.
- **H6.** The walk down is the price everywhere: in calm, a log-likelihood per step below
  the filter-as-it-stands' over the first third of a record.

**What would count as success:** H1 and H2 with H4, at a calm cost that decays. **What
would count as failure:** the ladder stars stalling far above the useful range (the blind
floor of `0017`, or leads that never accumulate because the nodes near the ceiling all
predict alike).
