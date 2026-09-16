# 0061 — The interior degradation: where, the simplest repro, and the reason

> **AI-generated, not peer-reviewed.** Scripts: [`0061_per_copy_through_potfail.py`](0061_per_copy_through_potfail.py)
> (each rung's own state error, likelihood and scales through POTFAIL on the arm), [`0061_chain_repro.py`](0061_chain_repro.py)
> (the single-joint chain rig), [`0061_chain_repro_with_mixing.py`](0061_chain_repro_with_mixing.py) and
> [`0061_state_mixing_patch.py`](0061_state_mixing_patch.py) (the completion and its test),
> [`0061_state_mixing_rigs.py`](0061_state_mixing_rigs.py) (the completion on every rig).

`0060` found the rate ladder's interior non-monotone in the state: POTFAIL 1.11 → 1.36 at five
rungs, a 1.77 m onset spike at nine. Non-monotone means not a budget but a piece of theory. This
note finds where the degradation is, builds its simplest reproduction, and names its cause — which
turns out to be a missing half of the switching model the copies were given.

## 1. Where

From the saved per-step errors (`0060`) and the per-copy diagnostics on the arm, seed 0,
five rungs:

- The POTFAIL excess is not spread over the phase. **69% of it is in steps 1100–1150 and 21% in
  1150–1200**, i.e. the 100 steps after the bank hands the mixture from the eager rung to the
  rate-0.18 rung (its weight passes 0.8 at step 1085). By 1200 the two ladders read the same.
- At the handover the rate-0.18 copy has a **better per-step predictive log-likelihood** than the
  eager copy — +24.77 against +24.72 — and a **worse state**: its own tip error is 0.0052 m against
  0.0035. It is stale: it lost the PROCESS regime, its own error in the calm before POTFAIL was
  0.0080 against the eager copy's 0.0020, and it was still recovering when the bank chose it.
- The never copy of the two-rung ladder is in the same position but 100× more stale (own error
  0.19–0.26 m, likelihood −7 to +8 against +21 to +27) and never competes. That is the difference
  in kind: the pair is "eager or hopeless", so the bank chooses by attribution alone; the interior
  is "eager or slightly stale", and a one-step density can prefer slightly stale.
- The nine-rung 1.77 m spike is at the BOTH onset (step 1452), where the weight jumps among
  interior rungs; the nine-rung POTFAIL onset bin (1050–1075) reads 0.21 m for the same reason.

## 2. The simplest reproduction

A one-dimensional level at its floor read by two sensors does **not** reproduce it (2 and 5 rungs
within noise; the interior copies' states are *better* there and the bank chooses right). What is
missing in one dimension: a stale state recovers in a few steps, and the clean second sensor reads
it directly, so the likelihood is not blind to it.

The chain does reproduce it — one joint of the arm, linear: position / velocity / acceleration
with jerk noise, a pot on position and an accelerometer on acceleration (the arm's own single-joint
model, constant `H`); PROCESS (jerk ×20), calm, POTFAIL (pot ×15), calm; 3 seeds:

| rungs | calm | PROCESS | calm | **POTFAIL** | calm | POTFAIL per seed |
|---|---|---|---|---|---|---|
| 2 | 1.11 | 1.08 | 1.11 | **2.63** | 1.11 | 1.69 / 2.77 / 6.26 |
| 5 | 1.11 | 1.10 | 1.11 | **3.04** | 1.11 | 2.57 / 2.84 / 5.42 |

Stepping the handover shows two mechanisms, one per kind of seed.

**The onset spike (seed 0, both ladders).** At the first POTFAIL step the stale copy — position
error 2.7 rad — takes 0.99 of the bank for one step: the eager copy's `S` is calibrated to calm pot
noise and pays −9.9 nats of surprise on the first ×15 innovation; the stale copy, wide already, pays
−1.5. The mixture jumps 2.7 rad for a step. This is the arm's 1.77 m BOTH spike in miniature, and
the 0007 mode-ladder catastrophe's mechanism: the widest member wins the onset step.

**Blind averaging (seeds 1 and 2, both ladders).** Through POTFAIL the copies' likelihoods tie to
within 0.1 nats per step: the pot is the only direct reader of position and is 225× noisier, and the
accelerometer sits two integrations from position, so the one-step density carries almost no
information about the state component the copies disagree on. With no evidence the switching
prior relaxes the weights toward its equilibrium (0.5 / 0.5, or spread over the interior), and the
mixture becomes the **average** of states that drift apart through the masked regime. More rungs,
more partners with more divergent states, worse average: 2.8 → 5.4× on seed 2.

Why the arm's SENSOR regime is the opposite case: there the masked channels are the accelerometers
and the clean pots read position directly, so the likelihood sees the state the copies disagree on
and chooses right (2.70 → 1.46). The evidence the copies provide is the divergence of their states
*as seen through the unmasked channels*; when those channels cannot see the component in question,
the bank has no evidence and defaults to its prior.

## 3. The reason

The bank mixes **weights** under a switching prior and never mixes **states**. A mixture of
*static* hypotheses (the class box, the split ladder) is right to keep states apart: the
hypotheses never switch, and the weights only forget. The attribution copies are *switching*
hypotheses — the regime can change — and a switching-hypothesis mixture is consistent only if states
mix at the rate the hypotheses switch (the Markov-switching / IMM structure): under no evidence
the copies' states should coalesce as fast as the regime is allowed to change, not be averaged from
wherever their histories left them. Mixing weights alone under no evidence averages divergent
states — that is the degradation, and it grows with the number of copies available to average.

**The completion, derived from the same object:** IMM mixing of each cell's copies at the
switching ladder's posterior-mean rate `r` (the symmetric kernel's mixing probabilities; no new
constant). On the chain rig it removes the effect entirely and makes the interior flat:

| rungs | POTFAIL, weights only | POTFAIL, weights + states mixed | calm |
|---|---|---|---|
| 2 | 2.63 | **0.84** | 1.11 → 1.04 |
| 5 | 3.04 | **0.84** | 1.11 → 1.04 |

(Below 1.0 against a Kalman told the schedule: that oracle is sequential-scalar and initialised
diffuse; the ratio is a comparison, not a bound.)

On the arm the same completion makes the interior flat (2 and 5 rungs read alike) and fixes the
regimes the degradation lived in, but it also gives back much of the SENSOR gain, because the
evidence the copies carry *is* the divergence of their states, and mixing them at the switching
rate erodes it over a 250-step burst:

| arm, seed 0 | calm | SENSOR | PROCESS | POTFAIL | BOTH |
|---|---|---|---|---|---|
| shipped | 1.14 | 2.70 | 1.16 | 1.11 | 1.06 |
| floor copy, 2 rungs, weights only (`0059`) | 1.11 | 1.46 | 1.16 | 1.11 | 1.06 |
| floor copy, 5 rungs, weights only (`0060`) | 1.11 | 1.40 | 1.16 | 1.36 | 1.11 |
| floor copy, 2 rungs, **weights + states mixed** | 1.12 | 2.30 | 1.19 | **1.01** | 1.39 |
| floor copy, 5 rungs, weights + states mixed | 1.13 | 2.31 | 1.19 | 1.03 | 1.34 |
| swapped schedule, 2 rungs, weights + states mixed: PROCESS-first / SENSOR-after / POTFAIL / BOTH | | 0.99 | 2.02 | 1.00 | 0.86 |

PATIENCE_SWITCH_PENDING

## 4. What this settles

- The interior is not a budget. Two rungs and infinitely many differ in kind because a bank that
  mixes weights without states is only consistent when the stale copy is hopeless enough never to
  tie; the interior supplies copies that tie.
- The consistent model is the switching filter: weights and states mixed at one rate, the rate the
  ladder already reads. Under it the interior is flat and the masked regimes are fixed, and the
  attribution gain shrinks to what state mixing at that rate leaves of the copies' divergence.
- What remains is a genuine tension inside one derived object, not a knob: the switching rate that
  best reads the regime (0.005 on the arm) mixes states too fast to keep the sensor-burst evidence
  and too slowly to keep the never copy from tying in a masked regime. Where the evidence lives
  (the state divergence) and where the consistency lives (the state mixing) are the same place.
  The next derivation is what a switching filter's mixing should be when the observation is
  *blind* to the component the hypotheses disagree on — mixing conditional on evidence, which is
  the IMM's missing case, not a constant.
