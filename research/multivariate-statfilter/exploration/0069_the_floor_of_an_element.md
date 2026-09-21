# 0069 — The floor of an element on the memory axis

> **AI-generated, not peer-reviewed.** Script: [`0069_posterior_floor_patch.py`](0069_posterior_floor_patch.py)
> (a posterior floor `ε` on the rung marginals of the attribution copies, and the elements of the axis as a
> list), run with the per-seed arm battery of 0065 and the onset data of 0067. Measurements in §3.

## 1. The observation that starts this

0067 §4's table has the patient copy's best cell at weight 1e⁻²²⁷ one step after the onset, and 0067 §6's
trace has it climbing from 1e⁻³⁰⁰ through 1e⁻¹⁴⁴, 1e⁻²⁹ to 0.9 over forty steps. Log-weights of minus
hundreds after one step mean that one observation carried hundreds of nats of evidence between two settings
of the *same* estimator. No observation with intrinsic noise can do that; a posterior that does has used the
observation to overrule its own prior about the world, and a mixture whose elements can be driven there is a
winner-take-all switch between "explain everything" and "explain nothing" — which is what two elements on
the axis are, and what a ladder of intermediate elements would have softened.

## 2. The floor, derived

The elements of the memory axis are not hypotheses about the world; they are timings of one model (0062).
The bank's own class says the world may leave any hypothesis with probability `r = 1/mem` per step — the
derived switching rate. So at every step there is prior probability `r` that the *other* element is the
right one from here on, whatever the observation says. A posterior that assigns an element less than that
after one step has let one noisy observation overrule the class's own switching prior, which one observation
cannot do: the per-step log-odds between elements is bounded by `log(1/r)`, and the floor of an element's
posterior weight is

    ε = r / (k − 1) = 1 / (mem (k − 1)),      k elements on the axis;   ε = 10⁻³ for the two-copy grid.

Today the switching kernel floors the *prior* at this value (`_attribution_mix`), and the likelihood then
multiplies the prior by e⁻¹⁰⁰⁰; the floor never reaches the posterior. The patch floors the posterior rung
marginal after the update and writes the floor back into the log-weights, so the bound holds step to step.

What this floor is *not*: a statement about which element is right. It is the statement that one step's
evidence between two timings of one estimator is worth at most the class's own switching odds. If the
measured optimum sits elsewhere, the derivation is wrong or incomplete, and §3 says which.

## 3. Measurements

(pending: ε ∈ {0, 10⁻⁶, 10⁻⁴, 10⁻³, 10⁻², 10⁻¹} on two elements; four elements {1, 0.1, 0.01, 0.001} at ε = 0 and
ε = 10⁻³/3; arm, 3 seeds: whole-burst RMSE, worst excursion, regime windows; seed 1's onset step by step.)
