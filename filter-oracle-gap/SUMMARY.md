# Current state

Closing the gap between the ODE filter and an oracle told the noise schedule
exactly — and finding the hole behind symptoms reported from three
workstreams: `0039`'s 80%-forced / 0%-as-fitted process-scale channel, the
self-confirming $s_P=0$ boundary, crypto's basin roulette, and fits whose
endpoints depend on which optimiser walked there.

**The hole is found, measured, and decomposed.** The GPB1 collapse — one
shared covariance for every grid node — makes the likelihood **flat along
the ridge $Q(s)=Q_{\text{eff}}e^{-s_P^2/2}$**: it measures the *mean*
process variance and cannot split it between a constant level and a
wandering scale ([`0005`](exploration/0005_the_hole_is_the_ridge.md)).
Ridge relief 0.0022 nats/pt shipped against 0.0101 with per-node
covariances (IMM — same model, no new parameters), whose argmin sits on the
generating $s_P$. The full accounting of the static-to-oracle span
([`0009`](exploration/0009_what_the_fit_does_with_the_sharper_likelihood.md)):
**80.0%** already captured by the forced shipped channel, **9.5%** the
collapse (repairable), **6.8%** the AR(1)-vs-regime channel model, **3.7%**
irreducible detection lag. The causal ceiling is **96.3%** of the oracle;
almost nothing about this gap is fundamental. A no-mixing bank closes 0.0%
— the kernel's forgetting is essential, not overhead.

**The fit follows the likelihood most of the way home.** The same staged
search on the IMM likelihood moves every fitted coordinate toward the truth
on every seed tried ($\hat s_P$ 1.07–1.28 against GPB1's 1.42–2.14, truth
0.8; $\hat\varphi_P$ 0.65–0.85 against 0.00–0.74, truth 0.9) — the ridge is
tilted, not abolished, at $n=900$.

**And the boundary is ill-posed for any point estimator** — the decisive
negative of the workstream so far. The kick control
([`0008`](exploration/0008_the_kick_control.py)) shows IMM-ML reads
$\hat s_P \in [0.3, 0.9]$ into kick-free homoscedastic windows, while
GPB1-ML stares at three 6-SD kicks and reports a dead-flat model on one
seed and $s_P=1.47$ on another. Fisher information in a spread parameter
vanishes at zero spread; the plug-in is the defect, not the search. This
lands on `0039`'s item 0a from the opposite direction, with the prices now
measured: the ill-posed small-positive estimate costs +0.0055 ± 0.0009
nats/pt (16× under the 0.0921 exposure); the boundary zero it replaces
costs the whole channel exactly when it is needed.

**The earned patch is one design, two parts**: per-node covariances in
`core.py`, and $s_P$ marginalised over a small $(\varphi_P, s_P)$ grid the
way every other nuisance in this filter already is — meaningful only now,
because a flat likelihood integrates to indifference and the IMM one does
not.

## The scoreboard

From `0039`, updated: beat 0.0025 premium against 0.0872 exposure —
**measured: +0.0055 ± 0.0009 against +0.0921**, 16×, and the premium is the
price of removing the catastrophic exposure; `0032`'s window no worse than
+0.0004 at fitted parameters — **passed, −0.0031 (better)**; close more of
the oracle gap than forced-80% — **89.5% forced, 96.3% ceiling**; break the
self-confirmation — **done for the profile** (argmin at truth from both
sides), replaced for the point fit by the marginalisation design above.

## Next, in order

0. ~~**The IMM patch.**~~ — **shipped**: `OdeFilter(collapse="imm")` and
   `fit(collapse="imm")` run the per-node recursion end to end (streaming,
   batched, predict, dynamics channel), bit-identical to the shipped one at
   $s=0$, default unchanged so downstream internals contracts hold. The
   battery that would have caught the holes is in
   `ode-adaptive-filter/output/tests/`: the ridge must separate, the forced
   channel must clear 85% of the oracle gap, the fitted $s_P$ must stay off
   the boundary. Through the core path the fit lands $\hat s_P = 0.87$
   against a truth of 0.8 at $n=600$.
0a. **Delete GPB1 once crypto migrates.** Decision recorded: `"imm"` is
   strictly superior (same model, strictly more of the evidence), so the
   `collapse` option exists only until `crypto-predictivity/output/mixture.py`
   reads the per-node state — its SUMMARY item 0a. Then the two modes
   collapse to the single most performant one and the flag disappears.
1. **The marginalised $(\varphi_P, s_P)$ grid** — the second half of the
   design; needs `Params`-level architecture (a hypothesis set is not a
   point) and its own premium/exposure/`0032` measurements.
2. **Fit-time cost.** An IMM fit runs ~107 s at $n=600$ against ~15 s
   shipped — the polish loop, not the evaluator (1.4×), is where the time
   goes; it needs the same stencil economics the speedup gave GPB1.
3. **Crypto's real basins** under the IMM likelihood — does 0.003
   in-sample become a decision on the actual BTC series.
4. The channel model's 6.8% (a regime-hazard or heavier-tailed chain in
   place of AR(1) — touches `FILTER-NOTES` §7's long-memory note).
5. Unassigned symptoms: `FILTER-NOTES` §8 (root nulls), `crypto/0022`
   (`dynamics` as a spread detector).

## Layout

- `exploration/` — numbered, later is more recent.
  [`0001`](exploration/0001_the_frame_and_the_ledger.md) frame and symptom
  ledger; [`0002`](exploration/0002_per_node_covariances.py)–
  [`0004`](exploration/0004_the_ridge.py) probes;
  [`0005`](exploration/0005_the_hole_is_the_ridge.md) the ridge finding;
  [`0006`](exploration/0006_the_fit_on_the_imm_likelihood.py)–
  [`0008`](exploration/0008_the_kick_control.py) the fit on the IMM
  likelihood and its gates;
  [`0009`](exploration/0009_what_the_fit_does_with_the_sharper_likelihood.md)
  the decomposition and the marginalisation case. Raw numbers in
  `exploration/figures/gap000N.json`.
- Patches, when earned, land in
  [`ode-adaptive-filter/output/odefilter/`](../ode-adaptive-filter/output/odefilter/)
  — the filter stays where it lives; this folder holds the investigation.
