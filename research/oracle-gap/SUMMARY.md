# Current state

Closing the gap between the ODE filter and an oracle told the noise schedule
exactly — and finding the hole behind symptoms reported from three
workstreams: `0039`'s 80%-forced / 0%-as-fitted process-scale channel, the
self-confirming $s_P=0$ boundary, an applied workstream's basin roulette, and fits whose
endpoints depend on which optimiser walked there.

**The hole is found, measured, and decomposed.** The GPB1 collapse — one
shared covariance for every grid node — makes the likelihood **flat along
the ridge $Q(s)=Q_{\text{eff}}e^{-s_P^2/2}$**: it measures the *mean*
process variance and cannot split it between a constant level and a
wandering scale ([`0005`](0005_the_hole_is_the_ridge.md)).

> **⚖️ ATTRIBUTION —** _GPB1 (single-Gaussian collapse per step) and its per-node replacement IMM are both textbook multiple-model filters; that GPB1's collapse loses discrimination relative to a per-node bank is the standard known cost of the collapse._ Prior art: Generalized Pseudo-Bayesian GPB1 — Ackerson & Fu 1970, Bar-Shalom; Interacting Multiple Model (IMM) — Blom & Bar-Shalom 1988; the underlying "which variance level vs spread" degeneracy is a stochastic-volatility identifiability question (Taylor 1986; Harvey/Ruiz/Shephard 1994). Status: REPRODUCTION.
Ridge relief 0.0022 nats/pt shipped against 0.0101 with per-node
covariances (IMM — same model, no new parameters), whose argmin sits on the
generating $s_P$. The full accounting of the static-to-oracle span
([`0009`](0009_what_the_fit_does_with_the_sharper_likelihood.md)):
**80.0%** already captured by the forced shipped channel, **9.5%** the
collapse (repairable), **6.8%** the AR(1)-vs-regime channel model, **3.7%**
irreducible detection lag. The causal ceiling is **96.3%** of the oracle;
almost nothing about this gap is fundamental. A no-mixing bank closes 0.0%
— the kernel's forgetting is essential, not overhead.

> **⚖️ ATTRIBUTION —** _The decomposition method (switch each error source independently, price the residual) is standard ablation; the value here is the specific measured budget on this synthetic rig — those numbers are original observations, though every phenomenon (collapse cost, model mismatch, irreducible causal detection lag) is known._ Prior art: oracle/clairvoyant-bound comparison and causal detection lag are standard in adaptive filtering / quickest-detection (Lorden 1971; Bar-Shalom); IMM vs GPB1 as above. Status: NEGATIVE-RESULT.

**The fit follows the likelihood most of the way home.** The same staged
search on the IMM likelihood moves every fitted coordinate toward the truth
on every seed tried ($\hat s_P$ 1.07–1.28 against GPB1's 1.42–2.14, truth
0.8; $\hat\varphi_P$ 0.65–0.85 against 0.00–0.74, truth 0.9) — the ridge is
tilted, not abolished, at $n=900$.

**And the boundary is ill-posed for any point estimator** — the decisive
negative of the workstream so far. The kick control
([`0008`](0008_the_kick_control.py)) shows IMM-ML reads
$\hat s_P \in [0.3, 0.9]$ into kick-free homoscedastic windows, while
GPB1-ML stares at three 6-SD kicks and reports a dead-flat model on one
seed and $s_P=1.47$ on another. Fisher information in a spread parameter
vanishes at zero spread; the plug-in is the defect, not the search.

> **⚖️ ATTRIBUTION —** _"Fisher information in a spread/variance parameter vanishes at zero spread, so a point estimate on the boundary is ill-posed" is the classic parameter-on-the-boundary problem — a known non-standard-asymptotics result, not new._ Prior art: testing/estimating a variance component at zero — Chernoff 1954; Self & Liang 1987; boundary MLE has a half-normal (one-sided) limiting distribution — standard result in mathematical statistics, specific source not verified. Status: REPRODUCTION. **A candidate cure is recorded in
[`0010`](0010_the_square_chart.py)** (contributed from the
quantum-mechanics sibling program, AI-generated, unverified in this
harness): the family depends on $s_P$ only through $s_P^2$ near the
boundary, so the defect is the *chart* — in $\tau=s_P^2$ the Fisher
information is finite and flat at the boundary ($I(\tau)\to c/4$,
measured $0.399$ on the one-step toy with analytic scores), and the
boundary estimate becomes the standard well-posed one-sided case (no
sign ambiguity; MLE demo included).

> **⚖️ ATTRIBUTION —** _Reparameterising by $\tau=s_P^2$ so the family enters through the variance and Fisher information is finite at the boundary is standard practice (estimate the variance, not the standard deviation, near zero); the "square is the well-posed coordinate" framing is imported by analogy from a physics sibling program and is decorative, not a new result._ Prior art: variance-vs-scale parameterisation and Fisher information transformation under reparameterisation are textbook (Jacobian $I(\tau)=I(s)/(ds/d\tau)^{-2}$); parameter-on-boundary as above. Status: REPRODUCTION. Untested here: the full GPB1/IMM
likelihood in $\tau$, and whether the $+0.0055$ nats/pt plug-in price
survives the chart change on the kick control. This
lands on `0039`'s item 0a from the opposite direction, with the prices now
measured: the ill-posed small-positive estimate costs +0.0055 ± 0.0009
nats/pt (16× under the 0.0921 exposure); the boundary zero it replaces
costs the whole channel exactly when it is needed.

**The earned patch is one design, two parts**: per-node covariances in
`core.py`, and $s_P$ marginalised over a small $(\varphi_P, s_P)$ grid the
way every other nuisance in this filter already is — meaningful only now,
because a flat likelihood integrates to indifference and the IMM one does
not.

> **⚖️ ATTRIBUTION —** _Marginalising a nuisance parameter over a grid rather than plugging in a point estimate is standard Bayesian model averaging; over a bank of noise-parameter hypotheses this is exactly Multiple-Model Adaptive Estimation._ Prior art: MMAE — Magill 1965; IMM — Blom & Bar-Shalom 1988; Bayesian nuisance marginalisation is textbook. Status: RECOMBINATION.

## The scoreboard

From `0039`, updated: beat 0.0025 premium against 0.0872 exposure —
**measured: +0.0055 ± 0.0009 against +0.0921**, 16×, and the premium is the
price of removing the catastrophic exposure; `0032`'s window no worse than
+0.0004 at fitted parameters — **passed, −0.0031 (better)**; close more of
the oracle gap than forced-80% — **89.5% forced, 96.3% ceiling**; break the
self-confirmation — **done for the profile** (argmin at truth from both
sides), replaced for the point fit by the marginalisation design above.

## The seam, attempted and settled

The follow-up bet — that removing `optimality-proof`'s log-loss/MSE
seam would take the residual gap to zero, the seam having been a happenstance
stabilizer — is scored in
[`optimality-proof/0036`](../optimality-proof/0036_the_seam_is_removed.md).
The seam **is removed** (Theorem A′: layer 1 transfers to code length by the
same three lines, verified), with zero behavioural change and no reversions —
but it was not carrying the gap: the 6.8% channel-model and 3.7%
detection-lag remainders were measured in code length already. The stabilizer
half was real and inverted: the hybrid's *log-likelihood* side is what
stabilises the directions MSE cannot see (`0027`), and it stays. What
removal buys is commensurability — theorems and gap numbers in one currency —
and a single located target for what remains (marginalisation).

## Next, in order

0. ~~**The IMM patch.**~~ — **shipped**: the per-node recursion runs end to
   end (streaming, batched, predict, dynamics channel), bit-identical to the
   old one at $s=0$. The battery that would have caught the holes is in
   `lucid/tests/`: the ridge must separate, the forced
   channel must clear 85% of the oracle gap, the fitted $s_P$ must stay off
   the boundary. Through the core path the fit lands $\hat s_P = 0.87$
   against a truth of 0.8 at $n=600$.
0a. ~~**Delete GPB1 once the downstream reader migrates.**~~ — **done**:
   the one downstream reader now reads the per-node state (the
   mixture is richer — every node carries its own accumulated covariance
   history), the `collapse` flag is gone, and the per-node recursion is the
   only one. Two knock-ons recorded in the odefilter README: the parent
   reduction narrows to exactness on the $s=0$ face (the parent is GPB1 by
   construction, ~6e-3 nats/pt apart with a live channel), and the unit disc
   is no longer walled off numerically — a detectable explosive $\alpha$ has
   a finite likelihood under per-node correction, so the disc is a
   commitment (`unit_roots`), not an emergent property.
1. **The marginalised $(\varphi_P, s_P)$ grid** — the second half of the
   design; needs `Params`-level architecture (a hypothesis set is not a
   point) and its own premium/exposure/`0032` measurements.
2. **Fit-time cost.** An IMM fit runs ~107 s at $n=600$ against ~15 s
   shipped — the polish loop, not the evaluator (1.4×), is where the time
   goes; it needs the same stencil economics the speedup gave GPB1.
3. **The applied workstream's real basins** under the IMM likelihood — does 0.003
   in-sample become a decision on the actual series.
4. The channel model's 6.8% (a regime-hazard or heavier-tailed chain in
   place of AR(1) — touches `FILTER-NOTES` §7's long-memory note).
5. Unassigned symptoms: the applied `FILTER-NOTES` §8 (root nulls), its §22
   (`dynamics` as a spread detector).

## Layout

- `exploration/` — numbered, later is more recent.
  [`0001`](0001_the_frame_and_the_ledger.md) frame and symptom
  ledger; [`0002`](0002_per_node_covariances.py)–
  [`0004`](0004_the_ridge.py) probes;
  [`0005`](0005_the_hole_is_the_ridge.md) the ridge finding;
  [`0006`](0006_the_fit_on_the_imm_likelihood.py)–
  [`0008`](0008_the_kick_control.py) the fit on the IMM
  likelihood and its gates;
  [`0009`](0009_what_the_fit_does_with_the_sharper_likelihood.md)
  the decomposition and the marginalisation case. Raw numbers in
  `exploration/figures/gap000N.json`.
- Patches, when earned, land in
  [`lucid/odefilter/`](../../lucid/odefilter)
  — the filter stays where it lives; this folder holds the investigation.
