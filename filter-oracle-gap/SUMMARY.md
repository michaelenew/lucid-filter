# Current state

Closing the gap between the ODE filter and an oracle told the noise schedule
exactly — and finding the hole behind symptoms reported from three
workstreams: `0039`'s 80%-forced / 0%-as-fitted process-scale channel, the
self-confirming $s_P=0$ boundary, crypto's basin roulette, and fits that land
at different $(\hat Q, \hat s_P)$ depending on which optimiser walked there.

**The hole is found and measured**
([`0005`](exploration/0005_the_hole_is_the_ridge.md)): the GPB1 collapse —
one shared covariance for every grid node — makes the likelihood **flat along
the ridge $Q(s)=Q_{\text{eff}}e^{-s_P^2/2}$**: it measures the *mean* process
variance and cannot split it between a constant level and a wandering scale.
Ridge relief on live-channel data: **0.0022 nats/pt** (shipped) against
**0.0101** with per-node covariances (IMM), whose argmin lands on the
generating $s_P$ exactly. The information that splits the ridge is the
accumulated covariance history, which collapsing to one $P$ erases every
step — `0038` §C measured the erasure and called it a tax; the ridge shows it
is the identification hole itself.

One mechanism then accounts for: the self-confirming boundary ($s_P=0$ is the
ridge's endpoint), the fitted 0% of oracle when the ×8 regime came, the
old-fit/new-fit inconsistency ($\hat s_P \to 0$ in `0029`, $\hat s_P = 1.44$
with $\hat Q e^{\hat s^2/2} \approx$ the true mean variance now — two
optimiser paths, two endpoints, one flat ridge), and crypto `FILTER-NOTES`
§4's 0.003-in / 0.042-out basin indifference (the ridge with $\varphi_P$ as a
second coordinate). The §1 screen correction is real but is a start-quality
fix on a rougher surface, not the hole (`0003` B: it moves nothing here).

**The repair measured so far** (reference IMM, no new parameters, no model
change; `0002`): oracle gap closed at forced $s_P$ 0.5/0.8/1.2:
**86.6 / 89.5 / 88.9%** against GPB1's 53.1 / 80.0 / 73.8% — better
everywhere and nearly flat across settings, so a wrong $s_P$ barely costs.
Calibration 1.012 vs 1.036. Wrong-way sharpness trebles (asserting a channel
that isn't there costs 3× more, so it is easier to reject). At $s=0$ the
recursion is the shipped one to machine precision; reference cost **1.4×**.

## The scoreboard (inherited from `ode-adaptive-filter/0039`)

Any repair must: beat 0.0025 nats/pt premium against 0.0872 exposure
(measured for IMM so far: +0.0056 / +0.0921 on one draw — needs seeds);
leave `0032`'s window no worse than +0.0004 (not yet run); close more of the
oracle gap than the forced channel's 80% (**done: 89.5%**); break the
self-confirmation — the $s_P$ profile through the repaired filter must argmin
at the generating value (**done, both datasets**).

## Next, in order

1. **A batched IMM** with `_loglik_batch`'s shape, so the actual staged fit
   can run on the IMM likelihood — the fitted-endpoint test (`0003` B's
   $\hat s_P=1.44$ should come back to ~0.8), then the `0032` do-no-harm
   gate, then the premium over seeds. That is the path to patching
   `odefilter/core.py`.
2. **Crypto's real basins** under the IMM likelihood — does 0.003-in-sample
   become an in-sample decision on the actual BTC series.
3. The unassigned symptoms: `FILTER-NOTES` §8 (root nulls — likely a
   different hole), `crypto/0022` (`dynamics` as a spread detector).
4. The screen fixes (§1 correction, wider `_S_SPLITS`, `0b`'s `_iv_alpha`
   precondition) — worth shipping as fit hygiene even though they are not
   the hole.

## Layout

- `exploration/` — numbered, later is more recent.
  [`0001`](exploration/0001_the_frame_and_the_ledger.md) is the frame and the
  symptom ledger; [`0002`](exploration/0002_per_node_covariances.py)–
  [`0004`](exploration/0004_the_ridge.py) the probes;
  [`0005`](exploration/0005_the_hole_is_the_ridge.md) the finding.
  Raw numbers in `exploration/figures/gap000N.json`.
- Patches, when earned, land in
  [`ode-adaptive-filter/output/odefilter/`](../ode-adaptive-filter/output/odefilter/)
  — the filter stays where it lives; this folder holds the investigation.
