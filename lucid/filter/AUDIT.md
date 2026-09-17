# Derivation audit of the live filter (`lucid.py`)

Every chunk of the shipped filter, tied to its derivation, with a grade for how far the
justification actually reaches.  The inline markers in `lucid.py` are grep-able as
`AUDIT[`; each points at an entry here.  Scope: `lucid/filter/lucid.py` — the single
public filter.  (`WalkingVectorFilter` and the odefilter are research specimens with their
own records, out of scope.)

## The rubric

The bar (the house rule): **derived from theory, then defended with simulation, no free
parameters.**  Grades, calibrated on review examples:

- **`AUDIT[derived]`** — clears the full bar.  The construction follows from the model
  class by theorem, exact computation, or a class definition, and simulation defends the
  derivation rather than substituting for it.
- **`AUDIT[proxy]`** — halfway.  Theoretically defensible by proxy or analogy (a resolution
  limit, a support convention, an approximation measured against an exact reference), but
  the sharp statement is missing.  Every proxy carries an open for the sharp version.
- **`AUDIT[measured]`** — does not clear the bar.  Justified by simulation only (an
  insensitivity, a flatness, a comparison); the derivation is absent.  Every such chunk
  references an open.
- **`AUDIT[budget]`** — a compute budget or numerical guard: monotone (more is never
  worse in accuracy, only in compute), claiming no theoretical content.  Non-monotone
  "budgets" are mis-graded knobs and belong above.
- **`AUDIT[convention]`** — a truth-free choice (units, initialisation, a reporting
  threshold, a tie-break) whose consequences are shown to wash out or to touch only a
  report.  A convention with unproven consequence-freedom carries an open.
- **`AUDIT[escape]`** — the one declared engineering parameter, `forget`.  Exactly one
  instance; see its parameter doc.

Reference resolution: bare `research 00NN` in older comments resolves to
`research/multivariate-statfilter/exploration/00NN_*`; `finding N` to
`research/adaptive-grid/SUMMARY.md`; workstream-prefixed paths are literal.

## The ledger

### Constants

| anchor | grade | justification | open |
|---|---|---|---|
| `_GAP_FACTOR = 1.5` | derived + budget | **Derived:** a uniform grid at spacing `c·b` on a Gaussian of SD `b` has relative aliasing `2e^{−2π²/c²}` (Poisson summation, `research/resolution-criterion/0002`), so `c = 1.5` reproduces the blur-width Gaussian to `ε₀ = 3.1e-4` — the sharp statement AUD-1 asked for, replacing the Sparrow optical analogy (finding 11). **Budget:** finer is monotonically more accurate at node cost only, so the tolerance is a priced compute choice, not a derivable optimum. Also measured at the optimum of the filter's own loss (`resolvable-regime/0014`, t = 10.5 / 6.4 vs factor-2 neighbours). The walk additionally has an **absolute** score-sign bound, `gap ≤ 0.89` nats at `ε₀` from the exact log-scale spectrum (`0001`/`0002`), consistent with finding 11's measured 0.7–0.8-nat dead zone (its first derivation); per member the large-`s` rungs exceed it and the bank covers it. | — |
| `_SPAN_S = 3.0` | proxy | ±3σ support of the class prior on the log-scale (99.7% mass); node count then follows from span/gap (budget). The support/tail-loss trade is not characterised. **Measured (resolvable-regime 0012–0017):** under code length the span is a *plateau* on the scalar hero rig (\|t\| ≤ 1.46 across 1.5–12, `0014`) with 6.0 the jump preference (`0015`). A wider span was tried and **reverted**: it wins the scalar rig and (with per-event reach scaling) the async rig, but NaNs the 15-DOF arm. The cause is **not** the walk cap — a walk-mean bound fails *worse* the tighter it is set (`0017`) — it is the per-node `Qg` the outer nodes inject into the Riccati step, which a coupled high-D rig compounds. 3.0 holds every rig; the derived fix is a resolution criterion for the grid (AUD-1), not a bigger span. | AUD-1 |
| `_RIDGE = 1e-4` | budget | Fisher stabiliser. Note: absolute units (not scaled to the axis Fisher); inert against `_Ifloor` on activation-floored axes, and the per-event use is guarded by the step budget clip — but the pair (ridge, clip) is what bounds a no-information Newton step, and that interplay is asserted, not derived. | AUD-8 |
| `_PHIS`, `_SS` | measured | Broad class box; the data down-weights unsupported corners; tracking measured flat along the identification ridge (adaptive-grid findings 13–16). `_SS` top end has a reach argument (largest one-step scale change the window represents); the ×2 ratio and `_PHIS` values are underived. Ridge flatness does not clear the bar. | AUD-2 |
| `_SERIES_REACH = 4.0` | budget | Switch radius for the pre-factored `Q(a)` series vs the exact Van Loan route; conservative by the stated tail criterion (`nrm·reach ≤ 1`); wrong only toward compute. | — |
| `_HAZARDS` (`_hazard_rungs`) | derived + budget | **Coordinate derived:** a rung is a diffusion rate in the offset/departure walkers, so a steady gain `K(ρ)`, and its information coordinate is the Whittle arclength `t = arccos(1 − K)` — the split ladder's (`resolution-criterion/0004`; the metric on `K` is the arcsine metric on `h = K/2`, an identity). Uniform in `t` on `[0, π/3]`: top the persistence boundary (derived), bottom exactly `ρ = 0` (**complete, no reach convention**). Spacing the shared budget `c√(2/mem)` at `_LADDER_MEM`. The retired log-ρ ladder was 3.4× non-uniform in `t` and four blurs short of zero; measured equivalent on the 0009 rig. | — |
| attribution grid (`rates`, `_rate`, `_held`, `_FLOOR_SHARE`) | derived + budget | Two copies per class cell differing only on the axes at their floor. **Derived:** *which axes* — the floor of the axis's coordinate, where the per-step Fisher vanishes and the per-step score is not the statistic (`resolution-criterion/0006`): SNR < 1, base share below `1/(φ_golden+1) = 0.382` (the share at process noise equal to its reading noise, from the steady Riccati); a confounded pair is the split ladder's; axes above the floor walk at the step on both copies. *The copy's rate* — the two derived ends of a rate ladder, the step and the memory's resolution floor `1/mem` (a copy moving less than one step's worth over the bank's memory is indistinguishable from never moving; the `_rung_odds` read). *Why copies at all* — no per-step estimator can attribute a burst away from a floor axis: the per-axis walk and the joint Fisher-scoring step both attribute to the wider prior (`multivariate-statfilter/0059`); only multi-step evidence across a copy that did not attribute can, each copy with its own state (0056). **Not a budget (0060/0061):** the rate ladder's interior is monotone in code length (30847 → 31548 at 2 → 9 rungs) but not in the state, and the reason is structural — weights mixed under a switching prior without states mixed; see AUD-10. Two rungs is the point where the stale copy is hopeless enough never to tie, which is why it works and why it is a difference in kind, not a resolution. **Measured** (a check, not a source): arm SENSOR 2.70 → 1.46× oracle, other regimes unchanged, ×2; scalar and learned-dynamics rigs bit-identical (no held axis); async whole 1.16 → 1.19. | AUD-10 |
| `_attribution_mix` (switching rate) | derived | **The rate is the bank's own memory, `1/mem` — not an inferred hazard** (`multivariate-statfilter/0062`). A uniform leak `r` floors a copy's weight at `~r` (revival costs `log(1/r)` nats) and caps the accumulated log-odds at a window of `1/r` steps. The binding bound is the second: a leak faster than the memory discards evidence the memory still holds, so `1/r ≥ mem`. Revival needs only `r > 0` (weak: ~7 nats, ~10 steps at `1/mem`). Take the largest admissible rate — `r = 1/mem`, uniquely — which is the patient copy's own rate, so the grid carries one derived number twice. Kernel: the symmetric chain on the copies, the only symmetric kernel on `k`, exact chain power over a gap. **Measured** (arm SENSOR, × oracle): flat 1.24 for every rate in `[1e-4, 2e-3]`, then 1.25/1.26/1.33/1.45 at 4.6e-3/1e-2/3e-2/1e-1 — flat on the admissible side, degrading past the bound; `r = 0` costs SENSOR 2.31 and BOTH 1.45. **A rate ladder is retired here:** uniform on the Bernoulli arcsine coordinate is Jeffreys but has mean 0.18 on `[0, ½]`, and it needs ~1000 steps to escape that prior while the burst it governs arrives at step 250 — every restricted-top ladder scored exactly where its prior mean fell on the pinned curve (tops ½/0.05/0.02 → prior means 0.18/0.0167/0.0066 → 1.46/1.33/1.27). And there was nothing to infer: the copies are two settings of one estimator, not two states of nature, so the generating process holds no such rate; what the ladder converged to tracked the test rig's phase schedule. | — |
| `_RANK_TOL`, `_LOG2PI` | budget | Numerical rank tolerance; constant. | — |
| `_OFFSET_CLASSES = 5` | budget + convention | Count is a budget; ladder floor **derived** (`V/T`, equal visibility over the memory), ceiling a **convention** (one noise sd per step) — bias-channels 0005/0012. | AUD-4 |
| `_LADDER_MEM = 1000` | budget | Caps the split-ladder rung count so `forget = 1` asks for a finite grid. Rung count believed monotone (finer quadrature of the split posterior) but unverified. | AUD-5 |

### Structure functions

| anchor | grade | justification | open |
|---|---|---|---|
| `_steady_Si` | derived | Steady-state Riccati at the balanced base; standard fixed point (400 iterations: budget). | — |
| `_scale_fisher` | derived | Gaussian score identity `I_ab = ½ tr(S⁻¹ dS_a S⁻¹ dS_b)`, exact at the steady state. | — |
| `_split_groups` | derived | Proposition 1 in coordinates: `H v_k` along one sensor axis ⇒ rank-1 scale-Fisher block ⇒ the split invisible at every step (sequence-demix 0001). | — |
| `_apply_split`, `_group_read/_group_write` | derived | The Fisher null direction integrates to `dQ = −dR`: the null manifold is the level set of the total; the flow moves only along it (sequence-demix 0001). | — |
| `_split_star` | proxy | Caltrop enumeration of the pair ladder (multivariate-statfilter 0013 by analogy); the shared-arm budget rule (`arms // n_pairs`, floor 2) is a budget split; completeness over [0, π/2] holds at any resolution. | AUD-6 |
| `_subset_groups` | derived | Single-sensor proportionality is exact by construction — Proposition 1 reached through packetisation (pointwise-streaming 0002/0003). | — |
| `_rung_odds` | derived + budget | Whittle MA(1) KL gives the exact arclength metric `t = arccos(1−K)` on splits (sequence-demix 0002); the blur at the memory the weights hold is `√(2/mem)` (`I = 1`/step, confirmed `resolvable-regime/0009`), and spacing `c = 1.5` on it is the **same aliasing theorem** as the walk grid at the same `ε₀` (`resolution-criterion/0002`). The `forget`-read prunes redundant rungs only (behavior-monotone, capped by `_LADDER_MEM`). | AUD-5 |
| `_mean_basis` | derived | Gauge/quotient analysis of constant offsets; the z = 1 generalized-eigenspace rule carries a drift only where its signature grows polynomially, whole towers only (bias-channels 0002/0003/0004/0007/0015, each decision measured). Cayley–Hamilton horizon 2n+2 exact. | — |
| `_MeanChannel` | derived + measured | Friedland two-stage is exact against augmentation (pinned 1e-12, bias-channels 0003); class ladder as `_OFFSET_CLASSES`; `q = hazard × class` is the class second moment per rung (0009); feedback OFF beside the dynamics channel is a measured decision with a structural rationale (equilibrium of two explanations, dynamics-learning 0008) — the equilibrium itself is not derived. | AUD-4, AUD-7 |

### Elapsed-time machinery

| anchor | grade | justification | open |
|---|---|---|---|
| `_expm`, `_sqrtm`, `_logm` | derived | Standard algorithms (scaling-and-squaring, Denman–Beavers, inverse scaling-and-squaring) with stated validity conditions; iteration caps are budgets. | — |
| `_Propagator.at` | derived | `F` as the a = 1 sampling of `A = log F`; forcing map `Φ(a)Φ(1)⁻¹B` continuous through the nominal step by the stated one-step semantics of `B` (pointwise-streaming 0004). | — |
| `_Propagator.spectral/series/accumulate`, `_base_Q` | derived | Van Loan accumulation exact; the recovered spectral density verified against `Q0` before use; series switch guarded; fallback (linear scaling) is exactly the pre-existing behaviour and exact for the random walk (pointwise-streaming 0005, 15× measured cost of the naive scaling). | — |
| `_kernel(a)` / gap handling | derived | OU sampling: `φ^a`, `s²(1−φ^2a)` — exact for the class. | — |

### The engine (`_WalkEngine`)

| anchor | grade | justification | open |
|---|---|---|---|
| activation rule | derived | Structural observability: a mode is live iff it carries base variance and is seen by `H` (multivariate-statfilter 0024/0036; the delocalisation freeze it replaces measured as the runaway). | — |
| `_Ifloor`, `_Pmu_cap` | derived | The 0010 localisation condition (`Var(μ) ≤ L²`) applied as a bound, never a freeze (multivariate-statfilter 0010). The no-information drift's saturation point (window bound vs stationary s²) has an existing open (pointwise-streaming, "the no-information drift's saturation"). | pw-opens |
| `K* = (1−φ)/4`, `q_mu` | derived | Critical damping of the walk loop pins the gain as a pure function of φ; `q_mu` follows (adaptive-grid 0030/0031, derivation verified against the loop). Known banked residual: uniform damping vs deep-quiet capture across the observability swing (adaptive-grid finding 18 open) — characterised, not hidden. | ag-opens |
| balanced-base Fisher | derived | Structure evaluated at the split-agnostic point so no hypothesis tunes its own walk (sequence-demix 0002, worth 1.230→1.138 measured). | — |
| window (`_build_window`) | derived + proxy | Node prior = the class prior itself, kernel exact AR(1) (derived); spacing is now the aliasing theorem (`_GAP_FACTOR`, derived + budget); the **span** (reach) remains the ±3σ support proxy. | AUD-1 |
| caltrop star + GPB1 collapse | proxy | The axial star in place of the tensor grid: structural argument (`dS_k` depends only on coordinate k) plus measured match to the exact grid for state tracking at linear cost (multivariate-statfilter 0013); the axial-uniform mixture in the star likelihood is part of the same approximation. No error bound. | AUD-6 |
| `_star_QR` congruence + rank-2 | derived | Congruence of PSD is PSD at every gap; reduces exactly to the eigen form at a = 1; rank-2 node update is exact algebra. | — |
| `_dS_axis` + live process time | derived + measured | Exact per-axis score; the zero-gap live-process-time semantics is a measured decision with the stated leading-term argument (pointwise-streaming 0003). | — |
| step budget (one gap per full row; `mo/m` share) | proxy | Guard against a Newton verdict on a near-singular Fisher; one grid gap is the largest step the window can represent. The linear `mo/m` share for partial events is a stated rationale, not derived. | AUD-8 |
| held splits on partial events | derived | The event's own confounded pairs move only in total — Proposition 1 exactly (pointwise-streaming 0002/0003). | — |
| null-excursion revert at φ | derived + measured | The excursion is a log-scale displacement, so the class's own kernel is its return law; both bounds measured load-bearing (sequence-demix 0002 §3; multivariate-statfilter 0053 lesson b). | — |
| `_cap_P` | derived | Symmetric congruence scaling: preserves PSD, bounds the diagonal, keeps the gain live (dynamics-learning 0003: latched freeze 20× worse). | — |
| init (`lstsq` at origin; diffuse `P`) | convention | Transient only; the diffuse scale is set from the model's own magnitudes and forgotten at the filter's own rate under observability. Consequence-freedom asserted, not measured. | AUD-8 |

### Stacked execution

| anchor | grade | justification | open |
|---|---|---|---|
| `_bank_key`, `_EngineBank`, `_LoopBank` | derived | An equivalence, not an approximation: identical math with a leading member axis, pinned step-for-step by `test_bank_matches_the_looped_members`. | — |
| `_inv_sym` / `_logdet_sym` singular guards (`_LOGDET_SINGULAR`) | budget | Numerical safety on a finite-but-numerically-singular innovation covariance (extreme scale node, collinear sensors, cond ~1e15). Bit-identical on the non-singular path (kernel pins enforce it); a singular node's likelihood limit is 0, so the guard makes it contribute nothing rather than crash. Reachable at the shipped defaults by a wide-`ss` call on a multi-sensor fault rig (`research/resolvable-regime/0016`). | — |

### The dynamics channel

| anchor | grade | justification | open |
|---|---|---|---|
| augmentation `(x, g)` | derived | Exact Jacobian of `F(g)x + B(g)u`; the noise machinery runs unchanged above it, which is what the Q↔F demix requires (dynamics-learning 0002). | — |
| `q_g = σ²ρ_j`, cap σ² | derived | The rung's own second moment; bounded never frozen (dynamics-learning 0003). | — |
| class units (σ = 1) | derived + measured | The only scale-free, dimensionally sound size statement; cost of violating the comparable-columns requirement measured (0008 units_control). Existing open: per-direction class size. | dl-opens |
| walk mask on `g`'s scale axes | derived | Structural: `[H|0]` cannot see `g`; the mask makes the activation rule exact under degeneracy (multivariate-statfilter 0024). | — |
| hazard ladder + per-rung walkers + `_wm` dedup | derived + budget + measured | See `_HAZARDS`; the dedup is arithmetic-free (shared rows share likelihoods exactly). | AUD-3 |
| Shiryaev kernel; exact gap power | derived | Hazard mixing is Shiryaev's rule for the jump class; the a-step kernel is the exact chain power in the shared eigenframe (0009-corrected). Uniform leak over k−1 alternatives is a max-entropy convention, unmeasured for k > 2. | AUD-9 |
| fault readout, rung-local reprice edges | convention + derived + measured | The readouts are posterior marginals (derived); the ½ crossing is a declared reporting convention. The 0003 restart is rung-local: each rung's own marginal edge re-prices its own walker — the global-edge variant self-oscillated and the restart-free variant lost 0003's derived calibration (both measured, 0009 addendum); the pinned form is the J = 1 case bit for bit. Jump-hold open stands. | dl-opens |
| hazard readout | derived | Posterior mean over the rung weights. | — |
| `_mean_src` mask (offset reads walkers out) | measured | Gain/covariance read off caller-space members only; measured consequence of mixing walkers in (0008: fault 0.37 vs 0.04). Structural rationale stated; not derived. | AUD-7 |
| `_dynamics_mean` | derived | Posterior-mean report over the weight rows. | — |

### Time and API

| anchor | grade | justification | open |
|---|---|---|---|
| `_elapsed`, per-step semantics | derived + convention | Everything supplied is per nominal step; elapsed maps take each to its exact power (pointwise-streaming 0001/0004). `R` unscaled: a reading's variance belongs to the reading (derived from the event model). | — |
| `forget` | escape | The one declared engineering parameter; documented at the parameter itself; nothing structural reads it (0009-corrected). Eliminable in principle (adaptive-grid open). | ag-opens |
| `filter`/`stream`/`observe`/`update` plumbing | convention | API surface; no inference content beyond what is graded above. | — |

## Opens raised by this audit

Logged in the owning workstream SUMMARYs; listed here for the grep.

- **AUD-1** (adaptive-grid — EXTENDS the existing open "the grid is *justified*, not
  proven optimal"). **Spacing: closed.** One theorem now sets all three Sparrow sites —
  a uniform grid at `c·b` on a Gaussian of SD `b` has aliasing `2e^{−2π²/c²}`, so `c = 1.5`
  is one tolerance `ε₀ = 3.1e-4` at the walk grid and `_rung_odds`; the hazard ladder is spaced
  in its own derived coordinate (`resolution-criterion/0004`) rather than in log-hazard
  (`research/resolution-criterion/0002`); the walk's absolute score-sign bound `0.89` nats
  gives finding 11's measured dead zone its first derivation. **What remains open is the
  span** (reach, `_SPAN_S`): the ±3σ support/tail-loss trade is still a proxy. Enforcing
  the absolute bound as a walk-gap cap with reach preserved was measured
  (`resolution-criterion/0003`): it wins the scalar jump and the async rig but regresses
  the arm 6× on its sensor-burst regime at 4.5× cost, so it does not ship. Twice now
  (span-6 in `resolvable-regime/0017`, the cap here) a grid change that helps a 1-D rig
  hurts the coupled 15-DOF arm through the reach/node structure, so the concrete form of
  the open is per-axis (per-member) node count — the rectangular axial array forces one
  `K` on every axis.
- **AUD-2** (adaptive-grid): derive the `(φ, s)` box from the class — both ends and the
  ratio; ridge flatness defends the interior, not the box.
- **AUD-3** (dynamics-learning): **closed** (`resolution-criterion/0004`). The hazard ladder's
  reach was a breadth convention because it was measured in log-hazard; in the walker's own
  arclength the range is bounded, `[0, π/3]`, and the ladder is complete over it — there is no
  reach to justify. Report crossing time is still priced at 1/KL per nat, a consumer convention.
- **AUD-4** (bias-channels — EXTENDS existing open 3, "drifts above twice the ladder's
  ceiling are under-served"): that open prices the ceiling from the practical side; the
  audit adds the theoretical half — the ceiling (one noise sd per step) is a convention,
  and a derivation should say what the class itself puts at the top.
- **AUD-5** (multivariate-statfilter): split-ladder resolution — verify rung-count
  monotonicity (finer never worse), and replace the memory-pruning rule's Sparrow factor
  per AUD-1.
- **AUD-6** (multivariate-statfilter — RELATED to pointwise-streaming's open "the residual
  pointwise/joint gap", which already attributes a measured 2–10% residual to the
  caltrop-plus-GPB1 construction): an error bound for the collapse against the exact tensor
  grid — 0013 measures the match on its rigs; nothing bounds it, and the pointwise residual
  is the measured signature of the missing bound.
- **AUD-7** (bias-channels): derive the feedback/feed-forward equilibrium beside the
  dynamics channel (0008's measured lock-up) instead of switching on a measured verdict;
  same for the `_mean_src` walker mask.
- **AUD-8** (pointwise-streaming): the small numerics with asserted consequence-freedom —
  the partial-event step-budget share `mo/m`, the (ridge, clip) guard pair, the diffuse
  init — either derive, or measure the assertions.
- **AUD-9** (dynamics-learning): the anchor-leak topology — uniform leak over k−1
  alternatives is max-entropy by convention; measure sensitivity with named anchors
  (k > 2) or derive the leak from the class.
- **AUD-10** (multivariate-statfilter 0057–0059): the attribution grid's remaining opens. The
  memory copy has no fast way back after a real process change (its weight after a PROCESS burst
  is ~0; the switching prior refreshes its prior, not its state); the first burst off the floor
  of either kind is the expensive one (PROCESS-first 2.04× on the swapped schedule, untouched);
  the rate ladder's interior is **not a budget** (0060/0061): monotone in code length but not in the state, because the bank mixes weights under the switching prior without mixing states, so in a regime whose unmasked channels cannot see the component the copies disagree on the bank averages divergent states; the consistent switching filter mixes states at the same rate (the IMM structure), which makes the interior flat and fixes the masked regimes but erodes the sensor-burst gain (arm SENSOR 1.46 → 2.30) because that gain *is* the copies' divergence — an open derivation, mixing conditional on evidence, not a constant;
  the departure specs carry an identical second copy (a per-spec cell count is plumbing); a cell
  with no held axis carries an identical copy too (the scalar rig pays ×2 for nothing); and the
  async rig pays 3% whole / 3% hot with its one floor mode held. The class-box form (separate
  process- and sensor-scale classes, the memory class at `φ = 1 − 1/mem`) was built and does not
  reproduce the effect — patience is not a class property, because a floor axis's step is
  budget-clipped whatever its φ (0059).

## Scoreboard

53 ledger entries; 49 inline markers in `lucid.py` (grep `AUDIT[`; API plumbing and the
`forget` escape are ledger-only — the escape's marker is its own parameter doc).  By primary
grade: **38 derived** (full bar), **3 proxy** (defensible, sharp
statement open), **2 measured** (below the bar, opens logged),
**10 budget/convention/escape** (no theoretical
claim; consequence-freedom owed in three places, AUD-8).  11 entries carry mixed grades
(a derived core with a proxy or measured edge — the box, the window, the offset channel).
Every proxy and measured element references an open; no chunk is unmarked.
