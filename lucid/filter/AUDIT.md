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
| `_GAP_FACTOR = 1.5` | proxy | Sparrow resolution limit: dead zone measured to open at gap ≈ 0.7–0.8 nats independent of order (adaptive-grid finding 11 / 0003); 1.5·s runs inside it with margin. Defensible as a resolution rule; the information-theoretic optimum is not characterised. | AUD-1 |
| `_SPAN_S = 3.0` | proxy | ±3σ support of the class prior on the log-scale (99.7% mass); node count then follows from span/gap (budget). The support/tail-loss trade is not characterised. | AUD-1 |
| `_RIDGE = 1e-4` | budget | Fisher stabiliser. Note: absolute units (not scaled to the axis Fisher); inert against `_Ifloor` on activation-floored axes, and the per-event use is guarded by the step budget clip — but the pair (ridge, clip) is what bounds a no-information Newton step, and that interplay is asserted, not derived. | AUD-8 |
| `_PHIS`, `_SS` | measured | Broad class box; the data down-weights unsupported corners; tracking measured flat along the identification ridge (adaptive-grid findings 13–16). `_SS` top end has a reach argument (largest one-step scale change the window represents); the ×2 ratio and `_PHIS` values are underived. Ridge flatness does not clear the bar. | AUD-2 |
| `_SERIES_REACH = 4.0` | budget | Switch radius for the pre-factored `Q(a)` series vs the exact Van Loan route; conservative by the stated tail criterion (`nrm·reach ≤ 1`); wrong only toward compute. | — |
| `_HAZARD_GAP = 1.5` | proxy | Per-event Fisher of a rare-event rate in log coordinates = event count ⇒ blur width 1/√n e-folds, n = 1 for the class; Sparrow factor 1.5 on it (research/dynamics-learning 0009). Same standing as `_GAP_FACTOR`, same missing sharp statement. | AUD-1 |
| `_HAZARDS` | derived + proxy + measured | Top 1/2 **derived** (the class's persistence boundary); gap **proxy** (above); reach **measured** (state tracking flat across and below the box; report crossing log-priced — 0009). | AUD-3 |
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
| `_rung_odds` | derived + proxy | Whittle MA(1) KL gives the exact arclength metric `t = arccos(1−K)` on splits (sequence-demix 0002); resolvability at the memory the weights hold is the Sparrow rule again (proxy). The `forget`-read here prunes redundant rungs only (behavior-monotone, capped by `_LADDER_MEM`), unlike the retired hazard floor which moved the launch. | AUD-1, AUD-5 |
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
| window (`_build_window`) | derived + proxy | Node prior = the class prior itself, kernel exact AR(1) (derived); spacing and span are the Sparrow/support proxies. | AUD-1 |
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

### The dynamics channel

| anchor | grade | justification | open |
|---|---|---|---|
| augmentation `(x, g)` | derived | Exact Jacobian of `F(g)x + B(g)u`; the noise machinery runs unchanged above it, which is what the Q↔F demix requires (dynamics-learning 0002). | — |
| `q_g = σ²ρ_j`, cap σ² | derived | The rung's own second moment; bounded never frozen (dynamics-learning 0003). | — |
| class units (σ = 1) | derived + measured | The only scale-free, dimensionally sound size statement; cost of violating the comparable-columns requirement measured (0008 units_control). Existing open: per-direction class size. | dl-opens |
| walk mask on `g`'s scale axes | derived | Structural: `[H|0]` cannot see `g`; the mask makes the activation rule exact under degeneracy (multivariate-statfilter 0024). | — |
| hazard box + per-rung walkers + `_wm` dedup | derived + proxy + measured | See `_HAZARDS`; the dedup is arithmetic-free (shared rows share likelihoods exactly). | AUD-3 |
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
  proven optimal"): that open already states it for the walk grid ("the criterion that sets
  it should be earned rather than borrowed"); the audit adds the two other Sparrow sites —
  the split-ladder step (`_rung_odds`) and the hazard gap (`_HAZARD_GAP`) — so one earned
  resolution criterion should replace all three at once.
- **AUD-2** (adaptive-grid): derive the `(φ, s)` box from the class — both ends and the
  ratio; ridge flatness defends the interior, not the box.
- **AUD-3** (dynamics-learning): the hazard box reach — a breadth convention, tracking-flat
  by measurement, report-priced at 1/KL per nat; a derivation would say how much standing
  readiness the *class* (not the consumer) requires.
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

## Scoreboard

50 ledger entries; 43 inline markers in `lucid.py` (grep `AUDIT[`; API plumbing and the
`forget` escape are ledger-only — the escape's marker is its own parameter doc).  By primary
grade: **33 derived** (full bar), **6 proxy** (defensible, sharp
statement open), **2 measured** (below the bar, opens logged),
**9 budget/convention/escape** (no theoretical
claim; consequence-freedom owed in three places, AUD-8).  11 entries carry mixed grades
(a derived core with a proxy or measured edge — the box, the window, the offset channel).
Every proxy and measured element references an open; no chunk is unmarked.
