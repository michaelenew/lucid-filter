# 0005 — The hole is a ridge: GPB1 measures the mean process variance and cannot split it

Probes: [`0002`](0002_per_node_covariances.py) ·
[`0003`](0003_the_search_half.py) · [`0004`](0004_the_ridge.py) ·
numbers: `figures/gap0002.json`, `gap0003.json`, `gap0004.json`

## The finding

Switching the process-scale channel on at fixed **mean** variance means moving
along the ridge $Q(s) = Q_{\text{eff}}\,e^{-s_P^2/2}$. Walking that ridge at
the generating $\varphi_P$ on data generated with a live channel
($s_P = 0.8$, $n = 900$):

| | ridge relief (nats/pt) | argmin |
|---|---|---|
| shipped filter (GPB1) | **0.0022** | 1.2 — wrong |
| per-node covariances (IMM) | **0.0101** | **0.8 — the truth** |

**The shipped likelihood is flat to two millinats along the coordinate that
decides how the process variance is split between a constant level and a
wandering scale.** The information that separates points on the ridge is the
*accumulated history* of the covariance — a node that has believed "high $Q$"
for thirty steps should carry a wider $P$ than one that has not — and the
GPB1 collapse hands every node the same $P$ every step, erasing exactly that.
`0038` §C measured the erasure (75% at $\sigma^2=9$) and called it a tax;
this measurement shows it is the hole: put the history back (one $(m_g,P_g)$
per node, mixed by the chain's own kernel — IMM, no new parameters, no model
change) and the likelihood identifies the split.

On data with **no** channel ($s_P = 0$), IMM's ridge relief is 0.0307 against
GPB1's 0.0113 — three times the penalty for asserting structure that is not
there — with the argmin at the foot (0.2 vs 0.0, a 0.0001 nats/pt difference,
below resolution). Sharper both ways.

## What this explains, symptom by symptom

- **`0039`'s self-confirming boundary.** $s_P{=}0$ is the ridge's endpoint.
  On a flat ridge, wherever the fit sits, the profile from that point endorses
  staying — that is what "self-confirming" *is*. Not a search pathology; a
  likelihood degeneracy.
- **The fitted 0% of the oracle gap** (`0033`'s 4.9× regression): an endpoint
  of the same flat ridge, reached by a fit the likelihood could not correct.
- **The inconsistency between fits.** `0029` (old fit) reported
  $\hat s_P \to 0$ on data generated with 0.8. The *current* fit on the same
  generator lands at $\hat s_P = 1.44$, $\hat Q = 0.39$
  ([`0003`](0003_the_search_half.py) B) — the opposite end of the slide, and
  $\hat Q e^{\hat s_P^2/2} = 1.11$: the mean variance is recovered, the split
  is not. Two fit pipelines, two endpoints, one flat ridge. **The reported
  "inconsistent behaviour" is the optimiser's path being the only thing that
  picks a point on it.**
- **Crypto `FILTER-NOTES` §4's basin roulette** — persistent-moderate against
  impulsive-large, 0.003 nats/pt apart in-sample, 0.042 out — is the same
  degeneracy with $\varphi_P$ as a second ridge coordinate: two splits of
  similar total variance the collapsed likelihood cannot tell apart.
  (`0003` A could **not** reproduce the indifference by generating from each
  basin and comparing fixed, well-separated hypothesis points — both filters
  separate those cleanly, GPB1 included. The indifference lives near the
  ridge, between hypotheses matched in $Q_{\text{eff}}$, which is exactly
  where fitted optima end up.)
- **`FILTER-NOTES` §1's screen correction** ($\log Q \mathrel{-}= s_P^2/2$)
  is, in this frame, "place the start *on* the ridge". On `0029`-style data
  it changes nothing (`0003` B: all four screen variants polish to the same
  endpoint — at $n=900$ the polish slides along the flat ridge unassisted).
  Its 10-nat value on BTC is real but belongs to a rougher surface than this
  one; the correction is a start-quality fix, not the hole.

## The oracle gap, revisited

On `0038` §B's exact data (same seed), forced channel at three settings:

| $s_P$ forced | GPB1 gap closed | **IMM gap closed** |
|---|---|---|
| 0.5 | 53.1% | **86.6%** |
| 0.8 | 80.0% | **89.5%** |
| 1.2 | 73.8% | **88.9%** |

IMM closes more of the gap at *every* setting, and is nearly flat across them
— under GPB1 you must guess $s_P$ well to get your 80%; under IMM a wrong
guess barely costs. Calibration at 0.8: 1.012 against GPB1's 1.036. Premium /
exposure ledger (`0002` D): GPB1 −0.0015 / +0.0823, IMM **+0.0056 / +0.0921**
— the IMM premium is real (a live grid carries mixture spread on data that
has none) and 16× under its exposure.

Validation and cost: at $s = 0$ the IMM recursion is the shipped one to
machine precision (identical to 10 decimals), and a reference implementation
runs at **1.4×** the shipped filter's wall clock at order 5 — the per-node
state is small and the recursion is dispatch-bound, same as everything else
about this filter.

## What is not established

- **The fitted endpoint under IMM.** The profiles argmin at the truth, but
  running the actual staged fit with the IMM likelihood needs a batched IMM
  (`_loglik_batch`'s shape) and has not been done. That, plus the `0032`
  do-no-harm gate (≤ +0.0004), is what stands between this finding and a
  patch to `core.py`.
- **Whether IMM's sharper likelihood un-flattens crypto's actual basin pair**
  — needs the real series, not `0003` A's synthetic.
- The IMM premium (+0.0056 on one draw) against the scoreboard's 0.0025 —
  needs seeds.
- `FILTER-NOTES` §8 (wide root nulls) and `crypto/0022` (the `dynamics`
  diagnostic tracking the spread) remain unassigned to this mechanism.
