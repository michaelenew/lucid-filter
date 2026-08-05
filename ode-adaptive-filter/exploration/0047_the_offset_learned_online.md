# 0047 — The offset channel learned online, trust as directed information, and what is specifically missing

From [`0046`](0046_online_learned_offset.py). Raw numbers in
`figures/ode046.json`. This discharges the "fixed, not learned" caveats of
[`0045`](0045_what_the_offset_probes_settle.md) §5 for the kernel, the gain,
and the null — and turns up two structural findings, one of which names the
next construction.

## 1. What was made online, and by which standing technique

| was fixed in `0044` | now | technique |
|---|---|---|
| $s_\tau=0.02$, $\varepsilon=10^{-3}$ | 12-member hyper-grid, Bayes-mixed online | grid the nuisance, prequential likelihood arbitrates |
| $c=1$ known | 5-node log-spaced grid crossed with $\tau$ | a gain is a scale → log coordinate |
| $v_{\text{null}}=\mathrm{Var}(y_2)$ over the whole series | matched null: independent same-class latent read by $y_2$ alone, amplitude gridded, mixed online | the null is a *model*, not a moment |

$s_\tau=0$ and $\varepsilon=0$ are explicit members — "the offset does not
move" is a hypothesis with a likelihood (the FLAT analogue). The
information-theoretic price of learning the kernel online is bounded *in
advance*: a Bayes mixture trails the best member in hindsight by at most
$\log 12=2.48$ nats, ever. Realized: **1.44 nats** (moving-$\tau$ run), 2.13
(static run). The other realized costs: relocation latency 4 points against
`0044`'s hand-set 3, and the gain posterior lands **all of its mass on the
true node** $c=0.7$ (posterior mean 0.700; the truth sat exactly on a grid
node, so read "resolved to grid resolution", not "exact").

On static-$\tau$ data the hyper-posterior puts **0.998** of its mass on
$s_\tau=0$ and its argmax on exactly $(0,0)$; on the jumping/ramping run the
$\varepsilon=0$ group is annihilated ($10^{-11}$ mass). The channel chooses
rigidity when the world is rigid and restarts when it jumps, with no
thresholds.

One implementation lesson worth keeping: a per-step Gaussian kernel **rounds
to the identity when $s_\tau$ is below the node spacing** ($e^{-50}$
off-diagonals at spacing/step = 10), silently deleting the slow-diffusion
members. The faithful discretisation at every scale is the **matrix
exponential of the diffusion generator** on the grid. After that fix the
static run's verdict sharpened from 0.50 to 0.998 — the inert members had
been free-riding on the FLAT members' likelihood.

## 2. Trust is a directed-information measurement

With the matched null,

$$\Lambda_T=\sum_t\Big[\log p(y^{(2)}_t\mid \text{both histories})-\log
p(y^{(2)}_t\mid y^{(2)}\text{'s own history})\Big]$$

is a prequential estimate of the **directed information rate** from series 1
to series 2 beyond self-prediction (conservative: the coupled model's improved
$y_1$ predictions are discarded). The numbers move exactly as this reading
demands:

- Coupled run: $\Lambda$ slope **0.565 nats/point** — against **2.53** with
  `0044`'s strawman null. The strawman inflated trust 4.5× by comparing
  against a model that could not predict $y_2$ from its own history at all
  (an oscillator is predictable from itself; only the *excess* is coupling
  evidence).
- The accounting identity checks: for calibrated Gaussians the rate should be
  $\tfrac12\mathbb E\log(S_\perp/S_\parallel)$; measured **0.599** against the
  realized slope **0.565** — within 6%, the residual being calibration error.
  The trust rate is a variance ratio, read off the two filters' own predicted
  variances.
- Control run: slope **−14.8** nats/point (was −357 against the strawman —
  that number was mutual overconfidence, not evidence). Trust collapses to 0
  either way; now the *rate* is also a meaningful KL.

## 3. The finding: prequentially optimal, yet undercovering — the missing persistence axis

Coverage of the $\tau$ posterior's central-90% band, by segment: **1.00**
(pre-jump), **0.95** (20 points at the jump), **1.00** (post-jump), **0.61**
(ramp). The run is prequentially near-optimal (§1's regret) *and* it
undercovers during the ramp. Two things are true at once:

1. **Members that undercover $\tau$ lose almost nothing in predictive score.**
   Adjacent-node predictive differences are small when tracking is good, so
   the likelihood barely penalises a too-tight $\tau$ posterior — the
   workstream's standing theme (a loss cannot regularise a coordinate it
   cannot see) reappearing one level up: *prequential likelihood on $y_2$
   under-polices the trusted distribution of $\tau$ itself.* If $\tau$ is the
   deliverable, its calibration must be scored directly, the way the parent
   scores $E[e^2/S]$.
2. **The hyper family spans impulse and undirected diffusion, and a ramp is
   neither.** The best member on the ramp run is restart-only
   $(s_\tau{=}0,\varepsilon{=}0.01)$: a deterministic drift of 0.0037/step
   crosses a node every 27 steps, while a diffusion matched to that rate would
   take ~730 — **directed drift is atypical under every random-walk kernel**,
   so the mixture tracks the ramp as a staircase of restarts (visible in
   fig34, top left) and is overconfident between steps. What is missing is
   exactly the parent's **persistence coordinate**: the $\tau$ channel has an
   impulsive end ($\varepsilon$) and a diffusive middle ($s_\tau$) but no
   persistent end — for a time-valued nuisance that is a **velocity member**,
   a $(\tau,\dot\tau)$ grid, the AR(1)-toward-drift structure every other
   channel in this filter already has. It was not built here; it is named as
   the next construction rather than improvised.

## 4. The specific missing techniques (requested report)

In order of how much they block "everything online, everything
information-theoretic":

1. **The saturated rung online — free coupling $b\in\mathbb R^p$ with the
   state latent.** The observation $y^{(2)}=b^\top z_t+v$ is *bilinear* in the
   two unknowns: conditional on $b$ it is exactly Kalman; jointly it is dual
   estimation. Gridding — the workstream's universal move — is exponential in
   $p$ here, and this is the first nuisance in either workstream where that
   happens. Candidate routes, none validated: (i) exploit that the delay
   manifold is a 2-surface in $b$-space and grid only a tube around it (turns
   the Occam bracket into a *local* test: "is the coupling within measurement
   distance of a pure delay?" — probably the honest question anyway); (ii)
   Rao-Blackwellise $b$ with a Gaussian prior conditional on the *smoothed*
   state at a lag, accepting a plug-in; (iii) the workstream's own IV/EIV
   closed forms as a consistent anchor plus likelihood polish, as in `fit()`.
   This needs a decision before the trust ladder's upper rail exists online.
2. **The persistence axis of the $\tau$ kernel** (§3): a $(\tau,\dot\tau)$
   grid with $\dot\tau$ mixed by the same hyper machinery. Doubles the grid
   dimension; the construction is standard given the appetite for compute.
3. **Joint online $(\alpha,\tau)$.** Everything here holds the dynamics $G$
   known. The univariate filter tracks $\alpha$ through the $g$-channel;
   crossing that channel with the $\tau$ grid multiplies the two grids, and —
   more importantly — $\tau$ is read *through* the modes, so a wrong $\alpha$
   biases $\hat\tau$ in an unmeasured way (the $(\mu,\tau)$ ridge of `0043`
   is the warning: mode-parameter errors and offset errors trade off along
   phase lines).
4. **Scoring the trusted distribution itself** (§3): a calibration term for
   the $\tau$ band alongside prequential $y$-likelihood, so that overconfident
   kernels are penalised in the coordinate that matters. Needs a proper-score
   formulation for a distribution over a nuisance (the parent's $E[e^2/S]$ is
   the template; the CRPS on $\tau$ against later-revealed truth is not
   available online — truth is never revealed — so this must be a
   self-consistency score, e.g. rolling PIT of the realized $y_2$ under each
   node band).
5. **Negative $\tau$ online** — the deferred-update ledger (`0042` §4) with
   per-node deferral interacting with IMM mixing order. Bookkeeping, but it
   must be built to say "which series leads" rather than assuming it.

## 5. State of the extension after 0042–0047

Framed (`0042`), measured offline (`0043`), tracked online with hand-set
knobs (`0044`), knobs learned online with bounded regret and trust upgraded to
a directed-information measurement (`0046`). The remaining constructions are
listed above in priority order; none is blocked on theory except 1 and 4,
which need a decision and a score respectively.
