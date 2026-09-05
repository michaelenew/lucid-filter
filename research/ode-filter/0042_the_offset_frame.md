# 0042 — The offset frame: two series, one clock, and where the lag lives

[`0001`](0001_the_frame.md) §5 closed "multi-variable" by instruction. Instruction
now opens it, in its minimal form: **two observed series, one latent process,
one time offset.** Not general MIMO — the new object is a single time-valued
coordinate $\tau$, and the question is whether the filter can *detect and track
online* the lead/lag relationship between two correlated series, including
couplings to derivatives, holding $\tau$ as a distribution with the parent's
trust semantics.

This note fixes the coordinates and records what is derivable before any probe
runs. Claims that a probe must settle are marked **(to measure)** and collected
in §7.

---

## 1. The model

One latent process from the target class — locally a second-order linear ODE
with offset, forced by noise — read twice:

$$y^{(1)}_t = x(t) + v^{(1)}_t,\qquad
  y^{(2)}_t = c\,x(t-\tau) + v^{(2)}_t$$

**Sign convention: $\tau>0$ means $y^{(2)}$ lags — $y^{(1)}$ leads.** $\tau$ is
time-valued and continuous: it need not be an integer number of timesteps, and
it may drift. $c$ is a gain. The symmetric case (both series noisy readings of
one latent at different times) and the driver–follower case (series 1 *is* the
latent, observed) differ only in which measurement variance is zero.

Everything univariate is inherited: the latent is the existing state (lag
vector, or continuous state $(x,\dot x)$ plus level), the dynamics are the
existing $\alpha$/roots, the noise channels are the existing channels. **The
only new object is the second observation row, and the whole subject of this
extension is what that row is as a function of $\tau$.**

## 2. Delay is the flow, and a derivative coupling is its tangent

> **⚖️ ATTRIBUTION —** _A fractional/continuous time-delay observation row is a fractional matrix power $F^{-\tau}=V\,\mathrm{diag}(z_i^{-\tau})\,V^{-1}$ of the transition (a matrix exponential $e^{-\tau G}$), and in the lag basis it is the Grünwald–Letnikov binomial series $(1-\Delta)^\tau$; delay being the exponential of the derivative operator is the classical shift/generator relation._ Prior art: matrix-exponential discretization $F=e^{A\,dt}$ (Van Loan 1978); Grünwald–Letnikov fractional differencing (fractional calculus; Granger–Joyeux/Hosking for fractional differencing of series); Lie-group generator of the shift/delay semigroup. Status: REPRODUCTION.

For the noiseless dynamics $\dot z = Gz$ (continuous form; $G$ the generator
whose eigenvalues are the $\lambda_i$, sampled roots $z_i=e^{\lambda_i}$):

$$x(t-\tau) = e_x^\top e^{-\tau G}\, z(t),\qquad\text{equivalently}\qquad
  h(\tau)^\top = e_x^\top F^{-\tau},\quad F^{-\tau} := V\,\mathrm{diag}(z_i^{-\tau})\,V^{-1}$$

Delay never touches the data; it is an observation-row modification — a
**fractional matrix power of the transition**, defined on the spectrum. This is
literally the machinery of the fractional-derivative program in the
[repository README](../../README.md#open-directions): $\Delta^\nu$ is a real
power of the difference operator, $z^{-\tau}$ is a real power of the shift, and
in the lag basis the delay row *is* the generalised-binomial series
$z^{-\tau}=(1-\Delta)^\tau=\sum_k\binom{\tau}{k}(-\Delta)^k$ — Grünwald–Letnikov
with $-\Delta$ in place of $\Delta$. Truncated at $p$ terms it is exact on
polynomials of degree $<p$; the modal form is exact on the whole solution space
and needs no truncation. The two programs share one operator calculus.

**And the derivative question is not a second question.** Since
$x(t-\tau)=e^{-\tau D}x(t)$, delay is the exponential of the derivative
operator: $D$ is the *generator* of the delay group. A coupling
$y^{(2)}\approx c_0x+c_1\dot x$ is the first-order (tangent-space) form of
$y^{(2)}=c\,x(t-\tau)$ with $c_0=c$, $c_1=-c\tau$. The general linear coupling
is $y^{(2)}_t=b^\top z_t+v_t$ with $b\in\mathbb R^p$ free; the **pure-delay
family $\{c\,e_x^\top e^{-\tau G}\}$ is a 2-dimensional submanifold of it**
(one curve, one scale). In modal coordinates $b_i=c\,z_i^{-\tau}$: the delay
hypothesis is the statement that the per-channel gains and phases are
*consistent with a single $\tau$*. That consistency is falsifiable exactly when
there are enough channels (§5), and evidence for it is what "there is a
lead/lag relationship" means.

The joint family — $\mu$-th fractional derivative read at lag $\tau$ — is the
two-parameter operator $D^\mu e^{-\tau D}$, acting per mode as
$\lambda^\mu e^{-\lambda\tau}$. One immediate consequence: **a derivative
coupling annihilates the offset channel** ($\lambda=0\Rightarrow\lambda^\mu=0$
for $\mu>0$), the continuous form of "differencing kills the unit root".

## 3. Two channels carry $\tau$, and they fail in opposite ways

The delay information splits along the same modal decomposition as everything
else in `0024`, but with a refinement the deterministic picture misses.

**The mean channel** (deterministic propagation, $z_i^{-\tau}$ per root):

- **Oscillator pair** $z=re^{\pm i\omega}$: $z^{-\tau}=r^{-\tau}e^{\mp
  i\omega\tau}$. A lag is a **phase rotation** of the oscillator channel — this
  is the "phase" coordinate `0024` flagged as having no parent analogue, now
  with a physical meaning: **the oscillator phase is what a lead/lag
  measurement reads.** But phase identifies $\tau$ only $\bmod\ 2\pi/\omega$ —
  an aliased comb — with the amplitude factor $r^{-\tau}$ breaking the tie only
  as fast as the damping (weak when $r\approx1$). **(to measure: the comb, and
  the nats separating true peak from alias, 0043c)**
- **Real root** $\rho\in(0,1)$: pure amplitude $\rho^{-\tau}$, unaliased but
  degenerate with the gain $c$ unless another channel pins $c$.
- **Unit root**: $1^{-\tau}=1$. The *deterministic* offset channel says nothing
  about $\tau$ at all.

**The bridge channel** (stochastic structure): the mean channel is not the
whole likelihood. A *stochastic* unit root — a random-walk level — does carry
$\tau$ information, through which increments the two series share: lag
identifies itself by the overlap pattern of innovations, with no periodicity
anywhere. So the earlier temptation "the unit root is blind to $\tau$" is wrong
as stated; correctly: **the mean channel of the unit root is $\tau$-blind; its
information lives entirely in the noise-correlation structure.**

This gives a two-channel structure worth naming: the oscillator gives a
**sharp but aliased** reading of $\tau$; the rough (unit-root) component gives
a **coarse but absolute** one. Combined they should resolve $\tau$ sharply and
absolutely — a Chinese-remainder flavour. **(to measure: RW-only profile has a
single unaliased peak; oscillator+RW resolves the comb, 0043c/0044)**

## 4. The bridge: reading between the samples

> **⚖️ ATTRIBUTION —** _Reading the latent at a fractional (between-sample) time via Gaussian conditioning on the bracketing states — mean linear in the endpoints, plus a state-independent "bridge" variance $R_b(\tau)$ that vanishes at integer $\tau$ — is the Brownian-bridge / Gaussian-process interpolation construction, and the $s(1-s)$ (and higher $s^{2m-1}$) endpoint behavior is the standard bridge variance._ Prior art: Brownian bridge / Gauss–Markov interpolation (textbook); continuous–discrete state-space smoothing (Jazwinski 1970). Status: REPRODUCTION.

With process noise, $x(t-\tau)$ for fractional $\tau$ is not a deterministic
functional of the sampled states — noise entered during the fractional
interval. Exactly, by Gaussian conditioning: for $\tau$ between stored samples,
$x(t-\tau)$ given the two bracketing states is Gaussian with mean **linear in
the bracketing states** and a variance $R_b(\tau)$ independent of them. So the
delayed observation is exactly linear-Gaussian on an augmented state:

$$y^{(2)}_t = c\left[w(\tau)^\top z_{\lceil t-\tau\rceil} + v(\tau)^\top
  z_{\lfloor t-\tau\rfloor}\right] + \mathcal N\!\left(0,\ c^2R_b(\tau) +
  \sigma_2^2\right)$$

and the lag-vector formulation is *naturally suited*: the state already stores
past values; fractional delay reads them through an interpolation row plus a
**bridge variance** $R_b(\tau)$ that vanishes exactly at integer $\tau$ and
humps between. Three properties, the first two derivable now:

1. **$R_b$ is periodic in $\tau$ with the sampling period, not the process
   period.** Integer offsets are privileged only by the sampling grid: $R_b$ is
   the exact price of not having sampled at the offset times.
2. **Near an integer, $R_b\sim s^{2m-1}$** where $s$ is the fractional part and
   $m$ counts integrations from the noise entry to the reading: $m{=}1$ (noise
   directly on the level, Brownian bridge) gives the classical $s(1-s)$; $m{=}2$
   (forcing in the acceleration, reading the position) gives a cubic-flat
   floor. The smoothness currency of `0007` §7 again: **smoother forcing entry
   makes fractional reads cheaper.** (to measure: the exponents, 0043b)
3. For $\tau$ *outside* the stored window the same row becomes extrapolation
   through $F^{-\tau}$ with variance growing on the memory scale
   $1/(1-|z|)$ — no special-casing, the Kalman machinery prices it.

**Leads are lags in processing time.** For $\tau<0$, $y^{(2)}_t$ reads
$x(t+|\tau|)$, whose bracketing states do not exist yet; treating the future
noise as measurement noise would throw away exactly the forecasting value a
leading indicator has. The correct handling is to *defer* the update: apply
$y^{(2)}_t$ when its bracket first exists, i.e. at time
$t+\lceil|\tau|\rceil$, with the same bridge row. The update ledger reorders;
nothing else changes. (Construction recorded; probes below run $\tau>0$ only.)

The bridge machinery generalises verbatim to any linear functional: reading
$D^\mu x$ at offset $\tau$ conditions the same way with $e_x^\top G^\mu$ in
place of $e_x^\top$.

## 5. Identifiability: when is "it is a delay" testable, and when is
## "delay, not derivative" answerable?

Counting, per the modal picture. A free coupling $b$ has $p$ real numbers: one
per real root, two (gain, phase) per complex pair. The delay family $(c,\tau)$
has two. The delayed-derivative family $(c,\mu,\tau)$ has three.

- **One complex pair, family $(c,\tau)$: just-identified.** Two equations
  (gain, phase), two unknowns. No degrees of freedom left over, so *delay-ness
  is untestable* — any (gain, phase) is some $(c,\tau)$, up to aliasing.
- **One complex pair, family $(c,\mu,\tau)$: under-determined.** Two equations,
  three unknowns: a one-parameter ridge of indistinguishable hypotheses. With
  $c$ free the gain constraint is absorbed and the ridge is the phase line
  $\mu\arg\lambda-\omega_d\tau=\text{const}$, slope

  $$\frac{d\tau}{d\mu}=\frac{\arg\lambda}{\omega_d}\ \xrightarrow{\ \zeta\to0\ }\ \frac{\pi/2}{\omega_d}
  = \text{a quarter period.}$$

  **A derivative is a quarter-period lead** — $\dot{}\,\sin\omega t =
  \omega\sin(\omega t+\pi/2)$ promoted to an exact statement about what the
  filter can and cannot distinguish. On a single oscillator, "does $y^{(2)}$
  couple to $x$'s derivative or to its past?" is unanswerable *through the mean
  channel*; the bridge channel should bend the ridge weakly (the
  noise-correlation kink at zero lag distinguishes shifted from
  differentiated autocovariances). **(to measure: ridge, slope, and how many
  nats/point the bridge channel puts along it, 0043d)**
- **Two channels or more: over-identified.** Each extra channel adds
  constraints without unknowns; the single-$\tau$ consistency of §2 becomes
  falsifiable, aliasing combs from incommensurate frequencies intersect in a
  single peak, and $(\mu,\tau)$ ridges from two pairs cross in a point.
  **This is `0024`'s "the channels are the roots" earning its keep:** the
  number of roots is the number of equations the lead/lag hypothesis must
  satisfy.

## 6. The trusted distribution, and the multivariate extension of trust

> **⚖️ ATTRIBUTION —** _Gridding the unknown lag $\tau$, running a conditional Kalman recursion per node, and mixing by marginal likelihood with a "null" (uncoupled) member is multiple-model adaptive estimation applied to a delay nuisance; reading $\Lambda=\sum\mathrm{LLR}$ against the null as evidence of coupling is a directed-information / transfer-entropy measurement; latency $\approx(\log\frac1\varepsilon+\Lambda)/\mathrm{KL}$ is the SPRT/confirmation-ledger arithmetic._ Prior art: MMAE/IMM (Magill 1965; Blom & Bar-Shalom 1988); directed information / transfer entropy (Massey 1990; Schreiber 2000); Wald SPRT 1947. Time-delay estimation itself is classical (Knapp & Carter 1976). Status: RECOMBINATION.

The parent's architecture, applied without new inventions: **grid the nuisance,
run the conditional recursion per node, let marginal likelihood arbitrate.**
What is new is only *what kind of thing* the nuisance is: every gridded nuisance
so far ($\lambda_P,\lambda_M,\lambda_A$) was a noise scale; $\tau$ is
time-valued and enters the **observation geometry**, not a variance. The nodes
differ in what they think the second series *is reading*, not in how loud
anything is.

- **The grid**: nodes $\tau_j$ over the plausible window, each carrying the
  bridge row $h(\tau_j)$ and floor $R_b(\tau_j)$. Per-node conditional Kalman
  recursions; the posterior over nodes is **the trusted distribution of the
  offset** — updated every step, no thresholds.
- **Drift**: $\tau$ moves under a transition kernel — random-walk steps of
  scale $s_\tau$ plus a small restart mass $\varepsilon$ (the impulsive end;
  the parent's $\varphi\to0$). Both are in-model quantities, learnable by the
  same marginal likelihood in principle; the probes fix them and say so.
- **The null member**: a node where the series are uncoupled ($c=0$,
  $y^{(2)}$ matched-marginal noise) — the FLAT of this channel. "The series
  are actually related" is then a hypothesis with a likelihood, and
  $\Lambda_t=\sum\mathrm{LLR}$ against the null converts to trust by
  $\sigma(\Lambda)$ — nats mean what they meant in the parent
  ([`theory/04`](../random-walk-filter/theory/04-nats-trust-influence.md)).
- **The saturated member**: the free coupling $b$ is the upper envelope. The
  full trust ladder is nested — $\{c{=}0\}\subset\{c,\tau\}\subset
  \{c,\mu,\tau\}\subset\{b\ \text{free}\}$ — and the honest evidence *for
  lead/lag specifically* is the parent's robust form extended: the min-KL row
  against **both neighbours**, beating the null from below and not being beaten
  (minus the Occam cost of its extra freedom) by the saturated model from
  above. That is the multivariate extension of the univariate trust
  distributions: a posterior over a vector-valued nuisance grid, bracketed by a
  null and a saturated envelope, with the same $\sigma(\Lambda)$ semantics.
- **Latency is priced in advance**: relocation after a jump in $\tau$ should
  take $\approx(\log\frac1\varepsilon+\Lambda_{\text{target}})/\mathrm{KL}$
  points, the parent's confirmation-ledger arithmetic with the per-point KL
  between the old and new nodes' predictive distributions. **(to measure
  against the arithmetic, 0044)**

## 7. What the probes must settle

| # | claim | probe |
|---|---|---|
| 1 | the delay row $e^{-\tau G}$ is exact on the noiseless solution space | 0043a |
| 2 | bridge row + $R_b(\tau)$ match Monte Carlo; endpoint exponents $2m-1$ | 0043b |
| 3 | one oscillator ⇒ aliased comb; damping separates the peaks only weakly | 0043c |
| 4 | a random-walk latent identifies $\tau$ unaliased, through the bridge channel alone | 0043c |
| 5 | two channels resolve the comb to a single peak | 0043c |
| 6 | $(\mu,\tau)$ ridge on one pair, slope $\arg\lambda/\omega_d$; broken by a second channel | 0043d |
| 7 | the gridded-$\tau$ mixture tracks a drifting/jumping offset online; trust behaves; latency matches the ledger arithmetic | 0044 |

Deferred, recorded: negative $\tau$ via deferred updates (§4); learning
$(s_\tau,\varepsilon,c)$ by marginal likelihood; the saturated-member Occam
comparison in the online filter; folding the channel into `output/odefilter`.
