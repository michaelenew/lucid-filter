# 0053 — The offset is a symmetry center: dynamics errors cannot move it

From [`0052`](0052_dynamics_error_and_the_offset.py); raw numbers in
`figures/ode052.json`. This probe went looking for the joint-$(\alpha,\tau)$
coupling that `0047` §4 item 3 flagged as a risk — and refuted it, in a way
that replaces the phase picture with something simpler that unifies three
earlier findings.

> **⚖️ ATTRIBUTION —** _The delay estimate is the symmetry center of the cross-covariance $c\,\gamma(s-\tau)$, and since every stationary autocovariance is even about its center, $\hat\tau$ is first-order immune to arbitrary dynamics (spectral-shape) misspecification — only an odd component (a derivative coupling) can move it. This is the delay operator commuting with the stationary flow, and it is why cross-correlation TDE is known to be robust to the signal's spectral shape. Prior art: evenness of stationary autocovariances (textbook); robustness of cross-correlation time-delay estimation to spectral shape (Knapp & Carter 1976). The "symmetry-center" unification is a clean local exposition. Status: RECOMBINATION (arguably a small PLAUSIBLY-NOVEL framing)._

## 1. The prediction, and its refutation

The phase argument said: the delay is read through each mode as the phase
$\omega\tau$, so the likelihood constrains only the product — a frequency
error should trade against the offset along $d\tau/d\omega=-\tau/\omega$
($-0.6$ at the truth), and a plug-in $\omega$ wrong by 2% should bias
$\hat\tau$ by $-0.024$.

Measured: the exchange slope is **zero to machine precision**
($-1.2\times10^{-15}$), the plug-in bias is **exactly zero in all 8 seeds**
(the $\hat\tau$ grid argmax does not move one 0.005-step), the marginalised
posterior is unbiased within noise ($-0.010\pm0.016$ across seeds) with **no
width cost at all** (0.01117 vs 0.01117), and a low-SNR sweep at
$\sigma\in\{1.5,3\}$ shows the exchange does not reappear when the path is
poorly tracked — slopes scatter with growing variance (means 0.02, 0.06;
individual seeds $-0.84$ to $+1.18$) and no systematic component.

## 2. The correct picture

Under a pure delay the cross-covariance is
$\mathrm{Cov}(y^{(2)}_t,\,y^{(1)}_{t+s})=c\,\gamma(s-\tau)$, and **every
stationary autocovariance is even about its center**. Every hypothesis in the
(dynamics, offset) family — whatever its $\gamma_\theta$ — also predicts an
even function centered at its own $\hat\tau$. Matching even-about-$\hat\tau$
to even-about-$\tau^*$ makes $\tau^*$ a stationary point of the fit by
symmetry, for *any* shape mismatch: **the offset estimate is first-order
immune to arbitrary dynamics misspecification.** A dynamics error changes
$\gamma$'s shape; it cannot change its evenness, because the delay operator
commutes with the flow. The residual scatter at low SNR is the second-order
term — asymmetric finite-sample noise — and it is unbiased, exactly as
measured.

The phase argument's error was reading one Fourier mode's phase as if it were
the covariance function. Phase alone *is* exchangeable; the even envelope
around the center is not.

## 3. What the symmetry-center picture unifies

- **The aliasing comb** (`0043`c): an oscillatory $\gamma$ is *approximately*
  even about every extremum — centers spaced by the period — with the
  approximation broken by the envelope. The comb is the set of approximate
  symmetry centers, and its separation rate is the envelope (damping) rate:
  the measured linear-in-$\gamma$ law is the envelope asymmetry accumulating.
- **The $(\mu,\tau)$ ridge** (`0043`d): a derivative is the one coupling that
  adds an **odd** component to the cross-covariance, which is precisely what
  *can* move an apparent symmetry center. The ridge is real for $\mu$ and
  absent for $\omega$ because odd-ness, not phase, is the exchangeable
  currency.
- **The anchor's quality** (`0048`/`0049`): the sliding cross-covariance with
  the $\gamma$-interpolant is nearly efficient because the estimand is a
  symmetry center — a shape-robust, noise-unbiased feature — not a model fit.
  Likewise `0048`'s $\hat\tau=1.2001$ off the manifold: the off-manifold
  component was along the normal direction, orthogonal to the odd (derivative)
  direction, so the center stayed put.

## 4. Consequences for the joint filter

`0047` §4 item 3 is **retired in its threatening form**: tracking $\alpha$
online does not contaminate $\hat\tau$ — the two posteriors decouple to first
order, so the joint $(\alpha,\tau)$ filter is a product of the machinery that
already exists, not a new estimation problem. The one genuine exchange to
carry into any joint design is $(\mu,\tau)$ — the odd axis — which the tube
grid already brackets ($\eta$ is normal to both the gain and the derivative
directions, and `0043`d measured the ridge's stochastic breaking at 0.12
nats/pt per pair, growing with channel count).

Caveats, stated: measured for frequency error on a single oscillator pair
plus the low-SNR sweep; damping errors and multi-mode mis-attribution
(assigning the coupling to the wrong root) were not swept — the evenness
argument covers them (any $\gamma_\theta$ is even), but "first-order immune"
has been *verified* only along $\omega$. And the argument is about the
*offset*; the gain $c$ and the coupling's mode composition remain entangled
with dynamics errors in the ordinary way.
