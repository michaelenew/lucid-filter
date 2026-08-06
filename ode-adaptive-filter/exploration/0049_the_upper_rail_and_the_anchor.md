# 0049 — The upper rail exists online, and the sliding window is the anchor

From [`0048`](0048_the_tube_grid.py); raw numbers in `figures/ode048.json`.
This builds `0047` §4's item 1 (the tube-grid route, as decided) and measures
the proposed static anchor. Two proposals from the same session are recorded
in §3 as directions, not results.

## 1. The tube grid: "is it a pure delay?" is now a tracked posterior

The free-coupling family $y^{(2)}=b^\top z+v$ grids exponentially in $p$ — but
the delay manifold organises $b$-space around itself. In the delayed frame the
manifold's tangent directions are `read` (the gain) and `read`$\cdot G$ (the
derivative — `0043`'s ridge, now a coordinate axis), so the **unit normal is
what "neither a delay nor a derivative" means to first order**. The tube nodes
$(\tau_j, c_k, \eta_l)$, with $\eta$ the normal offset, run as one mixture
with the same machinery as everything else; $P(\eta=0\mid\text{data})$ is the
trust ladder's upper rail, online. For this latent ($p=3$: level + one pair)
the tube is a *complete* reparameterisation of $b$-space; for larger $p$ it is
a genuine tube and the exponential grid is avoided by construction — the
question answered narrows from "what is $b$?" to "how far is $b$ from the
delay family?", which is the decidable and the useful question.

Measured, $n=700$, truth $\tau=1.2$, $c=0.7$:

| | A: pure delay ($\eta=0$) | B: off-manifold ($\eta=0.3$) |
|---|---|---|
| $P(\eta{=}0)$ final | **1.000** | **0.000** (mass on $\eta{=}{+}0.3$, sign right) |
| 99:1 verdict reached | **89 points** ("is a delay") | **24 points** ("is not") |
| coupling trust vs matched null | 0.632 nats/pt | **0.653 nats/pt — unharmed** |
| $\hat\tau$ | 1.2001 | **1.2001** |

Three things worth keeping:

- **The three-way verdict works online**: null / pure delay / related-but-not-
  a-delay, with no thresholds, and coupling trust is independent of
  delay-ness — run B keeps its full information rate while the delay claim
  collapses.
- **The 89-vs-24 asymmetry is the ledger again**: affirming the nested member
  accrues Occam evidence at the small KL of the nearest off-manifold node,
  refuting it accrues at the violation's own larger KL. Proving "it is exactly
  a delay" is intrinsically slower than catching an impostor, by the same
  arithmetic as the parent's confirmation ledger.
- **$\tau$ stays identified off the manifold** (1.2001 in both runs): the
  offset coordinate of a coupling survives the coupling not being a pure
  delay. Lead/lag detection does not require the delay hypothesis to hold.

Honest limits: $\eta$ was gridded at 5 fixed values, static, with the truth on
a node; the tube half-width (max $|\eta|$) is a window commitment like the
$\tau$ window; and at $p>3$ the normal space is $(p-2)$-dimensional, so "one
$\eta$ axis" becomes a choice of which violations to watch — the tube defers,
not defeats, the dimensionality of $b$, and says so.

## 2. The anchor: slide the windows, but interpolate with $\gamma$

The proposed static estimator — argmin of the covariance as the two windows
slide across one another — is measured and adopted as the closed-form start,
with one correction and one clean fact:

- **The fact**: independent measurement noises leave the *cross*-covariance
  unbiased at every lag (unlike the autocovariance, contaminated at lag 0), so
  the slide is a clean moment identity — the analogue of the parent's
  variogram and this filter's IV lags.
- **The correction**: at fractional $\tau$ the estimator is only sampled at
  integer lags, and the interpolant matters. The model says the interpolant
  *is* the autocovariance shape $\gamma(\cdot)$; the default parabola is
  wrong for an oscillatory $\gamma$. Measured over 8 seeds ($n=600$, RMSE of
  $\hat\tau$): parabola **0.075**, $\gamma$-shaped least squares **0.015**,
  full ML **0.012**. The $\gamma$-interpolated slide is 5× the parabola and
  within 25% of ML — a nearly efficient two-moment start at a fraction of the
  cost, exactly the role the variogram identity plays in the parent's `fit()`.

## 3. Recorded proposals (not probed)

Two further ideas from the session, kept precise for the next hands:

- **Lead-vs-coupling by rolling the filter backwards.** Time-reverse the
  recursion and compare forward and backward attributions. The stated caveat
  stands: it tests only correlation the ODE class can express. Note the
  in-model alternative already framed: leads are deferred updates (`0042` §4),
  so forward filtering alone can carry negative $\tau$ once the ledger is
  built; the backward roll would then be a *consistency check* rather than the
  estimator.
- **A compressed history to find $\tau$ directly from data.** The augmented
  state is a $K$-step window, so offsets beyond $K$ are invisible to the
  bridge rows; a compressed buffer (log-spaced lags or running multi-scale
  summaries of $y^{(1)}$) would let the sliding-window anchor of §2 run at
  large lags online, model-free, and hand its argmin to the $\tau$-grid as a
  restart proposal. This composes with §2's $\gamma$-interpolation and with
  the long-memory/fractional program (log-spaced lags are the natural
  quadrature for hyperbolic kernels). Unbuilt; the natural first probe is
  whether a restart-proposal channel fed by the anchor cuts the relocation
  latency for jumps that leave the tracked window.
