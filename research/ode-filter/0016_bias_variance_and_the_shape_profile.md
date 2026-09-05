# 0016 — The bias/unbiased split, and whether the drift shape is estimable

Two things: a reframing of [`0014`](0014_the_channel_and_the_withdrawal.md) §1
that came from the user and is sharper than what was there, and
[`0015`](0015_is_the_drift_shape_estimable.py), the probe
[`0014`](0014_the_channel_and_the_withdrawal.md) §4 said had to be run.

---

## 1. Fitting the dynamics is bias removal; the process noise is the residue

> **⚖️ ATTRIBUTION —** _Decomposing the predictive log density $-\log p=\tfrac12(e^2/S+\log S)+$const into a calibration term and a sharpness/error term, and accumulating it prequentially (each point scored before it is seen), is the standard proper-scoring-rule / prequential-MDL framework._ Prior art: prequential principle (Dawid 1984); MDL / stochastic complexity (Rissanen 1978, 1984); logarithmic score decomposition (Dawid; Gneiting & Raftery 2007). That squared error cannot see a variance-side parameter is a basic property of the score. Status: REPRODUCTION.

The reframing, in one line: **fitting $\alpha$ accounts for the *biased* portion
of the process variance, and $Q$ is the *unbiased* portion.**

Unpack it. Under any model, the innovation $e_t = y_t - \hat y_{t|t-1}$ has a
variance that splits into a part that was *predictable from the past* and a part
that was not. A filter that models no dynamics leaves the whole ODE-explainable
component in $e_t$, so $\mathbb E[e_t \mid \mathcal F_{t-1}] \ne 0$ — that
component is a **bias**, sitting in the variance budget only because the model
is too poor to have moved it out. Fitting $\alpha$ is precisely the operator
that moves mass from the predictable side to the unpredictable side, and $Q$ is
what remains when nothing predictable is left.

This reads straight onto the persistence dial of
[`0014`](0014_the_channel_and_the_withdrawal.md) §1, and explains it better than
that section did:

| | $\delta_t$ is | contributes to | visible in |
|---|---|---|---|
| $\varphi_A\to1$ | predictable | the **mean** — a bias to be removed | forecast-mean error |
| $\varphi_A\to0$ | unpredictable | the **variance** — irreducible | calibration |

So $\varphi_A$ is not merely "impulse versus regime" by analogy with the parent.
It is *the dial that splits the dynamics deviation into its biased and unbiased
portions*, which is what the parent's anomaly/regime split was all along.

### The loss this implies, and why it costs no parameters

[`0014`](0014_the_channel_and_the_withdrawal.md) §1 concluded that $\varphi_A$
needs a calibration loss because point forecasts provably cannot see it. The
worry is whether "calibration loss" smuggles in a free parameter. It does not,
and the reason is that the two halves of the split are already weighed against
each other by the predictive log-likelihood:

$$-\log p(y_t\mid\mathcal F_{t-1}) \;=\; \tfrac12\Big(\underbrace{\frac{e_t^2}{S_t}}_{\text{bias, in units of the model's own scale}} \;+\; \underbrace{\log S_t}_{\text{calibration}}\Big) \;+\; \tfrac12\log 2\pi$$

Squared error scores only the first term and drops the denominator; that is
exactly why it cannot see a parameter that lives in $S_t$. Log-loss scores both,
**in the ratio the model itself dictates** — there is no weight to choose
between them. And it is already what `fit()` maximises, so nothing new is
introduced.

The one remaining protocol choice is scoring out of sample, and even that is
removable: accumulate the **one-step-ahead predictive log score, each point
scored before it is seen** (the prequential / MDL construction). There is no
train-test split point, no held-out fraction, no window. That is the loss this
workstream should standardise on.

**One caveat, so this is not over-read.** `optimality-proof` has an open
leak (its "leak 2") where log-loss and squared error *disagree about the
geometry of the class* — max-entropy is least favourable under one and not the
other. Nothing here touches that. The argument above is about which loss can
*see* a parameter, not about which loss defines optimality. Choosing log-loss as
the scoring rule for this workstream does not resolve that seam and should not
be cited as if it did.

## 2. The drift shape is estimable — its magnitude, not its orientation

> **⚖️ ATTRIBUTION —** _Profiling the marginal likelihood over an anisotropy/orientation parameterization of a drift covariance to ask whether the shape carries readable evidence is standard variance-component / hyperparameter estimability analysis._ Prior art: marginal-likelihood (REML-style) estimation of variance components; Fisher-metric information geometry as above. The specific millinats-per-point readings are measurements on this rig. Status: RECOMBINATION (measurement).

Parameterise the drift covariance so scale and shape are separate coordinates,

$$\Sigma(\nu,\tau,\psi) = \nu^2\,R(\psi)\,\mathrm{diag}(\tau, 1/\tau)\,R(\psi)^{\top},
\qquad \det = \nu^4 \ \text{for every}\ (\tau,\psi)$$

and profile the marginal likelihood over $(\tau,\psi)$ with $\nu$ maximised out.
$\tau=1$ is isotropic. Two generating conditions, matched in determinant so they
differ **only** in shape: anisotropic ($\tau=4$, axis SDs 0.10 and 0.025,
$\psi=0.9$) and an isotropic control ($\tau=1$, both SDs 0.05). $p=2$,
$n=1200$, 6 seeds, 4356-node grid (fig11).

| generated | argmax $(\hat\tau,\hat\psi)$ | evidence over isotropic | $\psi$-profile span | $h{=}5$ MSE vs isotropic |
|---|---|---|---|---|
| **anisotropic** ($\tau{=}4$, $\psi{=}0.9$) | $(8,\ 1.571)$ | **6.69** millinats/pt | 12.2 mnats/pt | 0.9939 |
| **isotropic control** ($\tau{=}1$) | $(2,\ 2.618)$ | **0.38** millinats/pt | 3.1 mnats/pt | 1.0000 |

**The anisotropy magnitude is estimable.** 6.69 against a null floor of 0.38 —
a $17.6\times$ separation, so the evidence is not an artifact of maximising over
24 shape members. And 6.69 millinats/point is about **four times the parent's
$s_P$**, which it measured at 0.0017 nats/point and told callers not to read.
The user's expectation is confirmed: this is a readable parameter, unlike the
parent's weakest one.

**Three things it does not establish, and they matter.**

> **Point 1 is WITHDRAWN — see [`0020`](0020_orientation_is_readable.md) §2.**
> Profiling $\psi$ at $\tau=4$ rather than at the argmax $\hat\tau=8$ recovers
> the generating orientation, including at this very $\psi_{\rm true}=0.9$. The
> miss below is an artifact of profiling at $\tau=8$. Points 2 and 3 stand.

1. **~~The orientation is not recovered.~~** The profile peaks at $\psi=\pi/2$
   against a generating $\psi=0.9$ — 0.67 rad away, and it prefers $\pi/2$ over
   the grid node nearest the truth by 3.5 mnats/pt. The $\psi$ profile does
   carry $4\times$ the control's spread, so there *is* orientation information;
   it is not peaked where the drift actually happens. Two candidate
   explanations, **neither tested**:
   - *The visible axis is not the generating axis.* The likelihood sees drift
     only through how it changes the predictive distribution, and that mapping
     is the information metric $\tilde\Gamma$. At this base point
     $\tilde\Gamma = \begin{psmallmatrix}3.93&3.11\\3.11&3.93\end{psmallmatrix}$,
     with eigenvectors $(1,1)/\sqrt2$ (eigenvalue 7.04) and $(1,-1)/\sqrt2$
     (0.82) — an $8.6:1$ information anisotropy that would push the
     likelihood-preferred axis away from the generating one. If this is the
     explanation it is a nice one, because it says the Fisher metric governs
     *what can be seen* even though [`0014`](0014_the_channel_and_the_withdrawal.md)
     showed it does not govern *what moves*.
   - *Grid staircasing.* At $\hat\tau=8$ and $\hat\nu=0.025$ the minor axis SD
     is 0.0088, well below the grid step of 0.03, so the kernel is effectively a
     line — and a line at $\pi/2$ lies along a grid axis while one at 0.9 does
     not. The control's small $\psi$ spread argues against this being the whole
     story, but it does not rule it out.
2. **$\hat\tau=8$ is the top of the tested range**, so the concentration is
   unbounded above by this probe, and the true value is 4.
3. **The operational payoff is 0.6%.** Estimable is not the same as worth
   estimating. In this regime, learning the shape buys $0.9939$ at $h=5$ against
   simply using an isotropic kernel — real, one-sided, and small.

~~The honest position: the magnitude of the drift anisotropy is a readable
parameter and should be learned; the orientation is not yet demonstrated
readable, and until it is, learning $\tau$ with $\psi$ held isotropic-by-default
is the defensible middle.~~

**Superseded by [`0020`](0020_orientation_is_readable.md).** Both $\tau$ and
$\psi$ are readable; what is *not* established for either is that learning them
is worth anything operationally (forecast-MSE ratios 0.994–1.003).

## Next, in order

1. **Resolve the orientation.** Repeat `0015` at a base point where
   $\tilde\Gamma$ is nearly isotropic — if $\hat\psi$ then lands on the
   generating $\psi$, the metric explanation is right and orientation is
   readable after a known correction. If it still does not, test the grid
   explanation by halving the step. This is the probe that decides whether the
   shape is one learned number or three.
2. **Standardise the score.** Move every comparison in this workstream to
   prequential one-step predictive log-loss, per §1, and re-check the
   $\varphi_A$ result of [`0014`](0014_the_channel_and_the_withdrawal.md) §1
   under it — that claim currently rests on 3–5 in-sample nats.
3. **$p=3$: the full target class.** Architecture extends directly; the grid is
   the compute budget.
4. **Learn $Q$ and $\sigma^2$ jointly with $\alpha$.** Every probe so far holds
   them at truth.
5. **Order selection**, the way the offset root became testable in
   [`0011`](0011_the_drift_shape.md) §2.
