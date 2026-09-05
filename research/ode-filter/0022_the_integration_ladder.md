# 0022 — The integration ladder: the smooth square becomes a prism

The parent's square was (process / measurement) × (impulse / regime). Its
process-anomaly corner never sat comfortably: "the level jumped" and "a large
process-noise draw" are the *same event* there, so one corner of the square was
a description rather than a distinct mode.

**That is now explained, and the explanation generalises the square.**

---

## 1. Why the parent's odd corner was odd

Write the disturbance with a direction:

$$z_t = F z_{t-1} + u\,w_t$$

With one state dimension there is only one $u$, so "the level jumped" and "a
large $w$" cannot differ — they are the same vector. The parent had no room for
them to separate. In transfer-function terms $\alpha$ is the **poles** and $u$
is the **zeros**, and at $p=1$ there are no zeros.

Once the state is $(x, \dot x, \ddot x)$ the direction is a real degree of
freedom, and "$x$ jumped" splits into $N$ distinct events — one per derivative.

## 2. The direction axis saturates the model rather than inflating it

> **⚖️ ATTRIBUTION —** _Identifying the AR coefficients as the transfer function's poles and the disturbance-injection direction $u$ as its zeros, with $u$ costing $p-1$ numbers to complete the $2p+1$ identifiable ARMA($p,p$) content, is standard linear-systems / transfer-function theory._ Prior art: pole–zero / state-space realization theory (Kalman); ARMA spectral factorization. Status: REPRODUCTION.

[`0001`](0001_the_frame.md) §3 counted $p+2=5$ identifiable numbers in our model
against $2p+1=7$ for a scalar-observed linear system, and called the gap "a
modelling commitment". The gap is $p-1$, and

$$u \in \mathbb R^{p}\ \text{up to scale} \;=\; p-1\ \text{numbers},\qquad
(p) + (p-1) + 1 + 1 = 2p+1$$

**The injection direction is exactly the missing content.** Freeing it does not
over-parameterise; it saturates what a scalar observation can identify. (The
dimensions match; whether the map onto ARMA($p,p$) is *onto* is not checked —
$\sigma^2\ge0$ certainly restricts the image.)

## 3. The corners, and the ladder that orders them

At $p=3$ there are four named corners — three in the state, one in the
observation:

| corner | what moved | integrations before it reaches $y$ |
|---|---|---|
| MEASURE | the observation only; the state did not | — (never enters the state) |
| POSITION | $x$, with $\dot x,\ddot x$ unchanged | 0 |
| VELOCITY | $\dot x$ — a force impulse | 1 |
| ACCEL | $\ddot x$ — a force step | 2 |

Working in backward differences $v = Dz$, $D_{ij}=(-1)^jC(i,j)$, $D$ is an
involution, so the lag displacement realising a unit move of $v_i$ is column $i$
of $D$. The direction our model currently pins, $u=e_1$ ($w$ entering $x_t$), is
$D e_1 = (1,1,1)$ — position, velocity and acceleration moving together, i.e. a
kink. It is carried below as FORCING.

**The parent is the $p=1$ face**: corners MEASURE and POSITION, and nothing
else. Exactly its two channels.

### The single statistic that orders them

This is the user's framing — *among the information attributable to process
noise, how well do successive differences correlate* — made exact. The filter is
linear, so a disturbance produces a deterministic additive signature $g$ in the
**innovation** sequence. Its lag-1 autocorrelation:

| corner | MEASURE | POSITION | VELOCITY | ACCEL | FORCING |
|---|---|---|---|---|---|
| lag-1 autocorr of $g$ | $-0.368$ | $-0.053$ | $+0.426$ | $+0.788$ | $+0.788$ |

**Monotone along the ladder**, and it reads exactly as the parent's own
differentiator ("does the next point agree with the crazy one?"):

- MEASURE $\approx -0.37$: a spike — the next innovation *disagrees*.
- POSITION $\approx 0$: a step, absorbed by the filter — the next innovation is
  uninformative.
- VELOCITY $+0.43$, ACCEL $+0.79$: a ramp, then a smooth ramp — the next
  innovation *agrees*, more strongly the more integrations it has passed
  through.

So the parent's binary channel axis becomes a continuum, and the coordinate
along it is a correlation. **ACCEL and FORCING share the statistic to three
figures**, which is the first sign that they are not distinct corners.

## 4. The confusion ledger, measured

> **⚖️ ATTRIBUTION —** _Scoring how many post-event samples are needed to attribute a disturbance to one direction vs another, via the KL/likelihood-ratio between their innovation signatures, is failure-detection-and-isolation via filter innovations (GLR / multiple-model)._ Prior art: Willsky & Jones 1976 (GLR failure detection); Willsky 1976 survey; the POSITION≈MEASURE degeneracy is the repo's own optimality-proof Proposition 1 (jump-vs-glitch). Status: REPRODUCTION.

Exact linear algebra on the model's own covariance — no simulation, no fitting
([`0021`](0021_injection_directions.py), fig15). Unit root plus a damped
oscillator ($\rho=0.9489$, $\theta=0.346$, period 18.2 steps). Each event scaled
so its **detection** evidence is 8 nats — what a 4-SD single-point event earns
in the parent's ledger. Then $\text{attribution} = 16(1-\cos t)$ exactly, and
the table is the post-event points needed for 99:1 attribution:

| pair | $\kappa=0.25$ | $\kappa=1.0$ | attribution/detection at $m{=}24$, $\kappa{=}0.25$ |
|---|---|---|---|
| POSITION vs VELOCITY | 2 | 2 | 2.883 |
| POSITION vs ACCEL | 2 | 2 | 2.956 |
| POSITION vs FORCING | 2 | 2 | 2.169 |
| **POSITION vs MEASURE** | **never ≤24** | 2 | **0.549** |
| VELOCITY vs ACCEL | **4** | **5** | 1.396 |
| VELOCITY vs FORCING | 4 | 6 | 0.826 |
| VELOCITY vs MEASURE | 2 | 2 | 3.451 |
| **ACCEL vs FORCING** | **never ≤24** | **never ≤24** | **0.424** |
| ACCEL vs MEASURE | 2 | 2 | 2.542 |
| FORCING vs MEASURE | 2 | 2 | 2.458 |

(The ratio is bounded by 4, at antipodal signatures.) Three things.

**ACCEL ≡ FORCING.** They never separate at either noise level, and they share
the autocorrelation statistic. So the corner set is not five but **four**, which
is $p+1$ — and the direction our model already pins is, observationally, the
pure top-derivative corner. That is a mild but real result: calling $e_1$ "the
forcing-noise direction" is justified rather than merely conventional.

**POSITION ≈ MEASURE is the hard pair**, at 0.549 out of 4 — and at low
measurement noise it never reaches 99:1 within 24 points. This is
`optimality-proof`'s Proposition 1 reappearing in the new geometry: "the
level jumped" against "the sensor glitched" was the degeneracy that forced the
whole class definition, and it is still the degeneracy here. Worth flagging that
it is tied to the unit root — a position jump on an *integrated* process is
permanent and gets absorbed; on a stationary one it would decay and separate.
Note also the inversion: it gets *easier* at higher measurement noise (2 points
at $\kappa=1$), because a lower gain makes the filter absorb the level shift
more slowly, so the signature persists.

**VELOCITY vs ACCEL is the genuinely new distinction**, and it is affordable: 4
post-event points at $\kappa=0.25$, 5 at $\kappa=1.0$. A force impulse against a
force step — physically the two most common disturbances to a spring-mass
system — is separable in a handful of samples. **This is the content the parent
could not express at all.**

## 5. The shape of the extended object

The parent: 2 channels × persistence $\in[0,1]$ — a square.

Here: the direction $u$ lives in $\mathbb{RP}^{p-1}$ (two-dimensional at $p=3$)
for the state, plus the measurement channel, crossed with persistence. So the
square becomes a **prism**: a $(p-1)$-dimensional direction space × $[0,1]$.

The ladder of §3 is not the whole space — it is the image of a *scalar summary*
(the lag-1 autocorrelation) of a $(p-1)$-dimensional direction. That is the
right way to think about it, and it mirrors the parent exactly: one readable
scalar per channel, with whatever is orthogonal to it harder to read. Directions
sharing the autocorrelation are confusable to first order, which is precisely
what ACCEL/FORCING demonstrates.

**Not tested here: the persistence axis.** Every disturbance in `0021` is a
one-off impulse. The prism needs persistence crossed in — a disturbance that
recurs, versus one that does not — and that is the obvious next probe, because
it is the axis the parent already understands and the one that turns four
corners into eight.

## Next, in order

1. **Cross in persistence.** Repeat `0021` with disturbances that recur with
   persistence $\varphi$ rather than firing once. Gives the full prism and the
   eight-corner ledger, and it is the same exact linear algebra — cheap.
2. **Free $u$ in the filter and check the count empirically.** §2 argues $u$
   costs $p-1$ and saturates ARMA($p,p$). Fitting $(\alpha, u, Q, \sigma^2)$ and
   confirming the likelihood has $2p+1$ identifiable directions (and no more)
   would turn the count into a measurement.
3. **Does a free $u$ pay?** The whole ladder is currently a statement about what
   is *distinguishable*, not about what tracking gains. Same caution as the
   drift shape in [`0020`](0020_orientation_is_readable.md): estimable is not
   worth estimating.
4. Carry-overs, unchanged: the constant-Fisher-length direction sweep,
   prequential log-loss as the standard score, $p=3$ end to end, and joint
   $(Q,\sigma^2)$.
