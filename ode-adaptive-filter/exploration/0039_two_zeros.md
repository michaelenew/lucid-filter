# Two zeros, and neither one means what the summary said

`SUMMARY.md` has carried this as the top open item:

> **Fix the process-scale channel.** `0033` makes this the top item: $s_P$ fits
> to zero on smooth data, so a process-noise regime is mis-attributed to the
> measurement channel and costs **4.9×**. The diagnosis points at the fix —
> parameterise the scale on something the likelihood can see ($Q_{\text{eff}}$,
> or the innovation) rather than on $Q$ alone.

[`0038`](0038_why_the_process_channel_is_dead.py) went looking for that fix.
Every load-bearing clause above is wrong. The channel is not dead, the
conditioning story is not the mechanism, the proposed fix makes things worse,
and the two datasets that reported a zero were reporting two different things.

## The channel is not dead

The standing diagnosis is stated in the wrong currency. $\gamma_0$ is not what
the likelihood sees. What it sees is

$$S = a_{00} + Q e^{\lambda_P} + R e^{\lambda_M}$$

so the channel's leverage is $Q/S$ — which the filter already reports, as
`share_process`. Measured, on 900 points:

| data | prior | **process** | measurement |
|---|---|---|---|
| ODE, $Q{=}1$, $\sigma^2{=}9$ | 0.672 | **0.033** | 0.295 |
| WALK, $Q{=}1$, $\sigma^2{=}9$ | 0.203 | **0.080** | 0.718 |
| ODE at $\sigma^2 = 0.25$ | 0.600 | **0.320** | 0.080 |

The ODE case has 2.4× less leverage than the parent's own case. Less, not
none — and the parent's channel works on 0.080.

So: force the channel on and see. A ×8 process-noise regime over 160 of 900
points, scored against a Kalman filter **told $Q_t$ exactly** — an evidence
ceiling nothing can beat.

| filter | nll/pt | calibration | gap closed |
|---|---|---|---|
| oracle $Q_t$ | 3.1836 | 0.987 | 100% |
| static $Q = 1$ | 3.2864 | 1.317 | — |
| best constant $Q$, in hindsight | 3.2274 | 0.972 | 57.4% |
| grid, $s_P = 0$ (as fitted) | 3.2864 | 1.317 | **0.0%** |
| grid, $s_P = 0.5$ forced | 3.2318 | 1.149 | 53.1% |
| grid, $s_P = 0.8$ forced | 3.2041 | 1.036 | **80.0%** |
| grid, $s_P = 1.2$ forced | 3.2106 | 0.990 | 73.8% |

**Switched on, the channel closes 80% of the gap to an oracle and beats the
best single constant $Q$ chosen with hindsight.** Calibration goes 1.317 →
1.036. There is nothing wrong with the machinery. The entire loss is that the
fit sets $s_P$ to zero, which gets 0.0% of a gap it could get 80% of.

## What the collapse does cost — a tax, not the cause

One real mechanism did turn up, and it is worth keeping even though it is not
the answer. Under the GPB1 collapse every quadrature node shares one covariance
$P$, so two nodes hypothesising $Q$ and $8Q$ differ in $S$ by *one step* of
process noise. A node that remembered its own history would also differ by the
accumulated $a_{00}$:

$$\Delta_{\text{true}} = \underbrace{[a_{00}(8Q) - a_{00}(Q)]}_{\text{deleted every step}} + \underbrace{7Q}_{\text{kept}}$$

| $\sigma^2$ | $a_{00}(Q)$ | $a_{00}(8Q)$ | $\Delta_{\text{acc}}$ | $\Delta_{\text{step}}$ | kept |
|---|---|---|---|---|---|
| 0.25 | 1.87 | 3.10 | 1.22 | 7.00 | 85.1% |
| 1.00 | 4.81 | 9.12 | 4.32 | 7.00 | 61.9% |
| 9.00 | 20.50 | 41.59 | 21.09 | 7.00 | **24.9%** |
| 36.0 | 50.86 | 103.94 | 53.07 | 7.00 | 11.7% |

At this class's SNR the collapse throws away **75%** of the available
discrimination before the likelihood is evaluated, and it gets worse as
measurement noise grows. That is a genuine architectural tax and the honest
argument for an IMM-style per-node covariance. But the table above already
settles its status: what survives is still enough for 80%. **This is a tax, not
the cause.**

## The likelihood is not broken either

Profile the marginal likelihood in $s_P$, re-optimising $Q$ at each point —
because raising $s_P$ raises the mean process variance, so a profile that
forbids $Q$ to move is not the one the fitter sees:

| $s_P$ | ×8 regime, nll/pt | $\hat Q$ | AR(1) log-scale, nll/pt | $\hat Q$ |
|---|---|---|---|---|
| 0.0 | 3.22435 | 2.22 | 3.14547 | 1.13 |
| 0.4 | 3.21711 | 1.85 | 3.14494 | 1.03 |
| 0.8 | 3.20009 | 1.05 | 3.14358 | 0.77 |
| 1.0 | 3.19526 | 0.79 | **3.14345** | 0.64 |
| 1.4 | **3.19520** | 0.49 | 3.14483 | 0.44 |

Interior optima in both. The likelihood knows the answer.

## Two zeros

The two probes that reported $\hat s_P \to 0$ were not reporting the same thing.

**`0032`'s zero is correct.** Its filter was fitted on the baseline stretch,
$t < 620$ — and the ×8 process regime it later fails on sits at
$t \in [720, 850)$, **outside that window entirely**. Profiled on the data the
fit actually saw, the likelihood is monotone increasing in $s_P$ from zero:

| $s_P$ | 0.00 | 0.10 | 0.20 | 0.35 | 0.50 | 0.70 | 1.00 | 1.40 |
|---|---|---|---|---|---|---|---|---|
| vs $s_P{=}0$ | 0 | +0.00002 | +0.00007 | +0.00037 | +0.00125 | +0.00419 | +0.01709 | +0.08175 |

There is no process-scale variation in that window, and the filter correctly
says so. **The 4.9× regression is an out-of-sample failure, not a channel
failure** — the filter met a phenomenon it had no evidence for at fit time.
That is a different and much less damning claim than the one on record.

**`0029`'s zero is not correct.** Its data was generated *with* $s_P = 0.8$.
Profiled at the truth over a grid topping out at 1.2, the argmin lands at or
above it — never at zero:

| case | argmin $s_P$ | nll/pt at 0 | at the argmin |
|---|---|---|---|
| impulsive s31 | 1.2 | 3.28348 | 3.25560 |
| impulsive s32 | 0.8 | 3.20644 | 3.19931 |
| persistent s31 | 1.2 | 3.37906 | 3.29770 |
| persistent s32 | 1.2 | 3.31508 | 3.27249 |

and `fit_` returns 0 anyway. On the persistent seeds that is a gap of 0.08
nats/pt — 40 nats over the series — being left on the table.

## Why the fit returns zero, and what does not fix it

Profile $s_P$ **at the fitted parameters** rather than at the truth. The
boundary turns out to be **self-confirming**:

| case | $\hat s_P$, $n{=}500$ | $\hat s_P$, $n{=}900$ | argmin of the profile *at* the $n{=}900$ fit |
|---|---|---|---|
| impulsive s31 | 0.0000 | 0.0000 | 0.0 |
| impulsive s32 | 0.0000 | 0.0000 | 0.0 |
| persistent s31 | 0.0000 | 0.0000 | 0.0 |
| persistent s32 | 0.4870 | **1.5416** | **1.4** |

Wherever the fit lands on zero, the profile taken from that point says zero is
optimal; where it escapes, the profile follows it out. So the fit sits at a
genuine **local optimum**, not merely stopped short. Two candidate explanations
die together here. It is not a tolerance or step-size problem — three search
repairs change exactly nothing: an absolute one-octave initial simplex in place
of scipy's relative default, a dedicated $s_P$ scan before the joint search,
and fitting the two scale channels in sequence instead of together. And it is
not short evidence — **doubling $n$ rescues nothing in three of the four
cases.**

(On other data the fit finds $s_P$ without trouble — 1.13 on the ×8 regime
series and 1.31 on a $\varphi = 0.9$ log-scale series, both at $n = 900$, with
the dynamics stage leaving both intact. The failure is path-dependent, not a
function of sample size.)

The mechanism is a **coupling**. $\mathbb{E}[Q e^{\lambda}] = Q e^{s_P^2/2}$,
so raising $s_P$ at fixed $Q$ also raises the mean process variance. A fit that
has already put $\hat Q$ at the mean must move *both* coordinates together to
benefit — and each single-coordinate move from there is uphill. The profile
table above shows it directly: $\hat Q$ falls 2.22 → 0.49 as $s_P$ climbs.

That is exactly the coupling `SUMMARY.md`'s proposed fix was aimed at, and the
proposal is a good thought: hold $Q e^{s_P^2/2}$ fixed instead of $Q$, making
$s_P$ a pure spread coordinate. **Tested, it does not work.**

| | $s_P$ argmin, $Q$ fixed | $s_P$ argmin, $Qe^{s^2/2}$ fixed |
|---|---|---|
| impulsive, truth 0.8 | **0.8** | **0.0** |
| persistent, truth 0.8 | 1.0 | 0.8 |
| `0032` window, truth 0 | 0.0 | 0.0 |

On the impulsive case it moves the argmin the *wrong* way. On `0032`'s window
it keeps the argmin at 0 but flattens the penalty for being wrong sevenfold
(+0.011 at $s_P{=}1.4$, against +0.082). A reparameterisation that makes the
right answer harder to distinguish from the wrong one is not an improvement.

## What is actually wrong, stated so the next attempt can be scored

Strip out the mechanisms that turned out not to be the cause and one fact
survives, and it is a decision-theoretic one rather than a likelihood one.
**The loss around $s_P = 0$ is violently asymmetric.** Mean of three seeds,
900 points:

| carried $s_P$ | truth $s_P = 0$ | ×3 regime | ×8 regime |
|---|---|---|---|
| 0.0 | 3.1274 | 3.1787 | 3.3085 |
| 0.5 | +0.0002 | −0.0071 | −0.0591 |
| 0.8 | **+0.0025** | −0.0104 | **−0.0872** |
| 1.2 | +0.0082 | −0.0093 | −0.0920 |

Carrying $s_P = 0.8$ costs **+0.0025 nats/pt** when it is unnecessary and saves
**+0.0872** when it is not: an asymmetry of **35×**. And the likelihood is
nearly flat on the cheap side — on `0032`'s own fitting window, $s_P = 0.35$
costs +0.0004 nats/pt, four ten-thousandths of a nat.

So the real defect is not that the channel cannot see, nor that the search
cannot find, nor that the coordinates are coupled. It is that
**$s_P = 0$ is an attainable boundary and a plug-in maximum likelihood estimate
is willing to land on it.** $s_P = 0$ is not "a little scale variation"; it is
the claim that the process variance is constant *and always will be*, made from
a region where the likelihood is flat to four decimal places. Everywhere else
this filter integrates a nuisance out over a grid — that is what $\lambda_P$,
$\lambda_M$ and $\lambda_A$ *are*. $s_P$ is the one place a point estimate is
plugged in, and it is exactly where the plug-in is least defensible.

No fix is shipped here, because every candidate I tested either did nothing or
made it worse, and shipping one on that evidence would be worse than shipping
none. But the target is now quantified, which it was not before: **any repair
must beat 0.0025 nats/pt of insurance premium against 0.0872 nats/pt of
exposure**, and must leave `0032`'s window — where zero is the right answer —
no worse than +0.0004.

The obvious candidate, and the one consistent with the rest of the
architecture, is to stop plugging $s_P$ in at all and marginalise it the way
every other nuisance in this filter is marginalised. That is a real design
change with a real compute cost, and it needs its own probe.

## One thing did ship

`Step.pred_var` and `FilterResult.pred_var`: the variance of the one-step
predictive distribution, mixture spread included. The filter already computed
it for `whiteness` and did not report it, which meant nothing outside could
score a forecast as a distribution — the thing [`0036`](0036_three_corrections.md)
§2 argued everything should be scored on. Recovering it from `loglik` after the
fact is not possible: $e^2/S + \log S$ is not monotone in $S$, so the inversion
has two roots and picking the wrong one silently reports a filter as four times
worse than it is. It is pinned by an exact test — with both scale channels off
the grid is a single node, the predictive law is exactly Gaussian, and `loglik`
must equal its density at `innovation` to 1e-10.
