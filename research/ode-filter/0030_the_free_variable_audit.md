# 0029 — Every knob in the candidate, and which ones are free

The rule this workstream runs under is that there are **no theoretically
relevant free parameters**, with compute budgets explicitly exempt: a budget
trades a real cost against theoretical accuracy and nothing else. `0027`
delivered a filter without ever auditing it against that rule. This does the
audit, and [`0028`](0028_the_free_variable_audit.py) measures the parts of it
that are measurable rather than arguable.

## 1. Four kinds of number

Sorting every constant in `output/odefilter/core.py`:

| kind | test | verdict |
|---|---|---|
| **commitment** | changes the model | real; derive it or learn it |
| **scaffolding** | seeds a maximum-likelihood search | free *only if the answer depends on it* — measurable |
| **budget** | costs time, buys accuracy, monotonically | exempt by the stated rule |
| **guard** | keeps arithmetic in range | binds where the answer is already determined |

The middle two are where self-deception lives, because they look innocent and
are never checked. So `0028` varies each of them and refits.

**Counts: 4 commitments, 5 scaffolding, 3 budgets, 5 guards.** Three of the
four commitments bind at the default settings.

## 2. The commitments — the honest answer to "are there others?"

`p` is the known one. There are **two more**, and both are invisible in the
signature.

**`p` — the order of the recurrence.** Three is a second-order ODE plus a
constant offset, because the offset is a root at $z=1$. Since each root is a
channel (`0024`), choosing `p` is the same act as counting channels. §4 below
shows it is learnable.

**`u = e_1` — the injection direction.** Process noise enters the state through
the top slot only: `A[0,0] += Q` is $e_1e_1^\top Q$. Nothing in the model says
it has to. Unpinning it costs $p-1$ numbers and stays inside the identifiable
budget, because $p + (p-1) + 1 + 1 = 2p+1$ is exactly the identifiable content
of a scalar-observed linear system. **This is the gap `0001` §3 recorded as "a
modelling commitment" without naming it.** `0021` measured the consolation:
ACCEL and FORCING are never observationally separable, so the pin sits on a
corner the data could not have distinguished anyway.

**`α` is static.** Fitted once, never updated. The filter reports when this
has stopped being true — `whiteness`, 17.3 SE from control on a parameter
change (`0025`) — and does nothing about it.

**`scales` is a fourth entry that binds only when turned off.** `True` is the
unrestricted model; `False` pins $s_P = s_M = 0$. A nested submodel the
likelihood can choose itself.

**The parent has none of these.** Not because it is better designed, but
because a random walk has no order to choose, no injection direction (at $p=1$
there are no zeros — the reason its process-anomaly corner never sat right),
and no dynamics to go stale. **Three commitments is the price of the
extension, and it is exactly the price `0024` predicted.**

## 3. Things that look like knobs and are not

- **The offset root at $z=1$.** Derived. $\mathrm{span}\{1,e^{\lambda_1
  t},e^{\lambda_2 t}\}$ is annihilated by $(z-1)(z-z_1)(z-z_2)$. And `fit()`
  does not even pin it — it lets the root float, the weaker assumption, with
  `0011` §2 making the pin a testable hypothesis instead of a setting.
- **The lag basis.** $D$ is an involution with $|\det|=1$, so lags and
  derivatives carry identical information and the choice cannot move any answer.
- **The whiteness window.** There isn't one, deliberately: a half-life or a
  window length would be exactly the kind of free parameter this workstream
  refuses. The cost is real — a cumulative statistic dilutes a late change — and
  it is the open problem standing between the diagnostic and acting on it.
- **The loss.** Predictive log-likelihood introduces no constant, unlike AIC's
  2 or BIC's $\log n$. That is why §4 scores out of sample instead of
  penalising.

## 4. What `0028` measures

### A. The scaffolding, refit under each variant

Two seeds, $n=500$, $p=3$, `scales=False`. Truth: $|z_{\text{osc}}|=0.9489$,
$Q=1$, $\sigma^2=9$. $\Delta$ loglik is against the default; **positive means
the default found a worse optimum**, which is the signature of a hidden knob.

| variant | $\lvert z_{\text{osc}}\rvert$ | $Q$ | $\sigma^2$ | $\Delta$ loglik |
|---|---|---|---|---|
| default — `logspace(-2,1,13)` | 0.9248 | 1.030 | 8.303 | — |
| $Q$ window ×10⁻⁴…10² | 0.9276 | 0.992 | 8.325 | +0.09 |
| $Q$ window ×10⁻¹…10⁰·⁵ | 0.9276 | 0.993 | 8.323 | +0.09 |
| $Q$ grid 5 points | 0.9248 | 1.030 | 8.303 | +0.00 |
| $Q$ grid 31 points | 0.9282 | 0.993 | 8.310 | +0.09 |
| **no $Q$ scan at all** | 0.9276 | 0.992 | 8.325 | +0.09 |
| $\varphi_0=0.05$ | 0.9248 | 1.030 | 8.303 | +0.00 |
| $\varphi_0=0.95$ | 0.9248 | 1.030 | 8.303 | −0.00 |
| **IV $m=p$** | — | 409.0 | 7.59 | **diverged** |
| IV $m=4p$ | 0.9277 | 0.990 | 8.329 | +0.09 |

> **⚖️ ATTRIBUTION —** _An engineering audit of a specific implementation's constants (which scan/init/guard values move the fitted answer). The finding that an over-identified IV needs $m>p$ (just-identified IV diverges) is the standard identifiability requirement for instrumental variables._ Prior art: instrumental-variables identification order condition (standard econometrics/system-ID). These are NEGATIVE-RESULTs specific to this codebase. Status: NEGATIVE-RESULT.

**1. The $Q$ scan is inert, and should be deleted.** Widen the window by six
decades, narrow it to one and a half, resolve it at 5 or 31 points, or remove
it entirely: the fitted oscillator moves by $\le 0.003$ and the likelihood by
0.09 nats over 500 points. Maximum likelihood finds $Q$ from any start in the
range, so the stage buys nothing. Note the *sign* — every variant that moves
the start beats the default by the same 0.09, so the scan reliably picks a
slightly **worse** start than the moment estimate it was added to correct. It
costs 13 filter passes per fit to do so.

**2. The instrument count is a condition, not a dial.** At the just-identified
$m=p$ the fit **diverges** — the complex pair is lost, $\hat Q$ lands at 409
against a truth of 1. At $m=2p$ and $m=4p$ it agrees to 0.003. So there is no
interior optimum to tune: **over-identify, and the value stops mattering.**
`_iv_alpha` should assert this rather than default it.

**3. Two of these rows measure nothing, which is worth saying plainly.** With
`scales=False` the model has $s_P=s_M=0$, and $\varphi_c$ is *exactly*
unidentifiable when $s_c=0$ — the parent's own docstring says so. The
$\varphi_0$ rows are zero by construction, not by evidence. Tested properly in
§4.1.

### A2. The diffuse prior, over a millionfold range

| prior scaling | gap at $t=1$ | last step differing by >1e-6 SD |
|---|---|---|
| ×1 | 0.067 SD | $t=30$ |
| ×100 | 0.075 SD | $t=29$ |
| ×10⁴ | 0.075 SD | $t=29$ |

Against a ×0.01 reference: a $10^6$ range of initial state covariance moves the
first filtered mean by under 0.08 SD and is gone within 30 steps. **The `·p`
inflation is a transient, not a parameter.**

### B. Is $p$ learnable? Prequential log-loss

> **⚖️ ATTRIBUTION —** _Selecting the AR order by prequential predictive log-loss (fit on the first half, score the log density of the second, no explicit complexity penalty) is standard prequential/MDL order selection, of which AIC/BIC are the penalized-likelihood cousins._ Prior art: prequential model selection (Dawid 1984); MDL (Rissanen 1978, 1986); AIC (Akaike 1974), BIC (Schwarz 1978). Status: REPRODUCTION.

Fit on the first half, score the log predictive density of the second, three
seeds, $n=700$, nats/point, higher is better. **No complexity penalty** — AIC's
2 or BIC's $\log n$ would each import a free parameter into the very question
being asked.

| data | $p=1$ | $p=2$ | $p=3$ | $p=4$ | $p=5$ | verdict |
|---|---|---|---|---|---|---|
| ODE | −3.5247 ±.046 | −3.2532 ±.009 | **−3.1218** ±.014 | **−3.1213** ±.014 | **−3.1240** ±.013 | $p\ge3$ |
| WALK | **−2.6791** ±.002 | −2.6810 ±.001 | −2.6830 ±.001 | −2.6816 ±.001 | −2.6812 ±.001 | $p=1$ |

**$p$ is learnable from below, and only from below.** On ODE data the loss
climbs 0.40 nats/point from $p=1$ to $p=3$ and then goes flat — $p=3,4,5$ agree
to 0.003 against an SE of 0.014, so the argmax landing on 4 is noise, not a
preference. On a plain random walk it picks $p=1$: **the rule recovers the
parent on the parent's own data**, and over-fitting to $p=3$ costs 0.004
nats/point — the same near-free overhead `0026` measured as ±5%.

So prequential log-loss pins a **floor** on the order and is nearly blind above
it. That is enough to stop calling $p$ free — you cannot set it wrong in the
direction that hurts without the data telling you — but it is not a point
estimate, and "how many channels are there?" still has no sharp upper answer.

### 4.1 The $\varphi$ start, tested where $\varphi$ exists

[`0029`](0029_the_phi_start.py) runs the **full** path (`scales=True`) on data
generated with a genuinely varying process scale, $s_P=0.8$, at both ends of
the persistence axis. Two seeds each, $n=500$.

| data | $\varphi_0$ | $\hat\varphi_P$ | $\hat s_P$ | $\lvert z_{\text{osc}}\rvert$ | $\Delta$ loglik |
|---|---|---|---|---|---|
| impulsive, s31 | 0.50 / 0.05 / 0.95 | 0.406 / 1.000 / 0.167 | **0.000** | 0.9378 | 0.00 |
| impulsive, s32 | 0.50 / 0.05 / 0.95 | 0.501 / 1.000 / 0.000 | **0.000** | 0.9523 | 0.00 |
| persistent, s31 | 0.50 / 0.05 / 0.95 | 0.468 / 1.000 / 0.000 | **0.000** | 0.9096–0.9098 | 0.00 |
| persistent, s32 | 0.50 / 0.05 / 0.95 | 0.501 / 0.000 / 0.000 | **0.000** | 0.9560 | 0.00 |

**The $\varphi$ start is inert — but not for the reason the audit was looking
for.** Every fit drives $\hat s_P$ to zero, on data that genuinely has
$s_P=0.8$. With $s_P=0$ the persistence is exactly unidentifiable, so
$\hat\varphi_P$ wanders (0.000 to 1.000 across starts) while the likelihood and
the recovered dynamics do not move at all.

**That is a finding about the extension, not about the knob.** At the target
class's own signal-to-noise the **process-scale channel is fitted dead**: $Q$
is 0.66% of $\gamma_0$ for a process this smooth, so a log-scale wobble on $Q$
barely moves the predictive variance $S = A_{00} + Q_g + R_g$, which
$\sigma^2=9$ dominates. This is the same conditioning fact that makes
`_moment_noises` amplify $Q$ by 151×, wearing a third hat.

Two limits worth stating: this tests the **process** channel only — $s_M$ was
not varied, and $\sigma^2$ carries most of the variance, so the measurement
channel should still be alive. And the parent's own 1.3× failure was at $p=1$
on a random walk, where $Q$ *is* the signal. **So the missing $\varphi$ grid is
still a real gap; this shows it cannot be exercised on smooth ODE data, not
that it does not matter.**

## 5. Parent versus candidate

`statfilter` is this filter's $p=1$, $\alpha=1$ face, and the test suite checks
the two agree to 1e-8 on identical data rather than asserting it.

| | `statfilter` (parent) | `odefilter` (candidate) |
|---|---|---|
| state | scalar level $\theta_t$ | lag vector $(x_t,\dots,x_{t-p+1})$ |
| transition | $\theta_t=\theta_{t-1}+w_t$ | $x_t=\alpha\cdot(x_{t-1}\dots x_{t-p})+w_t$ |
| observation | $\theta_t+v_t$ | $x_t+v_t$ *(same)* |
| noise channels | 2, each log-AR(1) | 2, each log-AR(1) *(unchanged)* |
| learned numbers | 6 | $p+6$ → 9 at $p=3$ |
| settable model commitments | **none** | **three** — $p$, $u=e_1$, static $\alpha$ |
| grid | $\text{order}^2$ Gauss–Hermite nodes | identical |
| collapse | GPB1 on the level | GPB1 on the lag vector |
| process update | $P+\bar Q$ | $FPF^\top+e_1e_1^\top\bar Q$ — the extra term is `0004` |
| amplitude shares | 3, summing to 1 | *(unchanged)* |
| mode coordinates | 4: PA/PR/MA/MR | *(unchanged)* |
| `whiteness` | — | **new** |
| `roots` / `memory()` | — | **new** |
| `derivatives()` / `state_cov` | — | **new** |
| `predict(h)` | flat in $h$ | $F^h$ propagation — where $\alpha$ earns its keep |
| closed-form start | variogram $\gamma_0=Q+2\sigma^2$ | IV on lags $\ge p+1$; $\gamma_{k\ge1}$ for $\sigma^2$ |
| is that start admissible? | yes, by construction | **no** — $Q$ amplified ~151×, so it is scanned |
| $\varphi$ grid search | yes, 5×5 *(added after a 1.3× failure)* | **no** — starts at 0.5 |
| criterion switch | `loglik` / `pem` | `loglik` only |
| fit time, $n=900$ | ~22 s | ~120 s |

## 6. What this changes

**The rule survives the audit, with one correction and one deletion.**

1. **$p$ moves from "free" to "floored", and free is the right answer anyway.**
   It is learnable from below by a rule that imports no constant, and that rule
   reproduces the parent on the parent's data. SUMMARY item 10 is half
   discharged: the floor is measurable, the ceiling is not.

   And a free $p$ is a mild kind of free, because **it is a categorical axis,
   not a continuum**. There is no half-step to mis-set, the useful range is
   short — high-order ODEs are too high-frequency to be worth modelling at any
   realistic sampling rate — and the worst case is affordable: run several
   orders in parallel and let each one's own tracked predictive likelihood say
   which is fitting. That is not a new mechanism. It is exactly what §4.B does
   offline, and it is exactly the architecture the filter already uses one
   level down — grid the nuisance, weight by marginal likelihood, report the
   mixture. Order would become one more gridded coordinate, at the cost of a
   filter pass per candidate.

   The continuous version of the same idea is a real generalisation rather than
   a workaround; it is recorded as an open direction in the
   [repository README](../../README.md#open-directions).
2. **Delete the $Q$ scan** (`fit_` stage 1b). It is inert across every window
   and resolution tried, it costs 13 filter passes, and it reliably starts the
   search slightly worse than the closed form it was added to fix. Removing it
   removes the most arbitrary number in the file *and* makes the fit faster —
   which is the rare case where the principled move and the cheap move agree.
3. **`_iv_alpha` should require $m>p$, not default it.** Just-identified IV
   diverges. That is a precondition of the estimator, and stating it as one
   turns the last non-inert scaffolding choice into a theorem rather than a
   setting.
4. **Two commitments beyond $p$ are now named and sized**: the injection
   direction $u=e_1$ ($p-1$ numbers, inside the $2p+1$ budget) and the
   staticness of $\alpha$. Neither is a defect — both are the honest price of
   the extension — but neither was written down before.
5. **The $\varphi$-grid gap against the parent stays open**, for the reason in
   §4.1 rather than by neglect.

Everything else in the file is a budget, a guard, or a transient, and the
audit's real content is that this was checked rather than claimed.
