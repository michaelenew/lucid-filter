# stat-tracker

Adaptive filters with no theoretically relevant free parameters.

A compute budget is not a free parameter: it trades a real-world cost (time)
against theoretical accuracy and nothing else, so it is allowed and is always
labelled as such.

| workstream | state |
|---|---|
| [`adaptive-random-walk-filter/`](adaptive-random-walk-filter/SUMMARY.md) | delivered: a tuning-free filter for an unbiased random walk observed with noise |
| [`filter-optimality-proof/`](filter-optimality-proof/SUMMARY.md) | one layer proved, one measured, one open — where "optimal" does and does not hold |
| [`ode-adaptive-filter/`](ode-adaptive-filter/SUMMARY.md) | in progress: extending to processes locally described by a second-order linear ODE |
| [`crypto-predictivity/`](crypto-predictivity/SUMMARY.md) | the filters pointed at real series: no dynamics in a price at any frequency on any clock, half the volatility channel is the clock, and an oscillator in realised volatility |

## Open directions

### Fractional derivatives and an integral transform for the dynamics

The ODE workstream currently commits to an integer order $p$ — a recurrence
$x_t=\sum_{i=1}^{p}\alpha_i x_{t-i}+w_t$, whose characteristic roots are the
modes. The audit in
[`ode-adaptive-filter/exploration/0030`](ode-adaptive-filter/exploration/0030_the_free_variable_audit.md)
shows $p$ is learnable from below but not from above, and that it is the one
axis in the filter that is genuinely categorical: you pick an integer, and
picking between integers is not defined.

**Replacing the integer order with a fractional one would make degree a
continuous coordinate, and therefore learnable by the same marginal likelihood
as everything else.** The natural machinery is the one that already defines
fractional differentiation: the Grünwald–Letnikov series

$$\Delta^{\nu}x_t=\sum_{k\ge0}(-1)^k\binom{\nu}{k}x_{t-k}$$

is exactly a recurrence with *infinitely many* lags whose coefficients are a
smooth function of a single real $\nu$. Equivalently, in the transform domain,
$(1-z^{-1})^{\nu}$ — a filter with a branch point where the integer case has a
pole of integer multiplicity. So the dynamics would be specified by a *kernel*
against which the history is integrated, rather than by a finite coefficient
vector, and the model would be written as a transform integral rather than a
sum.

Why this is worth doing beyond removing a knob:

- **Degree becomes an estimate with an error bar.** "Is it second order?"
  stops being a model-selection question with no continuous answer and becomes
  a coordinate with a likelihood profile, like every other quantity in these
  filters. It also gives a principled reading of the in-between cases the
  integer model has to round.
- **It is the natural home for long memory.** $\nu\notin\mathbb Z$ gives
  hyperbolic rather than exponential decay of the impulse response, so
  $1/(1-|z|)$ — the memory law these filters keep rediscovering — would
  generalise to a power law. Processes whose autocorrelation refuses to fit an
  exponential are common and currently have to be forced into extra integer
  modes.
- **It may reorganise the channel structure.** `0024` established that the
  channels are the roots of the characteristic polynomial. Under a branch point
  there are no isolated roots, so "how many channels" would become a question
  about a continuous spectrum — which is either a much cleaner statement of the
  same fact or a sign that the discrete picture was an artifact of integer
  order. Either outcome is informative.

The obvious costs: the state is no longer finite-dimensional, so it has to be
truncated (a compute budget, which is allowed, but a new one), and the
errors-in-variables result that instruments at lags $\ge p+1$ annihilate the
measurement noise relies on the residual touching finitely many lags — it would
need restating.
