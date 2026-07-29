# stat-tracker

Tracking a drifting quantity through noise, without tuning anything.

The question this repository chases: a filter needs to know how fast the truth
moves and how noisy the sensor is, and every practical answer to that has a knob
in it somewhere. Can the knobs be removed — not hidden, not defaulted, actually
removed — by working out what information the data carries and what it does not?

## What's here

| | |
|---|---|
| **[`statfilter/`](statfilter/README.md)** | the filter. Importable, tested, no tuning parameters |
| **[`theory/`](theory/README.md)** | the derivation, in seven parts |
| `scripts/` | `THEORY-0{01..10}-*.py` compute every number in `theory/`; `FILTER-0*.py` are earlier attempts, kept for the record |
| `figures/` | 24 plots, plus the raw numbers as `theory00*.json` |
| `original_chat.md` | the conversation the work grew out of |

## The filter, in three lines

```python
from statfilter import AdaptiveFilter

f = AdaptiveFilter.fit(x)     # learns all six parameters
r = f.filter(x)               # r.mean, r.var, and why it moved
```

```
theta_t = theta_{t-1} + w_t     w_t ~ N(0, Q  * exp(lamP_t))
x_t     = theta_t     + v_t     v_t ~ N(0, s2 * exp(lamM_t))
lam_t   = phi * lam_{t-1} + noise            per channel
```

A local level observed with noise, where both noise scales are themselves
log-AR(1) processes. Two channels crossed with the two ends of each channel's own
autocorrelation covers the four ways a series can deviate:

| | `phi -> 0` impulsive | `phi -> 1` persistent |
|---|---|---|
| **process** | a jump in the level | a change in drift rate |
| **measurement** | an outlier | a change in noise level |

Not four hypotheses to test between — corners of one continuous space, with a
position reported at every step. No changepoint detection, no gate, no threshold.

Measured against a constant-gain Kalman filter whose gain is chosen *in
hindsight* for each series, over nine probes and four seeds: **geometric mean
0.678, worst case 1.017**, with nothing supplied. On stationary diffusions, where
a constant gain is genuinely optimal, 1.001–1.005.

## Three results worth the detour

**Influence is the square root of information.** The incremental nats a point at
lag $k$ contributes decay as $(1-K)^{2k}$; its optimal influence on the estimate
decays as $(1-K)^{k}$. Information is an energy, influence is an amplitude, and
allocating influence in proportion to nats is wrong by a square.

**Detection is strictly more expensive than not detecting.** Asking *where* a
change happened pays a null penalty growing like $\log n$; asking *how large the
deviation is at each $t$* pays a constant 1.353 nats with no $n$ in it. The scan
penalty buys a location estimate the filter never uses. That is the whole reason
there is no changepoint test in here.

**Nothing makes a finite memory optimal unless the parameters themselves drift.**
A buffer length is a statement about how fast the world's noise changes, not
about estimation efficiency — and once that is said out loud it can be estimated
instead of chosen. In the final filter it isn't even a length: the log-scale
state carries history recursively and $\varphi$ *is* the forgetting rate.

## The knob ledger

Gone: `epsilon`, `pairs_min`, `buf_cap`, the refit interval, the EWMA window
`W`, the robustness constant `c`, the gate strength `a_j`, a stray `6.0`, the
Student-$t$ shape `nu`, the jump prior `tau`, and the tail length `L`.

Remaining: six parameters the data chooses, a quadrature order (verified against
5/9/15/25), and a compute budget.

## Running it

```bash
pip install numpy scipy                  # scipy is needed only to fit
python -m pytest tests -q -m "not slow"  # 11 structural checks, 0.3 s
python -m pytest tests -q                # all 19, ~13 min
python scripts/THEORY-007-complete-allocator.py   # regenerate the headline results
```
