# 0010–0011 — The derivation closes for one channel and does not transfer to the other

> **AI-generated, not peer-reviewed.** Code
> [`0010_derived_ladder.py`](0010_derived_ladder.py),
> [`0011_window_scale.py`](0011_window_scale.py). README hero gate, 12 seeds,
> shipped `LucidFilter` with a patched copy pinned against it at `0.000e+00`.

## 1. The derivation, stated

[`0009`](0009_the_axiom_relocated.md) identified the structure: the log-scale
channel looks like the Q/R split channel one level up.

| | "state" | "process" | "sensor" | per-step identifiable? |
|---|---|---|---|---|
| split channel | `theta` | `Q` | `R` | only the **total** |
| scale channel | `lam` | `nu = s^2(1-phi^2)` | blur, variance `rho = 1/I = 2` | only the **level** |

In both, what is *not* per-step identifiable is a bank quantity. For the split
that is the ratio; for the scale it is how fast `lam` moves. So the `(phi, s)` box
ought to be the scale channel's own split ladder, built by the same rule with no
new constants:

```
q_lam = nu / rho,   K = gain(q_lam),   t = arccos(1 - K)  in  [0, pi/2]
ends    : t in [blur, pi/2 - blur],    blur = sqrt(2/N),  N = 1/(1 - forget)
spacing : the engine's own Sparrow factor, 1.5 * blur
count   : follows -- 23 rungs at forget = 0.999.  Nothing chosen.
```

Two further quantities are then forced, if the window is required to be the
channel's own predictive width. Writing `p` for the `lam` channel's one-step
predictive variance,

```
p^2 + p (rho(1 - phi^2) - nu) - nu rho = 0
```

and imposing `window = sqrt(p)` gives the closed form, exact to `1e-16`:

```
phi = sqrt(cos t),        s = sqrt(2 (sec t - 1))
```

a **one-parameter curve** through the `(phi, s)` plane, `t` the only coordinate.
It reaches `phi` from 0.9995 down to 0.224 — both far nearer 1 than the shipped
0.95 (which [`0007`](0007_phi_reach.md) wants) and far below its 0.70.

**As mathematics this closes.** Ends, spacing, count, persistence and window width
all follow from two numbers already in the filter: the blur `sqrt(2)` and the
memory `N`. No `_PHIS`, no `_SS`, no `_SPAN_S`.

## 2. It loses on the gate

RMSE ratio to an oracle Kalman told the truth, 12 seeds.

| variant | A steady | B jump | C sensor ×3 | settle | `E[e²/S]` |
|---|---|---|---|---|---|
| **shipped (15 cells)** | 1.017 | **0.694** | **1.078** | **35.0** | 0.698 |
| derived ladder (23) | 1.017 | 0.629 | 1.314 | 35.2 | 0.857 |
| window = `sqrt(p)` (15) | 1.021 | 1.049 | 1.230 | 42.4 | 0.807 |
| window = `sqrt(p)` + ladder (23) | 1.019 | 1.178 | 1.106 | 44.1 | 0.730 |

The ladder alone is better on the jump (0.629 vs 0.694) and better calibrated, and
**22% worse on the sensor-degradation window**. Re-keying the window to `sqrt(p)`
— the step that makes the derivation self-consistent — is worse than the shipped
filter on every column that matters.

## 3. Why, and this is the finding

Scale the window's geometry by a constant `c`, holding the class (prior and
kernel) fixed:

| `c` | A steady | B jump | C noisy | settle | `E[e²/S]` |
|---|---|---|---|---|---|
| 0.5 | 1.019 | 0.882 | 1.528 | 42.3 | 1.128 |
| 0.7 | 1.019 | 0.644 | 1.675 | 50.1 | 1.167 |
| **1.0 (shipped)** | 1.017 | 0.694 | 1.078 | **35.0** | 0.698 |
| 1.4 | 1.016 | 0.651 | **1.048** | 57.6 | 0.645 |
| 2.0 | **1.015** | 0.721 | **1.001** | 49.2 | 0.641 |
| 3.0 | 1.015 | 0.849 | 1.062 | 61.8 | 0.664 |

**The optimum is interior**, and different columns want different `c`: steady
state improves monotonically with `c`, regime C peaks near 1.4–2.0, settling
degrades with `c` throughout, and the jump is non-monotone with minima at 0.7 and
1.4. The shipped `c = 1` sits in the middle of that spread.

> **The window is not a resolution grid. It is doing two jobs at once — resolving
> where `lam` is, and reaching far enough to absorb a jump in one step — and they
> pull opposite ways.**

`sequence-demix/0002` had already measured the reach half ("the jump is a reach
problem... the shipped `s` box tops out at 0.8 — a factor of 11, where the jump
needs ~1000"). What is new here is that the resolution half is real too, and that
the trade has an interior optimum. **A resolution-only derivation necessarily
lands on the wrong side of it**: `sqrt(p)/s` runs 0.43–0.99 across the box, i.e.
`c < 1` throughout, which the sweep shows is the losing direction.

## 4. What this settles about AUD-2

AUD-2 asks for `_PHIS`/`_SS` to be derived. This workstream can now say what kind
of argument will *not* do it:

- **No resolution argument will derive the box**, because the box is not sized by
  resolution. That rules out the whole family of arguments this workstream has
  been pursuing, including the one that closes cleanly for the split channel.
- The split channel's ladder *is* fully derived, and the reason it is derivable is
  that its coordinate is compact **and the split cannot jump**. A split is a
  property of the world's noise structure that moves when regimes move; the scale
  itself jumps, and a jump has no resolution.
- What would derive the box is a **two-sided argument pricing reach against
  resolution**. §3's sweep is the first map of that trade and is the natural
  starting point.

## 5. Nothing was changed in the filter

Three candidate changes were tested and none clears the repository's bar
("derived from theory, then defended with simulation"):

| candidate | evidence | verdict |
|---|---|---|
| window keyed to `sqrt(p)` | derived; **measured worse** on 4 of 5 columns | rejected |
| the 23-rung derived ladder | derived; better on jump and calibration, **22% worse on regime C** | rejected |
| add `phi = 0.995` to `_PHIS` | [`0007`](0007_phi_reach.md): 2.9–3.5× better on a wandering scale, on a *research* grid filter; on the shipped filter's own hero gate it is **neutral** (1.018 vs 1.017, all other columns identical) | **not enough** |

The third is the one worth resolving, and the hero rig cannot resolve it: its
regime changes are **steps**, not a slowly wandering scale, so the near-unit-root
members have nothing to do. What would settle it is a rig whose noise level
*wanders* — [`0003`](0003_what_stationarity_costs.py)'s generator — run through
the shipped filter. Until that exists, adding a fourth `phi` rung costs 33% more
compute for a benefit measured only on a different implementation.

One thing worth recording even though it is not derived: on the repo's own gate
the shipped `c = 1` is **not** the best window scale for regime C. `c = 2` scores
1.001 against 1.078, and 1.015 against 1.017 in steady state, paying for it in
settling time (49.2 against 35.0). That is a measured trade in the shipped
filter's own defaults, and it belongs to the two-sided argument §4 asks for.
