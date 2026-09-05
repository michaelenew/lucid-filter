# 0010 — the spectral-truncation threshold, derived (no free parameter)

> **⚖️ ATTRIBUTION —** _Derives the freeze/truncation threshold as the point where the walk's steady spread (their finding-18 Var=(1−φ)/4I) exceeds its window — a self-contained internal derivation tying a known identifiability cut to the class (φ,s)._ Prior art: identifiability/observability of noise variances — Mehra 1972 (which modes are estimable); the specific threshold formula is internal. Status: RECOMBINATION.

`WalkingVectorFilter` freezes axes the data cannot resolve (otherwise their
unbounded walk integrates noise into a drift, 0006/0009). The freeze threshold was
a hand-picked `_TRUNC = 0.10·max(I_char)`; this derives it.

## Derivation

The walk on one axis is the finding-18 μ-loop: a dense window of half-span
`L = SPAN_S·s` centred on `mu`, whose centre integrates the grid-shift score with
gain `K*=(1−φ)/4` and drift variance `q_mu`. On static truth the window's restoring
pull holds `mu` near the truth **only while `mu` stays inside the window** — past
`|mu| > L` the score saturates (no node sees the truth), the pull vanishes, and `mu`
random-walks free. So an axis is walkable iff the walk's steady spread stays inside
its window. Finding 18 Theorem 2 gives that spread exactly, `Var(mu) = (1−φ)/(4 I)`,
so `Var(mu) < L²` gives

```
freeze  ⇔  I_char < I* = (1−φ) / (4 (SPAN_S·s)²).
```

`I*` is a pure function of the class `(φ, s)` and the coverage budget `SPAN_S` — no
free parameter. It replaces `_TRUNC`.

## Validation (the program)

The isolated μ-loop simulated on static truth across a sweep of the per-step Fisher
`I` (φ=0.9, s=0.4, `L=1.2`, `I*=0.0174`):

| I / I* | excursion std | localised (< L)? |
|---|---|---|
| 0.10 | 6.45 | no (drift) |
| 0.50 | 1.57 | no |
| 0.80 | 1.09 | yes |
| 1.00 | 0.93 | yes |
| 3.00 | 0.48 | yes |
| 20.0 | 0.19 | yes |

The excursion runs past the window half-span `L` below `I*` (delocalised → drift)
and settles inside above it — the knee sits at `I*` (slightly conservative: at
`I*` the spread is already `0.93 < L`, so the threshold errs toward freezing). The
functional form `I* ∝ (1−φ)/(SPAN_S·s)²` is confirmed.

For the shipped defaults (`φ=0.9, s=0.4, SPAN_S=3`) `I* = 0.017`, which freezes the
unidentifiable weak eigenmode (`I_char = 0.010`) and keeps every sensor
(`I_char ≥ 0.057`) — the right cut, now derived.

Code: `0010_truncation_derivation.py`.
