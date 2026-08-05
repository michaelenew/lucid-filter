# 0007 — The parent moved first, and the seam held

The parent's flagged structural change landed on main: `unit_roots=d` pins
$d$ characteristic roots at $z=1$ exactly, writing
$\alpha = \text{base} + \beta M$ with $(\text{base}, M)$ the fixed linear map
of multiplying $(z-1)^d$ into a free quotient polynomial $\beta$
([`ode-adaptive-filter/0041`](../../ode-adaptive-filter/exploration/0041_a_climbing_bias_is_a_pinned_root.md),
`core.py::_pin_maps`). This was the "linear offset to read directional bias"
this workstream was told to expect. This note records the merge verification
and one alignment that is better than compatibility.

## 1. The inheritability rule's first live test: passed, zero changes

The design rule (0001 §3) was that the fractional layer owns no recursion and
factors through the parent's published interface, so parent changes inherit
by delegation. Measured after the merge, with **no modification to any file
in this workstream**:

- The parent's fast test suite passes (24 tests).
- The prequential regression reproduces **bit-for-bit**: truth 1.3 seed 0
  gives FRAC $-1.6894$, $\hat\nu=1.256$, AR(2) $-1.7308$ — identical to
  0005-C's recorded values. (`_face_optimum` grew
  `(p, unit_roots)` parameters, defaulted so existing positional calls mean
  what they meant; `_face_profile` is unchanged.)
- The wide-$q$ profile point at truth 1.7 reproduces to the refine tolerance
  ($-2.3982$ against $-2.3983$ recorded).

## 2. The alignment: the split kernel *is* `unit_roots`, made continuous

0004's fix — compose the exact integer factor with the truncated fractional
one — and the parent's `unit_roots` are the **same structure, found
independently from opposite directions**. Verified to machine precision: for
every $\nu$ tried (0.7, 1.3, 1.7, 2.0, 2.3),

```
Params(alpha=gl_split(nu, K), unit_roots=floor(nu))
```

passes the parent's synthetic-division validation, and the free-coefficient
vector the parent recovers by `_vec()` **is exactly the truncated GL kernel
of the fractional part** ($\le 1.5\times10^{-16}$, including at integer
$\nu$, where the quotient is empty). So under the new parent interface the
fractional model reads:

> $\nu = m + f$: `unit_roots` $= m$, and the quotient polynomial pinned to
> the one-parameter GL curve $\beta = c(f)$ instead of left free.

The parent made the integer part of degree a pinned multiplicity; this
workstream makes the remainder continuous. Even the lessons match: 0041's
"a maximum-likelihood unit root lands just inside the circle, and a root at
$1-\varepsilon$ forecasts a drift that decays instead of one that continues"
is the $m$-side of 0004's "the integer part must be exact" — the same
integer-exactness fact, found there by forecasting failure and here by a
likelihood dragged toward the integer faces.

Two practical consequences, recorded for the next probe rather than acted on:

- The full fit can now be phrased entirely in parent vocabulary: for each
  $m\in\{0,1,2\}$, profile $f$ along the quotient curve, with the parent's
  `fit_(unit_roots=m)` machinery carrying the noise channels around it.
- `_face_optimum(y, b0, p=p, unit_roots=m)` — the quotient left **free** —
  is the right new baseline for prequential comparisons: it gets the unit
  roots for free (as FRAC does) but not the hyperbolic tail, so it isolates
  what the fractional curve itself buys. The free-AR baselines of 0003/0005
  conflate the two.

## 3. Status of 0006 §5's feedback requirements

1. **Overflow guards, not radius guards** — holds after the rework: the fit
   still rejects by the $-\infty$-on-overflow guard, and no radius
   admissibility test was added. Nothing to do.
2. **Multiplicity-aware root tolerance** — the parent's new validation
   independently adopted the right instrument: it checks pinned roots by
   synthetic division ("the ill-conditioning of the *question*, not of the
   construction", its test says), never by `np.roots`. The one place the
   original concern still stands is `Params.alpha_at`'s $1+10^{-9}$ radius
   test, unchanged in the rework: a split kernel with $m\ge2$ measures
   radius $1+3\times10^{-8}$ there and would be misread as explosive. It
   only matters when the dynamics channel runs over a fractional kernel,
   which nothing does yet — still flagged, not blocking.
