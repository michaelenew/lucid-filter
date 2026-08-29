# 0007 — the shipped `LucidFilter` on the research rigs, and the one real defect the wiring exposed

0001–0006 measured purpose-built prototypes.  This probe (`0007_shipped_filter.py`) runs
`lucid.LucidFilter` itself — the shipped object, with its full 15-cell `(phi, s)` noise bank
live underneath every dynamics hypothesis — on the same rigs and reports the same quantities.

## 1. The wiring: the dynamics channel is a state augmentation

`dynamics=None` / `faults=rho` is realized as the augmentation `(x, g)` with
`F = F0 + sum_j g_j A_j`, augmented observation `[H | 0]`, and augmented transition
`[[F(g), d(Fx+Bu)/dg], [0, I]]`.  Three things fall out for free, and they are the reason the
reduction is worth stating rather than just implementing:

- **The noise machinery runs on top unchanged.**  The caltrop scale walk, the structural
  activation, the `(phi, s)` bank — all of it operates on the augmented system without
  modification.  That is exactly what 0002 requires: a wrong `F` and an elevated `Q` compete as
  hypotheses *under a live noise walk*, not through a whiteness statistic bolted on the side.
- **The engine's own activation rule already excludes `g` from the scale walk.**  A departure
  axis is unobserved by `[H | 0]`, so research 0024's structural rule (a mode is live iff it
  carries base variance AND is seen by `H`) marks it inactive with no new code.  The mask is
  passed explicitly anyway, in eigenmode space, so it stays exact if a departure's drift
  happens to coincide with a process eigenvalue.
- **`g`'s drift is a class commitment, not a walked scale** — `q_g = sigma^2 rho`, capped at
  `sigma^2`.  Bounded, never frozen (0003): the gain stays live, so an axis the data cannot
  see today still moves when excitation arrives.

## 2. Measured, shipped

| | derived / prototype | **shipped `LucidFilter`** |
|---|---|---|
| 0001 scalar rig, detection delay | frontier **15**; prototype bank 13.5 ± 0.6 | **15.7 ± 1.7** (40/40 seeds) |
| 0005 blowout, detection | prototype (named anchors) 0.9 steps = 18 ms | **2.2 ± 0.2 steps = 43 ms** |
| 0005 blowout, `rL` recovered | true 0.30 | **0.303 ± 0.018** |
| 0005 blowout, `rR` (healthy) | true 1.00 | **1.043 ± 0.021** |
| 0005 blowout, settled RMSE / refit-oracle | prototype 1.011 | **1.037 ± 0.008** (frozen nominal: 5.06) |

The shipped filter sits on the derived frontier on the scalar rig.  On the blowout it is
~2.4× slower to detect than the prototype and ~3% further from the oracle, both for stated
reasons: this configuration carries **no named anchors** (it is mechanism (b), the physical
departure channel, not (a) — and 0005 already measured anchors as the fastest detector), and
every dynamics hypothesis is competing against a live 15-cell noise bank rather than a
2-point toy noise axis, which is 0004's masking rule applying to the shipped object.  43 ms
is still inside one control period of the thing it protects.

**The blowout rig is expressible in the shipped API at all** because the wheel radii enter a
*state-dependent control map*: `F = I` and `B(x)` rotates with the heading.  That needs two
things the first wiring did not have — dynamics supplied as a **callable** (`odefilter`'s
`linearized_dynamics`, lifted to the multivariate case and to `(F, B)`), and departure
**directions that are themselves callables**, since the direction a physical parameter pushes
in rotates with the operating point.  Both are now in the public API.  A *constant* callable
reduces to the matrix it returns to 1e-8 (`test_callable_dynamics_relinearises`), the same
reduction guarantee `odefilter` gives.

## 3. The defect this probe caught: "unit Frobenius" is not a class size

The first wiring scaled every departure direction to **unit Frobenius norm**, on the argument
that "the entries of a stable discrete transition are O(1)".  That is true for `F` and **false
for `B`**, whose entries carry the input's units: on a 50 Hz differential drive `|B| ~ 5e-3`,
so the blowout's true coefficient was `-0.0036` against a prior of standard deviation 1 —
four orders of magnitude inside the prior.  Measured consequence: detection still fired (any
mismatch shows up) but **recovery silently failed** — `rL` came back 0.999 instead of 0.30 and
settled tracking sat at 1.95× the oracle while *reporting* nothing wrong.

The fix is to make the class statement scale-free: a unit coefficient means **"this part of
the dynamics changed by about its own magnitude"**, i.e. a direction is scaled by the norm of
the nominal it perturbs (`||F0||` for an `F`-direction, `||B0||` for a `B`-direction, a
norm-weighted blend for a joint one).  With that, `rL` recovers to 0.303 ± 0.018 and tracking
to 1.037× oracle.

Worth recording as a general lesson for this repo's style of "derive every constant": a
derived constant is only as good as the *units* of the quantity it is derived for.  The
failure mode was silent — the filter reported a confident fault and a wrong parameter — which
is the worst kind, and only an end-to-end run of the shipped object against ground truth
exposed it.

## 4. Cost

Measured on the scalar rig: ~10.6 ms/step for 3 dynamics hypotheses × 15 `(phi, s)` cells =
45 engines, numpy.  The dominant multiplier is the `(phi, s)` bank, not the dynamics channel:
the dynamics hypotheses multiply an already-15× bank.  0053 §5's levers all still apply (the
bank is the first thing to spend, per-cluster factorisation, numpy→C).  For an embedded
target the honest statement is that the dynamics channel costs a factor equal to its
hypothesis count (here 3), on top of whatever the noise bank already costs.

## Opens

- Anchors + physical departures together on the blowout (this probe ran them separately:
  0005's prototype had anchors, this configuration has the departure channel).
- The drone rig through the shipped API: it needs a constant input channel to carry gravity
  (`u = (T, tau, 1)`), which works but was not measured here.
- Time-anchored spawns (0006) remain unshipped and uncredited.
