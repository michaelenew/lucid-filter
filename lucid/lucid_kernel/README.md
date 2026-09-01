# lucid_kernel

The filters' inner recursion, in C. **It returns the same bits as the NumPy
code it stands in for** — not the same number to a tolerance, the same
IEEE-754 payload — so a result computed with it and a result computed without
it are the same result, and switching it on is not a modelling decision.

```sh
python3 -m lucid_kernel.build        # or `pip install .`, which builds it too
```

Nothing else changes. `LUCID_KERNEL=0` in the environment turns it off; so
does not building it.

## Why

The recursion is *dispatch-bound*, not arithmetic-bound. `LucidFilter` runs its
members **stacked** — one recursion with a leading member axis over the (phi, s)
box crossed with the split ladder crossed with the hazard ladder's walkers,
which on a scalar rig is a few thousand members — so a step is around forty
einsums over `(M, G, n, n)` arrays with `n` and `m` in single digits. NumPy
spends longer deciding what to do than doing it.

So there is nothing clever to do. The arithmetic is already minimal; what costs
is asking NumPy for it forty times a step.

## What it covers, and what it buys

The compiled step is `lucid_bank_predict` and `lucid_bank_correct`: the stacked
bank step, split where LAPACK is. The first builds the predictive covariance
and stops at the innovation covariance `S`; NumPy's own `linalg` takes its
inverse and log-determinant, exactly as it always did; the second finishes the
correction, the per-axis window softmaxes and the two collapses. **Two compiled
calls and two NumPy ones per bank step, against forty.**

Measured on this machine (a 4-core x86-64 with AVX-512), against the same call
with the kernel off:

| call | shape | speed-up |
|---|---|---|
| `LucidFilter(dynamics=F, H=H).filter` | n = 2, m = 2, 400 points | 1.9× |
| `LucidFilter(dynamics=None).filter` | scalar, 400 points | 1.4× |
| the suite | 60 tests | 335s → 268s |

The learned-dynamics figure is lower than the two-sensor one for a reason worth
reading rather than averaging: that filter carries a *nominal* bank at n = 1
beside its walkers at n = 2, and n = 1 is one of the widths the kernel declines
(below).

**What it does not cover.** A state-dependent measurement map, a partially
observed row, and a step the offset channel rides on stay in NumPy — each is a
branch, not a cost. And the transcription is of the narrow widths only:
`2 <= n <= 4` with `1 <= m <= 3`. Outside them NumPy reduces the contractions
inside its vector registers, and the fold it uses there — which depends on the
register width the build dispatched to — is not one hand-written C reproduces.
The kernel declines those shapes rather than answering differently. Widening it
means either matching those folds or handing the four affected contractions back
to `np.einsum`, the way `mp_contract` already does for the same reason.

## How "identical" is arranged

Three of the operations in the recursion have an evaluation order that is a
property of the local NumPy build rather than of the arithmetic. Hand-written
C cannot guess them, so it does not try.

**`exp` and `log`.** NumPy ships its own SIMD implementations for float64 and
dispatches on CPU features, so they are *not* libm's: on this machine about
3% of `exp` arguments and 0.1% of `log` arguments land on a different last bit
than glibc's. The kernel therefore fetches NumPy's own inner loops out of the
ufunc objects at import and calls those. Identical by construction, on any CPU
and any NumPy build.

**The dot products in `update`.** `pi @ T`, `pi @ LP` and the rest are BLAS,
whose summation order is an OpenBLAS kernel detail — and differs again between
a contiguous vector and a strided one, which NumPy hands to BLAS as an
increment rather than copying. The kernel calls `PyArray_MatrixProduct2`, the
C entry point behind `np.dot`, with the same strides.

**`arr.sum(axis=-1)`.** NumPy's pairwise summation: eight accumulators up to a
128-element block, halving above that. This one *is* transcribed, because it
is plain C in NumPy too, and the tests check it against NumPy for every length
up to 1000.

Everything else contracts in NumPy's plain nested-loop order, which C
reproduces exactly — with three exceptions, which are the interesting part.

### The three contractions NumPy decides for itself

`np.einsum` chooses a reduction shape per contraction, and the choice moves
with the operand sizes rather than with the arithmetic:

| contraction | what NumPy does |
|---|---|
| `bgxw,bgwv,bgzv->bgxz` (`F P0 F'`) | one flat run over both contracted axes at p ≥ 3; a nested pair — inner sum over `v`, accumulated over `w` — at p = 2 |
| `bgxz,bgz->bgx` (`F m0`) | a contiguous reduction with a stride-0 output, which is the shape that goes vectorised: the products land in SIMD lanes and come back through a butterfly, `(t0 + t2) + (t1 + t3)`. Reproducible up to p = 4, where four and eight lanes agree; above that they part company and neither is what NumPy gives |
| `g,gxz->xz` and `g,gx,gz->xz` (the reporting collapse in `filter()`) | vectorised at p = 1, where the per-node covariance is contiguous in the node index, and left-to-right above it — and at p = 1 which way it goes depends on the *grid size* |

A table of what NumPy does would be a claim about every NumPy on every CPU.
So the kernel **asks**: `odefilter.core._einsum_orders` runs each contraction
both ways on data of the shape in hand and compares the bits, and where the
answer is beyond what C can reproduce (`F m0` above p = 4) the kernel hands
that one contraction back to NumPy — one call per step against O(B G p²)
arithmetic, which does not show up in the timings.

### And then the whole thing is checked

The probe is evidence; it is not the decision. Before the kernel is used for a
given problem shape it is run against the NumPy path on a short, deliberately
awkward series — a missing observation, parameter vectors spread far enough
apart that the nodes disagree — and every number that comes out is compared
bit for bit. For `filter()` that is all thirteen reported fields plus the
state mean and covariance, not just the likelihood. A shape that fails falls
back to NumPy and warns once.

There is no configuration in which a mismatched kernel is used. That is what
makes this safe across NumPy versions and CPUs: a future NumPy that
reassociates one of these sums does not silently change anyone's numbers, it
turns the kernel off.

The build uses `-ffp-contract=off`. That is a correctness flag, not an
optimisation one: with contraction on, the compiler may fuse `a * b + c` into
a single rounded operation, which is a different — and better — number than
the NumPy expression the kernel has to reproduce.

## Borrowing it

Another extension module can call the recursion rather than reimplement it,
which is the only way a second module can share the bit-exactness guarantee.
`_kernel.c` exports a versioned struct of function pointers through a capsule;
`lucid_kernel.h` is the interface and says how to import it.

## Tests

`../tests/test_kernel.py`. Everything there compares raw payloads through
`.view(np.uint64)`: `np.allclose` would pass on a kernel that had quietly
reassociated a sum, and would say nothing about whether a fit lands in the
same place. The bit comparison is also strict about the two things `==` gets
wrong — it separates `0.0` from `-0.0` and holds NaN equal to itself.

The slow one is the test worth reading: the same `fit()` run twice, once with
the kernel and once without, landing on the same nine parameters to the last
bit.
