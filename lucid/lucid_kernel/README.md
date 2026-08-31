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

The recursion is *dispatch-bound*, not arithmetic-bound. A step is a handful
of einsums over arrays small enough that NumPy spends longer deciding what to
do than doing it. Profiling one `OdeFilter.fit(y, p=3)` on 500 points:

| where | share of wall clock |
|---|---|
| `_loglik_batch` | 99% |
| — of which, inside `c_einsum` | 79% |

So there is nothing clever to do. The arithmetic is already minimal; what
costs is asking NumPy for it half a million times.

## What it covers, and what it buys

Measured on this machine (a 4-core x86-64 with AVX-512), against the same
call with the kernel off:

| call | shape | speed-up |
|---|---|---|
| `odefilter._loglik_batch` | p = 2, order 5, 104 vectors | 10.8× |
| `odefilter._loglik_batch` | p = 3, order 5, 104 vectors | 8.4× |
| `odefilter._loglik_batch` | p = 3, order 5, dynamics channel on | 7.9× |
| `odefilter._loglik_batch` | p = 5, order 5 | 5.1× |
| `odefilter._loglik_batch` | p = 1, order 7 | 2.1× |
| `OdeFilter.loglik` | p = 3, order 5 | 10.7× |
| `OdeFilter.filter` | p = 3, order 5 | 7.8× |
| `statfilter._loglik_batch` | order 3 … 9 | 8.2× … 1.7× |
| a whole `OdeFilter.fit(y, p=3)` | 600 points | **6.9×** (1196 s → 174 s) |
| a whole `OdeFilter.fit(y, p=3)` | 250 points | 3.7× (42 s → 11 s) |
| a whole `OdeFilter.fit(y, p=1)` | 250 points | 2.8× |

The two fit rows are the same code and differ only in the length of the
series, which is worth reading rather than averaging: the fit's fixed costs —
`scipy`'s optimiser bookkeeping, the grid rebuild per likelihood evaluation,
the one-off verification — do not shrink with the recursion, so on a short
series they are what is left, and on a long one they are not. Long series are
where a fit is actually slow, so 6.9× is the number to plan with.

At 250 points the largest single thing left is `_face_profile`, the scalar
Kalman pass over the `s = 0` face that pass 1 optimises: 0.4% of that fit
before the kernel and 22% of it after. It is the next thing worth compiling.

`OdeFilter.update` — one observation at a time — stays in NumPy: there is one
step's work to amortise a call over and nothing to win. So does a run with
caller-supplied transitions (`Fs=`, or `linearized_dynamics`), where the
dynamics come from a Python callable anyway.

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
