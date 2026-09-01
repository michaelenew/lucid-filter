/* _kernel.c -- the compiled transcription of the lucid filter recursion.
 *
 * WHAT THIS IS.  `LucidFilter` runs its members stacked, which is what makes it
 * dispatch-bound: a step is around forty einsums over (M, G, n, n) arrays with
 * n and m in single digits, so NumPy spends more time deciding what to do than
 * doing it.  This file is the same recursion with the dispatch taken out.
 *
 * WHAT IT IS NOT.  It is not a second implementation.  Every routine here is
 * bit-for-bit equal to the NumPy code it stands in for -- not equal to within
 * a tolerance, equal in the IEEE-754 bits -- and `tests/test_kernel.py`
 * asserts exactly that, field by field, on raw `.view(np.uint64)` buffers.
 * A result computed with the kernel and a result computed without it are the
 * same result, so turning it on is not a modelling decision and does not need
 * a note in a write-up.
 *
 * HOW THAT IS ARRANGED.  Three of the operations in the recursion have an
 * evaluation order that is a property of the local NumPy build rather than of
 * the arithmetic, and hand-written C cannot guess them:
 *
 *   exp, log      NumPy ships its own SIMD implementations for float64 and
 *                 dispatches on CPU features, so they are NOT libm's.  Around
 *                 3% of `exp` arguments and 0.1% of `log` arguments land on a
 *                 different last bit than glibc.  So the kernel does not call
 *                 libm: it fetches NumPy's own inner loops out of the ufunc
 *                 objects at import and calls those.  Identical by
 *                 construction, on any CPU and any NumPy build.
 *   the dot in    `pi @ T`, `pi @ LP` and friends are BLAS, whose summation
 *   `update`      order is an OpenBLAS kernel detail.  The kernel calls
 *                 `PyArray_MatrixProduct2` -- the C entry point behind
 *                 `np.dot` -- so the same BLAS does the same sum.
 *   `arr.sum(1)`  NumPy's pairwise summation, reproduced here exactly
 *                 (`nk_sum`): eight accumulators up to a 128-element block,
 *                 halving above it.  Checked against NumPy for every length
 *                 up to 1000 in the tests.
 *
 * Everything else contracts in NumPy's plain nested-loop order, which
 * hand-written C reproduces exactly, with one exception noted at
 * `mp_contract` below.
 *
 * The file is compiled with `-ffp-contract=off`: without it the compiler is
 * free to fuse `a*b + c` into an FMA, which is a different (better) number
 * and therefore the wrong one.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "lucid_kernel.h"

/* ------------------------------------------------------------ NumPy's own */

static PyUFuncGenericFunction g_exp_loop = NULL;
static PyUFuncGenericFunction g_log_loop = NULL;

/* -0.5 * log(2 pi), handed over from Python so that the constant is the one
 * the module already uses rather than one this file recomputes. */
static double g_half_log2pi = 0.0;

/* `np.einsum`, kept for the one contraction whose order C cannot predict. */
static PyObject *g_einsum = NULL;

static PyUFuncGenericFunction
find_double_loop(const char *name)
{
    PyObject *np = PyImport_ImportModule("numpy");
    if (np == NULL) {
        return NULL;
    }
    PyObject *uf = PyObject_GetAttrString(np, name);
    Py_DECREF(np);
    if (uf == NULL) {
        return NULL;
    }
    if (!PyObject_TypeCheck(uf, &PyUFunc_Type)) {
        Py_DECREF(uf);
        PyErr_Format(PyExc_RuntimeError, "numpy.%s is not a ufunc", name);
        return NULL;
    }
    PyUFuncObject *u = (PyUFuncObject *)uf;
    PyUFuncGenericFunction fn = NULL;
    for (int i = 0; i < u->ntypes; i++) {
        if (u->types[2 * i] == NPY_DOUBLE && u->types[2 * i + 1] == NPY_DOUBLE) {
            fn = u->functions[i];
            break;
        }
    }
    Py_DECREF(uf);
    if (fn == NULL) {
        PyErr_Format(PyExc_RuntimeError, "numpy.%s has no float64 loop", name);
    }
    return fn;
}

/* out[i] = exp(in[i]), by NumPy's own float64 loop.  The loop is elementwise
 * and chunk-invariant -- verified in the tests -- so calling it on a slice
 * gives what calling it on the whole array would have given. */
static void
nk_exp(const double *in, double *out, npy_intp n)
{
    char *data[2];
    npy_intp strides[2] = {(npy_intp)sizeof(double), (npy_intp)sizeof(double)};
    data[0] = (char *)in;
    data[1] = (char *)out;
    g_exp_loop(data, &n, strides, NULL);
}

static void
nk_log(const double *in, double *out, npy_intp n)
{
    char *data[2];
    npy_intp strides[2] = {(npy_intp)sizeof(double), (npy_intp)sizeof(double)};
    data[0] = (char *)in;
    data[1] = (char *)out;
    g_log_loop(data, &n, strides, NULL);
}

/* NumPy's pairwise summation for a contiguous float64 buffer, transcribed.
 * This is what `a.sum()` and `a.sum(axis=-1)` do, and it is not the same
 * number as a left-to-right sum for n >= 8. */
#define NK_BLOCK 128

static double
nk_sum(const double *a, npy_intp n)
{
    if (n < 8) {
        double res = 0.0;
        for (npy_intp i = 0; i < n; i++) {
            res += a[i];
        }
        return res;
    }
    if (n <= NK_BLOCK) {
        double r[8];
        npy_intp i;
        for (int k = 0; k < 8; k++) {
            r[k] = a[k];
        }
        for (i = 8; i < n - (n % 8); i += 8) {
            for (int k = 0; k < 8; k++) {
                r[k] += a[i + k];
            }
        }
        double res = ((r[0] + r[1]) + (r[2] + r[3]))
                   + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) {
            res += a[i];
        }
        return res;
    }
    npy_intp n2 = n / 2;
    n2 -= n2 % 8;
    return nk_sum(a, n2) + nk_sum(a + n2, n - n2);
}

/* `np.maximum.reduce` over a contiguous buffer: NaN wins, as it does there. */
static double
nk_max(const double *a, npy_intp n)
{
    double m = a[0];
    for (npy_intp i = 1; i < n; i++) {
        if (isnan(m)) {
            return m;
        }
        if (isnan(a[i]) || a[i] > m) {
            m = a[i];
        }
    }
    return m;
}

/* -------------------------------------------------------- BLAS, via NumPy */

/* A reusable float64 array shell over caller-owned memory.  Building one is a
 * single small allocation; the alternative -- copying into a persistent array
 * -- costs more than the dot product it feeds. */
static PyObject *
shell(const double *data, int nd, const npy_intp *dims)
{
    return PyArray_SimpleNewFromData(nd, (npy_intp *)dims, NPY_DOUBLE,
                                     (void *)data);
}

/* a @ b for two length-n vectors, through the BLAS NumPy itself would use.
 * `PyArray_MatrixProduct2` hands back a NumPy scalar for 1-D by 1-D, not a
 * 0-d array, so the value comes out through the number protocol. */
static double
nk_dot2(PyObject *A, PyObject *B)
{
    double out = NAN;
    if (A != NULL && B != NULL) {
        PyObject *r = PyArray_MatrixProduct2(A, B, NULL);
        if (r != NULL) {
            out = PyFloat_AsDouble(r);
            Py_DECREF(r);
        }
    }
    Py_XDECREF(A);
    Py_XDECREF(B);
    return out;
}

static double
nk_dot(const double *a, const double *b, npy_intp n)
{
    npy_intp dims[1] = {n};
    return nk_dot2(shell(a, 1, dims), shell(b, 1, dims));
}

/* --------------------------------------------------- the vectorised sums
 *
 * A contraction whose reduction axis is contiguous in BOTH operands with a
 * stride-0 output is the shape that sends NumPy's einsum down a vectorised
 * path: the products go into SIMD lanes and come back through a butterfly, so
 * the sum is not left-to-right.  For a reduction of length n the lanes hold
 * (t0..t_{n-1}, 0, ...) and the butterfly returns (t0 + t2) + (t1 + t3) --
 * the same answer whether the machine has four lanes or eight, which is why
 * `lane_dot` is not CPU-dependent for n <= 4.
 *
 * Above four the two widths part company, which is why the bank step probes
 * for the fold rather than assuming one, and declines the widths where no
 * candidate reproduces it.
 *
 * `lane_dot` uses fma() deliberately: the lane accumulator is a fused
 * multiply-add on every machine that has one, and the C library's fma() is
 * the same correctly-rounded operation.
 */

static inline double
lane_dot(const double *a, const double *b, int n)
{
    double acc[4] = {0.0, 0.0, 0.0, 0.0};
    int i = 0;
    while (i < n) {
        for (int k = 0; k < 4; k++) {
            if (i + k < n) {
                acc[k] = fma(a[i + k], b[i + k], acc[k]);
            }
        }
        i += 4;
    }
    return (acc[0] + acc[2]) + (acc[1] + acc[3]);
}

#if defined(__GNUC__)
#define LK_ALWAYS_INLINE static inline __attribute__((always_inline))
#else
#define LK_ALWAYS_INLINE static inline
#endif

/* One node's F P0 F^T.
 *
 * `np.einsum("bgxw,bgwv,bgzv->bgxz", ...)` sums its two contracted axes as one
 * flat run for p >= 3 and as a nested pair -- an inner sum over v, accumulated
 * over w -- at p = 2.  Which one it picks is an iterator decision inside
 * einsum, not something the arithmetic determines, so the caller probes NumPy
 * for the p in hand (`lucid_kernel.orders_for`) and passes the answer in.
 * LK_AP_FLAT and LK_AP_NESTED are the two; a p for which neither reproduces
 * NumPy never reaches this file, because the probe declines the kernel. */
#define LK_AP_FLAT   0
#define LK_AP_NESTED 1

/* The collapse over the NODES.  At a per-node block of one element the
 * covariance is contiguous in the node index, which is the shape that sends
 * einsum down its vectorised reduction; above that it is not and the sum is
 * left to right.  Probed, not assumed. */
#define LK_SC_NAIVE  0
#define LK_SC_EINSUM 1

LK_ALWAYS_INLINE void
ap_block_impl(const double *F, const double *P0, double *A, const int p,
              const int mode)
{
    for (int x = 0; x < p; x++) {
        for (int z = 0; z < p; z++) {
            double s = 0.0;
            if (mode == LK_AP_NESTED) {
                for (int w = 0; w < p; w++) {
                    double inner = 0.0;
                    for (int v = 0; v < p; v++) {
                        inner += F[x * p + w] * P0[w * p + v] * F[z * p + v];
                    }
                    s += inner;
                }
            }
            else {
                for (int w = 0; w < p; w++) {
                    for (int v = 0; v < p; v++) {
                        s += F[x * p + w] * P0[w * p + v] * F[z * p + v];
                    }
                }
            }
            A[x * p + z] = s;
        }
    }
}

/* ------------------------------------------------- the streaming entry point */

/* Pull a required (and contiguous, float64) array argument. */
static PyArrayObject *
as_arr(PyObject *o)
{
    return (PyArrayObject *)PyArray_FROM_OTF(o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
}

/* ------------------------------------------------------------------ module */

static PyObject *
py_configure(PyObject *Py_UNUSED(self), PyObject *args)
{
    double log2pi;
    if (!PyArg_ParseTuple(args, "d", &log2pi)) {
        return NULL;
    }
    /* the Python writes `... + mx - 0.5 * _LOG2PI`; IEEE subtraction is
     * addition of the negation, so this is the same last bit */
    g_half_log2pi = -(0.5 * log2pi);
    Py_RETURN_NONE;
}

/* The three NumPy-owned primitives, exposed so the tests can check them
 * against NumPy directly rather than only through the recursion. */
static PyObject *
py_np_prim(PyObject *Py_UNUSED(self), PyObject *args)
{
    const char *which;
    PyObject *a_o, *b_o = NULL;
    if (!PyArg_ParseTuple(args, "sO|O", &which, &a_o, &b_o)) {
        return NULL;
    }
    PyArrayObject *a = as_arr(a_o);
    if (a == NULL) {
        return NULL;
    }
    const npy_intp n = PyArray_SIZE(a);
    const double *ad = (const double *)PyArray_DATA(a);
    PyObject *out = NULL;
    if (strcmp(which, "sum") == 0) {
        out = PyFloat_FromDouble(nk_sum(ad, n));
    }
    else if (strcmp(which, "max") == 0) {
        out = PyFloat_FromDouble(nk_max(ad, n));
    }
    else if (strcmp(which, "dot") == 0) {
        PyArrayObject *b = as_arr(b_o);
        if (b != NULL) {
            out = PyFloat_FromDouble(
                nk_dot(ad, (const double *)PyArray_DATA(b), n));
            Py_DECREF(b);
        }
    }
    else if (strcmp(which, "lanedot") == 0) {
        PyArrayObject *b = as_arr(b_o);
        if (b != NULL) {
            out = PyFloat_FromDouble(
                lane_dot(ad, (const double *)PyArray_DATA(b), (int)n));
            Py_DECREF(b);
        }
    }
    else if (strcmp(which, "exp") == 0 || strcmp(which, "log") == 0) {
        npy_intp d[1] = {n};
        out = PyArray_SimpleNew(1, d, NPY_DOUBLE);
        if (out != NULL) {
            double *o = (double *)PyArray_DATA((PyArrayObject *)out);
            if (which[0] == 'e') {
                nk_exp(ad, o, n);
            }
            else {
                nk_log(ad, o, n);
            }
        }
    }
    else {
        PyErr_SetString(PyExc_ValueError, "unknown primitive");
    }
    Py_DECREF(a);
    return out;
}

/* ===================================================== the LucidFilter bank
 *
 * `LucidFilter` runs its members STACKED: one recursion with a leading member
 * axis over the (phi, s) box crossed with the split ladder crossed with the
 * hazard ladder's walkers, which on a scalar rig is a few thousand members.
 * That is what makes it dispatch-bound in the same way the two older
 * recursions were -- a step is ~40 einsums over (M, G, n, n) arrays with n and
 * m in single digits, so NumPy spends longer deciding than doing.  The two
 * routines below are that step.
 *
 * They are split where LAPACK is: `bank_predict` builds the predictive
 * covariance and stops at S, Python inverts it and takes its log-determinant
 * with the same LAPACK calls it always did, and `bank_correct` finishes.  Two
 * compiled calls and two NumPy ones per bank step, against forty.
 *
 * Everything else about the contract is the kernel's usual one: the reductions
 * whose order is NumPy's rather than the arithmetic's are probed and passed in
 * (`ap_mode`, `sc_mode`), the elementwise transcendentals are NumPy's own
 * loops, and the whole thing is verified bit-for-bit against the NumPy path
 * for the shape in hand before it is used.
 *
 * Restricted, deliberately, to the path that carries the cost: a fixed
 * measurement map, every sensor reporting, and no offset channel riding on
 * the step.  Anything else stays in NumPy, where it already was.
 */

/* Three of the contractions below reduce an axis that is CONTIGUOUS IN BOTH
 * operands with a stride-0 output, which is the shape that sends NumPy's
 * einsum down its vectorised path: the products land in SIMD lanes and come
 * back through a butterfly, so the sum is not left to right.  Up to width four
 * the butterfly and the plain loop are the same number; above it they part.
 * Which one this NumPy produces is asked, not assumed -- `LK_DOT_LOOP` and
 * `LK_DOT_LANES` are the two candidates, and a width where neither reproduces
 * it declines the kernel.  `lane_dot` above is the butterfly. */
#define LK_DOT_LOOP   0
#define LK_DOT_LANES  1     /* four lanes, the SSE/AVX2 width */
#define LK_DOT_LANE8  2     /* eight, with NumPy's halving horizontal sum */
#define LK_DOT_LANE8B 3     /* eight, summed in lane order */

/* NumPy's vector reduction folds the register in halves -- (0,4)(1,5)(2,6)(3,7),
 * then (0,2)(1,3), then one add -- so at eight lanes the tree is not the same
 * number as at four.  Which width this NumPy dispatches to is a CPU feature,
 * so it is asked rather than assumed; `LK_DOT_LANE8B` is the other plausible
 * fold, kept so a build that does not use the halving one is still reachable. */
LK_ALWAYS_INLINE double
lane8_reduce(const double *acc, int mode)
{
    if (mode == LK_DOT_LANE8) {
        return ((acc[0] + acc[4]) + (acc[2] + acc[6]))
             + ((acc[1] + acc[5]) + (acc[3] + acc[7]));
    }
    return ((acc[0] + acc[1]) + (acc[2] + acc[3]))
         + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
}

/* The same question one level up, for the reductions over the NODES: `Kbar`
 * and the two halves of the collapsed covariance walk `w` against an array
 * whose node stride is n*m or n*n, so they too become contiguous-in-both when
 * the per-node block is a single element -- which is exactly the scalar rig.
 * `node_mode` is asked the same way `dot_mode` is. */
LK_ALWAYS_INLINE double
bank_gsum(const double *w, const double *v, npy_intp G, npy_intp stride,
          int mode)
{
    if (mode == LK_DOT_LANES) {
        double acc[4] = {0.0, 0.0, 0.0, 0.0};
        for (npy_intp i = 0; i < G; i += 4) {
            for (int k = 0; k < 4 && i + k < G; k++) {
                acc[k] = fma(w[i + k], v[(i + k) * stride], acc[k]);
            }
        }
        return (acc[0] + acc[2]) + (acc[1] + acc[3]);
    }
    if (mode == LK_DOT_LANE8 || mode == LK_DOT_LANE8B) {
        double acc[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (npy_intp i = 0; i < G; i += 8) {
            for (int k = 0; k < 8 && i + k < G; k++) {
                acc[k] = fma(w[i + k], v[(i + k) * stride], acc[k]);
            }
        }
        return lane8_reduce(acc, mode);
    }
    double s = 0.0;
    for (npy_intp gi = 0; gi < G; gi++) {
        s += w[gi] * v[gi * stride];
    }
    return s;
}

/* The same reduction with the weight already folded in -- the second half of
 * the covariance collapse is a THREE-operand einsum, and einsum multiplies it
 * left to right, `(w * dm_i) * dm_j`, which is not `w * (dm_i * dm_j)`. */
LK_ALWAYS_INLINE double
bank_gsum1(const double *v, npy_intp G, int mode)
{
    if (mode == LK_DOT_LANES) {
        double acc[4] = {0.0, 0.0, 0.0, 0.0};
        for (npy_intp i = 0; i < G; i += 4) {
            for (int k = 0; k < 4 && i + k < G; k++) {
                acc[k] += v[i + k];
            }
        }
        return (acc[0] + acc[2]) + (acc[1] + acc[3]);
    }
    if (mode == LK_DOT_LANE8 || mode == LK_DOT_LANE8B) {
        double acc[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (npy_intp i = 0; i < G; i += 8) {
            for (int k = 0; k < 8 && i + k < G; k++) {
                acc[k] += v[i + k];
            }
        }
        return lane8_reduce(acc, mode);
    }
    double s = 0.0;
    for (npy_intp gi = 0; gi < G; gi++) {
        s += v[gi];
    }
    return s;
}

LK_ALWAYS_INLINE double
bank_dot(const double *a, const double *b, int n, int mode)
{
    if (mode == LK_DOT_LANES) {
        return lane_dot(a, b, n);
    }
    if (mode == LK_DOT_LANE8 || mode == LK_DOT_LANE8B) {
        double acc[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int i = 0; i < n; i += 8) {
            for (int k = 0; k < 8 && i + k < n; k++) {
                acc[k] = fma(a[i + k], b[i + k], acc[k]);
            }
        }
        return lane8_reduce(acc, mode);
    }
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
}

/* out[b,i,l] = sum_{j,k} F[b,i,j] P[b,j,k] F[b,l,k] -- the same contraction,
 * and the same two candidate orders, as `ap_block` above. */
static void
bank_fpft(const double *F, const double *P, double *out,
          npy_intp M, int n, int mode)
{
    const npy_intp nn = (npy_intp)n * n;
    for (npy_intp b = 0; b < M; b++) {
        ap_block_impl(F + b * nn, P + b * nn, out + b * nn, n, mode);
    }
}

/* bank_predict(F, P, Qg, rg, H, ap_mode)
 *     -> (Ppred (M,G,n,n), PHt (M,G,n,m), S (M,G,m,m))
 *
 *   FPFt  = einsum("bij,bjk,blk->bil", F, P, F)
 *   Ppred = FPFt[:, None] + Qg
 *   PHt   = einsum("bgij,kj->bgik", Ppred, H)
 *   S     = einsum("ij,bgjk->bgik", H, PHt);  S[..., d, d] += rg[..., d]
 */
static PyObject *
lucid_bank_predict(PyObject *self, PyObject *args)
{
    PyObject *oF, *oP, *oQg, *org, *oH;
    int ap_mode, dot_mode;
    if (!PyArg_ParseTuple(args, "OOOOOii", &oF, &oP, &oQg, &org, &oH,
                          &ap_mode, &dot_mode)) {
        return NULL;
    }
    PyArrayObject *aF = as_arr(oF), *aP = as_arr(oP), *aQg = as_arr(oQg);
    PyArrayObject *arg = as_arr(org), *aH = as_arr(oH);
    PyObject *ret = NULL;
    PyArrayObject *aPp = NULL, *aPHt = NULL, *aS = NULL;
    if (aF == NULL || aP == NULL || aQg == NULL || arg == NULL || aH == NULL) {
        goto done;
    }
    if (PyArray_NDIM(aQg) != 4 || PyArray_NDIM(aH) != 2
        || PyArray_NDIM(aF) != 3 || PyArray_NDIM(arg) != 3) {
        PyErr_SetString(PyExc_ValueError, "bank_predict: wrong ndim");
        goto done;
    }
    const npy_intp M = PyArray_DIM(aQg, 0);
    const npy_intp G = PyArray_DIM(aQg, 1);
    const int n = (int)PyArray_DIM(aQg, 2);
    const int m = (int)PyArray_DIM(aH, 0);
    if (PyArray_DIM(aF, 0) != M || PyArray_DIM(aF, 1) != n
        || PyArray_DIM(aP, 0) != M || PyArray_DIM(aP, 1) != n
        || PyArray_DIM(aH, 1) != n
        || PyArray_DIM(arg, 0) != M || PyArray_DIM(arg, 1) != G
        || PyArray_DIM(arg, 2) != m) {
        PyErr_SetString(PyExc_ValueError, "bank_predict: shapes disagree");
        goto done;
    }

    const npy_intp dPp[4] = {M, G, n, n};
    const npy_intp dPH[4] = {M, G, n, m};
    const npy_intp dS[4]  = {M, G, m, m};
    aPp  = (PyArrayObject *)PyArray_SimpleNew(4, (npy_intp *)dPp, NPY_DOUBLE);
    aPHt = (PyArrayObject *)PyArray_SimpleNew(4, (npy_intp *)dPH, NPY_DOUBLE);
    aS   = (PyArrayObject *)PyArray_SimpleNew(4, (npy_intp *)dS, NPY_DOUBLE);
    double *fp = (double *)malloc((size_t)M * n * n * sizeof(double));
    if (aPp == NULL || aPHt == NULL || aS == NULL || fp == NULL) {
        free(fp);
        PyErr_NoMemory();
        goto done;
    }

    const double *F = (const double *)PyArray_DATA(aF);
    const double *P = (const double *)PyArray_DATA(aP);
    const double *Qg = (const double *)PyArray_DATA(aQg);
    const double *rg = (const double *)PyArray_DATA(arg);
    const double *H = (const double *)PyArray_DATA(aH);
    double *Pp = (double *)PyArray_DATA(aPp);
    double *PHt = (double *)PyArray_DATA(aPHt);
    double *S = (double *)PyArray_DATA(aS);

    bank_fpft(F, P, fp, M, n, ap_mode);

    for (npy_intp b = 0; b < M; b++) {
        const double *fpb = fp + b * n * n;
        for (npy_intp gi = 0; gi < G; gi++) {
            const npy_intp off = (b * G + gi);
            const double *q = Qg + off * n * n;
            double *pp = Pp + off * n * n;
            for (int i = 0; i < n * n; i++) {
                pp[i] = fpb[i] + q[i];
            }
            /* PHt[i,k] = sum_j pp[i,j] * H[k,j] */
            double *ph = PHt + off * n * m;
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < m; k++) {
                    ph[i * m + k] = bank_dot(pp + i * n, H + k * n, n, dot_mode);
                }
            }
            /* S[i,k] = sum_j H[i,j] * PHt[j,k], then the diagonal gets rg */
            double *sb = S + off * m * m;
            if (m == 1) {                       /* PHt's j axis is contiguous too */
                sb[0] = bank_dot(H, ph, n, dot_mode);
            }
            else {
                for (int i = 0; i < m; i++) {
                    for (int k = 0; k < m; k++) {
                        double s = 0.0;
                        for (int j = 0; j < n; j++) {
                            s += H[i * n + j] * ph[j * m + k];
                        }
                        sb[i * m + k] = s;
                    }
                }
            }
            const double *rr = rg + off * m;
            for (int d = 0; d < m; d++) {
                sb[d * m + d] += rr[d];
            }
        }
    }
    free(fp);
    ret = Py_BuildValue("NNN", (PyObject *)aPp, (PyObject *)aPHt,
                        (PyObject *)aS);
    aPp = aPHt = aS = NULL;                     /* the tuple owns them now */
done:
    Py_XDECREF(aF); Py_XDECREF(aP); Py_XDECREF(aQg);
    Py_XDECREF(arg); Py_XDECREF(aH);
    Py_XDECREF(aPp); Py_XDECREF(aPHt); Py_XDECREF(aS);
    return ret;
}


/* bank_correct(Ppred, PHt, S, Si, logdet, e, mpred, H, pi, axwin, cap,
 *              mo, want_S, sc_mode)
 *     -> (w (M,G), ll (M,), m_new (M,n), P_new (M,n,n), Kbar (M,n,m), Sbar)
 *
 * `pi` is finished in place, as the Python does.  `cap` may be None, `Sbar` is
 * built only when `want_S`.
 */
static PyObject *
lucid_bank_correct(PyObject *self, PyObject *args)
{
    PyObject *oPp, *oPHt, *oSi, *oLd, *oe, *omp, *oH, *opi, *oax, *ocap;
    int mo, sc_mode, dot_mode, node_mode;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOiiii", &oPp, &oPHt, &oSi,
                          &oLd, &oe, &omp, &oH, &opi, &oax, &ocap,
                          &mo, &sc_mode, &dot_mode, &node_mode)) {
        return NULL;
    }
    if (sc_mode != LK_SC_NAIVE) {
        PyErr_SetString(PyExc_ValueError, "bank_correct: unsupported sc_mode");
        return NULL;
    }
    PyArrayObject *aPp = as_arr(oPp), *aPHt = as_arr(oPHt);
    PyArrayObject *aSi = as_arr(oSi), *aLd = as_arr(oLd), *ae = as_arr(oe);
    PyArrayObject *amp = as_arr(omp), *aH = as_arr(oH);
    PyArrayObject *acap = (ocap == Py_None) ? NULL : as_arr(ocap);
    PyArrayObject *api = NULL, *aax = NULL;
    PyObject *ret = NULL;
    PyArrayObject *aw = NULL, *all_ = NULL, *amn = NULL, *aPn = NULL;
    PyArrayObject *aKb = NULL;
    double *K = NULL, *lg = NULL, *logZ = NULL, *alpha = NULL, *buf = NULL;

    if (aPp == NULL || aPHt == NULL || aSi == NULL
        || aLd == NULL || ae == NULL || amp == NULL || aH == NULL
        || (ocap != Py_None && acap == NULL)) {
        goto done;
    }
    /* pi and axwin are written / read in place, so they must already be what
     * they claim to be rather than converted copies */
    if (!PyArray_Check(opi) || !PyArray_Check(oax)) {
        PyErr_SetString(PyExc_TypeError, "bank_correct: pi and axwin arrays");
        goto done;
    }
    api = (PyArrayObject *)opi;
    aax = (PyArrayObject *)oax;
    if (PyArray_TYPE(api) != NPY_DOUBLE || !PyArray_IS_C_CONTIGUOUS(api)
        || PyArray_TYPE(aax) != NPY_INTP || !PyArray_IS_C_CONTIGUOUS(aax)) {
        PyErr_SetString(PyExc_TypeError, "bank_correct: pi float64, axwin intp");
        goto done;
    }

    const npy_intp M = PyArray_DIM(aPp, 0);
    const npy_intp G = PyArray_DIM(aPp, 1);
    const int n = (int)PyArray_DIM(aPp, 2);
    const int m = (int)PyArray_DIM(aSi, 3);
    const npy_intp r = PyArray_DIM(api, 1);
    const npy_intp nw = PyArray_DIM(api, 2);
    if (PyArray_DIM(aax, 0) != r || PyArray_DIM(aax, 1) != nw
        || PyArray_DIM(api, 0) != M || mo != m) {
        PyErr_SetString(PyExc_ValueError, "bank_correct: shapes disagree");
        goto done;
    }

    const npy_intp dw[2] = {M, G}, dll[1] = {M}, dmn[2] = {M, n};
    const npy_intp dPn[3] = {M, n, n}, dKb[3] = {M, n, m};
    aw   = (PyArrayObject *)PyArray_SimpleNew(2, (npy_intp *)dw, NPY_DOUBLE);
    all_ = (PyArrayObject *)PyArray_SimpleNew(1, (npy_intp *)dll, NPY_DOUBLE);
    amn  = (PyArrayObject *)PyArray_SimpleNew(2, (npy_intp *)dmn, NPY_DOUBLE);
    aPn  = (PyArrayObject *)PyArray_SimpleNew(3, (npy_intp *)dPn, NPY_DOUBLE);
    aKb  = (PyArrayObject *)PyArray_SimpleNew(3, (npy_intp *)dKb, NPY_DOUBLE);
    K     = (double *)malloc((size_t)G * n * m * sizeof(double));
    lg    = (double *)malloc((size_t)G * sizeof(double));
    logZ  = (double *)malloc((size_t)(r ? r : 1) * sizeof(double));
    alpha = (double *)malloc((size_t)(r ? r : 1) * sizeof(double));
    buf   = (double *)malloc((size_t)(nw > (npy_intp)(m * n) ? nw : m * n)
                             * sizeof(double) + sizeof(double));
    if (aw == NULL || all_ == NULL || amn == NULL || aPn == NULL || aKb == NULL
        || K == NULL || lg == NULL || logZ == NULL
        || alpha == NULL || buf == NULL) {
        PyErr_NoMemory();
        goto done;
    }

    const double *Pp = (const double *)PyArray_DATA(aPp);
    const double *PHt = (const double *)PyArray_DATA(aPHt);
    const double *Si = (const double *)PyArray_DATA(aSi);
    const double *Ld = (const double *)PyArray_DATA(aLd);
    const double *e = (const double *)PyArray_DATA(ae);
    const double *mpred = (const double *)PyArray_DATA(amp);
    const double *H = (const double *)PyArray_DATA(aH);
    const double *cap = acap ? (const double *)PyArray_DATA(acap) : NULL;
    double *pi = (double *)PyArray_DATA(api);
    const npy_intp *axw = (const npy_intp *)PyArray_DATA(aax);
    double *W = (double *)PyArray_DATA(aw);
    double *LL = (double *)PyArray_DATA(all_);
    double *mnew = (double *)PyArray_DATA(amn);
    double *Pnew = (double *)PyArray_DATA(aPn);
    double *Kbar = (double *)PyArray_DATA(aKb);

    const npy_intp nn = (npy_intp)n * n, nm = (npy_intp)n * m, mm = (npy_intp)m * m;
    double *hpp = (double *)malloc((size_t)m * n * sizeof(double));
    double *Ppost = (double *)malloc((size_t)G * nn * sizeof(double));
    double *DM = (double *)malloc((size_t)G * n * sizeof(double));
    double *dd = (double *)malloc((size_t)G * sizeof(double));
    if (hpp == NULL || Ppost == NULL || DM == NULL || dd == NULL) {
        free(hpp); free(Ppost); free(DM); free(dd);
        PyErr_NoMemory();
        goto done;
    }

    for (npy_intp b = 0; b < M; b++) {
        const double *eb = e + b * m;
        /* the per-node log density */
        for (npy_intp gi = 0; gi < G; gi++) {
            const double *si = Si + (b * G + gi) * mm;
            double maha = 0.0;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    maha += eb[i] * si[i * m + j] * eb[j];
                }
            }
            /* `g_half_log2pi` is -log(2 pi)/2; both factors here are powers of
             * two, so recovering log(2 pi) from it is exact */
            lg[gi] = -0.5 * ((mo * (-2.0 * g_half_log2pi) + Ld[b * G + gi])
                             + maha);
        }
        /* per-axis: the window's softmax, and the axis's own evidence */
        double *wb = W + b * G;
        for (npy_intp gi = 0; gi < G; gi++) {
            wb[gi] = 0.0;
        }
        if (r) {
            for (npy_intp ax = 0; ax < r; ax++) {
                const npy_intp *idx = axw + ax * nw;
                double *pia = pi + (b * r + ax) * nw;
                for (npy_intp t = 0; t < nw; t++) {
                    buf[t] = lg[idx[t]];
                }
                const double mi = nk_max(buf, nw);
                for (npy_intp t = 0; t < nw; t++) {
                    buf[t] -= mi;
                }
                nk_exp(buf, buf, nw);
                for (npy_intp t = 0; t < nw; t++) {
                    buf[t] *= pia[t];
                }
                const double Zi = nk_sum(buf, nw);
                double lZ;
                nk_log(&Zi, &lZ, 1);
                for (npy_intp t = 0; t < nw; t++) {
                    pia[t] = buf[t] / Zi;
                }
                logZ[ax] = mi + lZ;
            }
            const double mz = nk_max(logZ, r);
            for (npy_intp ax = 0; ax < r; ax++) {
                alpha[ax] = logZ[ax] - mz;
            }
            nk_exp(alpha, alpha, r);
            const double asum = nk_sum(alpha, r);
            const double amean = asum / (double)r;
            double lmean;
            nk_log(&amean, &lmean, 1);
            LL[b] = mz + lmean;
            for (npy_intp ax = 0; ax < r; ax++) {
                alpha[ax] /= asum;
            }
            for (npy_intp ax = 0; ax < r; ax++) {
                const npy_intp *idx = axw + ax * nw;
                const double *pia = pi + (b * r + ax) * nw;
                for (npy_intp t = 0; t < nw; t++) {
                    wb[idx[t]] += alpha[ax] * pia[t];
                }
            }
        }
        else {
            LL[b] = lg[0];
            wb[0] = 1.0;
        }

        /* K, and the weighted mean gain */
        double *kb = Kbar + b * nm;
        for (npy_intp i = 0; i < nm; i++) {
            kb[i] = 0.0;
        }
        for (npy_intp gi = 0; gi < G; gi++) {
            const double *ph = PHt + (b * G + gi) * nm;
            const double *si = Si + (b * G + gi) * mm;
            double *kg = K + gi * nm;
            for (int i = 0; i < n; i++) {
                for (int l = 0; l < m; l++) {
                    double s = 0.0;
                    for (int k = 0; k < m; k++) {
                        s += ph[i * m + k] * si[k * m + l];
                    }
                    kg[i * m + l] = s;
                }
            }
        }
        for (npy_intp i = 0; i < nm; i++) {
            kb[i] = bank_gsum(wb, K + i, G, nm, node_mode);
        }
        const double *mpb = mpred + b * n;
        double *mnb = mnew + b * n;
        for (int i = 0; i < n; i++) {
            mnb[i] = mpb[i] + bank_dot(kb + i * m, eb, m, dot_mode);
        }

        /* the posterior covariance, node by node, and its two collapses */
        double *acc1 = Pnew + b * nn;
        for (npy_intp gi = 0; gi < G; gi++) {
            const double *pp = Pp + (b * G + gi) * nn;
            const double *kg = K + gi * nm;
            /* HPp[i,k] = sum_j H[i,j] Ppred[j,k] */
            for (int i = 0; i < m; i++) {
                for (int k = 0; k < n; k++) {
                    double s = 0.0;
                    for (int j = 0; j < n; j++) {
                        s += H[i * n + j] * pp[j * n + k];
                    }
                    hpp[i * n + k] = s;
                }
            }
            double *po = Ppost + gi * nn;
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < n; k++) {
                    double s = 0.0;
                    for (int l = 0; l < m; l++) {
                        s += kg[i * m + l] * hpp[l * n + k];
                    }
                    po[i * n + k] = pp[i * n + k] - s;
                }
            }
            double *dmg = DM + gi * n;
            for (int i = 0; i < n; i++) {
                dmg[i] = (mpb[i] + bank_dot(kg + i * m, eb, m, dot_mode))
                         - mnb[i];
            }
        }
        /* the two collapses, each a reduction over the nodes in its own right
         * -- the Python adds two finished arrays, so they are accumulated
         * apart and added once */
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                const npy_intp o = (npy_intp)i * n + k;
                for (npy_intp gi = 0; gi < G; gi++) {
                    dd[gi] = (wb[gi] * DM[gi * n + i]) * DM[gi * n + k];
                }
                acc1[o] = bank_gsum(wb, Ppost + o, G, nn, node_mode)
                          + bank_gsum1(dd, G, node_mode);
            }
        }
        /* symmetrise, then the class cap */
        for (int i = 0; i < n; i++) {
            for (int k = i; k < n; k++) {
                const double v = 0.5 * (acc1[i * n + k] + acc1[k * n + i]);
                acc1[i * n + k] = v;
                acc1[k * n + i] = v;
            }
        }
        if (cap != NULL) {
            for (int i = 0; i < n; i++) {
                const double d = acc1[i * n + i];
                buf[i] = 1.0;
                if (d > cap[i]) {
                    const double den = d > 1e-300 ? d : 1e-300;
                    buf[i] = sqrt(cap[i] / den);
                }
            }
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < n; k++) {
                    acc1[i * n + k] *= buf[i] * buf[k];
                }
            }
        }
    }
    free(hpp);
    free(Ppost);
    free(DM);
    free(dd);

    if (PyErr_Occurred()) {
        goto done;
    }
    ret = Py_BuildValue("NNNNN", (PyObject *)aw, (PyObject *)all_,
                        (PyObject *)amn, (PyObject *)aPn, (PyObject *)aKb);
    aw = all_ = amn = aPn = aKb = NULL;
done:
    free(K); free(lg); free(logZ); free(alpha); free(buf);
    Py_XDECREF(aPp); Py_XDECREF(aPHt); Py_XDECREF(aSi);
    Py_XDECREF(aLd); Py_XDECREF(ae); Py_XDECREF(amp); Py_XDECREF(aH);
    Py_XDECREF(acap);
    Py_XDECREF(aw); Py_XDECREF(all_); Py_XDECREF(amn); Py_XDECREF(aPn);
    Py_XDECREF(aKb);
    return ret;
}


static PyMethodDef methods[] = {
    {"configure", py_configure, METH_VARARGS,
     "configure(log2pi) -- hand over the module's own log(2 pi)."},
    {"lucid_bank_predict", lucid_bank_predict, METH_VARARGS,
     "lucid_bank_predict(F, P, Qg, rg, H, ap_mode, dot_mode)\n"
     "-> (Ppred, PHt, S) for a stacked bank of LucidFilter members."},
    {"lucid_bank_correct", lucid_bank_correct, METH_VARARGS,
     "lucid_bank_correct(Ppred, PHt, Si, logdet, e, mpred, H, pi, axwin,\n"
     "cap, mo, sc_mode, dot_mode, node_mode) -> (w, ll, m_new, P_new, Kbar);\n"
     "`pi` is finished in place."},
    {"np_prim", py_np_prim, METH_VARARGS,
     "np_prim(name, a[, b]) -- sum, max, dot, exp, log, lanedot: the\n"
     "primitives whose evaluation order the kernel has to match, exposed\n"
     "so the probe and the tests can check them against NumPy directly."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "lucid_kernel._kernel",
    "The compiled LucidFilter recursion.", -1, methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit__kernel(void)
{
    import_array();
    import_umath();

    g_exp_loop = find_double_loop("exp");
    g_log_loop = find_double_loop("log");
    if (g_exp_loop == NULL || g_log_loop == NULL) {
        return NULL;
    }
    PyObject *np = PyImport_ImportModule("numpy");
    if (np == NULL) {
        return NULL;
    }
    g_einsum = PyObject_GetAttrString(np, "einsum");
    Py_DECREF(np);
    if (g_einsum == NULL) {
        return NULL;
    }
    PyObject *mod = PyModule_Create(&moduledef);
    if (mod == NULL) {
        return NULL;
    }
    if (PyModule_AddIntConstant(mod, "ABI", LUCID_KERNEL_ABI) < 0) {
        Py_DECREF(mod);
        return NULL;
    }
    return mod;
}
