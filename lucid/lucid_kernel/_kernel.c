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
static PyObject *g_mp_spec = NULL;      /* "bgxz,bgz->bgx" */

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

/* `np.maximum(x, floor)` elementwise, NaN-propagating. */
static inline double
nk_clip_lo(double x, double floor_)
{
    if (isnan(x)) {
        return x;
    }
    return x > floor_ ? x : floor_;
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

/* v @ M for v of length n and M of shape (n, k), into out (length k). */
static int
nk_vecmat(const double *v, PyObject *M, npy_intp n, double *out, npy_intp k)
{
    npy_intp dv[1] = {n};
    npy_intp dk[1] = {k};
    PyObject *V = shell(v, 1, dv);
    PyObject *O = shell(out, 1, dk);
    int rc = -1;
    if (V != NULL && O != NULL) {
        PyObject *r = PyArray_MatrixProduct2(V, M, (PyArrayObject *)O);
        if (r != NULL) {
            Py_DECREF(r);
            rc = 0;
        }
    }
    Py_XDECREF(V);
    Py_XDECREF(O);
    return rc;
}

/* ------------------------------------------------------- the mp contraction
 *
 * `mp = np.einsum("bgxz,bgz->bgx", Fg, m0)` is the one contraction in the
 * recursion whose reduction axis is contiguous in BOTH operands with a
 * stride-0 output, which is the shape that sends NumPy's einsum down a
 * vectorised path: the products go into SIMD lanes and come back through a
 * butterfly, so the sum is not left-to-right.  For a reduction of length p
 * the lanes hold (t0..t_{p-1}, 0, ...) and the butterfly returns
 * (t0 + t2) + (t1 + t3) -- the same answer whether the machine has four lanes
 * or eight, which is why `lane_dot` below is not CPU-dependent for p <= 4.
 *
 * Above p = 4 the two lane widths part company and neither matches what NumPy
 * actually produces, so for p >= 5 the kernel stops guessing and calls
 * `np.einsum` itself, once per time step for the whole batch.  That is one
 * Python call against O(B G p^2) arithmetic and does not show up in the
 * timings; correctness is by construction rather than by transcription.
 *
 * `lane_dot` uses fma() deliberately: the lane accumulator is a fused
 * multiply-add on every machine that has one, and the C library's fma() is
 * the same correctly-rounded operation.
 */
#define LK_MP_LANES  0
#define LK_MP_EINSUM 1

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

/* ---------------------------------------------------------------- workspace */

typedef struct {
    npy_intp G;
    int p;
    double *pin;     /* (G,)      pi_pred                                   */
    double *muT;     /* (G, G)    P(came from i | now j), transposed        */
    double *m0;      /* (G, p)    mixed state                               */
    double *P0;      /* (G, p, p) mixed covariance                          */
    double *accA;    /* (p, p)    the two halves of P0 before they are added, */
    double *accB;    /* (p, p)    for orders too large to keep on the stack   */
    double *mp;      /* (G, p)    prior mean                                */
    double *Ap;      /* (G, p, p) prior covariance                          */
    double *S;       /* (G,)      predictive variance                       */
    double *lg;      /* (G,)      per-node log density                      */
    double *w;       /* (G,)      unnormalised posterior weights            */
    double *tmp;     /* (G,)      scratch for the reported dot products     */
    double *tmp2;    /* (G,)      the prior part of Aj[0, 0], for share_prior */
    double *K;       /* (p,)      the gain of one node                       */
} OdeWS;

static void
ode_ws_free(void *vws)
{
    OdeWS *ws = (OdeWS *)vws;
    if (ws == NULL) {
        return;
    }
    free(ws->pin);
    free(ws->muT);
    free(ws->m0);
    free(ws->P0);
    free(ws->accA);
    free(ws->accB);
    free(ws->mp);
    free(ws->Ap);
    free(ws->S);
    free(ws->lg);
    free(ws->w);
    free(ws->tmp);
    free(ws->tmp2);
    free(ws->K);
    free(ws);
}

static void *
ode_ws_alloc(npy_intp G, int p)
{
    OdeWS *ws = (OdeWS *)calloc(1, sizeof(OdeWS));
    if (ws == NULL) {
        return NULL;
    }
    ws->G = G;
    ws->p = p;
    ws->pin  = (double *)malloc((size_t)G * sizeof(double));
    ws->muT  = (double *)malloc((size_t)G * G * sizeof(double));
    ws->m0   = (double *)malloc((size_t)G * p * sizeof(double));
    ws->P0   = (double *)malloc((size_t)G * p * p * sizeof(double));
    ws->accA = (double *)malloc((size_t)p * p * sizeof(double));
    ws->accB = (double *)malloc((size_t)p * p * sizeof(double));
    ws->mp   = (double *)malloc((size_t)G * p * sizeof(double));
    ws->Ap   = (double *)malloc((size_t)G * p * p * sizeof(double));
    ws->S    = (double *)malloc((size_t)G * sizeof(double));
    ws->lg   = (double *)malloc((size_t)G * sizeof(double));
    ws->w    = (double *)malloc((size_t)G * sizeof(double));
    ws->tmp  = (double *)malloc((size_t)G * sizeof(double));
    ws->tmp2 = (double *)malloc((size_t)G * sizeof(double));
    ws->K    = (double *)malloc((size_t)p * sizeof(double));
    if (ws->pin == NULL || ws->muT == NULL
        || ws->m0 == NULL || ws->accA == NULL
        || ws->accB == NULL
        || ws->P0 == NULL || ws->mp == NULL || ws->Ap == NULL || ws->S == NULL
        || ws->lg == NULL || ws->w == NULL || ws->tmp == NULL
        || ws->tmp2 == NULL || ws->K == NULL) {
        ode_ws_free(ws);
        return NULL;
    }
    return ws;
}

/* ------------------------------------------------------- the shared stages
 *
 * Both recursions -- the batched one that a fit evaluates and the streaming
 * one that `update()` runs -- share their first half: mix the nodes by the
 * chain kernel, then push each node through its own transition.  The two
 * differ in how they weight the nodes afterwards and, unavoidably, in which
 * of NumPy's summation orders they inherit, so the shared part is written
 * once here and the halves that differ are written out separately below.
 */

/* mu, m0 and P0 from (pi_pred, pi, m, P).  `pin` must already hold pi_pred.
 *
 * The loops are ordered for locality, not written out the way the Python
 * reads.  What has to hold for the sums to be NumPy's is that each output
 * element accumulates over the nodes in ascending order; within that freedom
 * the node index is the OUTER loop here, so `mu` is walked along its rows --
 * the direction it is stored in -- rather than down its columns.  The two
 * halves of P0 are carried in separate accumulators because the Python adds
 * two finished arrays, and adding as it goes would be a different number.
 *
 * `dmix` is formed on the fly rather than materialised: it is a subtraction,
 * so recomputing it is exact, and the (G, G, p) array it would need costs
 * more in memory traffic than the arithmetic it saves.
 */
LK_ALWAYS_INLINE void
mix_nodes_impl(OdeWS *ws, const double *T, const double *pi, const double *m,
               const double *P, const int p)
{
    const npy_intp G = ws->G;
    const int pp = p * p;
    double *restrict muT = ws->muT;
    double *restrict m0 = ws->m0;

    /* mu = pi[:, None] * T / pi_pred[None, :], then transposed.  The
     * transpose is the point: every sum below runs over the node index, and
     * running it along a row instead of down a column is the difference
     * between the accumulators living in registers and living in memory.  It
     * changes nothing about the arithmetic -- each output element still
     * accumulates over the nodes in ascending order, which is what makes the
     * sum NumPy's. */
    for (npy_intp i = 0; i < G; i++) {
        const double c = pi[i];
        const double *restrict Ti = T + i * G;
        for (npy_intp j = 0; j < G; j++) {
            muT[j * G + i] = c * Ti[j] / ws->pin[j];
        }
    }

    for (npy_intp j = 0; j < G; j++) {
        const double *restrict muj = muT + j * G;
        double *restrict m0j = m0 + j * p;
        for (int x = 0; x < p; x++) {
            m0j[x] = 0.0;
        }
        for (npy_intp i = 0; i < G; i++) {
            const double c = muj[i];
            const double *restrict mi = m + i * p;
            for (int x = 0; x < p; x++) {
                m0j[x] += c * mi[x];
            }
        }
        /* The two halves of P0 are carried separately because the Python adds
         * two finished arrays; adding as it goes would be a different number. */
        double aA[36] = {0.0};
        double aB[36] = {0.0};
        double dx[6];
        double *restrict pA = aA;
        double *restrict pB = aB;
        double *restrict pdx = dx;
        if (p > 6) {                            /* the fixed-size fast path is
                                                 * for the orders that occur */
            pA = ws->accA;
            pB = ws->accB;
            pdx = ws->K;
            for (int k = 0; k < pp; k++) {
                pA[k] = 0.0;
                pB[k] = 0.0;
            }
        }
        for (npy_intp i = 0; i < G; i++) {
            const double c = muj[i];
            const double *restrict mi = m + i * p;
            const double *restrict Pi = P + i * pp;
            for (int k = 0; k < pp; k++) {
                pA[k] += c * Pi[k];
            }
            for (int x = 0; x < p; x++) {
                pdx[x] = mi[x] - m0j[x];
            }
            for (int x = 0; x < p; x++) {
                const double cdx = c * pdx[x];
                for (int z = 0; z < p; z++) {
                    pB[x * p + z] += cdx * pdx[z];
                }
            }
        }
        double *restrict P0j = ws->P0 + j * pp;
        for (int k = 0; k < pp; k++) {
            P0j[k] = pA[k] + pB[k];
        }
    }
}

/* The recurrence order is a handful of small integers in practice and the
 * inner loops are only a few iterations long, so the loop bounds are worth
 * more to the compiler as constants than the code duplication costs: at p = 1
 * the specialised form is the difference between a win and no win at all. */
static void
mix_nodes(OdeWS *ws, const double *T, const double *pi, const double *m,
          const double *P)
{
    switch (ws->p) {
    case 1: mix_nodes_impl(ws, T, pi, m, P, 1); return;
    case 2: mix_nodes_impl(ws, T, pi, m, P, 2); return;
    case 3: mix_nodes_impl(ws, T, pi, m, P, 3); return;
    case 4: mix_nodes_impl(ws, T, pi, m, P, 4); return;
    case 5: mix_nodes_impl(ws, T, pi, m, P, 5); return;
    case 6: mix_nodes_impl(ws, T, pi, m, P, 6); return;
    default: mix_nodes_impl(ws, T, pi, m, P, ws->p); return;
    }
}

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

static void
ap_block(const double *F, const double *P0, double *A, int p, int mode)
{
    switch (p) {                                /* see mix_nodes */
    case 1: ap_block_impl(F, P0, A, 1, mode); return;
    case 2: ap_block_impl(F, P0, A, 2, mode); return;
    case 3: ap_block_impl(F, P0, A, 3, mode); return;
    case 4: ap_block_impl(F, P0, A, 4, mode); return;
    case 5: ap_block_impl(F, P0, A, 5, mode); return;
    default: ap_block_impl(F, P0, A, p, mode); return;
    }
}

/* Ap[g] = Fg[g] P0[g] Fg[g]^T, with the process noise added to the top-left
 * and S = Ap[g, 0, 0] + Rg[g].  `mp` is filled by the caller (it is the one
 * contraction that has to be done for the whole batch at once). */
static void
propagate_cov(OdeWS *ws, const double *Fg, const double *Qg, const double *Rg,
              int ap_mode)
{
    const npy_intp G = ws->G;
    const int pp = ws->p * ws->p;

    for (npy_intp g = 0; g < G; g++) {
        double *A = ws->Ap + g * pp;
        ap_block(Fg + g * pp, ws->P0 + g * pp, A, ws->p, ap_mode);
        A[0] += Qg[g];
        ws->S[g] = A[0] + Rg[g];
    }
}

/* mp[g, x] = sum_z Fg[g, x, z] * m0[g, z], for one grid. */
static void
mp_lanes(OdeWS *ws, const double *Fg)
{
    const npy_intp G = ws->G;
    const int p = ws->p;
    for (npy_intp g = 0; g < G; g++) {
        for (int x = 0; x < p; x++) {
            ws->mp[g * p + x] = lane_dot(Fg + g * p * p + x * p,
                                         ws->m0 + g * p, p);
        }
    }
}

/* ============================================================================
 *                       the batched likelihood
 * ==========================================================================*/

/* One time step, for every parameter vector at once.  The arrays are the
 * batch's, laid out (B, ...) exactly as `_loglik_batch` lays them out; the
 * per-row work is independent, which is why it can be done row by row and
 * still give what NumPy's whole-batch expressions give. */
typedef struct {
    npy_intp B, G;
    int p;
    double *pi, *m, *P;         /* (B, G), (B, G, p), (B, G, p, p) */
    double *pin, *mp, *Ap, *S, *lg, *w, *tmp;
    double *muT, *m0, *P0, *accA, *accB;
    double *mx, *Z, *ll, *K;
    char *dead;
} BatchWS;

static void
batch_free(BatchWS *b)
{
    if (b == NULL) {
        return;
    }
    free(b->pi); free(b->m); free(b->P); free(b->pin); free(b->mp);
    free(b->Ap); free(b->S); free(b->lg); free(b->w); free(b->tmp);
    free(b->muT); free(b->m0); free(b->P0); free(b->accA); free(b->accB);
    free(b->mx); free(b->Z); free(b->ll); free(b->K); free(b->dead);
    free(b);
}

static BatchWS *
batch_alloc(npy_intp B, npy_intp G, int p)
{
    BatchWS *b = (BatchWS *)calloc(1, sizeof(BatchWS));
    if (b == NULL) {
        return NULL;
    }
    b->B = B; b->G = G; b->p = p;
    const size_t sd = sizeof(double);
    b->pi   = (double *)malloc((size_t)B * G * sd);
    b->m    = (double *)malloc((size_t)B * G * p * sd);
    b->P    = (double *)malloc((size_t)B * G * p * p * sd);
    b->pin  = (double *)malloc((size_t)B * G * sd);
    b->mp   = (double *)malloc((size_t)B * G * p * sd);
    b->Ap   = (double *)malloc((size_t)B * G * p * p * sd);
    b->S    = (double *)malloc((size_t)B * G * sd);
    b->lg   = (double *)malloc((size_t)B * G * sd);
    b->w    = (double *)malloc((size_t)B * G * sd);
    b->tmp  = (double *)malloc((size_t)B * G * sd);
    b->muT  = (double *)malloc((size_t)G * G * sd);
    b->m0   = (double *)malloc((size_t)B * G * p * sd);
    b->P0   = (double *)malloc((size_t)G * p * p * sd);
    b->accA = (double *)malloc((size_t)p * p * sd);
    b->accB = (double *)malloc((size_t)p * p * sd);
    b->mx   = (double *)malloc((size_t)B * sd);
    b->Z    = (double *)malloc((size_t)B * sd);
    b->ll   = (double *)malloc((size_t)B * sd);
    b->K    = (double *)malloc((size_t)p * sd);
    b->dead = (char *)calloc((size_t)B, 1);
    if (b->pi == NULL || b->m == NULL || b->P == NULL || b->pin == NULL
        || b->mp == NULL || b->Ap == NULL || b->S == NULL || b->lg == NULL
        || b->w == NULL || b->tmp == NULL || b->muT == NULL || b->m0 == NULL
        || b->accA == NULL || b->accB == NULL
        || b->P0 == NULL || b->mx == NULL || b->Z == NULL
        || b->ll == NULL || b->K == NULL || b->dead == NULL) {
        batch_free(b);
        return NULL;
    }
    return b;
}

/* mp for the whole batch.  p <= 4 goes through the lane model; above that,
 * through np.einsum, for the reason given at the top. */
static int
batch_mp(BatchWS *b, PyObject *Fg_obj, const double *Fg, PyObject *m0_arr,
         int mp_mode)
{
    const npy_intp B = b->B, G = b->G;
    const int p = b->p, pp = p * p;

    if (mp_mode == LK_MP_LANES) {
        for (npy_intp bb = 0; bb < B; bb++) {
            const double *F = Fg + bb * G * pp;
            const double *m0 = b->m0 + bb * G * p;
            double *mp = b->mp + bb * G * p;
            for (npy_intp g = 0; g < G; g++) {
                for (int x = 0; x < p; x++) {
                    mp[g * p + x] = lane_dot(F + g * pp + x * p, m0 + g * p, p);
                }
            }
        }
        return 0;
    }
    PyObject *args[3] = {g_mp_spec, Fg_obj, m0_arr};
    PyObject *r = PyObject_Vectorcall(
        g_einsum, args, 3 | PY_VECTORCALL_ARGUMENTS_OFFSET, NULL);
    if (r == NULL) {
        return -1;
    }
    memcpy(b->mp, PyArray_DATA((PyArrayObject *)r),
           (size_t)B * G * p * sizeof(double));
    Py_DECREF(r);
    return 0;
}

static PyObject *
ode_loglik_batch(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyObject *y_o, *T_o, *pi0_o, *Qg_o, *Rg_o, *Fg_o;
    int p, ap_mode, mp_mode;
    if (!PyArg_ParseTuple(args, "OOOOOOiii", &y_o, &T_o, &pi0_o, &Qg_o, &Rg_o,
                          &Fg_o, &p, &ap_mode, &mp_mode)) {
        return NULL;
    }
    PyArrayObject *y_a = (PyArrayObject *)PyArray_FROM_OTF(
        y_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *T_a = (PyArrayObject *)PyArray_FROM_OTF(
        T_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *pi0_a = (PyArrayObject *)PyArray_FROM_OTF(
        pi0_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *Qg_a = (PyArrayObject *)PyArray_FROM_OTF(
        Qg_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *Rg_a = (PyArrayObject *)PyArray_FROM_OTF(
        Rg_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *Fg_a = (PyArrayObject *)PyArray_FROM_OTF(
        Fg_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (y_a == NULL || T_a == NULL || pi0_a == NULL || Qg_a == NULL
        || Rg_a == NULL || Fg_a == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(T_a) != 3 || PyArray_NDIM(pi0_a) != 2
        || PyArray_NDIM(Fg_a) != 4 || PyArray_NDIM(y_a) != 1) {
        PyErr_SetString(PyExc_ValueError, "kernel: bad array rank");
        goto fail;
    }
    const npy_intp B = PyArray_DIM(pi0_a, 0);
    const npy_intp G = PyArray_DIM(pi0_a, 1);
    const npy_intp n = PyArray_DIM(y_a, 0);
    if (PyArray_DIM(T_a, 0) != B || PyArray_DIM(T_a, 1) != G
        || PyArray_DIM(T_a, 2) != G || PyArray_DIM(Fg_a, 0) != B
        || PyArray_DIM(Fg_a, 1) != G || PyArray_DIM(Fg_a, 2) != p
        || PyArray_DIM(Fg_a, 3) != p) {
        PyErr_SetString(PyExc_ValueError, "kernel: inconsistent shapes");
        goto fail;
    }

    const double *y = (const double *)PyArray_DATA(y_a);
    const double *T = (const double *)PyArray_DATA(T_a);
    const double *pi0 = (const double *)PyArray_DATA(pi0_a);
    const double *Qg = (const double *)PyArray_DATA(Qg_a);
    const double *Rg = (const double *)PyArray_DATA(Rg_a);
    const double *Fg = (const double *)PyArray_DATA(Fg_a);

    BatchWS *b = batch_alloc(B, G, p);
    if (b == NULL) {
        PyErr_NoMemory();
        goto fail;
    }
    PyObject *m0_arr = NULL;
    if (mp_mode == LK_MP_EINSUM) {
        npy_intp d3[3] = {B, G, p};
        m0_arr = shell(b->m0, 3, d3);
        if (m0_arr == NULL) {
            batch_free(b);
            goto fail;
        }
    }

    const int pp = p * p;
    const double y0 = (n > 0 && isfinite(y[0])) ? y[0] : 0.0;

    /* the diffuse start: eye(p) * ((Rg.max(1) + Qg.max(1)) * p) */
    memcpy(b->pi, pi0, (size_t)B * G * sizeof(double));
    for (npy_intp bb = 0; bb < B; bb++) {
        const double c = (nk_max(Rg + bb * G, G) + nk_max(Qg + bb * G, G))
                         * (double)p;
        for (npy_intp g = 0; g < G; g++) {
            for (int x = 0; x < p; x++) {
                b->m[(bb * G + g) * p + x] = y0;
                for (int z = 0; z < p; z++) {
                    b->P[(bb * G + g) * pp + x * p + z] = (x == z) ? c : 0.0;
                }
            }
        }
        b->ll[bb] = 0.0;
    }

    OdeWS scratch;
    scratch.G = G;
    scratch.p = p;
    scratch.muT = b->muT;
    scratch.P0 = b->P0;
    scratch.accA = b->accA;
    scratch.accB = b->accB;
    scratch.K = b->K;

    int rc = 0;
    for (npy_intp t = 0; t < n && rc == 0; t++) {
        const double v = y[t];
        const int v_finite = isfinite(v);

        /* Mixing and propagation are per row and share one scratch P0, so
         * each row's Ap is formed before the next row touches it. */
        for (npy_intp bb = 0; bb < B; bb++) {
            const double *Tb = T + bb * G * G;
            double *pi = b->pi + bb * G;
            double *pin = b->pin + bb * G;
            /* pi_pred = maximum(einsum("bi,bij->bj", pi, T), 1e-300) */
            for (npy_intp j = 0; j < G; j++) {
                pin[j] = 0.0;
            }
            for (npy_intp i = 0; i < G; i++) {
                const double c = pi[i];
                const double *Ti = Tb + i * G;
                for (npy_intp j = 0; j < G; j++) {
                    pin[j] += c * Ti[j];
                }
            }
            for (npy_intp j = 0; j < G; j++) {
                pin[j] = nk_clip_lo(pin[j], 1e-300);
            }
            scratch.pin = pin;
            scratch.m0 = b->m0 + bb * G * p;
            scratch.Ap = b->Ap + bb * G * pp;
            scratch.S = b->S + bb * G;
            mix_nodes(&scratch, Tb, pi, b->m + bb * G * p, b->P + bb * G * pp);
            propagate_cov(&scratch, Fg + bb * G * pp, Qg + bb * G, Rg + bb * G,
                          ap_mode);
        }

        if (batch_mp(b, (PyObject *)Fg_a, Fg, m0_arr, mp_mode) != 0) {
            rc = -1;
            break;
        }

        if (!v_finite) {
            /* missing: mix and propagate, no correction */
            memcpy(b->pi, b->pin, (size_t)B * G * sizeof(double));
            memcpy(b->m, b->mp, (size_t)B * G * p * sizeof(double));
            memcpy(b->P, b->Ap, (size_t)B * G * pp * sizeof(double));
        }
        else {
            for (npy_intp bb = 0; bb < B; bb++) {
                double *S = b->S + bb * G;
                int bad = 0;
                for (npy_intp g = 0; g < G; g++) {
                    if (!isfinite(S[g])) {
                        bad = 1;
                    }
                    if (!(S[g] > 0.0)) {
                        bad = 1;
                    }
                }
                if (bad) {
                    b->dead[bb] = 1;
                }
                for (npy_intp g = 0; g < G; g++) {
                    if (!(isfinite(S[g]) && S[g] > 0.0)) {
                        S[g] = 1.0;
                    }
                }
            }
            /* lg = -0.5 * (log(S) + e * e / S) */
            nk_log(b->S, b->tmp, B * G);
            for (npy_intp bb = 0; bb < B; bb++) {
                const double *mp = b->mp + bb * G * p;
                const double *S = b->S + bb * G;
                double *lg = b->lg + bb * G;
                const double *ls = b->tmp + bb * G;
                for (npy_intp g = 0; g < G; g++) {
                    const double e = v - mp[g * p];
                    lg[g] = -0.5 * (ls[g] + e * e / S[g]);
                }
                b->mx[bb] = nk_max(lg, G);
            }
            for (npy_intp bb = 0; bb < B; bb++) {
                const double mx = b->mx[bb];
                const double *lg = b->lg + bb * G;
                double *d = b->tmp + bb * G;
                for (npy_intp g = 0; g < G; g++) {
                    d[g] = lg[g] - mx;
                }
            }
            nk_exp(b->tmp, b->w, B * G);
            for (npy_intp bb = 0; bb < B; bb++) {
                const double *pin = b->pin + bb * G;
                double *w = b->w + bb * G;
                for (npy_intp g = 0; g < G; g++) {
                    w[g] = pin[g] * w[g];
                }
                b->Z[bb] = nk_sum(w, G);
            }
            nk_log(b->Z, b->tmp, B);
            for (npy_intp bb = 0; bb < B; bb++) {
                b->ll[bb] += b->tmp[bb] + b->mx[bb] + g_half_log2pi;
            }
            for (npy_intp bb = 0; bb < B; bb++) {
                const double *mp = b->mp + bb * G * p;
                const double *Ap = b->Ap + bb * G * pp;
                const double *S = b->S + bb * G;
                const double *w = b->w + bb * G;
                const double Z = b->Z[bb];
                double *m = b->m + bb * G * p;
                double *P = b->P + bb * G * pp;
                double *pi = b->pi + bb * G;
                double *K = b->K;
                for (npy_intp g = 0; g < G; g++) {
                    const double e = v - mp[g * p];
                    const double s = S[g];
                    for (int x = 0; x < p; x++) {
                        K[x] = Ap[g * pp + x * p] / s;
                    }
                    for (int x = 0; x < p; x++) {
                        m[g * p + x] = mp[g * p + x] + K[x] * e;
                        for (int z = 0; z < p; z++) {
                            P[g * pp + x * p + z] =
                                Ap[g * pp + x * p + z] - K[x] * Ap[g * pp + z];
                        }
                    }
                    pi[g] = w[g] / Z;
                }
            }
        }

        /* the dead rows are pulled back to a representable state, exactly as
         * the `np.where` block does, so that one dead row cannot poison the
         * arithmetic of the live ones */
        for (npy_intp bb = 0; bb < B; bb++) {
            if (!b->dead[bb]) {
                continue;
            }
            for (npy_intp g = 0; g < G; g++) {
                b->pi[bb * G + g] = 1.0 / (double)G;
                for (int x = 0; x < p; x++) {
                    b->m[(bb * G + g) * p + x] = y0;
                    for (int z = 0; z < p; z++) {
                        b->P[(bb * G + g) * pp + x * p + z] =
                            (x == z) ? 1.0 : 0.0;
                    }
                }
            }
        }
    }

    PyObject *out = NULL;
    if (rc == 0) {
        npy_intp dB[1] = {B};
        out = PyArray_SimpleNew(1, dB, NPY_DOUBLE);
        if (out != NULL) {
            double *o = (double *)PyArray_DATA((PyArrayObject *)out);
            for (npy_intp bb = 0; bb < B; bb++) {
                o[bb] = (b->dead[bb] || !isfinite(b->ll[bb])) ? -INFINITY
                                                              : b->ll[bb];
            }
        }
    }
    Py_XDECREF(m0_arr);
    batch_free(b);
    Py_DECREF(y_a); Py_DECREF(T_a); Py_DECREF(pi0_a);
    Py_DECREF(Qg_a); Py_DECREF(Rg_a); Py_DECREF(Fg_a);
    return out;

fail:
    Py_XDECREF(y_a); Py_XDECREF(T_a); Py_XDECREF(pi0_a);
    Py_XDECREF(Qg_a); Py_XDECREF(Rg_a); Py_XDECREF(Fg_a);
    return NULL;
}

/* ============================================================================
 *                       the streaming recursion
 * ==========================================================================*/
/* `OdeFilter._update_imm`, step for step.  It is a separate transcription
 * from the batched one above and not a special case of it, because the Python
 * is: `_loglik_batch` contracts `pi` against the kernel with `np.einsum` and
 * this one writes `pi @ T`, which is BLAS.  The two are the same recursion and
 * the same model, and they agree to round-off, but they do not agree bit for
 * bit -- so each has to be matched against the one it stands in for. */

/* A length-n float64 view over caller memory with an explicit stride, for the
 * two dot products whose right-hand operand is a column of an (G, p) array.
 * NumPy hands the stride to BLAS rather than copying, and BLAS sums a strided
 * vector in a different order than a contiguous one. */
static PyObject *
shell_strided(const double *data, npy_intp n, npy_intp stride_bytes)
{
    npy_intp dims[1] = {n};
    npy_intp strides[1] = {stride_bytes};
    return PyArray_New(&PyArray_Type, 1, dims, NPY_DOUBLE, strides,
                       (void *)data, 0, 0, NULL);
}

static double
nk_dot_strided(const double *a, const double *b, npy_intp n,
               npy_intp b_stride_bytes)
{
    npy_intp dims[1] = {n};
    return nk_dot2(shell(a, 1, dims), shell_strided(b, n, b_stride_bytes));
}

/* mp for one grid: the lane model up to p = 4, np.einsum above it. */
static PyObject *g_mp_spec1 = NULL;      /* "gxz,gz->gx"   */
static PyObject *g_sc_spec1 = NULL;      /* "g,gxz->xz"    */
static PyObject *g_sc_spec2 = NULL;      /* "g,gx,gz->xz"  */

/* The reporting collapse `filter()` does per step.  Same story as the two
 * contractions above: at p = 1 the per-node covariance is contiguous in g,
 * which is the shape that sends einsum down its vectorised reduction, and
 * above p = 1 it is not and the sum is left to right.  Probed, not assumed. */
#define LK_SC_NAIVE  0
#define LK_SC_EINSUM 1

static int
mp_contract(OdeWS *ws, const double *Fg, int mp_mode)
{
    const npy_intp G = ws->G;
    const int p = ws->p;
    if (mp_mode == LK_MP_LANES) {
        mp_lanes(ws, Fg);
        return 0;
    }
    npy_intp d3[3] = {G, p, p};
    npy_intp d2[2] = {G, p};
    PyObject *F = shell(Fg, 3, d3);
    PyObject *M = shell(ws->m0, 2, d2);
    int rc = -1;
    if (F != NULL && M != NULL) {
        PyObject *args[3] = {g_mp_spec1, F, M};
        PyObject *r = PyObject_Vectorcall(
            g_einsum, args, 3 | PY_VECTORCALL_ARGUMENTS_OFFSET, NULL);
        if (r != NULL) {
            memcpy(ws->mp, PyArray_DATA((PyArrayObject *)r),
                   (size_t)G * p * sizeof(double));
            Py_DECREF(r);
            rc = 0;
        }
    }
    Py_XDECREF(F);
    Py_XDECREF(M);
    return rc;
}

static void
ode_start(const LucidOdeGrid *g, LucidOdeState *st, double y0)
{
    const npy_intp G = g->G;
    const int p = g->p, pp = p * p;
    const double c = (nk_max(g->Rg, G) + nk_max(g->Qg, G)) * (double)p;
    memcpy(st->pi, g->pi0, (size_t)G * sizeof(double));
    for (npy_intp i = 0; i < G; i++) {
        for (int x = 0; x < p; x++) {
            st->m[i * p + x] = y0;
            for (int z = 0; z < p; z++) {
                st->P[i * pp + x * p + z] = (x == z) ? c : 0.0;
            }
        }
    }
    st->started = 1;
}

static double
ode_whiteness(const LucidOdeState *st)
{
    if (st->nw < 3 || st->e2 <= 0.0) {
        return 0.0;
    }
    return st->ee / st->e2;
}

static int
ode_update(void *vws, const LucidOdeGrid *g, LucidOdeState *st, double y,
           double phi_P, double phi_M, int want_step, LucidOdeStep *step)
{
    OdeWS *ws = (OdeWS *)vws;
    const npy_intp G = g->G;
    const int p = g->p, pp = p * p;

    npy_intp dTT[2] = {G, G};
    PyObject *T_obj = shell(g->T, 2, dTT);
    if (T_obj == NULL) {
        return -1;
    }
    /* pi_pred = maximum(pi @ T, 1e-300) */
    if (nk_vecmat(st->pi, T_obj, G, ws->pin, G) != 0) {
        Py_DECREF(T_obj);
        return -1;
    }
    Py_DECREF(T_obj);
    for (npy_intp j = 0; j < G; j++) {
        ws->pin[j] = nk_clip_lo(ws->pin[j], 1e-300);
    }

    mix_nodes(ws, g->T, st->pi, st->m, st->P);
    if (mp_contract(ws, g->Fg, g->mp_mode) != 0) {
        return -1;
    }

    /* Aj, with the prior part of the top-left kept before Q goes in: it is
     * what `share_prior` reports. */
    for (npy_intp i = 0; i < G; i++) {
        double *A = ws->Ap + i * pp;
        ap_block(g->Fg + i * pp, ws->P0 + i * pp, A, p, g->ap_mode);
        ws->tmp2[i] = A[0];                     /* a00 */
        A[0] += g->Qg[i];
        ws->S[i] = A[0] + g->Rg[i];
    }

    if (!isfinite(y)) {
        /* missing: mix and propagate */
        const double lamP = nk_dot(ws->pin, g->LP, G);
        const double lamM = nk_dot(ws->pin, g->LM, G);
        const double mbar = nk_dot_strided(ws->pin, ws->mp, G,
                                           (npy_intp)(p * sizeof(double)));
        for (npy_intp i = 0; i < G; i++) {
            const double d = ws->mp[i * p] - mbar;
            ws->tmp[i] = ws->Ap[i * pp] + d * d;
        }
        const double vbar = nk_dot(ws->pin, ws->tmp, G);
        const double dyn = 1.0 + nk_dot(ws->pin, g->LA, G);

        memcpy(st->pi, ws->pin, (size_t)G * sizeof(double));
        memcpy(st->m, ws->mp, (size_t)G * p * sizeof(double));
        memcpy(st->P, ws->Ap, (size_t)G * pp * sizeof(double));

        if (step != NULL) {
            step->mean = mbar;
            step->var = vbar;
            step->innovation = NAN;
            step->loglik = 0.0;
            step->share_prior = 1.0;
            step->share_process = 0.0;
            step->share_measurement = 0.0;
            step->process_anomaly = lamP - phi_P * st->prev_lamP;
            step->process_regime = phi_P * st->prev_lamP;
            step->measurement_anomaly = lamM - phi_M * st->prev_lamM;
            step->measurement_regime = phi_M * st->prev_lamM;
            step->whiteness = ode_whiteness(st);
            step->dynamics = dyn;
            step->pred_var = NAN;
        }
        st->prev_lamP = lamP;
        st->prev_lamM = lamM;
        return PyErr_Occurred() ? -1 : 0;
    }

    for (npy_intp i = 0; i < G; i++) {
        if (!isfinite(ws->S[i]) || ws->S[i] <= 0.0) {
            return -2;                          /* the _Numerical guard */
        }
    }

    const double ybar = nk_dot_strided(ws->pin, ws->mp, G,
                                       (npy_intp)(p * sizeof(double)));
    const double e_rep = y - ybar;

    nk_log(ws->S, ws->tmp, G);
    for (npy_intp i = 0; i < G; i++) {
        const double e = y - ws->mp[i * p];
        ws->lg[i] = -0.5 * (ws->tmp[i] + e * e / ws->S[i]);
    }
    const double mx = nk_max(ws->lg, G);
    for (npy_intp i = 0; i < G; i++) {
        ws->tmp[i] = ws->lg[i] - mx;
    }
    nk_exp(ws->tmp, ws->w, G);
    for (npy_intp i = 0; i < G; i++) {
        ws->w[i] = ws->pin[i] * ws->w[i];
    }
    const double Z = nk_sum(ws->w, G);
    const double ll = log(Z) + mx + g_half_log2pi;

    for (npy_intp i = 0; i < G; i++) {
        const double d = ws->mp[i * p] - ybar;
        ws->tmp[i] = ws->S[i] + d * d;
    }
    const double S_pred = nk_dot(ws->pin, ws->tmp, G);

    for (npy_intp i = 0; i < G; i++) {
        st->pi[i] = ws->w[i] / Z;
    }
    /* K = Aj[:, :, 0] / S[:, None];  m += K e;  P -= K Aj[0, :] */
    for (npy_intp i = 0; i < G; i++) {
        const double e = y - ws->mp[i * p];
        const double s = ws->S[i];
        for (int x = 0; x < p; x++) {
            ws->K[x] = ws->Ap[i * pp + x * p] / s;
        }
        for (int x = 0; x < p; x++) {
            const double k = ws->K[x];
            st->m[i * p + x] = ws->mp[i * p + x] + k * e;
            for (int z = 0; z < p; z++) {
                st->P[i * pp + x * p + z] =
                    ws->Ap[i * pp + x * p + z] - k * ws->Ap[i * pp + z];
            }
        }
    }

    st->loglik += ll;
    /* the whiteness accumulator is state, so it is kept whether or not the
     * caller wants the reported fields */
    const double r = e_rep / sqrt(1e-300 > S_pred ? 1e-300 : S_pred);
    st->ee += r * st->e_prev;
    st->e2 += r * r;
    st->nw += 1;
    st->e_prev = r;

    if (step != NULL) {
        step->loglik = ll;
        step->innovation = e_rep;
        step->pred_var = S_pred;
        step->whiteness = ode_whiteness(st);
        if (!want_step) {
            return 0;
        }
        const double mbar = nk_dot_strided(st->pi, st->m, G,
                                           (npy_intp)(p * sizeof(double)));
        for (npy_intp i = 0; i < G; i++) {
            const double d = st->m[i * p] - mbar;
            ws->tmp[i] = st->P[i * pp] + d * d;
        }
        const double vbar = nk_dot(st->pi, ws->tmp, G);
        const double lamP = nk_dot(st->pi, g->LP, G);
        const double lamM = nk_dot(st->pi, g->LM, G);
        for (npy_intp i = 0; i < G; i++) {
            ws->tmp[i] = ws->tmp2[i] / ws->S[i];
        }
        const double sh_prior = nk_dot(st->pi, ws->tmp, G);
        for (npy_intp i = 0; i < G; i++) {
            ws->tmp[i] = g->Qg[i] / ws->S[i];
        }
        const double sh_proc = nk_dot(st->pi, ws->tmp, G);
        for (npy_intp i = 0; i < G; i++) {
            ws->tmp[i] = g->Rg[i] / ws->S[i];
        }
        const double sh_meas = nk_dot(st->pi, ws->tmp, G);
        const double dyn = 1.0 + nk_dot(st->pi, g->LA, G);

        step->mean = mbar;
        step->var = vbar;
        step->share_prior = sh_prior;
        step->share_process = sh_proc;
        step->share_measurement = sh_meas;
        step->process_anomaly = lamP - phi_P * st->prev_lamP;
        step->process_regime = phi_P * st->prev_lamP;
        step->measurement_anomaly = lamM - phi_M * st->prev_lamM;
        step->measurement_regime = phi_M * st->prev_lamM;
        step->dynamics = dyn;
        st->prev_lamP = lamP;
        st->prev_lamM = lamM;
    }
    return PyErr_Occurred() ? -1 : 0;
}

/* `mixture._roll` / `OdeFilter.predict`: h steps of the same propagation with
 * no observation, leaving `st` untouched. */
static int
ode_roll(void *vws, const LucidOdeGrid *g, const LucidOdeState *st, int h,
         int with_process, double *pi_out, double *m_out, double *P_out)
{
    OdeWS *ws = (OdeWS *)vws;
    const npy_intp G = g->G;
    const int p = g->p, pp = p * p;

    memcpy(pi_out, st->pi, (size_t)G * sizeof(double));
    memcpy(m_out, st->m, (size_t)G * p * sizeof(double));
    memcpy(P_out, st->P, (size_t)G * pp * sizeof(double));

    npy_intp dTT[2] = {G, G};
    for (int step = 0; step < h; step++) {
        PyObject *T_obj = shell(g->T, 2, dTT);
        if (T_obj == NULL) {
            return -1;
        }
        if (nk_vecmat(pi_out, T_obj, G, ws->pin, G) != 0) {
            Py_DECREF(T_obj);
            return -1;
        }
        Py_DECREF(T_obj);
        for (npy_intp j = 0; j < G; j++) {
            ws->pin[j] = nk_clip_lo(ws->pin[j], 1e-300);
        }
        mix_nodes(ws, g->T, pi_out, m_out, P_out);
        if (mp_contract(ws, g->Fg, g->mp_mode) != 0) {
            return -1;
        }
        for (npy_intp i = 0; i < G; i++) {
            double *A = P_out + i * pp;
            ap_block(g->Fg + i * pp, ws->P0 + i * pp, A, p, g->ap_mode);
            if (with_process) {
                A[0] += g->Qg[i];
            }
        }
        memcpy(m_out, ws->mp, (size_t)G * p * sizeof(double));
        memcpy(pi_out, ws->pin, (size_t)G * sizeof(double));
    }
    return PyErr_Occurred() ? -1 : 0;
}

/* The reporting collapse through NumPy itself, for the shapes where einsum
 * does not sum left to right.  Two calls per step, and only when `filter()`
 * asked for the state covariance at all. */
static int
sc_einsum(const LucidOdeState *st, const double *dm, npy_intp G, int p,
          double *out)
{
    npy_intp dG[1] = {G};
    npy_intp dGp[2] = {G, p};
    npy_intp dGpp[3] = {G, p, p};
    PyObject *PI = shell(st->pi, 1, dG);
    PyObject *PP = shell(st->P, 3, dGpp);
    PyObject *DM = shell(dm, 2, dGp);
    int rc = -1;
    if (PI != NULL && PP != NULL && DM != NULL) {
        PyObject *a1[3] = {g_sc_spec1, PI, PP};
        PyObject *r1 = PyObject_Vectorcall(
            g_einsum, a1, 3 | PY_VECTORCALL_ARGUMENTS_OFFSET, NULL);
        if (r1 != NULL) {
            PyObject *a2[4] = {g_sc_spec2, PI, DM, DM};
            PyObject *r2 = PyObject_Vectorcall(
                g_einsum, a2, 4 | PY_VECTORCALL_ARGUMENTS_OFFSET, NULL);
            if (r2 != NULL) {
                const double *d1 = (const double *)PyArray_DATA((PyArrayObject *)r1);
                const double *d2 = (const double *)PyArray_DATA((PyArrayObject *)r2);
                for (int k = 0; k < p * p; k++) {
                    out[k] = d1[k] + d2[k];
                }
                Py_DECREF(r2);
                rc = 0;
            }
            Py_DECREF(r1);
        }
    }
    Py_XDECREF(PI);
    Py_XDECREF(PP);
    Py_XDECREF(DM);
    return rc;
}

/* ------------------------------------------------- the streaming entry point */

/* Pull a required (and contiguous, float64) array argument. */
static PyArrayObject *
as_arr(PyObject *o)
{
    return (PyArrayObject *)PyArray_FROM_OTF(o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
}

/* `OdeFilter._run`: the whole series through `_update_imm`, with the reported
 * fields collected as `filter()` collects them.
 *
 * Returns None when the recursion hits the `_Numerical` guard, which is what
 * `_run` turns into -inf (loglik) or into a raised exception (filter).
 */
static PyObject *
ode_filter(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyObject *y_o, *T_o, *pi0_o, *Qg_o, *Rg_o, *Fg_o, *LP_o, *LM_o, *LA_o;
    double phi_P, phi_M;
    int p, want, ap_mode, mp_mode, sc_mode;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOddiiiii", &y_o, &T_o, &pi0_o, &Qg_o,
                          &Rg_o, &Fg_o, &LP_o, &LM_o, &LA_o, &phi_P, &phi_M,
                          &p, &want, &ap_mode, &mp_mode, &sc_mode)) {
        return NULL;
    }
    PyArrayObject *aa[9];
    PyObject *objs[9] = {y_o, T_o, pi0_o, Qg_o, Rg_o, Fg_o, LP_o, LM_o, LA_o};
    for (int i = 0; i < 9; i++) {
        aa[i] = as_arr(objs[i]);
        if (aa[i] == NULL) {
            for (int j = 0; j < i; j++) {
                Py_DECREF(aa[j]);
            }
            return NULL;
        }
    }
    const npy_intp n = PyArray_SIZE(aa[0]);
    const npy_intp G = PyArray_SIZE(aa[2]);
    const int pp = p * p;

    LucidOdeGrid grid;
    grid.G = G;
    grid.p = p;
    grid.T = (const double *)PyArray_DATA(aa[1]);
    grid.pi0 = (const double *)PyArray_DATA(aa[2]);
    grid.Qg = (const double *)PyArray_DATA(aa[3]);
    grid.Rg = (const double *)PyArray_DATA(aa[4]);
    grid.Fg = (const double *)PyArray_DATA(aa[5]);
    grid.LP = (const double *)PyArray_DATA(aa[6]);
    grid.LM = (const double *)PyArray_DATA(aa[7]);
    grid.LA = (const double *)PyArray_DATA(aa[8]);
    grid.ap_mode = ap_mode;
    grid.mp_mode = mp_mode;

    const double *y = (const double *)PyArray_DATA(aa[0]);

    OdeWS *ws = (OdeWS *)ode_ws_alloc(G, p);
    LucidOdeState st;
    memset(&st, 0, sizeof(st));
    st.pi = (double *)malloc((size_t)G * sizeof(double));
    st.m = (double *)malloc((size_t)G * p * sizeof(double));
    st.P = (double *)malloc((size_t)G * pp * sizeof(double));
    double *sm = NULL, *sc = NULL, *cols = NULL, *dm = NULL;
    PyObject *out = NULL;
    const int NCOL = 13;

    if (ws == NULL || st.pi == NULL || st.m == NULL || st.P == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    if (want) {
        cols = (double *)malloc((size_t)NCOL * n * sizeof(double));
        sm = (double *)malloc((size_t)n * p * sizeof(double));
        sc = (double *)malloc((size_t)n * pp * sizeof(double));
        dm = (double *)malloc((size_t)G * p * sizeof(double));
        if (cols == NULL || sm == NULL || sc == NULL || dm == NULL) {
            PyErr_NoMemory();
            goto done;
        }
    }

    ode_start(&grid, &st, (n > 0 && isfinite(y[0])) ? y[0] : 0.0);

    double total = 0.0;
    npy_intp dGp[2] = {G, p};
    for (npy_intp t = 0; t < n; t++) {
        LucidOdeStep step;
        const int rc = ode_update(ws, &grid, &st, y[t], phi_P, phi_M, want,
                                  &step);
        if (rc == -2) {
            out = Py_None;
            Py_INCREF(out);
            goto done;
        }
        if (rc != 0) {
            goto done;
        }
        total += step.loglik;
        if (!want) {
            continue;
        }
        /* the column order `_run` builds, which leaves `loglik` out */
        cols[ 0 * n + t] = step.mean;
        cols[ 1 * n + t] = step.var;
        cols[ 2 * n + t] = step.innovation;
        cols[ 3 * n + t] = step.share_prior;
        cols[ 4 * n + t] = step.share_process;
        cols[ 5 * n + t] = step.share_measurement;
        cols[ 6 * n + t] = step.process_anomaly;
        cols[ 7 * n + t] = step.process_regime;
        cols[ 8 * n + t] = step.measurement_anomaly;
        cols[ 9 * n + t] = step.measurement_regime;
        cols[10 * n + t] = step.whiteness;
        cols[11 * n + t] = step.dynamics;
        cols[12 * n + t] = step.pred_var;
        /* the per-node mixture collapsed for reporting, as `_run` collapses it */
        PyObject *m_obj = shell(st.m, 2, dGp);
        if (m_obj == NULL) {
            goto done;
        }
        if (nk_vecmat(st.pi, m_obj, G, sm + t * p, p) != 0) {
            Py_DECREF(m_obj);
            goto done;
        }
        Py_DECREF(m_obj);
        for (npy_intp i = 0; i < G; i++) {
            for (int x = 0; x < p; x++) {
                dm[i * p + x] = st.m[i * p + x] - sm[t * p + x];
            }
        }
        if (sc_mode == LK_SC_NAIVE) {
            for (int x = 0; x < p; x++) {
                for (int z = 0; z < p; z++) {
                    double a = 0.0;
                    for (npy_intp i = 0; i < G; i++) {
                        a += st.pi[i] * st.P[i * pp + x * p + z];
                    }
                    double b = 0.0;
                    for (npy_intp i = 0; i < G; i++) {
                        b += st.pi[i] * dm[i * p + x] * dm[i * p + z];
                    }
                    sc[t * pp + x * p + z] = a + b;
                }
            }
        }
        else if (sc_einsum(&st, dm, G, p, sc + t * pp) != 0) {
            goto done;
        }
    }

    if (!want) {
        out = PyFloat_FromDouble(total);
    }
    else {
        npy_intp dc[2] = {NCOL, n};
        npy_intp ds[2] = {n, p};
        npy_intp dv[3] = {n, p, p};
        PyObject *c_arr = PyArray_SimpleNew(2, dc, NPY_DOUBLE);
        PyObject *s_arr = PyArray_SimpleNew(2, ds, NPY_DOUBLE);
        PyObject *v_arr = PyArray_SimpleNew(3, dv, NPY_DOUBLE);
        if (c_arr != NULL && s_arr != NULL && v_arr != NULL) {
            memcpy(PyArray_DATA((PyArrayObject *)c_arr), cols,
                   (size_t)NCOL * n * sizeof(double));
            memcpy(PyArray_DATA((PyArrayObject *)s_arr), sm,
                   (size_t)n * p * sizeof(double));
            memcpy(PyArray_DATA((PyArrayObject *)v_arr), sc,
                   (size_t)n * pp * sizeof(double));
            out = Py_BuildValue("(NNNd)", c_arr, s_arr, v_arr, total);
        }
        else {
            Py_XDECREF(c_arr);
            Py_XDECREF(s_arr);
            Py_XDECREF(v_arr);
        }
    }

done:
    free(cols); free(sm); free(sc); free(dm);
    free(st.pi); free(st.m); free(st.P);
    ode_ws_free(ws);
    for (int i = 0; i < 9; i++) {
        Py_DECREF(aa[i]);
    }
    return out;
}

/* ============================================================================
 *                     statfilter's batched likelihood
 * ==========================================================================*/
/* `statfilter._loglik_batch`.  The parent's per-node state is a scalar, so
 * there is no mixing stage and no covariance: the whole step is elementwise
 * work plus three reductions.  It is here because it is the same fit, and
 * because at order 7 the parent's grid is 49 nodes wide and pays the same
 * dispatch tax per step that the ODE filter does. */
static PyObject *
stat_loglik_batch(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyObject *x_o, *T_o, *pi0_o, *Qg_o, *Rg_o;
    if (!PyArg_ParseTuple(args, "OOOOO", &x_o, &T_o, &pi0_o, &Qg_o, &Rg_o)) {
        return NULL;
    }
    PyArrayObject *x_a = as_arr(x_o), *T_a = as_arr(T_o), *pi0_a = as_arr(pi0_o);
    PyArrayObject *Qg_a = as_arr(Qg_o), *Rg_a = as_arr(Rg_o);
    PyObject *out = NULL;
    double *pi = NULL, *m = NULL, *P = NULL, *S = NULL, *Pp = NULL;
    double *lg = NULL, *w = NULL, *tmp = NULL, *ll = NULL, *mx = NULL,
           *Z = NULL, *Kb = NULL;
    if (x_a == NULL || T_a == NULL || pi0_a == NULL || Qg_a == NULL
        || Rg_a == NULL) {
        goto done;
    }
    {
        const npy_intp B = PyArray_DIM(pi0_a, 0);
        const npy_intp G = PyArray_DIM(pi0_a, 1);
        const npy_intp n = PyArray_SIZE(x_a);
        const double *x = (const double *)PyArray_DATA(x_a);
        const double *T = (const double *)PyArray_DATA(T_a);
        const double *pi0 = (const double *)PyArray_DATA(pi0_a);
        const double *Qg = (const double *)PyArray_DATA(Qg_a);
        const double *Rg = (const double *)PyArray_DATA(Rg_a);

        pi  = (double *)malloc((size_t)B * G * sizeof(double));
        m   = (double *)malloc((size_t)B * sizeof(double));
        P   = (double *)malloc((size_t)B * sizeof(double));
        S   = (double *)malloc((size_t)B * G * sizeof(double));
        Pp  = (double *)malloc((size_t)B * G * sizeof(double));
        lg  = (double *)malloc((size_t)B * G * sizeof(double));
        w   = (double *)malloc((size_t)B * G * sizeof(double));
        tmp = (double *)malloc((size_t)B * G * sizeof(double));
        ll  = (double *)malloc((size_t)B * sizeof(double));
        mx  = (double *)malloc((size_t)B * sizeof(double));
        Z   = (double *)malloc((size_t)B * sizeof(double));
        Kb  = (double *)malloc((size_t)B * sizeof(double));
        if (pi == NULL || m == NULL || P == NULL || S == NULL || Pp == NULL
            || lg == NULL || w == NULL || tmp == NULL || ll == NULL
            || mx == NULL || Z == NULL || Kb == NULL) {
            PyErr_NoMemory();
            goto done;
        }
        memcpy(pi, pi0, (size_t)B * G * sizeof(double));
        const double x0 = (n > 0 && isfinite(x[0])) ? x[0] : 0.0;
        for (npy_intp b = 0; b < B; b++) {
            m[b] = x0;
            P[b] = nk_max(Rg + b * G, G) + nk_max(Qg + b * G, G);
            ll[b] = 0.0;
        }

        for (npy_intp t = 0; t < n; t++) {
            const double v = x[t];
            for (npy_intp b = 0; b < B; b++) {
                const double *Tb = T + b * G * G;
                double *pib = pi + b * G;
                double *o = tmp + b * G;
                for (npy_intp j = 0; j < G; j++) {
                    o[j] = 0.0;
                }
                for (npy_intp i = 0; i < G; i++) {
                    const double c = pib[i];
                    const double *Ti = Tb + i * G;
                    for (npy_intp j = 0; j < G; j++) {
                        o[j] += c * Ti[j];
                    }
                }
            }
            memcpy(pi, tmp, (size_t)B * G * sizeof(double));

            if (!isfinite(v)) {
                for (npy_intp b = 0; b < B; b++) {
                    for (npy_intp g = 0; g < G; g++) {
                        tmp[b * G + g] = pi[b * G + g] * Qg[b * G + g];
                    }
                    P[b] = P[b] + nk_sum(tmp + b * G, G);
                }
                continue;
            }
            for (npy_intp b = 0; b < B; b++) {
                for (npy_intp g = 0; g < G; g++) {
                    Pp[b * G + g] = P[b] + Qg[b * G + g];
                    S[b * G + g] = P[b] + (Qg[b * G + g] + Rg[b * G + g]);
                }
            }
            nk_log(S, tmp, B * G);
            for (npy_intp b = 0; b < B; b++) {
                const double e = v - m[b];
                for (npy_intp g = 0; g < G; g++) {
                    lg[b * G + g] =
                        -0.5 * (tmp[b * G + g] + (e * e) / S[b * G + g]);
                }
                mx[b] = nk_max(lg + b * G, G);
                for (npy_intp g = 0; g < G; g++) {
                    tmp[b * G + g] = lg[b * G + g] - mx[b];
                }
            }
            nk_exp(tmp, w, B * G);
            for (npy_intp b = 0; b < B; b++) {
                for (npy_intp g = 0; g < G; g++) {
                    w[b * G + g] = pi[b * G + g] * w[b * G + g];
                }
                Z[b] = nk_sum(w + b * G, G);
            }
            nk_log(Z, tmp, B);
            for (npy_intp b = 0; b < B; b++) {
                ll[b] += tmp[b] + mx[b] + g_half_log2pi;
            }
            for (npy_intp b = 0; b < B; b++) {
                const double e = v - m[b];
                double *pib = pi + b * G;
                for (npy_intp g = 0; g < G; g++) {
                    pib[g] = w[b * G + g] / Z[b];
                }
                /* K = Pp / S;  Kbar = sum pi K */
                for (npy_intp g = 0; g < G; g++) {
                    tmp[b * G + g] = pib[g] * (Pp[b * G + g] / S[b * G + g]);
                }
                Kb[b] = nk_sum(tmp + b * G, G);
                m[b] = m[b] + Kb[b] * e;
                for (npy_intp g = 0; g < G; g++) {
                    const double K = Pp[b * G + g] / S[b * G + g];
                    tmp[b * G + g] = pib[g] * ((1.0 - K) * Pp[b * G + g]);
                }
                const double lhs = nk_sum(tmp + b * G, G);
                for (npy_intp g = 0; g < G; g++) {
                    const double K = Pp[b * G + g] / S[b * G + g];
                    const double d = K - Kb[b];
                    tmp[b * G + g] = pib[g] * (d * d);
                }
                P[b] = lhs + e * e * nk_sum(tmp + b * G, G);
            }
        }
        npy_intp dB[1] = {B};
        out = PyArray_SimpleNew(1, dB, NPY_DOUBLE);
        if (out != NULL) {
            memcpy(PyArray_DATA((PyArrayObject *)out), ll,
                   (size_t)B * sizeof(double));
        }
    }

done:
    free(pi); free(m); free(P); free(S); free(Pp); free(lg); free(w);
    free(tmp); free(ll); free(mx); free(Z); free(Kb);
    Py_XDECREF(x_a); Py_XDECREF(T_a); Py_XDECREF(pi0_a);
    Py_XDECREF(Qg_a); Py_XDECREF(Rg_a);
    return out;
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


static LucidKernelAPI g_api;

static PyMethodDef methods[] = {
    {"configure", py_configure, METH_VARARGS,
     "configure(log2pi) -- hand over the module's own log(2 pi)."},
    {"ode_loglik_batch", ode_loglik_batch, METH_VARARGS,
     "ode_loglik_batch(y, T, pi0, Qg, Rg, Fg, p) -> (B,) log-likelihoods."},
    {"ode_filter", ode_filter, METH_VARARGS,
     "ode_filter(y, T, pi0, Qg, Rg, Fg, LP, LM, LA, phi_P, phi_M, p, want,\n"
     "ap_mode, mp_mode, sc_mode) -- the whole series through the streaming\n"
     "recursion; None when the numerical guard fires."},
    {"stat_loglik_batch", stat_loglik_batch, METH_VARARGS,
     "stat_loglik_batch(x, T, pi0, Qg, Rg) -> (B,) log-likelihoods."},
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
    "The compiled recursion behind odefilter and statfilter.", -1, methods,
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
    g_mp_spec = PyUnicode_FromString("bgxz,bgz->bgx");
    g_mp_spec1 = PyUnicode_FromString("gxz,gz->gx");
    g_sc_spec1 = PyUnicode_FromString("g,gxz->xz");
    g_sc_spec2 = PyUnicode_FromString("g,gx,gz->xz");
    if (g_mp_spec == NULL || g_mp_spec1 == NULL || g_sc_spec1 == NULL
        || g_sc_spec2 == NULL) {
        return NULL;
    }

    PyObject *mod = PyModule_Create(&moduledef);
    if (mod == NULL) {
        return NULL;
    }
    g_api.abi = LUCID_KERNEL_ABI;
    g_api.ode_alloc = ode_ws_alloc;
    g_api.ode_free = ode_ws_free;
    g_api.ode_start = ode_start;
    g_api.ode_update = ode_update;
    g_api.ode_roll = ode_roll;
    g_api.np_exp = nk_exp;
    g_api.np_log = nk_log;
    g_api.np_sum = nk_sum;
    g_api.np_dot = nk_dot;
    PyObject *cap = PyCapsule_New(&g_api, "lucid_kernel._kernel._C_API", NULL);
    if (cap == NULL || PyModule_AddObject(mod, "_C_API", cap) < 0) {
        Py_XDECREF(cap);
        Py_DECREF(mod);
        return NULL;
    }
    if (PyModule_AddIntConstant(mod, "ABI", LUCID_KERNEL_ABI) < 0) {
        Py_DECREF(mod);
        return NULL;
    }
    return mod;
}
