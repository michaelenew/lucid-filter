/* lucid_kernel.h -- the exported C interface of the lucid filter kernel.
 *
 * The kernel is a compiled transcription of the recursions in `odefilter`
 * and `statfilter`.  It exists to make the same arithmetic cheap, not to be
 * a second implementation of it: every routine below is bit-for-bit equal to
 * the NumPy code it replaces (see README.md for how that is arranged and
 * what it rests on).
 *
 * Another extension module may borrow these routines rather than reimplement
 * them -- which is the only way a second module can share the bit-exactness
 * guarantee.  Import the capsule `lucid_kernel._kernel._C_API` and check
 * `abi` against LUCID_KERNEL_ABI:
 *
 *     #include "lucid_kernel.h"
 *     const LucidKernelAPI *api = PyCapsule_Import("lucid_kernel._kernel._C_API", 0);
 *     if (api == NULL || api->abi != LUCID_KERNEL_ABI) { ... }
 *
 * The ABI is bumped whenever a signature or a field order changes; it is not
 * a version number and carries no compatibility promise beyond equality.
 */
#ifndef LUCID_KERNEL_H
#define LUCID_KERNEL_H

#include <Python.h>
#include <numpy/ndarraytypes.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LUCID_KERNEL_ABI 1

/* The grid an `OdeFilter` carries, as flat C arrays.  Laid out exactly as
 * `OdeFilter._build()` lays it out, so the Python side passes its own arrays
 * straight through with no copy and no reordering.
 *
 *   G   number of quadrature nodes (order * order * nA)
 *   p   order of the recurrence
 *   T   (G, G) chain kernel          pi0 (G,)   stationary weights
 *   Qg  (G,)  per-node process var   Rg  (G,)   per-node measurement var
 *   Fg  (G, p, p) per-node transition (already expanded over the A index)
 *   LP, LM, LA  (G,) the three log-scale coordinates
 */
typedef struct {
    npy_intp G;
    int p;
    const double *T;
    const double *pi0;
    const double *Qg;
    const double *Rg;
    const double *Fg;
    const double *LP;
    const double *LM;
    const double *LA;
    /* Which of NumPy's two observed reduction shapes to use for the two
     * contractions whose order einsum decides rather than the arithmetic.
     * Fill these from `lucid_kernel.orders_for(p)` -- never by guessing. */
    int ap_mode;
    int mp_mode;
} LucidOdeGrid;

/* The streaming state of one `OdeFilter`, as flat C arrays owned by the
 * caller.  `pi` is (G,), `m` is (G, p), `P` is (G, p, p). */
typedef struct {
    double *pi;
    double *m;
    double *P;
    double prev_lamP;
    double prev_lamM;
    double e_prev;
    double ee;
    double e2;
    long   nw;
    double loglik;
    int    started;
} LucidOdeState;

/* Everything `Step` carries, in declaration order. */
typedef struct {
    double mean, var, innovation, loglik;
    double share_prior, share_process, share_measurement;
    double process_anomaly, process_regime;
    double measurement_anomaly, measurement_regime;
    double whiteness, dynamics, pred_var;
} LucidOdeStep;

typedef struct {
    int abi;

    /* Scratch for one filter of this size.  Free with `ode_free`. */
    void *(*ode_alloc)(npy_intp G, int p);
    void  (*ode_free)(void *ws);

    /* Set `st` to the diffuse start `update()` uses on its first observation.
     * `y0` is that observation (0.0 when it is not finite). */
    void  (*ode_start)(const LucidOdeGrid *g, LucidOdeState *st, double y0);

    /* One observation.  Returns 0, or -1 when the predictive variance leaves
     * the range the recursion can represent (the `_Numerical` guard); `step`
     * is then not written.  `phi_P`/`phi_M` are the two persistences, needed
     * only for the mode coordinates the step reports.  `want_step` = 0 fills
     * only `step->loglik`, which is all a likelihood needs and is the cheap
     * path: the reported quantities cost a dot product each. */
    int   (*ode_update)(void *ws, const LucidOdeGrid *g, LucidOdeState *st,
                        double y, double phi_P, double phi_M,
                        int want_step, LucidOdeStep *step);

    /* The mixture after `h` steps of the filter's own propagation, without
     * touching `st`: `pi_out` (G,), `m_out` (G, p), `P_out` (G, p, p). */
    int   (*ode_roll)(void *ws, const LucidOdeGrid *g, const LucidOdeState *st,
                      int h, int with_process,
                      double *pi_out, double *m_out, double *P_out);

    /* The primitives whose result is a property of NumPy's build rather than
     * of the arithmetic, exposed so a borrower gets the same bits. */
    void   (*np_exp)(const double *in, double *out, npy_intp n);
    void   (*np_log)(const double *in, double *out, npy_intp n);
    double (*np_sum)(const double *a, npy_intp n);
    double (*np_dot)(const double *a, const double *b, npy_intp n);
} LucidKernelAPI;

#ifdef __cplusplus
}
#endif

#endif /* LUCID_KERNEL_H */
