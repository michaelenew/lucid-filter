/* lucid_kernel.h -- the compiled LucidFilter step.
 *
 * The kernel is a compiled transcription of the stacked bank step in
 * `lucid.filter.lucid`.  It exists to make the same arithmetic cheap, not to
 * be a second implementation of it: every routine is bit-for-bit equal to the
 * NumPy code it replaces (see README.md for how that is arranged and what it
 * rests on).
 *
 * There is no C API to borrow: the two entry points take and return NumPy
 * arrays and are reached from Python, through `lucid.filter.lucid`, which
 * verifies them against the NumPy path for a problem shape before using them
 * for it.  ABI is exported so a caller can tell one build from another.
 */
#ifndef LUCID_KERNEL_H
#define LUCID_KERNEL_H

#define LUCID_KERNEL_ABI 2

#endif /* LUCID_KERNEL_H */
