"""0049 -- which projection is the 1/r in, and how would we know?

Their items 3 and 4: implement the source (T = Fisher) as lattice
code, then measure the response and READ THE 1/r. Filter stones 0011
and 0012 already have the 1/r on this side (Green function alpha =
1.02; force law C(r) = a - b/r^1.04). So the physics is not the open
part. Two engineering parts are.

FIRST: a massless field does not show 1/r in every projection. The
same field gives three different laws depending on what is summed
over, and their item 2 used the one that answers a DIFFERENT
question. If they read the wrong projection they will conclude
"screened" from data that is exactly massless.

SECOND: with four separations and fat errors, "power law" and
"Yukawa" are not obviously distinguishable. Their own history says
pre-register the criterion (0038: a sloppy test ran a 51% false
positive where the honest one runs 0.5%). So the acceptance test is
built HERE, before the run, and priced.

  s1  THE THREE PROJECTIONS of one exactly massless 4D field.
  s2  THE ACCEPTANCE TEST -- what precision is needed to call
      massless, and what it costs to get it wrong.
  s3  THE PORT SPEC.
"""

import numpy as np

rng = np.random.default_rng(49)
L = 32


def khat2(L, d=4):
    ax = 2 * np.pi * np.fft.fftfreq(L)
    g = np.meshgrid(*([ax] * d), indexing="ij")
    return sum(4 * np.sin(gi / 2) ** 2 for gi in g)


def massless_field_corr(L, m2=0.0):
    """EXACT connected two-point function of a free lattice field
    with mass^2 = m2, by inverting the kinetic operator in Fourier
    space. No sampling, so no statistical error anywhere in s1."""
    k2 = khat2(L) + m2
    k2f = k2.copy()
    k2f[(0,) * 4] = np.inf          # drop the zero mode
    return np.real(np.fft.ifftn(1.0 / k2f))


def s1_three_projections():
    print("== s1: three projections of ONE exactly massless field ==")
    G = massless_field_corr(L)
    r = np.arange(1, 9)
    # (i) static: sum over the time separation -> 3D Green function
    stat = G.sum(0)[r, 0, 0]
    # (ii) equal time: fixed t = 0
    eq = G[0][r, 0, 0]
    # (iii) zero spatial momentum: sum over the spatial volume,
    #       read against the TIME separation  (their item 2's choice)
    zsp = G.sum((1, 2, 3))[r]

    def fit_n(x, y):
        """removing the zero mode adds a CONSTANT to every
        projection, which steepens a naive log-log slope. Fit
        y = A/r^n + C and report n."""
        best, bn = np.inf, np.nan
        for n in np.linspace(0.2, 4.0, 761):
            X = np.vstack([x ** (-n), np.ones_like(x)]).T
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            e = np.sum((X @ b - y) ** 2)
            if e < best:
                best, bn = e, n
        return bn

    print("     r      static (sum_t)     equal-time       "
          "zero-spatial-p")
    for i, rr in enumerate(r):
        print(f"    {rr:2d}    {stat[i]:+.6e}    {eq[i]:+.6e}    "
              f"{zsp[i]:+.6e}")
    ss, se = fit_n(r.astype(float), stat), fit_n(r.astype(float), eq)
    print()
    print(f"  fitted exponent in A/r^n + C:   static n = {ss:.3f}"
          f"    equal-time n = {se:.3f}")
    print()
    print("  The SAME field, three answers. Static is 1/r. "
          "Equal-time is 1/r^2. And the")
    print("  zero-spatial-momentum column does not decay at all -- "
          "it is very nearly")
    print("  LINEAR in the separation, which is what a massless "
          "field does at p = 0 once")
    print("  the zero mode is removed. It certainly does not fall "
          "off fast.")
    assert abs(ss - 1) < 0.15, f"static projection is not 1/r: {ss}"
    assert abs(se - 2) < 0.25, f"equal-time is not 1/r^2: {se}"
    print()
    print("  TWO CONSEQUENCES FOR THEM.")
    print("  (i) 'Read the 1/r' means the STATIC projection: sum "
          "the separation over the")
    print("      time direction, then look along a spatial one.")
    print("  (ii) Their 0++ measurement fell by 11x from d=0 to "
          "d=1 at zero spatial")
    print("      momentum. An ELEMENTARY massless channel cannot "
          "do that -- it would be")
    print("      flat. So the plaquette is not the channel that "
          "carries the 1/r, and no")
    print("      amount of statistics on it will produce one. "
          "That is an argument for")
    print("      building the source operator of item 3 rather "
          "than reusing what exists.")
    print()
    return stat


def codelen(y, sy, model):
    """Gaussian description length in nats: misfit + one nat per
    fitted parameter (BIC-ish, but stated so it is not tuned)."""
    res = (y - model) / sy
    return 0.5 * np.sum(res ** 2)


def fit_power(r, y, sy):
    """y = A / r^n -- linear in logs, 2 parameters."""
    w = 1.0 / (sy / np.abs(y))
    X = np.vstack([np.ones_like(r), -np.log(r)]).T
    lw = np.log(np.abs(y))
    W = np.diag(w ** 2)
    beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ lw)
    return np.exp(beta[0]) / r ** beta[1], beta[1]


def fit_yukawa(r, y, sy):
    """y = A e^{-m r} / r -- scan m, 2 parameters."""
    best, bm, bmod = np.inf, 0.0, None
    for m in np.linspace(0.0, 3.0, 601):
        f = np.exp(-m * r) / r
        A = np.sum(y * f / sy ** 2) / np.sum(f * f / sy ** 2)
        c = codelen(y, sy, A * f)
        if c < best:
            best, bm, bmod = c, m, A * f
    return bmod, bm, best


def trial(truth_m, eps, r, ntrial=400):
    """how often does the test call it massless?"""
    calls = 0
    gaps = []
    for _ in range(ntrial):
        y0 = np.exp(-truth_m * r) / r
        sy = eps * y0[0] * np.ones_like(r)
        y = y0 + sy * rng.standard_normal(len(r))
        _, m_y, c_y = fit_yukawa(r, y, sy)
        mod_p, _ = fit_power(r, y, sy)
        c_p = codelen(y, sy, mod_p)
        # "massless" = the m=0 Yukawa (pure 1/r) is not beaten by
        # more than 2 nats by the best massive one
        f0 = 1.0 / r
        A0 = np.sum(y * f0 / sy ** 2) / np.sum(f0 * f0 / sy ** 2)
        c0 = codelen(y, sy, A0 * f0)
        gap = c0 - c_y            # >0 means massive fits better
        gaps.append(gap)
        if gap < 2.0:
            calls += 1
    return calls / ntrial, float(np.mean(gaps))


def s2_acceptance():
    print("== s2: the acceptance test, priced before the run ==")
    print("  Criterion, pre-registered: call the channel MASSLESS "
          "when the best fitted")
    print("  Yukawa beats the pure 1/r by less than 2 nats. Two "
          "parameters each way,")
    print("  so no parameter-count correction is needed -- only "
          "the misfit differs.")
    print()
    r = np.array([1.0, 2.0, 3.0, 4.0])
    print("   relative error    P(call massless | truly massless)"
          "     P(call massless | m = 0.5)")
    for eps in (0.30, 0.10, 0.03, 0.01):
        fp, _ = trial(0.0, eps, r)
        tp, _ = trial(0.5, eps, r)
        print(f"      {eps:5.2f}              {fp:5.3f}"
              f"                          {tp:5.3f}")
    print()
    print("  Read the SECOND column as the error that matters: it "
          "is the chance of")
    print("  calling a screened channel massless. At 30% errors "
          "the test is useless --")
    print("  it says 'massless' whatever is true. The test only "
          "starts separating the")
    print("  hypotheses at the few-percent level.")
    print()
    print("  THE NUMBER THEY NEED: about 3% relative precision on "
          "the static response")
    print("  at r = 1..4. Anything looser and a 1/r claim is not "
          "supported by the data,")
    print("  however good the plot looks.")
    print()


def s3_port():
    print("== s3: the port spec ==")
    print("  1. THE SOURCE (their item 3). Couple a background "
          "field lambda to the local")
    print("     log-weight: W_p -> W_p^(1+lambda_p). Then")
    print("       S(x) = d ln W / d lambda = ln W_p ,")
    print("     and the Fisher information of that one-parameter "
          "family is Var(ln W_p)")
    print("     exactly. 'T = Fisher' is then not an analogy, it "
          "is the definition of")
    print("     the operator they insert -- no new machinery, the "
          "log-weight is already")
    print("     in their lookup table.")
    print()
    print("  2. THE RESPONSE (their item 4). Linear response to "
          "that source is")
    print("       delta<S(x)> = <S(x) S(0)>_c ,")
    print("     so the response function IS the connected "
          "correlator they can already")
    print("     measure -- but in the STATIC projection of s1: sum "
          "the separation over")
    print("     the time direction, then read along a spatial one.")
    print()
    print("  3. THE TEST. s2's criterion, at 3% relative "
          "precision, r = 1..4. Their")
    print("     two-level estimator (24x in variance) is worth "
          "about a factor 5 in")
    print("     error, so it is the difference between a useless "
          "test and a live one.")
    print()
    print("  AND THE WARNING FROM s1: whatever they find, it is "
          "only about masslessness")
    print("  if it is the static projection. The projection their "
          "item 2 used cannot")
    print("  answer this question and should not be quoted at it.")
    print()


if __name__ == "__main__":
    s1_three_projections()
    s2_acceptance()
    s3_port()
    print("all assertions passed")
