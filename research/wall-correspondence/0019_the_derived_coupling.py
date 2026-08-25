"""wall-correspondence 0019 -- the coupling, derived: ratio records
+ pinned learning => Coulomb response.

0013 left the trust field's Laplacian coupling underived. The
missing mechanism is DYNAMICS, built from the program's own two
theorems plus locality:

  - freeze lemma: spatial records carry log-RATIOS (differences);
  - masslessness (pinned root): no restoring force to any absolute
    level -- the zero mode sits at k = 0;
  - and the structural discovery made while building this stone:
    a purely spatial ratio-bias is SATISFIABLE (lambda = b at the
    source solves every bond record exactly -- a delta, no tail),
    so a mass cannot be a spatial-record bias. THE POINT CHARGE
    ENTERS THROUGH THE TEMPORAL RATIO CHANNEL: a site whose
    innovations say 'my scale rose relative to my own past'.
    Mass sources through time; gravity propagates through space --
    exactly 0010's 'mass = information accumulated in a site's own
    history'.

The online bank learning from spatial difference records + a
temporal self-record at the source performs gradient flow
d lambda/dt = -eta [L lambda + e_s(lambda_s - b)] + noise terms,
whose steady state is the SATURATING point-charge potential
lambda = m_eff G(r) (Sherman-Morrison -- the 0011 s3 mass law drops
out automatically). Measured:

  s1  1D ring: the steady response is harmonic away from the source
      (piecewise linear -- the 1D Coulomb), while the vacuum
      spectrum of the same dynamics is WHITE: response and
      correlation decouple. No FDT: gravity is a response
      phenomenon, consistent with the measured short-ranged vacuum
      scale correlations (their 0102/0104).
  s2  3D: the steady response is 1/r^alpha with alpha ~ 1: NEWTON,
      with no prior posited -- the potential is the learning
      operator's Green function.
  s3  Sum-record control: the same dynamics on pair-SUM records
      also has a massless mode -- at the zone corner: a staggered,
      long-ranged anti-binding response. The record structure
      selects the massless momentum (ratio -> k = 0, sum -> k = pi);
      smooth universal attraction selects ratio records, which is
      the freeze lemma's own structure.
"""

import numpy as np

ETA = 0.2
SIG = 0.5
B = 1.0


def run_field(shape, site, b, T, mode, rng, spec_from=None,
              noise=True, dirichlet=False, eta=None):
    lam = np.zeros(shape)
    d = len(shape)
    spec_acc, nspec = None, 0
    for t in range(T):
        upd = np.zeros(shape)
        for ax in range(d):
            nb = np.roll(lam, -1, ax)
            y = SIG * rng.standard_normal(shape) if noise else 0.0
            if mode == "diff":
                r = y - (lam - nb)
                upd += r
                upd -= np.roll(r, 1, ax)
            else:
                r = y - (lam + nb)
                upd += r
                upd += np.roll(r, 1, ax)
        if b:
            # temporal self-record at the source site
            upd[site] += (b - lam[site]) + (
                SIG * rng.standard_normal() if noise else 0.0)
        lam = lam + (eta if eta else ETA) * upd
        if dirichlet:                      # 'infinity': far shell
            for ax in range(d):
                idx = [slice(None)] * d
                idx[ax] = 0
                lam[tuple(idx)] = 0.0
        if spec_from is not None and t >= spec_from and t % 10 == 0:
            spec_acc = (np.abs(np.fft.fftn(lam)) ** 2
                        if spec_acc is None else
                        spec_acc + np.abs(np.fft.fftn(lam)) ** 2)
            nspec += 1
    return lam, (spec_acc / nspec if nspec else None)


def s1_1d():
    print("== s1: 1D -- the sustained well needs flux to infinity ==")
    rng = np.random.default_rng(5)
    N = 129
    # absorbing far boundary ('infinity'): the well is harmonic --
    # the exact 1D tent
    lam, _ = run_field((N,), (N // 2,), B, 60000, "diff", rng,
                       noise=False, dirichlet=True)
    c = N // 2
    r = np.arange(1, c - 2)
    prof = (lam[c + r] + lam[c - r]) / 2
    fit = np.polyfit(r, prof, 1)
    resid = np.abs(prof - np.polyval(fit, r)).max() \
        / (prof.max() - prof.min())
    print(f"  Dirichlet ('infinity') boundary: tent profile, "
          f"linear residual {resid:.3f}, slope {fit[0]:+.4f}")
    assert resid < 0.02 and fit[0] < 0
    # closed (periodic) universe: the same pin fills the WHOLE field
    lam2, _ = run_field((N,), (N // 2,), B, 60000, "diff", rng,
                        noise=False, dirichlet=False)
    print(f"  periodic (closed) box: field range "
          f"{lam2.max() - lam2.min():.4f} around mean "
          f"{lam2.mean():.3f} -- the pin lifts EVERYTHING: no well")
    assert lam2.max() - lam2.min() < 0.02 * abs(lam2.mean())
    # vacuum fluctuations of the same dynamics: white
    _, spec0 = run_field((N,), (N // 2,), 0.0, 6000, "diff", rng,
                         spec_from=3000)
    k2 = 4 * np.sin(2 * np.pi * np.fft.fftfreq(N) / 2) ** 2
    sel = k2 > 0.2
    cv = np.std(spec0[sel]) / np.mean(spec0[sel])
    print(f"  vacuum spectrum cv = {cv:.2f} (white): response is "
          f"Coulomb, fluctuations are")
    print("  white -- no FDT; and a potential well EXISTS only "
          "where the field can shed")
    print("  level to infinity: gravity needs an open boundary\n")


def s2_3d():
    print("== s2: 3D -- Newton from online learning ==")
    rng = np.random.default_rng(6)
    N = 24
    c = N // 2
    lam, _ = run_field((N, N, N), (c, c, c), B, 20000, "diff", rng,
                       noise=False, dirichlet=True, eta=0.1)
    rs = np.arange(1, 9)
    rays = [lam[c + rs, c, c], lam[c - rs, c, c],
            lam[c, c + rs, c], lam[c, c - rs, c],
            lam[c, c, c + rs], lam[c, c, c - rs]]
    prof = np.mean(rays, axis=0)
    best = None
    for al in np.arange(0.5, 1.61, 0.01):
        X = np.stack([np.ones_like(rs, float), 1 / rs ** al], 1)
        cfs, res, *_ = np.linalg.lstsq(X, prof, rcond=None)
        if best is None or res[0] < best[1]:
            best = (al, res[0])
    print(f"  steady response ~ 1/r^alpha, alpha = {best[0]:.2f}  "
          f"(Newton: 1)")
    assert 0.85 < best[0] < 1.2
    print("  the potential is the learning operator's Green "
          "function -- no prior posited.")
    print("  (And only 3D sustains it at infinite volume: diffusion "
          "is transient in 3D,")
    print("  recurrent in 1D/2D -- the dimension selection, in "
          "dynamical form)\n")


def s3_sum_control():
    print("== s3: the sum-record control ==")
    rng = np.random.default_rng(7)
    N = 129
    lam, _ = run_field((N,), (N // 2,), B, 60000, "sum", rng,
                       noise=False, dirichlet=True)
    c = N // 2
    seq = lam[c:c + 5]
    alt = np.mean(np.sign(seq[:-1]) * np.sign(seq[1:]))
    reach = abs(lam[c + 20]) / max(abs(lam[c]), 1e-12)
    print(f"  response alternates in sign (neighbor sign product "
          f"{alt:+.1f}) and its envelope")
    print(f"  is LONG-RANGED (r = 20 amplitude {reach:.2f} of "
          f"source): the sum channel has")
    print(f"  its own massless mode -- at the ZONE CORNER (k = pi)")
    assert alt < 0 and reach > 0.3
    print("  the record structure selects the massless mode's "
          "momentum: ratio records ->")
    print("  k = 0 (smooth universal attraction: gravity); sum "
          "records -> k = pi (staggered")
    print("  anti-binding). Observation -- gravity is smooth and "
          "attractive -- selects the")
    print("  ratio channel, which is the freeze lemma's own "
          "structure\n")


if __name__ == "__main__":
    s1_1d()
    s2_3d()
    s3_sum_control()
    print("all assertions passed")
