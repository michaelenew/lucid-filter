"""wall-correspondence 0021 -- the hot boundary: Unruh for the
half-web observer.

The horizon front, partially unblocked. The sibling proved (their
0083, to 1e-4) that the half-space vacuum of a free field is
boost-thermal at beta = 2 pi. Through the time tier (Euclidean =
smoother), that has a filter reading: the joint posterior of a
massless (pinned) Gaussian bank, MARGINALIZED to an observer who
holds only half the web, is a thermal ensemble with respect to the
boost reweighting -- the missing half is experienced as HEAT at
temperature 1/2pi per rapidity. This module verifies the compact
version:

  - full-chain vacuum (ground-state covariances of a light-mass
    chain), reduced to a region touching the cut;
  - boost-thermal covariances on the region (site weights d_j,
    link weights d_link, walled far end), at beta in {pi, 2pi, 4pi};
  - the reduced state matches the boost-thermal state near the cut,
    and beta = 2pi wins the temperature scan decisively.

What remains blocked (stated): the accelerated-NODE protocol -- a
single filter whose data schedule mimics constant acceleration --
which would make the temperature operational per-observer rather
than per-region. No principled schedule map yet.
"""

import numpy as np

M2 = 0.01
R = 128
NFULL = 2 * R + 128
CMP = 24                       # near-cut comparison block


def chain_M(n, m2, weights=None, site_w=None):
    Mm = np.zeros((n, n))
    for j in range(n - 1):
        w = 1.0 if weights is None else weights[j]
        for (a, b, s) in ((j, j, 1), (j + 1, j + 1, 1),
                          (j, j + 1, -1), (j + 1, j, -1)):
            Mm[a, b] += s * w
    for j in range(n):
        w = 1.0 if site_w is None else site_w[j]
        Mm[j, j] += m2 * w
    return Mm


def sqrtm_sym(A):
    w, U = np.linalg.eigh(A)
    w = np.maximum(w, 1e-14)
    return U @ np.diag(np.sqrt(w)) @ U.T, \
        U @ np.diag(1 / np.sqrt(w)) @ U.T


if __name__ == "__main__":
    print("== the hot boundary: the half-web posterior is "
          "boost-thermal ==")
    Mf = chain_M(NFULL, M2)
    S, Si = sqrtm_sym(Mf)
    Cxx = 0.5 * Si            # vacuum <xx>
    Cpp = 0.5 * S             # vacuum <pp>
    c = NFULL // 2            # the cut: trace out the left half
    rx = Cxx[c:c + R, c:c + R]
    rp = Cpp[c:c + R, c:c + R]
    dsite = np.arange(R) + 0.5
    dlink = np.arange(1, R)
    Mk = chain_M(R, M2, weights=dlink, site_w=dsite)
    D = np.diag(dsite)
    Dh = np.diag(np.sqrt(dsite))
    Dhi = np.diag(1 / np.sqrt(dsite))
    Om2 = Dh @ Mk @ Dh
    w, U = np.linalg.eigh(Om2)
    w = np.maximum(w, 1e-14)
    Om = np.sqrt(w)
    print("   beta/2pi   rel err <xx>   rel err <pp>   (near-cut "
          f"{CMP}x{CMP} block)")
    errs = {}
    for bfac in (0.5, 1.0, 2.0):
        beta = bfac * 2 * np.pi
        cth = 1.0 / np.tanh(beta * Om / 2)
        Xt = U @ np.diag(cth / (2 * Om)) @ U.T
        Pt = U @ np.diag(cth * Om / 2) @ U.T
        Txx = Dh @ Xt @ Dh
        Tpp = Dhi @ Pt @ Dhi
        ex = np.linalg.norm(Txx[:CMP, :CMP] - rx[:CMP, :CMP]) \
            / np.linalg.norm(rx[:CMP, :CMP])
        ep = np.linalg.norm(Tpp[:CMP, :CMP] - rp[:CMP, :CMP]) \
            / np.linalg.norm(rp[:CMP, :CMP])
        errs[bfac] = ex + ep
        print(f"   {bfac:.1f}       {ex:.4f}        {ep:.4f}")
    assert errs[1.0] < 0.05
    assert errs[1.0] < 0.3 * errs[0.5]
    assert errs[1.0] < 0.3 * errs[2.0]
    print("  the reduced (half-web) posterior IS the boost-thermal "
          "ensemble, and the")
    print("  temperature scan picks beta = 2 pi decisively: a "
          "filter that holds half the")
    print("  web experiences the other half as heat -- the horizon "
          "is hot, in filter terms.")
    print("  Remaining blocked: the accelerated-NODE protocol (a "
          "data schedule realizing")
    print("  constant acceleration for a single filter) -- stated, "
          "not run.")
    print("all assertions passed")
