"""wall-correspondence 0014 -- the radiative ledger: which tier
carries gravitational waves.

0011 measured the record-tier trust field to be diffusive (front
~ t^0.5): statics right, no radiation, and predicted that radiation
lives in the SOURCE ledger. This stone runs the two tiers side by
side with the SAME coupling, differing in exactly one symbol -- the
generator is real (record: relaxation toward neighbors) or imaginary
(source: the polar/phase composition of the two-ledger theorem,
their 0086: source contributions compose as phases, so the field's
generator is anti-Hermitian). Both evolved exactly (split-step FFT).

  s1  FRONTS. The same pulse: record front ~ t^0.5 (diffusive);
      source front ~ t^1.0 (ballistic). One imaginary unit is the
      entire difference between screening and radiation.
  s2  THE HERTZ EXPERIMENT. An oscillating source at the origin
      (frequency w0), steady state: the record tier's response
      decays EXPONENTIALLY (skin depth sqrt(2 eta/w0): quasistatic
      screening); the source tier's response decays as a 1/r
      radiating shell. Fitted both; the record tier's far field is
      orders of magnitude below the source tier's.
  s3  The ported statement: gravitational statics (Newton, 0011-12)
      is record-tier inference; gravitational RADIATION is
      source-tier (phase) dynamics -- undetectable by any
      record-side (probability-mixing) observer per 0005, but
      certified by the 0006-style detector. A falsifiable
      structural prediction of the correspondence.
"""

import numpy as np

L = 72
ETA = 0.2
k = 2 * np.pi * np.fft.fftfreq(L)
kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
K2 = 4 * (np.sin(kx / 2) ** 2 + np.sin(ky / 2) ** 2
          + np.sin(kz / 2) ** 2)


def prop(imag):
    return np.exp((-1j if imag else -1) * ETA * K2)


def front_exponent(imag):
    """Energy-median radius of a pulse vs time."""
    f = np.zeros((L, L, L), dtype=complex)
    f[0, 0, 0] = 1.0
    fk = np.fft.fftn(f)
    P = prop(imag)
    rs, ts = [], []
    r = np.arange(1, L // 2)
    for t in range(1, 82):
        fk = fk * P
        if t % 9 == 0 and t >= 18:
            ray = np.abs(np.fft.ifftn(fk))[1:L // 2, 0, 0]
            wgt = r ** 2 * ray ** 2
            cum = np.cumsum(wgt) / wgt.sum()
            rs.append(r[np.searchsorted(cum, 0.5)])
            ts.append(t)
    return np.polyfit(np.log(ts), np.log(np.array(rs, float)),
                      1)[0]


def s1_fronts():
    print("== s1: one imaginary unit separates screening from "
          "radiation ==")
    sr = front_exponent(False)
    ss = front_exponent(True)
    print(f"  record tier (real generator)     : energy-median "
          f"radius ~ t^{sr:.2f}  (diffusive 0.5)")
    print(f"  source tier (imaginary generator): energy-median "
          f"radius ~ t^{ss:.2f}  (ballistic 1)")
    assert sr < 0.65 and ss > 0.85
    print("  same coupling, same pulse; the phase ledger radiates, "
          "the record ledger screens\n")


def s2_hertz():
    print("== s2: the Hertz experiment (oscillating source, open "
          "boundary, steady state) ==")
    w0 = 0.121
    ix = np.minimum(np.arange(L), L - np.arange(L))
    X, Y, Z = np.meshgrid(ix, ix, ix, indexing="ij")
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    mask = np.where(R < 22, 1.0, np.exp(-0.01 * (R - 22) ** 2))
    rbin = np.clip(R.astype(int), 0, 40)
    prof = {}
    for imag in (False, True):
        P = prop(imag)
        fk = np.zeros((L, L, L), dtype=complex)
        acc = np.zeros((L, L, L))
        for t in range(480):
            drive = np.exp(1j * w0 * t) if imag else np.cos(w0 * t)
            fk = fk * P + drive
            psi = np.fft.ifftn(fk) * mask
            fk = np.fft.fftn(psi)
            if t >= 420:
                acc += np.abs(psi) ** 2
        num = np.bincount(rbin.ravel(), weights=acc.ravel(),
                          minlength=41)
        cnt = np.bincount(rbin.ravel(), minlength=41)
        prof[imag] = np.sqrt(num / cnt)
    rr = np.arange(2, 10)
    er = np.polyfit(rr, np.log(np.maximum(prof[False][rr],
                                          1e-300)), 1)
    delta = -1 / er[0]
    rs_ = np.arange(4, 19)
    pw = np.polyfit(np.log(rs_), np.log(prof[True][rs_]), 1)[0]
    print(f"  record tier: exponential over r = 2..9, skin depth = "
          f"{delta:.2f}  (theory sqrt(2 eta/w0) ~ "
          f"{np.sqrt(2 * ETA / w0):.2f})")
    print(f"  source tier: power law over r = 4..18, amplitude ~ "
          f"r^{pw:.2f}  (radiating shell: -1)")
    far = prof[True][18] / max(prof[False][18], 1e-300)
    print(f"  far-field amplitude ratio at r = 18, source/record = "
          f"{far:.1e}")
    assert delta < 4
    assert -1.4 < pw < -0.7
    assert far > 1e2
    print("  the record field is quasistatically screened; the "
          "source field radiates\n")


def s3_statement():
    print("== s3: the ported statement ==")
    print("  Newtonian statics = record-tier inference (0011-0012); "
          "gravitational")
    print("  radiation = source-tier phase dynamics. Per 0005 no "
          "record-side observer can")
    print("  score the phase on classical streams; per 0006 the "
          "coherent detector can.")
    print("  Falsifiable structure: if the sibling's full theory "
          "radiates in a record-tier")
    print("  observable, this split is wrong\n")


if __name__ == "__main__":
    s1_fronts()
    s2_hertz()
    s3_statement()
    print("all assertions passed")
