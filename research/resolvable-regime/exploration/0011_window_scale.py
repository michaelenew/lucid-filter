"""0011 -- decouple the window's GEOMETRY from the class's stationary SD.

0010 built the derived ladder and it lost 22% on the sensor-degradation window.
The cause is a conflation in the derivation, and it is also a real defect in the
shipped filter.

`s` plays TWO roles in `lucid.py`:

  (a) the class's stationary SD of `lam` -- it sets the node prior and, through
      `nu = s^2 (1 - phi^2)`, the AR(1) kernel.  Correct.
  (b) the window's GEOMETRIC scale -- `gap = 1.5 s`, span `3 s`, and the caps
      `_Ifloor` / `_Pmu_cap` / `_Pmu`.  **Wrong**, and it is what breaks.

The window represents `lam`'s PREDICTIVE distribution, whose SD is `sqrt(p)` where
`p` is the one-step predictive variance of the `lam` channel -- a local-level /
AR(1) channel observed with noise `rho = 1/I = 2`, since `I(lam) = 1/2`:

    p^2 + p (rho (1 - phi^2) - nu) - nu rho = 0
    p = ( -b + sqrt(b^2 + 4 nu rho) ) / 2,      b = rho (1 - phi^2) - nu

The two coincide only when the class is comfortably stationary.  **As `phi -> 1`
at fixed `nu`, `s = sqrt(nu/(1-phi^2))` diverges while `sqrt(p)` stays finite.**
So a near-unit-root class member gets a window with nodes spaced `1.5 s` apart --
hundreds of nats -- and degenerates to a single node, i.e. a fixed filter.

That is a mechanism for 0007's measured finding. `_PHIS` cannot reach near 1 not
because the class does not want it -- the fit is rail-pinned on every seed and
reaching 0.995 is worth 2.9-3.5x -- but because **the window geometry breaks
there**.

At the shipped defaults the change is modest (`sqrt(p)/s` = 0.68-0.76 across the
box), so the question is whether it (i) holds the gates where the box already
works, and (ii) unlocks the near-1 reach that 0007 wants.

Run: python3 0011_window_scale.py
"""
import math
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from lucid import LucidFilter as ShippedFilter                     # noqa: E402

SRC = os.path.join(ROOT, "lucid", "filter", "lucid.py")
PATCHES = [
    ("        self.gap = _GAP_FACTOR * self.s_ax",
     "        self.w_ax = (self.s_ax if WINDOW_SCALE is None\n"
     "                     else WINDOW_SCALE(self.phi_ax, self.s_ax))\n"
     "        self.gap = _GAP_FACTOR * self.w_ax"),
    ("        self._Ifloor = (1.0 - self.phi_ax) / (4.0 * (_SPAN_S * self.s_ax) ** 2)",
     "        self._Ifloor = (1.0 - self.phi_ax) / (4.0 * (_SPAN_S * self.w_ax) ** 2)"),
    ("        self._Pmu_cap = (_SPAN_S * self.s_ax) ** 2",
     "        self._Pmu_cap = (_SPAN_S * self.w_ax) ** 2"),
    ("        self._Pmu = self.s_ax ** 2",
     "        self._Pmu = self.w_ax ** 2"),
]
CELL_PATCH = (
    "        cells = [(ph, sv, bq, br) for ph in phis for sv in ss for (bq, br) in bases]",
    "        cells = [(ph, sv, bq, br) for ph in phis for sv in ss for (bq, br) in bases]\n"
    "        if CELL_PAIRS is not None:\n"
    "            cells = [(ph, sv, bq, br) for (ph, sv) in CELL_PAIRS\n"
    "                     for (bq, br) in bases]")

RHO = 2.0          # 1/I for a log-scale, I(lam) = 1/2


def predictive_sd(phi, s):
    """SD of the lam channel's one-step predictive distribution."""
    phi, s = np.asarray(phi, float), np.asarray(s, float)
    nu = np.maximum(s * s * (1.0 - phi * phi), 1e-12)
    b = RHO * (1.0 - phi * phi) - nu
    p = 0.5 * (-b + np.sqrt(b * b + 4.0 * nu * RHO))
    return np.sqrt(p)


def load_patched():
    src = open(SRC).read()
    for old, new in PATCHES + [CELL_PATCH]:
        if src.count(old) != 1:
            raise SystemExit(f"patch anchor moved:\n{old}")
        src = src.replace(old, new)
    src = src.replace('from __future__ import annotations',
                      'from __future__ import annotations\n'
                      'WINDOW_SCALE = None\nCELL_PAIRS = None', 1)
    mod = types.ModuleType("lucid_ws")
    mod.__file__ = SRC
    sys.modules["lucid_ws"] = mod
    mod.__dict__["__name__"] = "lucid_ws"
    exec(compile(src, SRC, "exec"), mod.__dict__)
    return mod


def derived_ladder(forget=0.999, stride=1):
    """t-ladder; phi = sqrt(cos t) and s = sqrt(2 (sec t - 1)) is the local-level
    solution of `window = predictive SD`, so the ladder is self-consistent."""
    N = 1.0 / (1.0 - forget)
    blur = math.sqrt(2.0 / N)
    gap = 1.5 * blur * stride
    lo, hi = blur, math.pi / 2.0 - blur
    out = []
    for j in range(int((hi - lo) // gap) + 1):
        c = math.cos(lo + j * gap)
        out.append((math.sqrt(c), math.sqrt(2.0 * (1.0 / c - 1.0))))
    return out


SEED, N_T, JUMP_AT, JUMP, NOISE_AT = 11, 900, 380, 9.0, 600
Q_TRUE, S2_A, S2_C = 0.02, 1.0, 9.0
NSEED = 12
PHIS, SS = (0.70, 0.85, 0.95), (0.20, 0.40, 0.80, 1.60, 3.20)
PHIS4 = (0.70, 0.85, 0.95, 0.995)


def generate(seed):
    rng = np.random.default_rng(seed)
    th = np.cumsum(rng.normal(0.0, math.sqrt(Q_TRUE), N_T))
    th[JUMP_AT:] += JUMP
    sd = np.where(np.arange(N_T) < NOISE_AT, math.sqrt(S2_A), math.sqrt(S2_C))
    return th, th + rng.normal(0.0, sd)


def kalman(y, Q, R):
    m, P, out = y[0], R, np.empty(N_T)
    for t in range(N_T):
        P += Q
        g = P / (P + R)
        m = m + g * (y[t] - m)
        P = (1.0 - g) * P
        out[t] = m
    return out


def regimes():
    return [("A steady", slice(80, JUMP_AT)),
            ("B jump+40", slice(JUMP_AT, JUMP_AT + 40)),
            ("C noisy", slice(NOISE_AT + 40, N_T))]


def settle(err):
    seg = np.abs(err[JUMP_AT:JUMP_AT + 120]) < 1.0
    for i in range(seg.size):
        if seg[i:].all():
            return i
    return seg.size


if __name__ == "__main__":
    print(__doc__.split("Run:")[0])
    mod = load_patched()

    _, y = generate(SEED)
    mod.WINDOW_SCALE, mod.CELL_PAIRS = None, None
    a = np.asarray(mod.LucidFilter(phis=PHIS, ss=SS).filter(y.reshape(-1, 1)).mean)
    b = np.asarray(ShippedFilter(phis=PHIS, ss=SS).filter(y.reshape(-1, 1)).mean)
    dev = float(np.max(np.abs(a - b)))
    print(f"pin, both overrides off vs shipped: max|dev| = {dev:.3e}")
    if dev > 0.0:
        raise SystemExit("patched copy is not the shipped filter")
    print("  exact.\n")

    print("sqrt(p) / s across the shipped box (how much the window narrows):")
    for ph in PHIS:
        row = " ".join(f"{float(predictive_sd(ph, sv)) / sv:6.3f}" for sv in SS)
        print(f"    phi={ph:.2f}: {row}")
    print("  and at phi = 0.995: " +
          " ".join(f"{float(predictive_sd(0.995, sv)) / sv:6.3f}" for sv in SS)
          + "   <- where s diverges and sqrt(p) does not\n")

    lad = derived_ladder()
    VARIANTS = {
        "shipped        (15)": (None, None, PHIS, SS),
        "window=sqrt(p) (15)": (predictive_sd, None, PHIS, SS),
        "shipped  +.995 (20)": (None, None, PHIS4, SS),
        "win+p    +.995 (20)": (predictive_sd, None, PHIS4, SS),
        f"win+p  ladder ({len(lad)})": (predictive_sd, lad, PHIS, SS),
    }

    acc = {k: {r[0]: [] for r in regimes()} for k in VARIANTS}
    for k in VARIANTS:
        acc[k]["settle"], acc[k]["calib"] = [], []
    orc = {r[0]: [] for r in regimes()}

    for seed in range(SEED, SEED + NSEED):
        theta, y = generate(seed)
        km = kalman(y, Q_TRUE, S2_A)
        for nm, sl in regimes():
            orc[nm].append(float(np.mean((km[sl] - theta[sl]) ** 2)))
        for k, (ws, pairs, ph, ss) in VARIANTS.items():
            mod.WINDOW_SCALE, mod.CELL_PAIRS = ws, pairs
            r = mod.LucidFilter(phis=ph, ss=ss).filter(y.reshape(-1, 1))
            m = np.asarray(r.mean).reshape(-1)
            v = np.asarray(r.var).reshape(-1)
            for nm, sl in regimes():
                acc[k][nm].append(float(np.mean((m[sl] - theta[sl]) ** 2)))
            acc[k]["settle"].append(settle(m - theta))
            sl = regimes()[2][1]
            acc[k]["calib"].append(float(np.mean((m[sl] - theta[sl]) ** 2 / v[sl])))
    mod.WINDOW_SCALE, mod.CELL_PAIRS = None, None

    print(f"RMSE ratio to an oracle Kalman told the truth, {NSEED} seeds.")
    print("gates: steady <= 1.10x, jump rise <= 4, calibration in [0.6, 1.5]\n")
    print(f"{'variant':22s}{'A steady':>10s}{'B jump':>9s}{'C noisy':>9s}"
          f"{'settle':>8s}{'E[e2/S]':>9s}")
    for k in VARIANTS:
        row = [math.sqrt(np.mean(acc[k][nm]) / np.mean(orc[nm])) for nm, _ in regimes()]
        print(f"{k:22s}{row[0]:10.3f}{row[1]:9.3f}{row[2]:9.3f}"
              f"{np.mean(acc[k]['settle']):8.1f}{np.mean(acc[k]['calib']):9.3f}")


def reach_sweep():
    """Is the window a RESOLUTION grid or a REACH budget?

    Scale every axis's window geometry by a constant c, holding the class (prior
    and kernel) fixed.  If the window is a reach budget, the jump metric improves
    monotonically with c.  If it is a resolution grid, it degrades.
    """
    mod = load_patched()
    print("\n\n=== is the window resolution, or reach? ===")
    print("window geometry scaled by c, class held fixed:\n")
    print(f"{'c':>6s}{'A steady':>10s}{'B jump':>9s}{'C noisy':>9s}"
          f"{'settle':>8s}{'E[e2/S]':>9s}")
    orc = {r[0]: [] for r in regimes()}
    data = []
    for seed in range(SEED, SEED + NSEED):
        theta, y = generate(seed)
        km = kalman(y, Q_TRUE, S2_A)
        for nm, sl in regimes():
            orc[nm].append(float(np.mean((km[sl] - theta[sl]) ** 2)))
        data.append((theta, y))
    for c in (0.5, 0.7, 1.0, 1.4, 2.0, 3.0):
        mod.WINDOW_SCALE = (lambda ph, s, c=c: c * np.asarray(s, float))
        mod.CELL_PAIRS = None
        acc = {r[0]: [] for r in regimes()}
        st, cb = [], []
        for theta, y in data:
            r = mod.LucidFilter(phis=PHIS, ss=SS).filter(y.reshape(-1, 1))
            m = np.asarray(r.mean).reshape(-1)
            v = np.asarray(r.var).reshape(-1)
            for nm, sl in regimes():
                acc[nm].append(float(np.mean((m[sl] - theta[sl]) ** 2)))
            st.append(settle(m - theta))
            sl = regimes()[2][1]
            cb.append(float(np.mean((m[sl] - theta[sl]) ** 2 / v[sl])))
        row = [math.sqrt(np.mean(acc[nm]) / np.mean(orc[nm])) for nm, _ in regimes()]
        print(f"{c:6.1f}{row[0]:10.3f}{row[1]:9.3f}{row[2]:9.3f}"
              f"{np.mean(st):8.1f}{np.mean(cb):9.3f}")
    mod.WINDOW_SCALE = None
