"""Render the LucidFilter 3D drone demo to a GIF for the main README.

Real output of the PUBLIC filter (``from lucid import LucidFilter``) on the 3D quadrotor rig
of `drone3d.py`: a delivery drone flies a job, picks a heavy crate up OFF CENTRE mid-flight,
carries it through a gust and a GPS dropout, sets it down, and flies home.  The filter is
told the empty vehicle and nothing else.  It has to work out, live and from the same twelve
noisy channels, both *which sensor to stop trusting* and *that the aircraft it is flying is
no longer the one it was given* -- and then, when the crate goes, that it is again.

The regime that separates them is the GPS one, and it is the canonical avionics failure
rather than a rig contrivance: the fix does not merely get noisy, it CUTS OUT -- position and
velocity together, because they come out of one receiver -- and comes back degraded.  While
it is gone there is nothing to distrust and no noise level to have tuned: the only thing
carrying the estimate across the gap is the model of the aircraft.  The fixed filter's model
is the empty airframe it was handed; a 0.42 kg crate is aboard, and the mass error alone is
worth 3.7 m/s^2 of dead-reckoned vertical acceleration.  `drone3d.dropouts` schedules it and
is off by default, so the acceptance rig 0008 measures is untouched.

**The filter is IN THE LOOP, and so is its opponent.**  Two aircraft fly the same job in the
same air on the same sensor noise, and each is flown by its own estimator -- nothing else
reaches the autopilot.  The opponent is not raw GPS: nothing flies on raw GPS, and racing
against something nobody would deploy proves nothing.  It is a fixed Kalman filter on the
same nominal airframe, with its noise levels **tuned in hindsight on this very mission**
(`drone3d.tune_fixed`) -- the best a fixed filter could have been set to if you had already
flown the flight and kept the truth.  What it cannot do is change its mind partway through,
and it is never told about the crate.

Panels: a rotating 3D view of the two aircraft against the path they were both commanded to
fly, the per-component "which noise is hot" chip grid, the payload the lucid filter reports
(mass and centre-of-mass offset, read off the public ``.control``), and a position-error
scope racing the raw GPS fix and the hindsight-tuned Kalman against the lucid estimate.  The
camera sweeps a full 360 degrees over the loop, so the loop is seamless.

    python make_drone3d_lucid_gif.py        # writes ../figures/drone3d-lucid.gif (+ .mp4)

Two knobs at the top set the pacing, and they are the ones to reach for when the animation is
hard to read: ``REGIME`` runs every regime (and the mission carrying it) that many times
longer, and ``STEP`` is how many simulated steps one frame advances -- lower is slower and
smoother on screen, and costs frames.
"""
import math
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from PIL import Image                                                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import drone3d as R                                                    # noqa: E402

OUT = os.path.join(HERE, "..", "figures", "drone3d-lucid.gif")

# ------------------------------------------------------------------------- pacing
REGIME = 2.0            # every regime, hover and mission stage runs this many times longer
STEP = 12               # simulated steps per frame -- the playback rate, with FRAME_MS
FRAME_MS = 92           # milliseconds per frame in the GIF
SEED = 0
COLORS = 96             # GIF palette size, shared by every frame (see palette_for)
DPI = 92
GPS_ON, GPS_OFF = 0.5, 0.5     # seconds of fix, seconds of nothing, through the GPS regime

R.set_regime(REGIME)

# ---------------------------------------------------------------------------- style
BG, PANEL, LINE, INK, MUT = "#0e1217", "#161c24", "#2a3542", "#dbe3ec", "#7c8a99"
C_GPS, C_LU, C_FIX, C_WARN = "#e2607b", "#33e0a6", "#8fa2ff", "#f5b23d"
C_CRATE, C_CRATE_DIM = "#d9a05b", "#6d5637"
NORM = 5.0                                    # chip scale: x12 -> 2 ln 12 = 5.0
EL = math.radians(24)
TRAIL = int(round(130 * REGIME))

PHASE_TXT = {"calm": ("calm — clear air, clean fix", MUT),
             "WIND": ("gust — the air is shoving the airframe", C_WARN),
             "MULTIPATH": ("GPS dropout — the fix cuts out, and comes back bad", C_WARN),
             "VIBRATION": ("rotor damage — the rate gyros go noisy", C_WARN)}


def phase_at(i):
    for nm, a, b in R.PHASES:
        if a <= i < b:
            return nm
    return "calm"


# ------------------------------------------------------------------- 3D scene geometry
PAD_A = np.array([-1.22, -0.79])              # the pick-up pad
PAD_B = np.array([1.40, 0.94])                # the set-down pad
CRATE = 0.22                                  # crate edge (m)
GRID = np.linspace(-2.1, 2.1, 7)


def project(Pw, az):
    """Orthographic projection of world points, azimuth ``az`` and a fixed elevation."""
    x, y, z = Pw[..., 0], Pw[..., 1], Pw[..., 2]
    sx = -x * math.sin(az) + y * math.cos(az)
    sy = z * math.cos(EL) - (x * math.cos(az) + y * math.sin(az)) * math.sin(EL)
    return np.stack([sx, sy], -1)


def az_at(i):
    return math.radians(34) + 2.0 * math.pi * i / R.T


_ROT = np.array([[math.cos(a), math.sin(a)] for a in np.radians([45, 135, 225, 315])])
_DISC = np.stack([np.cos(np.linspace(0, 2 * math.pi, 17)),
                  np.sin(np.linspace(0, 2 * math.pi, 17))], 1) * 0.105


def airframe(p, att):
    """The drone's drawable parts in world coordinates: (arms, 4 rotor discs, body)."""
    Rm = R.Rmat(att)
    hub = np.concatenate([_ROT * R.ARM, np.zeros((4, 1))], 1)
    arms = []
    for h in hub:
        arms.append(np.stack([p, p + Rm @ h]))
    discs = [(p + Rm @ h)[None] + _DISC @ Rm[:, :2].T for h in hub]
    body = np.array([[0.07, 0.07, 0], [-0.07, 0.07, 0], [-0.07, -0.07, 0],
                     [0.07, -0.07, 0], [0.07, 0.07, 0]]) @ Rm.T + p
    return arms, discs, body


def box(centre, edge, Rm=None):
    """The 12 edges of an axis-aligned (or rotated) cube, as a list of 2-point segments."""
    h = edge / 2.0
    v = np.array([[sx * h, sy * h, sz * h] for sx in (-1, 1) for sy in (-1, 1)
                  for sz in (-1, 1)])
    if Rm is not None:
        v = v @ Rm.T
    v = v + np.asarray(centre)
    ed = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7),
          (4, 5), (4, 6), (5, 7), (6, 7)]
    return [np.stack([v[a], v[b]]) for a, b in ed]


def palette_for(frames):
    """One palette for the whole animation, with the colours that carry meaning nailed down.

    A per-frame adaptive palette makes consecutive frames differ in every pixel, which costs
    about twice the file size once frames are stored as differences.  A shared palette fixes
    that, but a quantiser handed a screenful of dark panels spends its entries on the dark
    and merges the accents -- median cut moved the blue aircraft to a grey, which is the one
    error this animation cannot afford, since telling the blue line from the green line IS
    the figure.  So the accents, their blends against both backgrounds, and the chip ramp are
    placed by hand, and only the leftover entries are chosen from the frames.
    """
    rgb = matplotlib.colors.to_rgb
    bg, panel = np.array(rgb(BG)), np.array(rgb(PANEL))
    fixed = []

    def ramp(base, col, n):
        c = np.array(rgb(col))
        for a in np.linspace(1.0 / n, 1.0, n):
            fixed.append(tuple(np.round(255 * (base + a * (c - base))).astype(int)))

    ramp(bg, C_WARN, 10)                                  # the chip heat scale
    for col in (C_GPS, C_LU, C_FIX, C_CRATE, C_CRATE_DIM, INK, MUT, LINE):
        ramp(bg, col, 4)                                  # strokes on the dark ground
    for col in (C_GPS, C_LU, C_FIX, INK, MUT, LINE):
        ramp(panel, col, 3)                               # ... and on a panel
    fixed += [tuple(np.round(255 * bg).astype(int)), tuple(np.round(255 * panel).astype(int))]
    seen, cols = set(), []
    for c in fixed:
        if c not in seen:
            seen.add(c); cols.append(c)

    keep = list(frames[::max(1, len(frames) // 12)])
    spare = max(1, COLORS - len(cols))
    extra = Image.fromarray(np.concatenate(keep, 0)).convert(
        "P", palette=Image.ADAPTIVE, colors=spare)
    for c in map(tuple, np.array(extra.getpalette()[:3 * spare]).reshape(-1, 3)):
        if c not in seen:
            seen.add(c); cols.append(c)
    pal = Image.new("P", (1, 1))
    flat = [int(v) for c in cols for v in c]
    pal.putpalette(flat + flat[-3:] * (256 - len(cols)))   # pad with a real colour, not black
    return pal


def crate_pose(k, X):
    """(centre, rotation) of the crate at step k -- on a pad, or hanging off the drone."""
    if k < R.T_PICK:
        return np.array([PAD_A[0], PAD_A[1], CRATE / 2]), np.eye(3)
    if k >= R.T_DROP:
        return np.array([PAD_B[0], PAD_B[1], CRATE / 2]), np.eye(3)
    Rm = R.Rmat(X[k, R.AT])
    return X[k, R.PX] + Rm @ R.D_P, Rm


def render():
    sv, sw = R.schedule()
    W, V = R.draw_noise(SEED, sv, sw)
    sense = R.dropouts(GPS_ON, GPS_OFF)

    print("hindsight-tuning the fixed Kalman filter on this mission...")
    t0 = time.time()
    tuned, tuned_rms = R.tune_fixed(W, V, sense=sense)
    print(f"  (q, r_gps, r_vel, r_ahrs, r_gyro) = {tuned}   its own RMSE {tuned_rms:.4f} m"
          f"   [{time.time() - t0:.0f}s]")

    print("flying the mission on the hindsight-tuned Kalman filter...")
    U_fx, X_fx, Y_fx, XH_fx, hold = R.fly(R.fixed_pilot(tuned), W, V, sense=sense)

    print(f"flying the same mission on LucidFilter ({R.T} steps) -- a few minutes...")
    t0 = time.time()
    rec = {"measurement_scale": np.zeros((R.T, R.M)), "process_scale": np.zeros((R.T, R.N)),
           "control": np.zeros((R.T, R.N, R.P)), "fault": np.zeros(R.T)}
    U_lu, X_lu, Y_lu, XH_lu, hold = R.fly(R.lucid_pilot(rec), W, V, sense=sense)
    print(f"  [{time.time() - t0:.0f}s]")
    m_hat, _, c_hat = R.read_payload(rec["control"])

    # Each aircraft's error is against ITS OWN flight: that is what its autopilot was handed.
    p_lu, p_fx = X_lu[:, R.PX], X_fx[:, R.PX]
    err_lu = np.linalg.norm(XH_lu[:, R.PX] - p_lu, axis=1)
    err_fx = np.linalg.norm(XH_fx[:, R.PX] - p_fx, axis=1)
    err_gps = np.linalg.norm(Y_lu[:, 0:3] - p_lu, axis=1)
    apart = np.linalg.norm(p_lu - p_fx, axis=1)

    ref_p = R.reference()[0][:, :3]

    # -------- the chip grid: 3 rows (absolute / rate / disturbance) x 6 axis columns
    ms, ps = rec["measurement_scale"], rec["process_scale"]
    chip = np.zeros((R.T, 3, 6))
    chip[:, 0, 0:3] = ms[:, 0:3]                          # GPS position
    chip[:, 1, 0:3] = ms[:, 3:6]                          # GPS velocity
    chip[:, 0, 3:6] = ms[:, 6:9]                          # AHRS attitude
    chip[:, 1, 3:6] = ms[:, 9:12]                         # rate gyro
    for a in range(6):
        chip[:, 2, a] = ps[:, R.MODE_OF_AXIS[a]]          # wind force / disturbance torque
    # a channel that is not reading is not a channel that is noisy -- the grid says so
    CHIP_SENSE = np.ones((R.T, 3, 6), bool)
    CHIP_SENSE[:, 0, 0:3] = sense[:, 0:3]
    CHIP_SENSE[:, 1, 0:3] = sense[:, 3:6]
    CHIP_SENSE[:, 0, 3:6] = sense[:, 6:9]
    CHIP_SENSE[:, 1, 3:6] = sense[:, 9:12]

    # -------- the payload readout: raw per-step, and a 0.6 s running mean
    def smooth(v, w=60):
        k = np.ones(w) / w
        pad = np.concatenate([np.full(w - 1, v[0]), v])
        return np.convolve(pad, k, mode="valid")

    pay_raw = m_hat - R.M0
    pay = smooth(pay_raw)
    off = smooth(100.0 * np.linalg.norm(c_hat, axis=1))
    pay_true = np.where(hold, R.M_P, 0.0)
    off_true = 100.0 * np.linalg.norm(R.C_FULL[:2])

    print("laying out...")
    plt.rcParams.update({"font.family": "monospace", "font.size": 9, "text.color": INK,
                         "axes.edgecolor": LINE, "xtick.color": MUT, "ytick.color": MUT})
    fig = plt.figure(figsize=(11.0, 5.7), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax_s = fig.add_axes([0.002, 0.015, 0.475, 0.845]); ax_s.set_facecolor(PANEL)
    ax_d = fig.add_axes([0.515, 0.590, 0.475, 0.230]); ax_d.set_facecolor(PANEL)
    ax_p = fig.add_axes([0.575, 0.295, 0.415, 0.155]); ax_p.set_facecolor(PANEL)
    ax_e = fig.add_axes([0.575, 0.068, 0.415, 0.147]); ax_e.set_facecolor(PANEL)

    fig.text(0.004, 0.950, "LucidFilter — a delivery drone picks up a heavy crate, "
             "off centre", color=INK, fontsize=14.5, weight="bold", family="sans-serif")
    fig.text(0.004, 0.900, "GPS + AHRS + gyro · the filter is told the EMPTY aircraft · "
             "nothing tuned · from lucid import LucidFilter", color=MUT, fontsize=8.0)
    regime = fig.text(0.515, 0.532, "", fontsize=9.2, weight="bold", family="monospace",
                      va="center", ha="left")
    event = fig.text(0.012, 0.815, "", fontsize=10.5, weight="bold", family="monospace",
                     va="center", ha="left")
    for lx, lab, c in [(0.575, "raw GPS fix", C_GPS), (0.700, "fixed Kalman", C_FIX),
                       (0.820, "lucid", C_LU)]:
        fig.text(lx, 0.238, "▬ " + lab, color=c, fontsize=7.6, va="center")

    # ---- scene ----
    seen = np.concatenate([p_lu, p_fx, np.column_stack([p_lu[:, 0], p_lu[:, 1],
                                                        np.zeros(R.T)])])
    sp = np.concatenate([project(seen[::7], az_at(i)) for i in range(0, R.T, 40)])
    pad = R.ARM + 0.12
    ax_s.set_xlim(sp[:, 0].min() - pad, sp[:, 0].max() + pad)
    ax_s.set_ylim(sp[:, 1].min() - pad, sp[:, 1].max() + pad)
    ax_s.set_aspect("equal"); ax_s.axis("off")
    ax_s.text(0.02, 0.075, "both aircraft fly the same job, in the same air, on the same "
              "sensors —", transform=ax_s.transAxes, color=MUT, fontsize=7.6, va="bottom")
    ax_s.text(0.02, 0.036, "each one flown by its own filter, and by nothing else",
              transform=ax_s.transAxes, color=MUT, fontsize=7.6, va="bottom")
    for lx, lab, c in [(0.02, "● lucid", C_LU),
                       (0.155, "● hindsight-tuned Kalman", C_FIX),
                       (0.505, "-- commanded path", MUT),
                       (0.785, "✕ raw GPS fix", C_GPS)]:
        ax_s.text(lx, -0.002, lab, transform=ax_s.transAxes, color=c, fontsize=8.0,
                  va="bottom")

    glines = [ax_s.plot([], [], "-", color=LINE, lw=0.6, alpha=0.5, zorder=1)[0]
              for _ in range(2 * len(GRID))]
    pads = [ax_s.plot([], [], "-", color=MUT, lw=1.3, alpha=0.8, zorder=2)[0]
            for _ in range(2)]
    shadow, = ax_s.plot([], [], "o", color="#000000", ms=7, alpha=0.35, zorder=2)
    drop, = ax_s.plot([], [], "-", color=LINE, lw=0.8, alpha=0.6, zorder=2)
    crate_fx = [ax_s.plot([], [], "-", color=C_CRATE_DIM, lw=1.4, zorder=4)[0]
                for _ in range(12)]
    crate_lu = [ax_s.plot([], [], "-", color=C_CRATE, lw=1.7, zorder=6)[0] for _ in range(12)]
    teth_fx, = ax_s.plot([], [], "-", color=C_CRATE_DIM, lw=0.9, alpha=0.7, zorder=4)
    teth_lu, = ax_s.plot([], [], "-", color=C_CRATE, lw=1.0, alpha=0.7, zorder=6)
    fx_l = ([ax_s.plot([], [], "-", color=C_FIX, lw=2.0, alpha=.85, zorder=3)[0]
             for _ in range(4)]
            + [ax_s.plot([], [], "-", color=C_FIX, lw=1.3, alpha=.75, zorder=3)[0]
               for _ in range(5)])
    lu_l = ([ax_s.plot([], [], "-", color=C_LU, lw=2.8, zorder=5)[0] for _ in range(4)]
            + [ax_s.plot([], [], "-", color=C_LU, lw=1.7, zorder=5)[0] for _ in range(5)])
    gps_m, = ax_s.plot([], [], "x", color=C_GPS, ms=8, mew=1.8, alpha=.9, zorder=7)
    trail_fx, = ax_s.plot([], [], "-", color=C_FIX, lw=1.0, alpha=.28, zorder=2)
    trail_lu, = ax_s.plot([], [], "-", color=C_LU, lw=1.1, alpha=.32, zorder=2)

    # ---- the commanded path, as a ghost.  With the two aircraft on screen at once, what
    # makes the comparison readable is the third thing: the line both were asked to fly.
    cmd, = ax_s.plot([], [], "--", color=MUT, lw=1.0, alpha=.55, zorder=2)

    # ---- chips: rows are the sensor (and the disturbance), columns are the axis ----
    COL = [0.0, 1.0, 2.0, 4.85, 5.85, 6.85]
    ax_d.set_xlim(-1.85, 7.55); ax_d.set_ylim(-1.35, 3.25); ax_d.invert_yaxis()
    ax_d.axis("off")
    ax_d.set_title("which noise is hot — learned online, per component", color=MUT,
                   fontsize=9, loc="left", pad=6)
    chips = {}
    for row, (left, right) in enumerate([("GPS fix", "AHRS"), ("GPS vel", "gyro"),
                                         ("wind", "torque")]):
        ax_d.text(-0.55, row, left, color=MUT, fontsize=7.8, va="center", ha="right")
        ax_d.text(4.30, row, right, color=MUT, fontsize=7.8, va="center", ha="right")
        for a in range(6):
            rect = plt.Rectangle((COL[a] - 0.36, row - 0.32), 0.72, 0.64, facecolor=BG,
                                 edgecolor=LINE, lw=1, zorder=2)
            ax_d.add_patch(rect); chips[(row, a)] = rect
    for a, nm in enumerate(["x", "y", "z", "roll", "pitch", "yaw"]):
        ax_d.text(COL[a], -0.68, nm, color=MUT, fontsize=7.6, ha="center")
    ax_d.text(1.0, -1.18, "POSITION", color=INK, fontsize=7.8, ha="center", alpha=.75)
    ax_d.text(-1.80, -1.18, "dotted = no reading", color=C_GPS, fontsize=6.6, alpha=.9)
    ax_d.text(5.85, -1.18, "ATTITUDE", color=INK, fontsize=7.8, ha="center", alpha=.75)
    ax_d.plot([3.55, 3.55], [-0.95, 2.4], "-", color=LINE, lw=1)
    q, rp, rv, ra, rg = tuned
    ax_d.text(-1.80, 2.72, "what the fixed filter gets instead: its noise tuned in hindsight "
              "on this very flight —", color=C_FIX, fontsize=7.2, va="center")
    ax_d.text(-1.80, 3.05, f"Q ×{q:g},  σ ×{math.sqrt(rp):g} GPS fix,  "
              f"×{math.sqrt(rv):g} GPS vel,  ×{math.sqrt(ra):g} AHRS,  "
              f"×{math.sqrt(rg):g} gyro — one setting, held all mission",
              color=C_FIX, fontsize=7.2, va="center")

    # ---- payload ----
    tt = np.arange(R.T) * R.DT
    ax_p.set_xlim(0, R.T * R.DT); ax_p.set_ylim(-0.44, 0.86)
    ax_p.set_ylabel("payload (kg)", color=MUT, fontsize=8.2)
    ax_p.set_yticks([0.0, 0.5])
    ax_p.tick_params(length=3, labelsize=7.5, labelbottom=False)
    ax_p.set_title("what it says it is carrying — read off .control, live",
                   color=MUT, fontsize=8.4, loc="left", pad=3)
    ax_p.plot(tt, pay_true, "--", color=INK, lw=1.2, alpha=.55)
    pr, = ax_p.plot([], [], "-", color=C_LU, lw=0.7, alpha=.28)
    pm, = ax_p.plot([], [], "-", color=C_LU, lw=1.9)
    ptxt = ax_p.text(0.014, 0.045, "", transform=ax_p.transAxes, color=C_LU, fontsize=8.6,
                     va="bottom", family="monospace")

    # ---- error scope ----
    ax_e.set_xlim(0, R.T * R.DT)
    ax_e.set_yscale("log"); ax_e.set_ylim(0.005, 20.0)
    ax_e.set_yticks([0.01, 0.1, 1.0, 10.0]); ax_e.set_yticklabels(["1cm", "10cm", "1m", "10m"])
    for nm, a, b in R.PHASES:
        if nm != "calm":
            ax_e.axvspan(a * R.DT, b * R.DT, color=C_WARN, alpha=0.10, lw=0)
    for k in (R.T_PICK, R.T_DROP):
        ax_e.axvline(k * R.DT, color=C_CRATE, lw=1.0, alpha=0.55, ls=":")
        ax_p.axvline(k * R.DT, color=C_CRATE, lw=1.0, alpha=0.55, ls=":")
    ax_e.set_ylabel("pos err (m)", color=MUT, fontsize=8.2)
    ax_e.set_xlabel("time (s)", color=MUT, fontsize=8.2)
    ax_e.tick_params(length=3, labelsize=7.5)
    e_gps, = ax_e.plot([], [], color=C_GPS, lw=0.9, alpha=.5)
    e_fx, = ax_e.plot([], [], color=C_FIX, lw=1.2, alpha=.85)
    e_lu, = ax_e.plot([], [], color=C_LU, lw=1.7)
    head = ax_e.axvline(0, color=INK, lw=1, alpha=.5)

    def heat(v):
        a = float(np.clip(v / NORM, 0.0, 1.0))
        c0 = np.array(matplotlib.colors.to_rgb(BG))
        c1 = np.array(matplotlib.colors.to_rgb(C_WARN))
        return tuple(c0 + a * (c1 - c0))

    def draw_craft(lines, p, att):
        arms, discs, body = airframe(p, att)
        for j, a in enumerate(arms):
            q = project(a, draw.az); lines[j].set_data(q[:, 0], q[:, 1])
        for j, d in enumerate(discs):
            q = project(d, draw.az); lines[4 + j].set_data(q[:, 0], q[:, 1])
        q = project(body, draw.az); lines[8].set_data(q[:, 0], q[:, 1])

    def draw_crate(segs, tether, i, X):
        cc, cR = crate_pose(i, X)
        for j, seg in enumerate(box(cc, CRATE, cR)):
            q = project(seg, draw.az); segs[j].set_data(q[:, 0], q[:, 1])
        if hold[i]:
            q = project(np.stack([X[i, R.PX], cc]), draw.az)
            tether.set_data(q[:, 0], q[:, 1])
        else:
            tether.set_data([], [])

    def draw(i):
        draw.az = az = az_at(i)
        for j, g in enumerate(GRID):
            a3 = np.array([[g, GRID[0], 0.0], [g, GRID[-1], 0.0]])
            b3 = np.array([[GRID[0], g, 0.0], [GRID[-1], g, 0.0]])
            qa, qb = project(a3, az), project(b3, az)
            glines[j].set_data(qa[:, 0], qa[:, 1])
            glines[j + len(GRID)].set_data(qb[:, 0], qb[:, 1])
        for j, pd in enumerate((PAD_A, PAD_B)):
            sq = np.array([[pd[0] + dx, pd[1] + dy, 0.0] for dx, dy in
                           [(-.32, -.32), (.32, -.32), (.32, .32), (-.32, .32), (-.32, -.32)]])
            q = project(sq, az); pads[j].set_data(q[:, 0], q[:, 1])
        draw_craft(fx_l, p_fx[i], X_fx[i, R.AT])
        draw_craft(lu_l, p_lu[i], X_lu[i, R.AT])
        g = project(Y_lu[i, 0:3][None], az); gps_m.set_data(g[:, 0], g[:, 1])
        sh = project(np.array([[p_lu[i, 0], p_lu[i, 1], 0.0]]), az)
        shadow.set_data(sh[:, 0], sh[:, 1])
        dl = project(np.array([p_lu[i], [p_lu[i, 0], p_lu[i, 1], 0.0]]), az)
        drop.set_data(dl[:, 0], dl[:, 1])
        draw_crate(crate_fx, teth_fx, i, X_fx)
        draw_crate(crate_lu, teth_lu, i, X_lu)
        lo = max(0, i - TRAIL)
        tr = project(p_lu[lo:i + 1], az); trail_lu.set_data(tr[:, 0], tr[:, 1])
        tr = project(p_fx[lo:i + 1], az); trail_fx.set_data(tr[:, 0], tr[:, 1])
        tr = project(ref_p[lo:i + 1], az); cmd.set_data(tr[:, 0], tr[:, 1])

        txt, col = PHASE_TXT[phase_at(i)]
        regime.set_text("ACTIVE REGIME: " + txt); regime.set_color(col)
        span = int(round(190 * REGIME))
        if 0 <= i - R.T_PICK < span:
            event.set_text("▲ CRATE ATTACHED"); event.set_color(C_CRATE)
        elif 0 <= i - R.T_DROP < span:
            event.set_text("▼ CRATE RELEASED"); event.set_color(C_CRATE)
        else:
            event.set_text("")
        for row in range(3):
            for a in range(6):
                rect = chips[(row, a)]
                live = CHIP_SENSE[i, row, a]
                rect.set_facecolor(heat(chip[i, row, a]) if live else PANEL)
                rect.set_edgecolor(LINE if live else C_GPS)
                rect.set_linestyle("-" if live else ":")
        pr.set_data(tt[:i + 1], pay_raw[:i + 1])
        pm.set_data(tt[:i + 1], pay[:i + 1])
        ptxt.set_text(f"carrying {pay[i]:+.2f} kg, {off[i]:.1f} cm off centre"
                      if pay[i] > 0.12 else "carrying nothing")
        e_gps.set_data(tt[:i + 1], err_gps[:i + 1])
        e_fx.set_data(tt[:i + 1], err_fx[:i + 1])
        e_lu.set_data(tt[:i + 1], err_lu[:i + 1])
        head.set_xdata([i * R.DT, i * R.DT])

    frames = list(range(0, R.T, STEP))
    print("rendering", len(frames), "frames...")
    fig.canvas.draw()
    raws = []
    for i in frames:
        draw(i); fig.canvas.draw()
        raws.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    master = palette_for(raws)
    imgs = [Image.fromarray(rr).quantize(palette=master, dither=Image.NONE) for rr in raws]
    # disposal=1 leaves each frame in place and stores the next as a difference; with the
    # shared palette above that is worth about half the file.
    imgs[0].save(OUT, save_all=True, append_images=imgs[1:], duration=FRAME_MS, loop=0,
                 disposal=1, optimize=True)
    print("wrote", os.path.relpath(OUT), "-", os.path.getsize(OUT) // 1024, "KB,",
          len(frames), "frames,", f"{len(frames) * FRAME_MS / 1000:.0f}s")
    try:
        import imageio
        h, w = raws[0].shape[:2]
        if h % 2 or w % 2:
            bg = np.array([14, 18, 23], dtype=np.uint8)
            H2, W2 = h + h % 2, w + w % 2
            raws = [np.pad(rr, ((0, H2 - h), (0, W2 - w), (0, 0))) for rr in raws]
            for rr in raws:
                rr[h:, :] = bg; rr[:, w:] = bg
        mp4 = OUT[:-4] + ".mp4"
        imageio.mimwrite(mp4, raws, fps=1000.0 / FRAME_MS, codec="libx264", quality=8,
                         pixelformat="yuv420p")
        print("wrote", os.path.relpath(mp4), "-", os.path.getsize(mp4) // 1024, "KB")
    except ImportError:
        print("imageio not installed -- skipped the .mp4")

    # ---------------------------------------------------------- the numbers the README cites
    rm = lambda e, s: float(np.sqrt(np.nanmean(e[s] ** 2)))                 # noqa: E731
    off_fx = np.linalg.norm(p_fx - ref_p, axis=1)      # how far off the job each one flew
    off_lu = np.linalg.norm(p_lu - ref_p, axis=1)
    print(f"\n{'':12s}{'--- how far from the truth ---':>38}"
          f"{'-- how far off the job --':>27}")
    print(f"{'window':12s}{'raw GPS':>10}{'fixed KF':>10}{'lucid':>10}{'ratio':>8}"
          f"{'fixed KF':>11}{'lucid':>8}{'apart':>8}")
    wins = [("whole run", slice(0, R.T))] + [(nm, slice(a, b)) for nm, a, b in R.PHASES]
    for nm, s in wins:
        print(f"{nm:12s}{rm(err_gps, s):10.3f}{rm(err_fx, s):10.4f}{rm(err_lu, s):10.4f}"
              f"{rm(err_fx, s) / rm(err_lu, s):8.2f}{rm(off_fx, s):11.3f}"
              f"{rm(off_lu, s):8.3f}{apart[s].max():8.3f}")
    settle = int(round(400 * REGIME))
    carry = slice(R.T_PICK + settle, R.T_DROP)
    after = slice(R.T_DROP + settle, R.T)
    print(f"\npayload  carried {pay[carry].mean():.3f} kg (true {R.M_P})   "
          f"after the drop {pay[after].mean():.3f} kg (true 0.0)")
    print(f"offset   carried {off[carry].mean():.2f} cm (true {off_true:.2f})   "
          f"after the drop {off[after].mean():.2f} cm (true 0.0)")
    fault = rec["fault"]
    cr = np.flatnonzero(fault[R.T_PICK:] > 0.5)
    print(f"fault marginal crosses 0.5 {cr[0] if cr.size else 'never'} steps after the "
          f"pick-up; {int((fault[:R.T_PICK] > 0.5).sum())} steps flagged before it")
    print(f"lowest altitude flown: lucid {X_lu[:, 2].min():.2f} m, "
          f"fixed Kalman {X_fx[:, 2].min():.2f} m")


if __name__ == "__main__":
    render()
