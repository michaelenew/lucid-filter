"""Render the LucidFilter 5-DOF arm demo to a GIF for the main README.

Real output of the PUBLIC filter (`from lucid import LucidFilter`) on the 3D 5-DOF arm rig of
`arm5dof.py` -- the common chain (yaw + shoulder pitch at the base, elbow pitch + forearm
roll, one wrist flex), a bad potentiometer plus a link-mounted MEMS accelerometer per joint,
driven along a commanded trajectory by a servo that flies on the potentiometers, with noise
arriving in phases:

    calm | SENSOR (accelerometers swamped) | calm | PROCESS (disturbance jerk) |
    calm | POT FAILURE (one joint's potentiometer dies) | calm | BOTH | calm

The measurement map is a CALLABLE, and has to be: a linear accelerometer reads gravity in a
link frame that moves with every joint below it, configuration-dependent lever arms on the
joint accelerations, and terms quadratic in the rates -- there is no constant `H`, so the map
is linearised at every step, exactly as `F` would be.  `arm5dof.py` builds it and `0054` pins
the complex-step Jacobian against central differences and measures what freezing the
linearisation costs (`../exploration/0054_physical_sensors.md`).

Panels: a rotating 3D view of the arm (true vs raw-pot vs lucid estimate), a per-component
"which noise is hot" chip grid (the learned log-scales, per joint: pots / accels / process
modes), and a tip-error scope racing the raw pot and a fixed-noise KF against the lucid
estimate.  The camera sweeps a full 360 degrees over the loop, so the loop is seamless.

    python make_arm5dof_lucid_gif.py            # writes ../figures/arm5dof-lucid.gif
"""
import os
import sys
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
import arm5dof as R  # noqa: E402

OUT = os.path.join(HERE, "..", "figures", "arm5dof-lucid.gif")

NJ, ORDER, DT = R.NJ, R.ORDER, R.DT
POT_MULT, ACC_MULT, JERK_MULT = 15.0, 15.0, 20.0
FAIL_J = 2                                   # the joint whose potentiometer dies in POTFAIL
T = 1900
PHASES = [("calm", 0, 250), ("SENSOR", 250, 500), ("calm", 500, 650),
          ("PROCESS", 650, 900), ("calm", 900, 1050), ("POTFAIL", 1050, 1300),
          ("calm", 1300, 1450), ("BOTH", 1450, 1700), ("calm", 1700, 1900)]
MODE_OF_JOINT = R.MODE_OF_JOINT
joints3d = R.joints3d
EL = math.radians(22)


def schedule():
    jstd = np.full(T, R.JERK)
    pot_s = np.full((T, NJ), R.POT)
    acc_s = np.full((T, NJ), R.ACC)
    for name, a, b in PHASES:
        if name == "SENSOR":
            acc_s[a:b] = R.ACC * ACC_MULT
        elif name == "PROCESS":
            jstd[a:b] = R.JERK * JERK_MULT
        elif name == "POTFAIL":
            pot_s[a:b, FAIL_J] = R.POT * POT_MULT
        elif name == "BOTH":
            acc_s[a:b] = R.ACC * ACC_MULT
            jstd[a:b] = R.JERK * JERK_MULT
    return jstd, pot_s, acc_s


def simulate(seed=0):
    jstd, pot_s, acc_s = schedule()
    U, S, Y = R.simulate(seed, jstd, pot_s, acc_s)
    return U, S, Y, jstd, pot_s, acc_s


def fixed_kf(U, Y):
    """The same model frozen at the base noise -- and given the SAME live measurement map,
    so the comparison is about the noise and not about handing one contender a better
    sensor model than the other."""
    return R.kalman(U, Y)


def project(P, az):
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    sx = -x * math.sin(az) + y * math.cos(az)
    sy = z * math.cos(EL) - (x * math.cos(az) + y * math.sin(az)) * math.sin(EL)
    return np.stack([sx, sy], -1)


def az_at(i):                                     # full turn over the loop -> seamless
    return math.radians(38) + 2.0 * math.pi * i / T


# ---------------------------------------------------------------- style
BG, PANEL, LINE, INK, MUT = "#0e1217", "#161c24", "#2a3542", "#dbe3ec", "#7c8a99"
C_TRUE, C_POT, C_LU, C_FIX, C_WARN = "#c3d0de", "#e2607b", "#33e0a6", "#8fa2ff", "#f5b23d"

# plain-language regime names, intelligible from the animation alone (fit the label width)
PHASE_TXT = {"calm": ("calm — everything nominal", MUT),
             "SENSOR": ("all accelerometers get noisy", C_WARN),
             "PROCESS": ("vibration shakes the arm itself", C_WARN),
             "POTFAIL": (f"position sensor J{FAIL_J + 1} starts failing", C_WARN),
             "BOTH": ("vibration + noisy accelerometers", C_WARN)}


def phase_at(i):
    for nm, a, b in PHASES:
        if a <= i < b:
            return nm
    return "calm"


def render():
    print("simulating + filtering (LucidFilter, full bank)...")
    U, S, Y, jstd, pot_s, acc_s = simulate(0)
    res = R.make_filter().filter(Y, U=U)
    fx = fixed_kf(U, Y)
    th_true = S.reshape(T, NJ, ORDER)[:, :, 0]
    th_lu = res.mean.reshape(T, NJ, ORDER)[:, :, 0]
    th_fx = fx.reshape(T, NJ, ORDER)[:, :, 0]
    th_pot = Y[:, 0::3]

    print("kinematics...")
    P_true = np.array([joints3d(th_true[k]) for k in range(T)])      # (T, 6, 3)
    P_pot = np.array([joints3d(th_pot[k]) for k in range(T)])
    P_lu = np.array([joints3d(th_lu[k]) for k in range(T)])
    P_fx = np.array([joints3d(th_fx[k]) for k in range(T)])
    err_pot = np.sqrt(((P_pot[:, -1] - P_true[:, -1]) ** 2).sum(1))
    err_lu = np.sqrt(((P_lu[:, -1] - P_true[:, -1]) ** 2).sum(1))
    err_fx = np.sqrt(((P_fx[:, -1] - P_true[:, -1]) ** 2).sum(1))

    ms, ps = res.measurement_scale, res.process_scale
    chip = np.zeros((T, 3, NJ))                                      # rows: pot, accel, process
    chip[:, 0] = ms[:, 0::3]
    chip[:, 1] = np.maximum(ms[:, 1::3], ms[:, 2::3])   # hotter of the IMU's two axes
    for j in range(NJ):
        chip[:, 2, j] = ps[:, MODE_OF_JOINT[j]]
    NORM = 6.0                                                       # x15 -> 2 ln 15 = 5.4

    plt.rcParams.update({"font.family": "monospace", "font.size": 9, "text.color": INK,
                         "axes.edgecolor": LINE, "xtick.color": MUT, "ytick.color": MUT})
    fig = plt.figure(figsize=(9.6, 4.9), dpi=110); fig.patch.set_facecolor(BG)
    ax_arm = fig.add_axes([0.005, 0.03, 0.47, 0.80]); ax_arm.set_facecolor(PANEL)
    ax_diag = fig.add_axes([0.55, 0.50, 0.43, 0.30]); ax_diag.set_facecolor(PANEL)
    ax_scope = fig.add_axes([0.55, 0.11, 0.43, 0.25]); ax_scope.set_facecolor(PANEL)
    # the evolving-regime label, between the channel grid and the error scope
    regime_lbl = fig.text(0.55, 0.425, "", fontsize=9.5, weight="bold", family="monospace",
                          va="center", ha="left")

    fig.text(0.005, 0.94, "LucidFilter — a 5-DOF arm, all noise inferred online", color=INK,
             fontsize=15, weight="bold", family="sans-serif")
    fig.text(0.005, 0.885, "bad potentiometer + link-mounted accelerometer per joint · H "
             "linearised every step · nothing tuned · from lucid import LucidFilter",
             color=MUT, fontsize=8.2)

    # arm panel: fixed workspace box over all frames and cameras -- sized to the true and
    # lucid arms; the raw-pot arm may clip out of frame when its sensor is failing (the point)
    half = float(np.abs(np.concatenate([P_true, P_lu])).max()) * 1.06
    ax_arm.set_xlim(-half, half); ax_arm.set_ylim(-half * 0.72, half * 1.05)
    ax_arm.set_aspect("equal"); ax_arm.axis("off")
    for lx, lab, c in [(0.02, "true", C_TRUE), (0.18, "raw pot", C_POT), (0.40, "lucid", C_LU)]:
        ax_arm.text(lx, 0.02, "● " + lab, transform=ax_arm.transAxes, color=c, fontsize=8.5,
                    va="bottom")
    # floor grid (re-projected per frame -- the rotation cue)
    gl = []
    for _ in range(14):
        (ln,) = ax_arm.plot([], [], "-", color=LINE, lw=0.6, alpha=0.55, zorder=1)
        gl.append(ln)
    gxy = np.linspace(-0.8, 0.8, 7)
    (l_true,) = ax_arm.plot([], [], "-", color=C_TRUE, lw=2, alpha=.55, solid_capstyle="round",
                            zorder=3)
    (l_pot,) = ax_arm.plot([], [], "-o", color=C_POT, lw=1.6, ms=4, alpha=.85, zorder=5)
    (l_lu,) = ax_arm.plot([], [], "-o", color=C_LU, lw=3.5, ms=5, alpha=.95, zorder=4)
    (trail,) = ax_arm.plot([], [], "-", color=C_LU, lw=1.2, alpha=.3, zorder=2)

    # diagnostics: 3 x NJ chip grid of the learned per-component log-scales
    ax_diag.set_xlim(-1.6, NJ); ax_diag.set_ylim(-0.75, 2.6); ax_diag.invert_yaxis()
    ax_diag.set_xticks([]); ax_diag.set_yticks([]); ax_diag.axis("off")
    ax_diag.set_title("which noise is hot — learned online, per component", color=MUT,
                      fontsize=9, loc="left", pad=6)
    chips = {}
    for r, lab in enumerate(["pots", "accels", "process"]):
        ax_diag.text(-1.5, r, lab, color=MUT, fontsize=8, va="center")
        for j in range(NJ):
            rect = plt.Rectangle((j - 0.38, r - 0.32), 0.76, 0.64, facecolor=BG,
                                 edgecolor=LINE, lw=1, zorder=2)
            ax_diag.add_patch(rect); chips[(r, j)] = rect
    for j in range(NJ):
        ax_diag.text(j, -0.62, f"J{j + 1}", color=MUT, fontsize=7.5, ha="center")

    # scope
    tt = np.arange(T) * DT
    ax_scope.set_xlim(0, T * DT)
    # cap the scope so lucid-vs-fixed stays readable; the raw pot going off-chart
    # during the pot failure is the story, not an axis bug
    EM = min(max(float(err_pot.max()), float(err_fx.max()), 0.09) * 1.05, 0.40)
    ax_scope.set_ylim(0, EM)
    for nm, a, b in PHASES:
        if nm != "calm":
            ax_scope.axvspan(a * DT, b * DT, color=C_WARN, alpha=0.10, lw=0)
    ax_scope.set_ylabel("tip err (m)", color=MUT, fontsize=8.5)
    ax_scope.set_xlabel("time (s)", color=MUT, fontsize=8.5)
    ax_scope.tick_params(length=3, labelsize=8)
    (s_pot,) = ax_scope.plot([], [], color=C_POT, lw=1.0, alpha=.55, label="raw pot")
    (s_fx,) = ax_scope.plot([], [], color=C_FIX, lw=1.2, alpha=.75, label="fixed noise")
    (s_lu,) = ax_scope.plot([], [], color=C_LU, lw=1.8, label="lucid")
    head = ax_scope.axvline(0, color=INK, lw=1, alpha=.5)
    ax_scope.legend(loc="upper right", facecolor=PANEL, edgecolor=LINE, labelcolor=INK,
                    fontsize=7.5, ncols=3, columnspacing=1.0, handlelength=1.2)

    def heat(v):
        """MUT -> amber colour ramp on the clipped normalised scale."""
        a = float(np.clip(v / NORM, 0.0, 1.0))
        c0 = np.array(matplotlib.colors.to_rgb(BG)); c1 = np.array(matplotlib.colors.to_rgb(C_WARN))
        return tuple(c0 + a * (c1 - c0))

    STEP = 8
    frames = list(range(0, T, STEP))

    def draw(i):
        az = az_at(i)
        for k, g in enumerate(gxy):     # floor grid at z=0
            a3 = np.array([[g, -0.8, 0.0], [g, 0.8, 0.0]])
            b3 = np.array([[-0.8, g, 0.0], [0.8, g, 0.0]])
            pa, pb = project(a3, az), project(b3, az)
            gl[k].set_data(pa[:, 0], pa[:, 1]); gl[k + 7].set_data(pb[:, 0], pb[:, 1])
        jt = project(P_true[i], az); jp = project(P_pot[i], az); jl = project(P_lu[i], az)
        l_true.set_data(jt[:, 0], jt[:, 1])
        l_pot.set_data(jp[:, 0], jp[:, 1])
        l_lu.set_data(jl[:, 0], jl[:, 1])
        lo = max(0, i - 70)
        tr = project(P_lu[lo:i + 1, -1], az)
        trail.set_data(tr[:, 0], tr[:, 1])
        txt, col = PHASE_TXT[phase_at(i)]
        regime_lbl.set_text("ACTIVE REGIME: " + txt); regime_lbl.set_color(col)
        for r in range(3):
            for j in range(NJ):
                chips[(r, j)].set_facecolor(heat(chip[i, r, j]))
        s_pot.set_data(tt[:i + 1], err_pot[:i + 1])
        s_fx.set_data(tt[:i + 1], err_fx[:i + 1])
        s_lu.set_data(tt[:i + 1], err_lu[:i + 1])
        head.set_xdata([i * DT, i * DT])

    print("rendering", len(frames), "frames...")
    fig.canvas.draw()
    imgs, raws = [], []
    for i in frames:
        draw(i); fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        raws.append(rgb)
        imgs.append(Image.fromarray(rgb).convert("P", palette=Image.ADAPTIVE, colors=64))
    imgs[0].save(OUT, save_all=True, append_images=imgs[1:], duration=60, loop=0,
                 disposal=2, optimize=True)
    print("wrote", os.path.relpath(OUT), "-", os.path.getsize(OUT) // 1024, "KB,",
          len(frames), "frames")
    # the same animation as an H.264 video: GitHub's player gives pause / scrub / 0.25-2x speed
    try:
        import imageio
        h, w = raws[0].shape[:2]
        if h % 2 or w % 2:                       # yuv420p needs even dimensions -- pad with BG
            H2, W2 = h + h % 2, w + w % 2
            bg = np.array([14, 18, 23], dtype=np.uint8)   # BG "#0e1217"
            raws = [np.pad(r, ((0, H2 - h), (0, W2 - w), (0, 0)),
                           constant_values=0) + 0 for r in raws]
            for r in raws:
                r[h:, :] = bg; r[:, w:] = bg
        mp4 = OUT[:-4] + ".mp4"
        imageio.mimwrite(mp4, raws, fps=17, codec="libx264", quality=8,
                         pixelformat="yuv420p")
        print("wrote", os.path.relpath(mp4), "-", os.path.getsize(mp4) // 1024, "KB")
    except ImportError:
        print("imageio not installed -- skipped the .mp4 (pip install imageio imageio-ffmpeg)")
    # numbers for the README claim
    on = np.zeros(T, bool)
    for nm, a, b in PHASES:
        if nm != "calm":
            on[a:b] = True
    print(f"tip RMSE over bursts: raw pot {np.sqrt((err_pot[on]**2).mean()):.4f}  "
          f"fixed {np.sqrt((err_fx[on]**2).mean()):.4f}  lucid {np.sqrt((err_lu[on]**2).mean()):.4f}")
    print(f"tip RMSE calm:        raw pot {np.sqrt((err_pot[~on]**2).mean()):.4f}  "
          f"fixed {np.sqrt((err_fx[~on]**2).mean()):.4f}  lucid {np.sqrt((err_lu[~on]**2).mean()):.4f}")


if __name__ == "__main__":
    render()
