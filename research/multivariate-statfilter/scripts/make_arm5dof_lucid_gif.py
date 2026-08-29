"""Render the LucidFilter 5-DOF arm demo to a GIF for the main README.

Real output of the PUBLIC filter (`from lucid import LucidFilter`) on the 3D 5-DOF arm rig
(probes 0026/0052): a bad potentiometer + a good accelerometer per joint fused through `H`,
driven along a commanded trajectory (known forcing B.u), with noise arriving in phases:

    calm | SENSOR (accelerometers swamped) | calm | PROCESS (disturbance torque) |
    calm | POT FAILURE (one joint's potentiometer dies) | calm | BOTH | calm

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
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
from lucid import LucidFilter  # noqa: E402

OUT = os.path.join(HERE, "..", "figures", "arm5dof-lucid.gif")

# ---------------------------------------------------------------- rig (0026/0052 constants)
NJ, ORDER, DT = 5, 3, 0.01
POT, ACC, JERK = 0.06, 0.02, 0.6
POT_MULT, ACC_MULT, JERK_MULT = 15.0, 15.0, 20.0
FAIL_J = 2                                   # the joint whose potentiometer dies in POTFAIL
T = 1900
PHASES = [("calm", 0, 250), ("SENSOR", 250, 500), ("calm", 500, 650),
          ("PROCESS", 650, 900), ("calm", 900, 1050), ("POTFAIL", 1050, 1300),
          ("calm", 1300, 1450), ("BOTH", 1450, 1700), ("calm", 1700, 1900)]

Fb = np.eye(ORDER)
for i in range(ORDER):
    for j in range(i + 1, ORDER):
        Fb[i, j] = DT ** (j - i) / math.factorial(j - i)
G = np.array([DT ** (ORDER - i) / math.factorial(ORDER - i) for i in range(ORDER)])
F = np.kron(np.eye(NJ), Fb)
Q0 = np.kron(np.eye(NJ), JERK ** 2 * np.outer(G, G) + 1e-12 * np.eye(ORDER))
B = np.kron(np.eye(NJ), G[:, None])
rows = []
for d in range(NJ):
    for di in (0, 2):                        # pot reads theta, accel reads alpha
        e = np.zeros(ORDER * NJ); e[d * ORDER + di] = 1.0; rows.append(e)
H = np.array(rows)
R0 = np.tile([POT ** 2, ACC ** 2], NJ)
N, M = ORDER * NJ, 2 * NJ

# eigenmode -> joint map for the process-scale diagnostics (block-diag Q0 -> localised modes)
_lam, _V = np.linalg.eigh(Q0)
_top = np.argsort(_lam)[-NJ:]                # the NJ jerk modes (eigh sorts ascending)
MODE_OF_JOINT = {int(np.argmax([np.abs(_V[j * ORDER:(j + 1) * ORDER, k]).sum()
                                for j in range(NJ)])): k for k in _top}


def schedule():
    jstd = np.full(T, JERK)
    pot_s = np.full((T, NJ), POT); acc_s = np.full((T, NJ), ACC)
    for name, a, b in PHASES:
        if name == "SENSOR":
            acc_s[a:b] = ACC * ACC_MULT
        elif name == "PROCESS":
            jstd[a:b] = JERK * JERK_MULT
        elif name == "POTFAIL":
            pot_s[a:b, FAIL_J] = POT * POT_MULT
        elif name == "BOTH":
            acc_s[a:b] = ACC * ACC_MULT; jstd[a:b] = JERK * JERK_MULT
    return jstd, pot_s, acc_s


def simulate(seed=0):
    rng = np.random.default_rng(seed); t = np.arange(T) * DT
    U = np.zeros((T, NJ))
    for j in range(NJ):
        for (a, w, ph) in [(2.0, 0.35 + 0.1 * j, j), (1.2, 0.7 + 0.13 * j, 2 * j)]:
            U[:, j] += a * np.sin(2 * np.pi * w * t + ph)
    jstd, pot_s, acc_s = schedule()
    s = np.zeros(N); S = np.zeros((T, N)); Y = np.zeros((T, M))
    for k in range(T):
        s = F @ s + B @ U[k] + B @ (jstd[k] * rng.standard_normal(NJ)); S[k] = s
        std = np.empty(M); std[0::2] = pot_s[k]; std[1::2] = acc_s[k]
        Y[k] = H @ s + std * rng.standard_normal(M)
    return U, S, Y, jstd, pot_s, acc_s


def fixed_kf(U, Y):
    Qc = JERK ** 2 * (B @ B.T); Rc = np.diag(R0)
    m0 = np.zeros(N); P = np.eye(N); out = np.zeros((T, N))
    for k, y in enumerate(Y):
        mp = F @ m0 + B @ U[k]; Pp = F @ P @ F.T + Qc
        K = Pp @ H.T @ np.linalg.inv(H @ Pp @ H.T + Rc)
        m0 = mp + K @ (y - H @ mp); P = Pp - K @ H @ Pp; out[k] = m0
    return out


# ---------------------------------------------------------------- 3D arm + rotating camera
AXES = ["z", "y", "y", "y", "x"]; L = [0.30, 0.50, 0.40, 0.25, 0.15]
HOME = np.array([0.25, 0.55, -0.95, 0.55, 0.0])   # articulated home pose; theta is the deviation
EL = math.radians(22)


def joints3d(theta):
    def rot(ax, a):
        c, s = math.cos(a), math.sin(a)
        if ax == "z": return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        if ax == "y": return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])
        return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])
    Rm = np.eye(3); pos = np.zeros(3); pts = [pos.copy()]
    for j in range(NJ):
        Rm = Rm @ rot(AXES[j], HOME[j] + theta[j])
        pos = pos + Rm @ np.array([0, 0, L[j]]); pts.append(pos.copy())
    return np.array(pts)


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

PHASE_TXT = {"calm": ("○ calm — nominal noise", MUT),
             "SENSOR": ("⚠ SENSOR — accelerometers swamped ×15", C_WARN),
             "PROCESS": ("⚠ PROCESS — disturbance torque ×20", C_WARN),
             "POTFAIL": (f"⚠ POT FAILURE — joint {FAIL_J + 1} potentiometer dies", C_WARN),
             "BOTH": ("⚠ BOTH — sensor + process at once", C_WARN)}


def phase_at(i):
    for nm, a, b in PHASES:
        if a <= i < b:
            return nm
    return "calm"


def render():
    print("simulating + filtering (LucidFilter, full bank)...")
    U, S, Y, jstd, pot_s, acc_s = simulate(0)
    f = LucidFilter(dynamics=F, control=B, H=H, process=Q0, measurement=R0)
    res = f.filter(Y, U=U)
    fx = fixed_kf(U, Y)
    th_true = S.reshape(T, NJ, ORDER)[:, :, 0]
    th_lu = res.mean.reshape(T, NJ, ORDER)[:, :, 0]
    th_fx = fx.reshape(T, NJ, ORDER)[:, :, 0]
    th_pot = Y[:, 0::2]

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
    chip[:, 0] = ms[:, 0::2]
    chip[:, 1] = ms[:, 1::2]
    for j in range(NJ):
        chip[:, 2, j] = ps[:, MODE_OF_JOINT[j]]
    NORM = 6.0                                                       # x15 -> 2 ln 15 = 5.4

    plt.rcParams.update({"font.family": "monospace", "font.size": 9, "text.color": INK,
                         "axes.edgecolor": LINE, "xtick.color": MUT, "ytick.color": MUT})
    fig = plt.figure(figsize=(9.6, 4.9), dpi=110); fig.patch.set_facecolor(BG)
    ax_arm = fig.add_axes([0.005, 0.03, 0.47, 0.80]); ax_arm.set_facecolor(PANEL)
    ax_diag = fig.add_axes([0.55, 0.47, 0.43, 0.33]); ax_diag.set_facecolor(PANEL)
    ax_scope = fig.add_axes([0.55, 0.11, 0.43, 0.25]); ax_scope.set_facecolor(PANEL)

    fig.text(0.005, 0.94, "LucidFilter — a 5-DOF arm, all noise inferred online", color=INK,
             fontsize=15, weight="bold", family="sans-serif")
    fig.text(0.005, 0.885, "bad potentiometer + good accelerometer per joint · commanded "
             "trajectory · nothing tuned · from lucid import LucidFilter",
             color=MUT, fontsize=8.2)

    # arm panel: fixed workspace box over all frames and cameras
    half = float(np.abs(np.concatenate([P_true, P_pot, P_lu])).max()) * 1.02
    ax_arm.set_xlim(-half, half); ax_arm.set_ylim(-half * 0.72, half * 1.05)
    ax_arm.set_aspect("equal"); ax_arm.axis("off")
    for lx, lab, c in [(0.02, "true", C_TRUE), (0.18, "raw pot", C_POT), (0.40, "lucid", C_LU)]:
        ax_arm.text(lx, 0.02, "● " + lab, transform=ax_arm.transAxes, color=c, fontsize=8.5,
                    va="bottom")
    banner = ax_arm.text(0.5, 0.99, "", transform=ax_arm.transAxes, ha="center", va="top",
                         fontsize=10.5, weight="bold", family="monospace")
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
        txt, col = PHASE_TXT[phase_at(i)]; banner.set_text(txt); banner.set_color(col)
        for r in range(3):
            for j in range(NJ):
                chips[(r, j)].set_facecolor(heat(chip[i, r, j]))
        s_pot.set_data(tt[:i + 1], err_pot[:i + 1])
        s_fx.set_data(tt[:i + 1], err_fx[:i + 1])
        s_lu.set_data(tt[:i + 1], err_lu[:i + 1])
        head.set_xdata([i * DT, i * DT])

    print("rendering", len(frames), "frames...")
    fig.canvas.draw()
    imgs = []
    for i in frames:
        draw(i); fig.canvas.draw()
        imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert(
            "P", palette=Image.ADAPTIVE, colors=64))
    imgs[0].save(OUT, save_all=True, append_images=imgs[1:], duration=60, loop=0,
                 disposal=2, optimize=True)
    print("wrote", os.path.relpath(OUT), "-", os.path.getsize(OUT) // 1024, "KB,",
          len(frames), "frames")
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
