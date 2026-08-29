"""
5-DOF Robotic Arm: Online Noise Learning with LucidFilter
==========================================================
Usage:
    python examples/arm5dof_demo.py          # writes figures/arm5dof_lucid.gif

Hand it the arm's kinematics; it learns every noise level online:

    from lucid import LucidFilter

    F, B, H, Q0, R0 = build_arm_model()      # ordinary state-space matrices
    f = LucidFilter(dynamics=F, control=B, H=H, process=Q0, measurement=R0)
    r = f.filter(sensor_readings, U=commanded_positions)
    # r.mean               -> filtered joint states
    # r.measurement_scale  -> per-sensor log noise (rises when a sensor fails)
    # r.process_scale      -> per-mode log process noise (rises on a disturbance)

No Q schedule, no R schedule, no thresholds, no forgetting factor to choose;
``Q0``/``R0`` set only the base magnitude the scales walk away from.  Give them
the right order of magnitude -- see the caveat under "What this demo does and
does not show" below.

The whole arm is ONE filter -- 10 states, 10 sensors, 20 noise channels learned
jointly.  That is only possible because the scale posterior is carried on an
axial stencil rather than a tensor grid; the tensor product over 20 channels
would need 5**20 = 1e14 nodes, while the stencil uses about a thousand.

Model (per joint, block-diagonal over the five joints)
    state    [theta, omega]     -- joint angle and rate
    dynamics closed-loop PD servo tracking a commanded angle (stable, so the
             joint stays bounded)
    input    commanded joint angle
    sensors  potentiometer -> theta,  tachometer -> omega
    process  independent disturbances on both channels

Six phased regimes exercise both halves of the problem:

    SENSOR HOT     -- one potentiometer fails catastrophically (x25 noise)
    DYNAMICS HOT   -- large disturbance torque / vibration (x20 process noise)
    BOTH HOT       -- simultaneous sensor and dynamics failure
    SENSOR NOISY   -- all potentiometers moderately elevated (x5)
    DYNAMICS NOISY -- all process channels moderately elevated (x5)
    calm           -- nominal; the filter recovers after each event

Three estimators are compared on tip position error:
    LucidFilter -- kinematics in, all noise learned online (this library)
    Static KF   -- same kinematics, fixed Q/R tuned on the calm regime
    Raw pot     -- no filter; the potentiometer readings alone

What this demo does and does not show
-------------------------------------
Does: the learned log-scales track the injected noise closely with nothing
pre-tuned -- e.g. DYNAMICS NOISY (a x5 process burst, true log-scale 3.22) is
recovered at ~3.4, and a x25 pot failure (true 6.44) at ~5.7-6.2, within a few
samples of onset and released again afterwards.

Does not: beat a *correctly tuned* static Kalman filter on overall tip RMSE in
this scenario.  Two honest reasons, both worth knowing before reaching for this
filter:

  1. A single innovation only ever sees ``S = H(FPF' + Q)H' + R``.  Q and R are
     therefore not separable one step at a time, and during a burst the filter
     inflates both -- so the Kalman gain does not drop as sharply as it should
     when it is specifically the sensor that failed.  Separating them needs
     multi-step (innovation-autocorrelation) information the per-step score
     does not carry.
  2. The identifiability gate is evaluated once at the supplied base.  Hand it
     a base whose Q/R ratio is off by ~10x and it freezes precisely the
     channels that would have walked back, so it stays put next to an equally
     mis-tuned static filter.

The static baseline here is given the exact true calm Q and R, which is a
stronger baseline than any practitioner actually has.
"""
from __future__ import annotations
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D    # noqa: F401  registers the "3d" projection
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from lucid import LucidFilter              # noqa: E402

# -- arm ---------------------------------------------------------------------
NJ        = 5                              # 5-DOF
DT        = 0.02                           # 50 Hz
KP, KD    = 25.0, 7.0                      # joint servo gains (closed loop)
NS        = 2                              # states per joint: [theta, omega]
LINK_AXES = ["z", "y", "y", "y", "x"]
LINK_LEN  = [0.30, 0.50, 0.40, 0.25, 0.15]

# -- true noise levels (the filter is never told these) ----------------------
POT_STD  = 0.03                            # potentiometer, rad
TACH_STD = 0.10                            # tachometer, rad/s
Q_STD    = np.array([0.02, 0.15])          # process std on [theta, omega]

# -- failure schedule --------------------------------------------------------
T = 2200
REGIMES = [
    ("calm",             0,  230),
    ("SENSOR HOT",     230,  560),
    ("calm",           560,  760),
    ("DYNAMICS HOT",   760, 1100),
    ("calm",          1100, 1300),
    ("BOTH HOT",      1300, 1630),
    ("calm",          1630, 1800),
    ("SENSOR NOISY",  1800, 2000),
    ("DYNAMICS NOISY", 2000, 2200),
]
HOT_JOINT = 1                              # the joint whose pot fails
SENSOR_MULT, DYN_MULT = 25.0, 20.0

# -- theme -------------------------------------------------------------------
BG, PANEL, LINE = "#0e1217", "#161c24", "#2a3542"
INK, MUT = "#dbe3ec", "#7c8a99"
C_TRUE, C_POT, C_STATIC, C_LUCID = "#c3d0de", "#e2607b", "#f5b23d", "#33e0a6"
C_PROC = "#8fa2ff"

REGIME_COLOR = {"calm": MUT, "SENSOR HOT": C_POT, "DYNAMICS HOT": C_PROC,
                "BOTH HOT": C_STATIC, "SENSOR NOISY": C_POT,
                "DYNAMICS NOISY": C_PROC}
REGIME_LABEL = {
    "calm":           "CALM  -  nominal operation",
    "SENSOR HOT":     "SENSOR HOT  -  joint-2 pot failed (x25 noise)",
    "DYNAMICS HOT":   "DYNAMICS HOT  -  disturbance torque (x20 process)",
    "BOTH HOT":       "BOTH HOT  -  sensor and dynamics failed together",
    "SENSOR NOISY":   "SENSOR NOISY  -  all pots elevated (x5)",
    "DYNAMICS NOISY": "DYNAMICS NOISY  -  process noise elevated (x5)",
}


def regime_at(k):
    for name, a, b in REGIMES:
        if a <= k < b:
            return name
    return "calm"


# -- model -------------------------------------------------------------------

def build_arm_model():
    """Block-diagonal state space for the whole arm.

    Returns F (10,10), B (10,5), H (10,10), Q0 (10,10), R0 (10,).
    Per joint the closed-loop servo is

        theta' = theta + dt*omega
        omega' = omega + dt*(-KP*(theta - u) - KD*omega)

    which is stable, so the joint tracks its command and stays bounded --
    unlike a free integrator chain, whose angle random-walks away.

    Both states are sensed, so every one of the 20 noise channels is
    identifiable from a single innovation and the filter can learn all of
    them.  (A channel that no sensor sees -- an unmeasured disturbance torque,
    say -- has zero one-step innovation sensitivity; the filter detects that
    and freezes it at the base rather than inventing a value for it.)
    """
    Fj = np.array([[1., DT],
                   [-KP * DT, 1 - KD * DT]])
    Bj = np.array([[0.], [KP * DT]])
    Hj = np.eye(2)                      # pot -> theta,  tach -> omega
    F = np.kron(np.eye(NJ), Fj)
    B = np.kron(np.eye(NJ), Bj)
    H = np.kron(np.eye(NJ), Hj)
    Q0 = np.diag(np.tile(Q_STD, NJ) ** 2)
    R0 = np.tile([POT_STD ** 2, TACH_STD ** 2], NJ)
    return F, B, H, Q0, R0


def joints3d(theta):
    """theta (NJ,) -> (NJ+1, 3) joint positions in the world frame."""
    def rot(ax, ang):
        c, s = math.cos(ang), math.sin(ang)
        if ax == "z":
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
        if ax == "y":
            return np.array([[c, 0, s], [0, 1., 0], [-s, 0, c]])
        return np.array([[1., 0, 0], [0, c, -s], [0, s, c]])

    R, p, pts = np.eye(3), np.zeros(3), [np.zeros(3)]
    for j in range(NJ):
        R = R @ rot(LINK_AXES[j], theta[j])
        p = p + R @ np.array([0., 0., LINK_LEN[j]])
        pts.append(p.copy())
    return np.array(pts)


class StaticKalman:
    """Fixed-noise Kalman filter, Q and R tuned once on the calm regime.

    The standard approach, and optimal exactly while the arm stays calm.
    """

    def __init__(self, F, B, H, Q, R_diag):
        self.F, self.B, self.H = F, B, H
        self.Q, self.R = Q, np.diag(R_diag)

    def filter(self, Y, U):
        n = self.F.shape[0]
        m, P, I = np.zeros(n), np.eye(n), np.eye(n)
        out = np.zeros((len(Y), n))
        for k, (y, u) in enumerate(zip(Y, U)):
            mp = self.F @ m + self.B @ u
            Pp = self.F @ P @ self.F.T + self.Q
            S = self.H @ Pp @ self.H.T + self.R
            Kg = Pp @ self.H.T @ np.linalg.inv(S)
            m = mp + Kg @ (y - self.H @ mp)
            P = (I - Kg @ self.H) @ Pp
            out[k] = m
        return out


# -- simulation --------------------------------------------------------------

def simulate(seed=0):
    """Drive the arm through the six regimes; return commands, truth, sensors."""
    F, B, H, _, _ = build_arm_model()
    rng = np.random.default_rng(seed)
    t = np.arange(T) * DT

    U = np.zeros((T, NJ))                       # commanded joint angles
    for j in range(NJ):
        U[:, j] = (0.55 * np.sin(2 * np.pi * (0.16 + 0.035 * j) * t + 0.7 * j)
                   + 0.30 * np.sin(2 * np.pi * (0.31 + 0.045 * j) * t + 1.4 * j))

    q_mul = np.ones(T)
    pot_mul = np.ones((T, NJ))
    for name, a, b in REGIMES:
        if name in ("SENSOR HOT", "BOTH HOT"):
            pot_mul[a:b, HOT_JOINT] *= SENSOR_MULT
        if name in ("DYNAMICS HOT", "BOTH HOT"):
            q_mul[a:b] *= DYN_MULT
        if name == "SENSOR NOISY":
            pot_mul[a:b] *= 5.0
        if name == "DYNAMICS NOISY":
            q_mul[a:b] *= 5.0

    qs = np.tile(Q_STD, NJ)
    S = np.zeros((T, NJ * NS))
    Y = np.zeros((T, NJ * 2))
    s = np.zeros(NJ * NS)
    for k in range(T):
        s = F @ s + B @ U[k] + (q_mul[k] * qs) * rng.standard_normal(NJ * NS)
        S[k] = s
        Y[k, 0::2] = s[0::NS] + POT_STD * pot_mul[k] * rng.standard_normal(NJ)
        Y[k, 1::2] = s[1::NS] + TACH_STD * rng.standard_normal(NJ)
    return U, S, Y, q_mul, pot_mul


def run(seed=0):
    """Run all three estimators over the whole trajectory."""
    U, S, Y, q_mul, pot_mul = simulate(seed)
    F, B, H, Q0, R0 = build_arm_model()

    # ---- one LucidFilter for the entire arm: 25 noise channels, learned -----
    f = LucidFilter(dynamics=F, control=B, H=H, process=Q0, measurement=R0)
    r = f.filter(Y, U=U)

    # ---- same kinematics, Q/R frozen at their calm values ------------------
    static = StaticKalman(F, B, H, Q0, R0).filter(Y, U)

    theta_true = S[:, 0::NS]
    theta_pot = Y[:, 0::2]
    theta_lucid = r.mean[:, 0::NS]
    theta_static = static[:, 0::NS]

    def allpts(theta):
        return np.array([joints3d(theta[k]) for k in range(T)])

    pts_true = allpts(theta_true)

    def tip_err(pts):
        return np.linalg.norm(pts[:, -1] - pts_true[:, -1], axis=1)

    pts_lucid, pts_static, pts_pot = (allpts(theta_lucid), allpts(theta_static),
                                      allpts(theta_pot))
    return dict(pts_true=pts_true, pts_lucid=pts_lucid, pts_static=pts_static,
                pts_pot=pts_pot, err_lucid=tip_err(pts_lucid),
                err_static=tip_err(pts_static), err_pot=tip_err(pts_pot),
                meas_scale=r.measurement_scale, proc_scale=r.process_scale,
                q_mul=q_mul, pot_mul=pot_mul, nodes=sum(m._G for m in f._members))


# -- animation ---------------------------------------------------------------

def render(data, out_path, fps=25, step=8):
    pts_t, pts_l = data["pts_true"], data["pts_lucid"]
    pts_s, pts_p = data["pts_static"], data["pts_pot"]
    err_l, err_s, err_p = data["err_lucid"], data["err_static"], data["err_pot"]

    plt.rcParams.update({"font.family": "monospace", "font.size": 9,
                         "text.color": INK, "axes.edgecolor": LINE,
                         "xtick.color": MUT, "ytick.color": MUT})

    fig = plt.figure(figsize=(11.5, 5.2), dpi=100)
    fig.patch.set_facecolor(BG)
    gs = GridSpec(1, 2, figure=fig, left=0.005, right=0.975, top=0.875,
                  bottom=0.115, width_ratios=[1.12, 1.0], wspace=0.06)

    # ---- 3D arm ------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 0], projection="3d")
    ax3.set_facecolor(BG)
    for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(LINE)
    ax3.grid(False)
    ax3.view_init(elev=24, azim=40)
    ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])

    span = np.concatenate([pts_t.reshape(-1, 3), pts_l.reshape(-1, 3)])
    lo, hi = span.min(0), span.max(0)
    cx, cy, cz = 0.5 * (lo + hi)
    half = float(0.60 * (hi - lo).max())
    zfloor = max(0.0, cz - half)
    ax3.set_xlim(cx - half, cx + half)
    ax3.set_ylim(cy - half, cy + half)
    ax3.set_zlim(zfloor, cz + half)
    for g in np.linspace(cx - half, cx + half, 9):
        ax3.plot([g, g], [cy - half, cy + half], [zfloor, zfloor],
                 color=LINE, lw=0.5, alpha=0.30)
    for g in np.linspace(cy - half, cy + half, 9):
        ax3.plot([cx - half, cx + half], [g, g], [zfloor, zfloor],
                 color=LINE, lw=0.5, alpha=0.30)

    def arm(color, lw, alpha, ms=0.0):
        (ln,) = ax3.plot3D([], [], [], "-o" if ms else "-", color=color, lw=lw,
                           alpha=alpha, markersize=ms, solid_capstyle="round")
        return ln

    l_true   = arm(C_TRUE,   1.4, 0.42)
    l_pot    = arm(C_POT,    1.6, 0.68, 3.2)
    l_static = arm(C_STATIC, 1.6, 0.68, 3.2)
    l_lucid  = arm(C_LUCID,  3.0, 0.95, 4.2)
    (trail,) = ax3.plot3D([], [], [], "-", color=C_LUCID, lw=0.9, alpha=0.22)
    drops = [ax3.plot3D([], [], [], "--", color=c, lw=0.7, alpha=0.25)[0]
             for c in (C_POT, C_STATIC, C_LUCID)]
    for i, (lab, c) in enumerate([("truth", C_TRUE), ("raw pot", C_POT),
                                  ("static KF", C_STATIC), ("LucidFilter", C_LUCID)]):
        ax3.text2D(0.015 + i * 0.245, 0.015, "-- " + lab, transform=ax3.transAxes,
                   color=c, fontsize=8.5, va="bottom")

    # ---- tip-error scope ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor(PANEL)
    tt = np.arange(T) * DT
    ymax = max(float(np.percentile(err_p, 99)) * 1.10, 0.05)
    ax.set_xlim(0, T * DT); ax.set_ylim(0, ymax)
    for name, a, b in REGIMES:
        if name != "calm":
            ax.axvspan(a * DT, b * DT, color=REGIME_COLOR[name], alpha=0.10, lw=0)
            ax.text((a + b) * DT / 2, ymax * 0.965, name.replace(" ", "\n"),
                    ha="center", va="top", color=REGIME_COLOR[name],
                    fontsize=6.4, alpha=0.85, linespacing=1.15)
    ax.set_title("tip position error", color=MUT, fontsize=9, loc="left", pad=6)
    ax.set_ylabel("error (m)", color=MUT, fontsize=8)
    ax.set_xlabel("time (s)", color=MUT, fontsize=8)
    ax.tick_params(length=3, labelsize=7.5)
    (ln_p,) = ax.plot([], [], color=C_POT, lw=1.0, alpha=0.55, label="raw pot")
    (ln_s,) = ax.plot([], [], color=C_STATIC, lw=1.2, alpha=0.80, label="static KF")
    (ln_l,) = ax.plot([], [], color=C_LUCID, lw=2.0, label="LucidFilter")
    head = ax.axvline(0, color=INK, lw=0.8, alpha=0.30)
    ax.legend(loc="upper right", facecolor=PANEL, edgecolor=LINE,
              labelcolor=INK, fontsize=7.5, handlelength=1.4, framealpha=0.95)
    rms_txt = ax.text(0.015, 0.955, "", transform=ax.transAxes, va="top",
                      ha="left", fontsize=7.8, family="monospace", color=MUT)

    fig.text(0.5, 0.983, "5-DOF arm   |   LucidFilter: give it the kinematics, "
             "it learns every noise level online", ha="center", va="top",
             color=INK, fontsize=12.5, weight="bold", family="sans-serif")
    banner = fig.text(0.5, 0.930, "", ha="center", va="top", fontsize=10.5,
                      weight="bold", family="monospace")
    TRAIL = 70

    def draw(i):
        for ln, pts in ((l_true, pts_t), (l_pot, pts_p),
                        (l_static, pts_s), (l_lucid, pts_l)):
            p = pts[i]
            ln.set_data_3d(p[:, 0], p[:, 1], p[:, 2])
        tips = pts_l[max(0, i - TRAIL):i + 1, -1]
        trail.set_data_3d(tips[:, 0], tips[:, 1], tips[:, 2])
        for ln, pts in zip(drops, (pts_p, pts_s, pts_l)):
            tx, ty, tz = pts[i, -1]
            ln.set_data_3d([tx, tx], [ty, ty], [zfloor, tz])
        ph = regime_at(i)
        banner.set_text(REGIME_LABEL[ph]); banner.set_color(REGIME_COLOR[ph])
        ln_p.set_data(tt[:i + 1], err_p[:i + 1])
        ln_s.set_data(tt[:i + 1], err_s[:i + 1])
        ln_l.set_data(tt[:i + 1], err_l[:i + 1])
        head.set_xdata([i * DT, i * DT])
        rms = lambda e: math.sqrt(float((e[:i + 1] ** 2).mean()))   # noqa: E731
        rms_txt.set_text(f"RMSE so far   pot {rms(err_p):.3f}   "
                         f"static {rms(err_s):.3f}   lucid {rms(err_l):.3f} m")

    frames = list(range(0, T, step))
    print(f"rendering {len(frames)} frames ...")
    imgs = []
    for n, i in enumerate(frames):
        draw(i)
        fig.canvas.draw()
        imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()))
                    .convert("P", palette=Image.ADAPTIVE, colors=80))
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(frames)}")
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                 duration=round(1000 / fps), loop=0, disposal=2, optimize=True)
    print(f"wrote {out_path}  ({os.path.getsize(out_path) // 1024} KB, "
          f"{len(frames)} frames)")


def report(data):
    print(f"\none LucidFilter, {NJ*NS} states / {NJ*2} sensors / "
          f"{NJ*NS + NJ*2} noise channels, {data['nodes']} stencil nodes")
    print("\ntip-error RMSE (m) by regime")
    print(f"  {'regime':16s} {'raw pot':>9s} {'static KF':>10s} {'Lucid':>9s}")
    for name, a, b in REGIMES:
        rms = lambda e: float(np.sqrt((e[a:b] ** 2).mean()))       # noqa: E731
        print(f"  {name:16s} {rms(data['err_pot']):9.4f} "
              f"{rms(data['err_static']):10.4f} {rms(data['err_lucid']):9.4f}")
    rms = lambda e: float(np.sqrt((e ** 2).mean()))                # noqa: E731
    print(f"  {'OVERALL':16s} {rms(data['err_pot']):9.4f} "
          f"{rms(data['err_static']):10.4f} {rms(data['err_lucid']):9.4f}")

    ms, ps = data["meas_scale"], data["proc_scale"]
    hp = 2 * HOT_JOINT                                   # failing pot's channel
    print(f"\nlearned log-scales (0 = the base it was handed)")
    print(f"  {'regime':16s} {'pot j2':>8s} {'pots avg':>9s} {'process':>8s}")
    for name, a, b in REGIMES:
        print(f"  {name:16s} {ms[a:b, hp].mean():8.2f} "
              f"{ms[a:b, 0::2].mean():9.2f} {ps[a:b].max(1).mean():8.2f}")


if __name__ == "__main__":
    print("simulating 5-DOF arm through six noise regimes ...")
    data = run(seed=0)
    report(data)
    out = os.path.join(os.path.dirname(HERE), "figures", "arm5dof_lucid.gif")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    render(data, out)
