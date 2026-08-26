"""Render the 5-DOF arm stress test to a GIF for the README.

Real AdaptiveKalmanFilter output on the 3D 5-DOF arm (probe 0026): a bad potentiometer + good
accelerometer per joint, driven along a commanded trajectory, with noise arriving in phases
(sensor -> process -> both).  Panels: a 3/4 view of the arm (true vs raw-pot vs adaptive
estimate), a live "which noise is hot" diagnostic (the learned scales), and a tip-error scope.
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
sys.path.insert(0, os.path.join(HERE, "..", "exploration"))
import importlib.util
spec = importlib.util.spec_from_file_location("p26", os.path.join(HERE, "..", "exploration", "0026_arm5dof.py"))
p26 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p26)
NJ, ORDER, DT, T, PHASES = p26.NJ, p26.ORDER, p26.DT, p26.T, p26.PHASES

BG, PANEL, LINE, INK, MUT = "#0e1217", "#161c24", "#2a3542", "#dbe3ec", "#7c8a99"
C_TRUE, C_POT, C_AD, C_WARN, C_PROC = "#c3d0de", "#e2607b", "#33e0a6", "#f5b23d", "#8fa2ff"
OUT = os.path.join(HERE, "..", "figures", "arm5dof-adaptive.gif")

AXES = ["z", "y", "y", "y", "x"]; L = [0.30, 0.50, 0.40, 0.25, 0.15]


def joints3d(theta):                                    # theta (5,) -> (6,3) joint positions
    def rot(ax, a):
        c, s = math.cos(a), math.sin(a)
        if ax == "z": return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        if ax == "y": return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])
        return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])
    Rm = np.eye(3); pos = np.zeros(3); pts = [pos.copy()]
    for j in range(NJ):
        Rm = Rm @ rot(AXES[j], theta[j]); pos = pos + Rm @ np.array([0, 0, L[j]]); pts.append(pos.copy())
    return np.array(pts)


AZ, EL = math.radians(38), math.radians(20)


def project(P):                                          # (...,3) -> (...,2) 3/4 view
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    sx = -x * math.sin(AZ) + y * math.cos(AZ)
    sy = z * math.cos(EL) - (x * math.cos(AZ) + y * math.sin(AZ)) * math.sin(EL)
    return np.stack([sx, sy], -1)


def gen():
    f, F, B, H, U, S, Y, jstd, pot_s, acc_s = p26.simulate(0)
    ad = f.filter(Y, U=U)
    th_true = S.reshape(T, NJ, ORDER)[:, :, 0]
    th_ad = ad.mean.reshape(T, NJ, ORDER)[:, :, 0]
    th_pot = Y[:, 0::2]                                   # raw potentiometer angle
    tip_true = np.array([joints3d(th_true[k])[-1] for k in range(T)])
    tip_ad = np.array([joints3d(th_ad[k])[-1] for k in range(T)])
    err_pot = np.sqrt(((np.array([joints3d(th_pot[k])[-1] for k in range(T)]) - tip_true) ** 2).sum(1))
    err_ad = np.sqrt(((tip_ad - tip_true) ** 2).sum(1))
    ms = ad.measurement_scale; ps = ad.process_scale
    diag = dict(accel=ms[:, 1::2].mean(1), pot=ms[:, 0::2].mean(1),
                proc=ps.reshape(T, NJ, ORDER)[:, :, 2].mean(1))
    return th_true, th_ad, th_pot, err_pot, err_ad, diag


PHASE_TXT = {"calm": ("○ calm — nominal noise", MUT),
             "SENSOR": ("⚠ SENSOR NOISE — accelerometers swamped", C_WARN),
             "PROCESS": ("⚠ PROCESS NOISE — disturbance torque", C_WARN),
             "BOTH": ("⚠ BOTH — sensor + process noise", C_WARN)}


def phase_at(i):
    for nm, a, b in PHASES:
        if a <= i < b:
            return nm
    return "calm"


def render():
    th_true, th_ad, th_pot, err_pot, err_ad, diag = gen()
    # arm 2D projections per frame
    j_true = np.array([project(joints3d(th_true[k])) for k in range(T)])
    j_pot = np.array([project(joints3d(th_pot[k])) for k in range(T)])
    j_ad = np.array([project(joints3d(th_ad[k])) for k in range(T)])
    tip_ad = j_ad[:, -1, :]

    plt.rcParams.update({"font.family": "monospace", "font.size": 9, "text.color": INK,
                         "axes.edgecolor": LINE, "xtick.color": MUT, "ytick.color": MUT})
    fig = plt.figure(figsize=(9.6, 4.7), dpi=110); fig.patch.set_facecolor(BG)
    ax_arm = fig.add_axes([0.005, 0.05, 0.47, 0.80]); ax_arm.set_facecolor(PANEL)
    ax_diag = fig.add_axes([0.55, 0.50, 0.43, 0.30]); ax_diag.set_facecolor(PANEL)
    ax_scope = fig.add_axes([0.55, 0.13, 0.43, 0.26]); ax_scope.set_facecolor(PANEL)

    fig.text(0.005, 0.93, "5-DOF Arm — IMU fusion under phased noise", color=INK, fontsize=15,
             weight="bold", family="sans-serif")
    fig.text(0.005, 0.875, "a bad potentiometer + a good accelerometer per joint · commanded trajectory",
             color=MUT, fontsize=8.5)

    # arm view (fixed workspace box)
    allx = np.concatenate([j_true[..., 0].ravel(), j_pot[..., 0].ravel()])
    ally = np.concatenate([j_true[..., 1].ravel(), j_pot[..., 1].ravel()])
    cx, cy = 0.5 * (allx.min() + allx.max()), 0.5 * (ally.min() + ally.max())
    half = 0.55 * max(allx.max() - allx.min(), ally.max() - ally.min())
    ax_arm.set_xlim(cx - half, cx + half); ax_arm.set_ylim(cy - half, cy + half)
    ax_arm.set_aspect("equal"); ax_arm.axis("off")
    # legend chips
    for lx, lab, c in [(0.02, "true", C_TRUE), (0.20, "raw pot", C_POT), (0.44, "adaptive", C_AD)]:
        ax_arm.text(lx, 0.02, "● " + lab, transform=ax_arm.transAxes, color=c, fontsize=8.5, va="bottom")
    # phase banner inside the panel, top-centre (no title collision)
    banner = ax_arm.text(0.5, 0.99, "", transform=ax_arm.transAxes, ha="center", va="top",
                         fontsize=10.5, weight="bold", family="monospace")
    (l_true,) = ax_arm.plot([], [], "-", color=C_TRUE, lw=2, alpha=.55, solid_capstyle="round", zorder=3)
    (l_pot,) = ax_arm.plot([], [], "-o", color=C_POT, lw=1.6, ms=4, alpha=.85, zorder=5)
    (l_ad,) = ax_arm.plot([], [], "-o", color=C_AD, lw=3.5, ms=5, alpha=.95, zorder=4)
    (trail,) = ax_arm.plot([], [], "-", color=C_AD, lw=1.2, alpha=.3, zorder=2)

    # diagnostic bars
    labels = ["accelerometers", "potentiometers", "process (dynamics)"]
    ax_diag.set_xlim(0, 1); ax_diag.set_ylim(-0.5, 2.5); ax_diag.set_yticks([])
    ax_diag.set_xticks([]); ax_diag.invert_yaxis()
    ax_diag.set_title("which noise is hot  (learned)", color=MUT, fontsize=9, loc="left", pad=6)
    bars = ax_diag.barh([0, 1, 2], [0, 0, 0], height=0.5, color=MUT, zorder=3)
    for yi, lab in zip([0, 1, 2], labels):
        ax_diag.text(0.01, yi - 0.42, lab, color=MUT, fontsize=8, va="center")
        ax_diag.add_patch(plt.Rectangle((0, yi - 0.25), 1, 0.5, facecolor="none",
                          edgecolor=LINE, lw=1, zorder=2))
    NORM = 6.0

    # scope
    ax_scope.set_xlim(0, T * DT); EM = max(err_pot.max(), 0.09) * 1.05; ax_scope.set_ylim(0, EM)
    for nm, a, b in PHASES:
        if nm != "calm":
            ax_scope.axvspan(a * DT, b * DT, color=C_WARN, alpha=0.11, lw=0)
    ax_scope.set_ylabel("tip err (m)", color=MUT, fontsize=8.5); ax_scope.set_xlabel("time (s)", color=MUT, fontsize=8.5)
    ax_scope.tick_params(length=3, labelsize=8)
    (s_pot,) = ax_scope.plot([], [], color=C_POT, lw=1.1, alpha=.6, label="raw pot")
    (s_ad,) = ax_scope.plot([], [], color=C_AD, lw=1.8, label="adaptive")
    head = ax_scope.axvline(0, color=INK, lw=1, alpha=.5)
    ax_scope.legend(loc="upper right", facecolor=PANEL, edgecolor=LINE, labelcolor=INK, fontsize=7.5)

    STEP = 8
    frames = list(range(0, T, STEP))
    tt = np.arange(T) * DT

    def draw(i):
        l_true.set_data(j_true[i, :, 0], j_true[i, :, 1])
        l_pot.set_data(j_pot[i, :, 0], j_pot[i, :, 1])
        l_ad.set_data(j_ad[i, :, 0], j_ad[i, :, 1])
        lo = max(0, i - 60); trail.set_data(tip_ad[lo:i + 1, 0], tip_ad[lo:i + 1, 1])
        ph = phase_at(i)
        lv = [np.clip(diag["accel"][i] / NORM, 0, 1), np.clip(diag["pot"][i] / NORM, 0, 1),
              np.clip(diag["proc"][i] / NORM, 0, 1)]
        for bar, v in zip(bars, lv):
            bar.set_width(max(v, 0.001))
            bar.set_color(C_WARN if v > 0.18 else MUT)
        txt, col = PHASE_TXT[ph]; banner.set_text(txt); banner.set_color(col)
        s_pot.set_data(tt[:i + 1], err_pot[:i + 1]); s_ad.set_data(tt[:i + 1], err_ad[:i + 1])
        head.set_xdata([i * DT, i * DT])
        return ()

    fig.canvas.draw()
    imgs = []
    for i in frames:
        draw(i); fig.canvas.draw()
        imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert(
            "P", palette=Image.ADAPTIVE, colors=64))
    imgs[0].save(OUT, save_all=True, append_images=imgs[1:], duration=60, loop=0,
                 disposal=2, optimize=True)
    print("wrote", os.path.relpath(OUT), "-", os.path.getsize(OUT) // 1024, "KB,", len(frames), "frames")


if __name__ == "__main__":
    render()
