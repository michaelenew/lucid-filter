"""Render the crusher-watch animation to a GIF for the README.

Data is the real AdaptiveKalmanFilter output (see exploration/crusher_anim.json, produced by
the 2-DOF crusher scenario).  Two panels: the robot cell (true arm + three tip estimates +
the crusher, which shakes while it runs) and an oscilloscope of tip error with the crusher
window shaded.  The story: when the crusher fires and swamps the encoders, the raw and
fixed-noise estimates jitter while the adaptive estimate stays locked on the truth.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "..", "exploration", "crusher_anim.json")))
OUT = os.path.join(HERE, "..", "figures", "crusher-adaptive.gif")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

BG, PANEL, LINE, INK, MUT = "#0e1217", "#161c24", "#2a3542", "#dbe3ec", "#7c8a99"
C_TRUE, C_RAW, C_ASSUMED, C_FILT, C_WARN = "#8895a5", "#e2607b", "#f0a24e", "#33e0a6", "#f5b23d"

d = DATA
T = len(d["true"]); L1, L2 = d["L1"], d["L2"]; dt = d["dt"]; onA, onB = d["on"]
true = np.array(d["true"]); enc = np.array(d["enc"])
assumed = np.array(d["assumed"]); filt = np.array(d["filtered"])
er, ea, ef = np.array(d["err_raw"]), np.array(d["err_assumed"]), np.array(d["err_filtered"])


def fk(a):
    x1 = L1 * np.cos(a[0]); y1 = L1 * np.sin(a[0])
    return (x1, y1), (x1 + L2 * np.cos(a[0] + a[1]), y1 + L2 * np.sin(a[0] + a[1]))


plt.rcParams.update({"font.family": "monospace", "font.size": 9, "text.color": INK,
                     "axes.edgecolor": LINE, "xtick.color": MUT, "ytick.color": MUT})
fig = plt.figure(figsize=(9.2, 4.2), dpi=110)
fig.patch.set_facecolor(BG)
# explicit panel positions [left, bottom, width, height] -- no gridspec/aspect fights
axc = fig.add_axes([0.015, 0.10, 0.40, 0.72])     # robot cell (left)
axs = fig.add_axes([0.52, 0.16, 0.465, 0.60])     # oscilloscope (right)
for ax in (axc, axs):
    ax.set_facecolor(PANEL)

# ---- static scope frame ----
axs.set_xlim(0, (T - 1) * dt); EMAX = 0.9; axs.set_ylim(0, EMAX)
axs.axvspan(onA * dt, onB * dt, color=C_WARN, alpha=0.12, lw=0)
axs.axvline(onA * dt, color=C_WARN, ls=(0, (2, 3)), lw=1, alpha=.6)
axs.axvline(onB * dt, color=C_WARN, ls=(0, (2, 3)), lw=1, alpha=.6)
axs.text(onA * dt + 0.1, EMAX * 0.93, "CRUSHER RUNNING", color=C_WARN, fontsize=8, weight="bold")
for s in axs.spines.values():
    s.set_color(LINE)
axs.set_ylabel("tip error (m)", color=MUT, fontsize=9)
axs.set_xlabel("time (s)", color=MUT, fontsize=9)
axs.tick_params(length=3)
(ln_a,) = axs.plot([], [], color=C_ASSUMED, lw=1.6, label="fixed-noise filter")
(ln_r,) = axs.plot([], [], color=C_RAW, lw=1.2, alpha=.6, label="raw encoder")
(ln_f,) = axs.plot([], [], color=C_FILT, lw=2.2, label="adaptive filter")
head = axs.axvline(0, color=INK, lw=1, alpha=.5)
axs.legend(loc="upper right", facecolor=PANEL, edgecolor=LINE, labelcolor=INK,
           fontsize=8, framealpha=.9)

# ---- titles (top band, above both panels) ----
fig.text(0.015, 0.92, "Crusher Watch", color=INK, fontsize=17, weight="bold", family="sans-serif")
fig.text(0.015, 0.855, "AdaptiveKalmanFilter — encoders swamped ×250 while the crusher runs",
         color=MUT, fontsize=8.5)
flag = fig.text(0.985, 0.92, "", ha="right", va="center", fontsize=9, weight="bold",
                family="monospace")

# ---- robot-cell axes set up ONCE (no per-frame cla / aspect reflow) ----
XL, YL = (-0.8, 2.15), (-0.45, 1.95)
axc.set_aspect("equal", adjustable="box"); axc.set_xlim(*XL); axc.set_ylim(*YL); axc.axis("off")
axc.text(XL[0], YL[1], "ROBOT CELL  ·  tip tracking", color=MUT, fontsize=9, va="top")
# static crusher (colour toggles with state; position fixed)
CX, CY, CW, CH = 1.6, 0.0, 0.42, 1.35
warn_bg = plt.Rectangle((XL[0], YL[0]), XL[1] - XL[0], YL[1] - YL[0], color=C_WARN, alpha=0, lw=0, zorder=0)
crush = plt.Rectangle((CX - CW / 2, CY), CW, CH, facecolor=PANEL, edgecolor=LINE, lw=2, zorder=2)
axc.add_patch(warn_bg); axc.add_patch(crush)
teeth = []
for k in range(5):
    tx = CX - CW / 2 + (k + .5) * CW / 5
    (poly,) = axc.fill([tx - CW / 12, tx + CW / 12, tx], [CY + CH, CY + CH, CY + CH - .13],
                       color="#4d5a68", zorder=3)
    teeth.append(poly)
axc.text(CX, CY - 0.16, "CRUSHER", ha="center", color=MUT, fontsize=8)
axc.plot(0, 0, "o", color=LINE, ms=10, zorder=2)
(arm_true,) = axc.plot([], [], color=C_TRUE, lw=6, alpha=.85, solid_capstyle="round",
                       ls=(0, (1, 2)), zorder=4)
(trail_f,) = axc.plot([], [], color=C_FILT, lw=1.6, alpha=.35, zorder=5)
(trail_a,) = axc.plot([], [], color=C_ASSUMED, lw=1.4, alpha=.22, zorder=5)
(m_true,) = axc.plot([], [], "o", color=C_TRUE, ms=4, zorder=6)
(m_raw,) = axc.plot([], [], marker="+", color=C_RAW, ms=9, mew=2, ls="", zorder=7)
(m_as,) = axc.plot([], [], marker="o", mfc="none", mec=C_ASSUMED, ms=9, mew=2.2, ls="", zorder=7)
(m_fi,) = axc.plot([], [], marker="o", color=C_FILT, ms=8, ls="", zorder=8)
(m_halo,) = axc.plot([], [], marker="o", mfc="none", mec=C_FILT, ms=15, mew=1.5, alpha=.35, ls="", zorder=8)

STEP = 4
frames = list(range(0, T, STEP))


def draw(i):
    on = onA <= i < onB
    warn_bg.set_alpha(0.08 if on else 0.0)
    crush.set_edgecolor(C_WARN if on else LINE)
    for poly in teeth:
        poly.set_color(C_WARN if on else "#4d5a68")
    (e, t) = fk(true[i])
    arm_true.set_data([0, e[0], t[0]], [0, e[1], t[1]])
    m_true.set_data([t[0]], [t[1]])
    lo = max(0, i - 40)
    ft = np.array([fk(filt[k])[1] for k in range(lo, i + 1)])
    at = np.array([fk(assumed[k])[1] for k in range(lo, i + 1)])
    trail_f.set_data(ft[:, 0], ft[:, 1]); trail_a.set_data(at[:, 0], at[:, 1])
    tr = fk(enc[i])[1]; ta = fk(assumed[i])[1]; tf = fk(filt[i])[1]
    m_raw.set_data([tr[0]], [tr[1]]); m_as.set_data([ta[0]], [ta[1]])
    m_fi.set_data([tf[0]], [tf[1]]); m_halo.set_data([tf[0]], [tf[1]])
    tt = np.arange(i + 1) * dt
    ln_a.set_data(tt, ea[:i + 1]); ln_r.set_data(tt, er[:i + 1]); ln_f.set_data(tt, ef[:i + 1])
    head.set_xdata([i * dt, i * dt])
    flag.set_text("● CRUSHER RUNNING" if on else "○ crusher idle")
    flag.set_color(C_WARN if on else MUT)
    return ()


# ---- capture each frame at the fixed canvas size, assemble GIF with explicit disposal ----
fig.canvas.draw()
imgs = []
for i in frames:
    draw(i)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    imgs.append(Image.fromarray(buf).convert("P", palette=Image.ADAPTIVE, colors=128))
imgs[0].save(OUT, save_all=True, append_images=imgs[1:], duration=50, loop=0,
             disposal=2, optimize=True)
print("wrote", os.path.relpath(OUT), "-", os.path.getsize(OUT) // 1024, "KB,", len(frames), "frames")
