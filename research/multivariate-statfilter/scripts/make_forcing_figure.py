"""Static before/after figure: known forcing (B u) removes the lag on a driven arm.

A single joint is driven along a commanded sinusoidal-acceleration reach and measured by a
noisy encoder.  Without the command the constant-velocity filter LAGS (position and, badly,
velocity).  Feeding the same commanded acceleration as forcing collapses the lag to the sensor
floor.  Real AdaptiveKalmanFilter output.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
from statfilter import AdaptiveKalmanFilter  # noqa: E402

BG, PANEL, LINE, INK, MUT = "#0e1217", "#161c24", "#2a3542", "#dbe3ec", "#7c8a99"
C_TRUE, C_NOF, C_FOR = "#8895a5", "#f0a24e", "#33e0a6"
OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "forcing-lag-fix.png")

T, dt = 300, 0.05
rng = np.random.default_rng(0); t = np.arange(T) * dt
F = np.array([[1.0, dt], [0.0, 1.0]]); G = np.array([[0.5 * dt * dt], [dt]]); H = np.array([[1.0, 0.0]])
acmd = 1.2 * np.sin(2 * np.pi * t / (T * dt))
x = np.zeros(2); X = np.zeros((T, 2)); Y = np.zeros((T, 1))
for k in range(T):
    x = F @ x + (G * acmd[k]).ravel(); X[k] = x; Y[k] = H @ x + 0.02 * rng.standard_normal()
Q0 = 1e-4 * (G @ G.T) + 1e-9 * np.eye(2); R0 = [0.02 ** 2 * 4]
no_f = AdaptiveKalmanFilter(Q0, R0=R0, H=H, F=F, s=0.5).filter(Y).mean
fc = AdaptiveKalmanFilter(Q0, R0=R0, H=H, F=F, B=G, s=0.5)
with_f = fc.filter(Y, U=acmd[:, None]).mean

plt.rcParams.update({"font.family": "monospace", "font.size": 9, "text.color": INK,
                     "axes.edgecolor": LINE, "xtick.color": MUT, "ytick.color": MUT})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 4.8), dpi=110, sharex=True)
fig.patch.set_facecolor(BG); fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.11, hspace=0.18)
fig.text(0.09, 0.93, "Known forcing removes the tracking lag", color=INK, fontsize=15,
         weight="bold", family="sans-serif")
fig.text(0.09, 0.865, "arm driven along a commanded reach · noisy encoder measures position only",
         color=MUT, fontsize=9)
for ax, col, title, tru, a, b in [
        (ax1, 0, "position (rad)", X[:, 0], no_f[:, 0], with_f[:, 0]),
        (ax2, 1, "velocity (rad/s)", X[:, 1], no_f[:, 1], with_f[:, 1])]:
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.plot(t, tru, color=C_TRUE, lw=3, alpha=.9, label="true")
    ax.plot(t, a, color=C_NOF, lw=1.8, label="no forcing (lags)")
    ax.plot(t, b, color=C_FOR, lw=1.8, label="+ known forcing")
    ax.set_ylabel(title, color=MUT, fontsize=9); ax.tick_params(length=3)
    rn = np.sqrt(((a - tru) ** 2).mean()); rf = np.sqrt(((b - tru) ** 2).mean())
    ax.text(0.985, 0.06, f"RMSE  {rn:.3f} → {rf:.3f}", transform=ax.transAxes, ha="right",
            color=INK, fontsize=9, weight="bold")
ax1.legend(loc="upper right", facecolor=PANEL, edgecolor=LINE, labelcolor=INK, fontsize=8, ncol=3)
ax2.set_xlabel("time (s)", color=MUT, fontsize=9)
fig.savefig(OUT, facecolor=BG)
print("wrote", os.path.relpath(OUT), "-", os.path.getsize(OUT) // 1024, "KB")
