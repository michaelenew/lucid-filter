"""The six-panel figure of 0065-0068 (regime windows, whole-burst RMSE, seed 1 onset, the spike step inside the bank, the copies' per-step evidence, the derived kernels).  Needs the npz files from 0067_figure_data.py for main, the grid commit and grid + lag-1 weights."""
import sys, math, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "."); sys.path.insert(0, "research/multivariate-statfilter/scripts")
S="/tmp/claude-0/-home-user-lucid-filter/56bbf4e5-3fe2-5e9c-93f7-d50a0493cfbc/scratchpad/"
BLUE, ORANGE, AQUA, YELLOW, MAG = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#8a8983", "#fcfcfb", "#e6e5e0"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.titlecolor": INK})
fig, ax = plt.subplots(2, 3, figsize=(16, 9.6)); fig.subplots_adjust(hspace=0.55, wspace=0.3, left=0.05, right=0.99, top=0.9, bottom=0.07)
def style(a):
    a.grid(axis="y", color=GRID, lw=0.8); a.set_axisbelow(True); a.tick_params(length=0)
# ---- A: arm regimes (3 seeds, ratio to oracle)
a = ax[0, 0]; regimes = ["calm", "SENSOR", "PROCESS", "POTFAIL", "BOTH"]
rows = {"main": [0.94, 2.77, 1.11, 1.07, 2.10], "branch head\n(grid + memory ladder)": [0.91, 1.16, 1.11, 1.08, 1.90], "memory ladder alone": [0.94, 2.77, 1.11, 1.07, 2.10]}
cols = [BLUE, ORANGE, AQUA]; x = np.arange(5); w = 0.26
for i, (nm, v) in enumerate(rows.items()):
    b = a.bar(x + (i - 1) * (w + 0.02), v, w, color=cols[i], label=nm)
    for xi, vi in zip(x + (i - 1) * (w + 0.02), v):
        if nm.startswith("branch") and vi < 2: a.text(xi, vi + 0.05, f"{vi:.2f}", ha="center", fontsize=7.5, color=INK2)
a.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (3, 3))); a.text(-0.55, 1.06, "oracle = 1", color=MUTED, fontsize=7.5, ha="left")
a.set_xticks(x); a.set_xticklabels(regimes); a.set_ylabel("tip RMSE / oracle (3 seeds)"); a.set_title("A  Arm regime windows: the windows improve", loc="left", fontweight="bold")
a.legend(frameon=False, fontsize=7.5, loc="upper right"); style(a); a.set_ylim(0, 3.2)
# ---- B: whole-burst RMSE (onsets included)
a = ax[0, 1]; names = ["main", "grid\ncommit", "branch\nhead", "memory\nladder alone"]; vals = [0.0190, 0.0711, 0.0684, 0.0190]; errs = [0.0040, 0.05, 0.0522, 0.0040]
bc = [BLUE, ORANGE, ORANGE, AQUA]
a.bar(names, vals, 0.55, color=bc, yerr=errs, error_kw=dict(ecolor=INK2, lw=1, capsize=3))
for i, v in enumerate(vals): a.text(i, v + errs[i] + 0.004, f"{v:.3f} m", ha="center", fontsize=8, color=INK2)
a.set_ylabel("tip RMSE over every burst step, m"); a.set_title("B  ...but the onsets, which the windows skip, do not", loc="left", fontweight="bold"); style(a); a.set_ylim(0, 0.15)
a.text(1.5, 0.135, "one seed: a 1.08 m excursion for 40 steps\nat the first sensor onset (the grid's)", ha="center", fontsize=7.5, color=INK2)
# ---- C: seed-1 onset, tip error per step
a = ax[0, 2]
srcs = [("main", "main", BLUE), ("grid", "grid commit", ORANGE), ("gridlag1", "grid + lag-1 weights", YELLOW)]
t = np.arange(235, 340); data = {}
for tag, lab, c in srcs:
    d = np.load(S + f"fig_{tag}.npz"); data[tag] = d
    a.plot(t, d["err"][235:340], color=c, lw=2, label=lab)
a.plot(t, data["main"]["oerr"][235:340], color=MUTED, lw=1.5, ls=(0, (3, 3)), label="oracle")
a.axvspan(250, 340, color=ORANGE, alpha=0.06, lw=0); a.text(252, 2.0, "accelerometers ×15 from t = 250", fontsize=7.5, color=INK2, va="top")
a.set_yscale("log"); a.set_xlabel("step"); a.set_ylabel("tip error, m (log)"); a.set_title("C  Seed 1, the first sensor onset, step by step", loc="left", fontweight="bold")
a.legend(frameon=False, fontsize=7.5, loc="lower left"); style(a); a.set_ylim(1e-3, 3)
# ---- D: the spike step inside the bank: joint-1 position state of every eager-copy member
a = ax[1, 0]
for j, (tag, lab, c) in enumerate([("main", "main", BLUE), ("grid", "grid commit", ORANGE)]):
    d = data[tag]; th = d["th1"]; wt = d["wts"]; M = th.shape[1]; ncell = 15
    for k, tt in enumerate((249, 250, 251)):
        xs = j * 4 + k + np.zeros(ncell); ys = th[tt, :ncell]; ww = wt[tt].reshape(-1, ncell) if tag == "main" else wt[tt].reshape(2, ncell)
        wv = ww[0] if tag == "main" else ww[0]
        a.scatter(xs + np.linspace(-0.25, 0.25, ncell), ys, s=8 + 160 * wv, color=c, alpha=0.75, edgecolor=SURF, lw=0.8)
    a.text(j * 4 + 1, 5.6, lab, ha="center", fontsize=8.5, color=c, fontweight="bold")
a.axhline(0.15, color=MUTED, lw=0.8, ls=(0, (3, 3))); a.text(6.9, 0.3, "truth ≈ 0.15 rad", color=MUTED, fontsize=7.5, ha="right")
a.set_xticks([0, 1, 2, 4, 5, 6]); a.set_xticklabels(["t=249", "250", "251", "t=249", "250", "251"]); a.set_ylabel("joint-1 position state of each cell, rad")
a.set_title("D  Inside the spike step (dot size = weight): two cells jump to\n     4.9 rad in both; main ignores them, the grid hands them the weight", loc="left", fontweight="bold", fontsize=8.5); style(a); a.set_ylim(-0.5, 6.2)
# ---- E: the copies' per-step log-likelihood difference across the onset (grid commit)
a = ax[1, 1]
dll = [-0.25, -0.04, 0.40, 0.09, 0.49, -1212.16, -417.75, -np.inf, -np.inf, -16.59, -np.inf, -29.06, -41.67, -12.69, -132.64, 11.55, -19.29, 17.18, 11.77, 17.85, 22.75, 22.16, 24.02, 27.22, 19.28, 8.59, -85.41, 26.01, 24.10, 21.97, 17.40, 8.82, 21.20, 21.56, 26.58, 23.03, 28.81, 30.43, 27.14, 26.84, 27.71, 19.38, 30.20, 22.67, 30.96, 3.10, 0.63, 0.13, 0.54, 0.47]
lags = np.arange(-5, 45); v = np.array(dll); inf = ~np.isfinite(v); v2 = np.clip(np.where(inf, -1500, v), -1500, 40)
colors = [AQUA if x > 0 else ORANGE for x in v2]
a.bar(lags, v2, 0.8, color=colors)
for l, isinf in zip(lags, inf):
    if isinf: a.text(l, -1300, "-inf", ha="center", fontsize=6, color=SURF, rotation=90, va="top")
a.set_yscale("symlog", linthresh=10); a.set_xlabel("lag after the onset (steps)"); a.set_ylabel("log-likelihood, patient − eager (symlog)")
a.set_title("E  Grid copies' evidence per step: the spike step and the next\n     ten cost the patient copy thousands of nats; lags ≥ 15 favour it", loc="left", fontweight="bold", fontsize=8.5); style(a)
a.axhline(0, color=MUTED, lw=0.8); a.set_ylim(-2000, 60)
# ---- F: the derived kernels D(l) against the stand-in
a = ax[1, 2]
def steady(F, H, Q, R):
    n = F.shape[0]; P = Q + np.eye(n); I = np.eye(n)
    for _ in range(20000):
        Pp = F @ P @ F.T + Q; Sm = H @ Pp @ H.T + R; K = Pp @ H.T @ np.linalg.solve(Sm, np.eye(Sm.shape[0]))
        Pn = (I - K @ H) @ Pp @ (I - K @ H).T + K @ R @ K.T; Pn = 0.5 * (Pn + Pn.T)
        if np.max(np.abs(Pn - P)) < 1e-12 * (1 + np.max(np.abs(P))): break
        P = Pn
    return K, Sm
def profile(F, H, K, Sm, bdir, chan, L=120):
    n = F.shape[0]; A = (np.eye(n) - K @ H) @ F; s = np.zeros(H.shape[0]); s[chan] = 1.0
    x = bdir.copy(); xs = K @ s; D = np.zeros(L)
    for l in range(1, L):
        x = A @ x; xs = A @ xs; D[l] = (H @ x)[chan] + (H @ xs)[chan]
    return np.abs(D) / np.abs(D).max()
lag = np.arange(120)
K, Sm = steady(np.eye(1), np.eye(1), np.array([[0.02]]), np.array([[1.0]])); a.plot(lag, profile(np.eye(1), np.eye(1), K, Sm, np.ones(1), 0), color=BLUE, lw=2, label="scalar level (q=0.02, r=1): mode 1, tail 7.6")
import arm5dof as AR
F, H, Q, R = AR.F, AR.H_CHAR, AR.Q0, np.diag(AR.R0); K, Sm = steady(F, H, Q, R); bj = AR.B[:, 2] / np.linalg.norm(AR.B[:, 2])
a.plot(lag, profile(F, H, K, Sm, bj, 7), color=ORANGE, lw=2, label="arm joint 2: jerk vs its accelerometer (n = 1): mode 1")
a.plot(lag, profile(F, H, K, Sm, bj, 6), color=AQUA, lw=2, label="arm joint 2: jerk vs its pot (a chain): mode ≈ 20")
l2 = np.arange(0, 40, dtype=float); chi = np.where(l2 > 0, l2 ** 0.5 * np.exp(-l2 / 2), 0); a.plot(l2, chi / chi.max(), color=INK2, lw=1.2, ls=(0, (3, 3)), label="chi-squared stand-in used in stage A (mode 1, scale 2)")
a.set_xlabel("lag l after an innovation"); a.set_ylabel("|D(l)| / max: discriminating impulse response"); a.set_xlim(0, 120)
a.set_title("F  The derived kernel D(l): zero at the spike step,\n     then the loop's own impulse response", loc="left", fontweight="bold", fontsize=8.5); a.legend(frameon=False, fontsize=7, loc="center right"); style(a)
fig.suptitle("lucid-filter: main vs the branch, and why — the arm's sensor onset and the lag kernel (0065–0068)", fontsize=12, color=INK, x=0.05, ha="left", y=0.965)
out = S + "fig_results.png"; fig.savefig(out, dpi=150); print("saved", out)
