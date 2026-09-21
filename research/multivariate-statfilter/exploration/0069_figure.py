import re, glob, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
S="/tmp/claude-0/-home-user-lucid-filter/56bbf4e5-3fe2-5e9c-93f7-d50a0493cfbc/scratchpad/"
BLUE, ORANGE, AQUA, YELLOW, MAG, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"
INK, INK2, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#8a8983", "#fcfcfb", "#e6e5e0"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.titlecolor": INK})
def parse(path):
    out = []
    for line in open(path):
        m = re.match(r"seed (\d): burst RMSE lucid ([\d.]+) oracle ([\d.]+) \| worst tip error ([\d.]+) m at step (\d+) \((\w+)@(\d+)\).*calm ([\d.]+)\s+SENSOR ([\d.]+)\s+PROCESS ([\d.]+)\s+POTFAIL ([\d.]+)\s+BOTH ([\d.]+)", line)
        if m: out.append([float(m.group(i)) for i in (2, 3, 4)] + [float(m.group(i)) for i in (8, 9, 10, 11, 12)])
    return np.array(out)   # rows: seeds; cols: burst, oracle burst, worst, calm, SENSOR, PROCESS, POTFAIL, BOTH
eps = ["0", "1e-6", "1e-4", "1e-3", "1e-2", "1e-1"]; R = {e: parse(S + f"floor_fl_{e}.log") for e in eps}
R4 = {"0": parse(S + "floor_fl4_0.log"), "1e-3": parse(S + "floor_fl4_1e-3.log")}
main = np.array([[0.0123, 0.0089, 0.043, 1.14, 2.70, 1.16, 1.11, 1.06], [0.0187, 0.0097, 0.069, 1.10, 3.48, 1.03, 1.16, 1.72], [0.0262, 0.0075, 0.105, 0.56, 2.14, 1.16, 0.94, 3.52]])
x = np.array([1e-7, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1]); xl = ["0", "1e-6", "1e-4", "1e-3", "1e-2", "1e-1"]
fig, ax = plt.subplots(2, 3, figsize=(16, 9.4)); fig.subplots_adjust(hspace=0.5, wspace=0.3, left=0.05, right=0.99, top=0.9, bottom=0.08)
def style(a):
    a.grid(axis="y", color=GRID, lw=0.8); a.set_axisbelow(True); a.tick_params(length=0)
def logx(a):
    a.set_xscale("log"); a.set_xticks(x); a.set_xticklabels(xl); a.set_xlabel("posterior floor ε of an element (0 = today's grid)")
    a.axvline(1e-3, color=MUTED, lw=1, ls=(0, (3, 3))); a.text(1e-3, a.get_ylim()[1] * 0.98, " derived ε = 1/mem", color=INK2, fontsize=7.5, va="top")
# A burst RMSE per seed vs eps
a = ax[0, 0]
for s, c in zip(range(3), (BLUE, ORANGE, AQUA)):
    a.plot(x, [R[e][s, 0] for e in eps], color=c, lw=2, marker="o", ms=5, label=f"seed {s}")
    a.axhline(main[s, 0], color=c, lw=1, ls=(0, (3, 3)))
a.plot(x, [R[e][:, 0].mean() for e in eps], color=INK, lw=2.5, marker="s", ms=6, label="mean of 3 seeds")
a.axhline(main[:, 0].mean(), color=INK, lw=1, ls=(0, (3, 3))); a.text(1.2e-7, main[:, 0].mean() + 0.002, "dashed: main", color=INK2, fontsize=7.5)
a.set_ylabel("tip RMSE over every burst step, m"); a.set_title("A  Whole-burst RMSE vs the floor (two elements)", loc="left", fontweight="bold"); style(a); logx(a); a.legend(frameon=False, fontsize=7.5, loc="upper right")
# B worst excursion per seed
a = ax[0, 1]
for s, c in zip(range(3), (BLUE, ORANGE, AQUA)):
    a.plot(x, [R[e][s, 2] for e in eps], color=c, lw=2, marker="o", ms=5, label=f"seed {s}"); a.axhline(main[s, 2], color=c, lw=1, ls=(0, (3, 3)))
a.set_yscale("log"); a.set_ylabel("worst tip error in the run, m (log)"); a.set_title("B  Worst excursion vs the floor", loc="left", fontweight="bold"); style(a); logx(a); a.legend(frameon=False, fontsize=7.5)
# C regime windows (mean over seeds) vs eps
a = ax[0, 2]
for j, (nm, c) in enumerate(zip(["calm", "SENSOR", "PROCESS", "POTFAIL", "BOTH"], (BLUE, ORANGE, AQUA, YELLOW, MAG))):
    a.plot(x, [R[e][:, 3 + j].mean() for e in eps], color=c, lw=2, marker="o", ms=4, label=nm); a.axhline(main[:, 3 + j].mean(), color=c, lw=0.8, ls=(0, (3, 3)))
a.set_ylabel("regime window RMSE / oracle (mean of 3 seeds)"); a.set_title("C  Regime windows vs the floor (dashed: main)", loc="left", fontweight="bold"); style(a); logx(a); a.legend(frameon=False, fontsize=7.5, ncol=2)
# D four elements vs two
a = ax[1, 0]; labels = ["main", "2 elements\nε = 0", "2 elements\nε = 1e-3", "4 elements\nε = 0", "4 elements\nε = 1e-3/3"]
vals = [main[:, 0].mean(), R["0"][:, 0].mean(), R["1e-3"][:, 0].mean(), R4["0"][:, 0].mean(), R4["1e-3"][:, 0].mean()]
worst = [main[:, 2].max(), R["0"][:, 2].max(), R["1e-3"][:, 2].max(), R4["0"][:, 2].max(), R4["1e-3"][:, 2].max()]
a.bar(labels, vals, 0.55, color=[BLUE, ORANGE, ORANGE, VIOLET, VIOLET])
for i, (v, w) in enumerate(zip(vals, worst)): a.text(i, v + 0.002, f"{v:.3f} m\nworst {w:.2f}", ha="center", fontsize=7.5, color=INK2)
a.set_ylabel("tip RMSE over every burst step, m"); a.set_title("D  Two elements vs four on the axis", loc="left", fontweight="bold"); style(a); a.set_ylim(0, max(vals) * 1.5)
# E onset traces seed 1
a = ax[1, 1]; t = np.arange(235, 340)
for tag, lab, c in (("grid", "ε = 0 (grid)", ORANGE), ("fl1e6", "ε = 1e-6", YELLOW), ("fl1e3", "ε = 1e-3 (derived)", AQUA), ("fl1e1", "ε = 1e-1", MAG), ("fl4", "4 elements, ε = 1e-3/3", VIOLET), ("main", "main", BLUE)):
    try:
        d = np.load(S + f"fig_{tag}.npz"); a.plot(t, d["err"][235:340], color=c, lw=1.8 if tag != "main" else 2.2, label=lab)
    except Exception as ex: print("missing", tag, ex)
d = np.load(S + "fig_main.npz"); a.plot(t, d["oerr"][235:340], color=MUTED, lw=1.4, ls=(0, (3, 3)), label="oracle")
a.set_yscale("log"); a.set_xlabel("step"); a.set_ylabel("tip error, m (log)"); a.set_title("E  Seed 1's first sensor onset under each floor", loc="left", fontweight="bold"); style(a); a.legend(frameon=False, fontsize=7, loc="lower left"); a.set_ylim(1e-3, 3)
a.axvspan(250, 340, color=ORANGE, alpha=0.06, lw=0)
# F SENSOR and BOTH per seed vs eps
a = ax[1, 2]
for s, c in zip(range(3), (BLUE, ORANGE, AQUA)):
    a.plot(x, [R[e][s, 4] for e in eps], color=c, lw=2, marker="o", ms=5, label=f"SENSOR, seed {s}")
    a.plot(x, [R[e][s, 7] for e in eps], color=c, lw=1.4, marker="^", ms=5, ls=(0, (4, 2)), label=f"BOTH, seed {s}")
a.set_ylabel("window RMSE / oracle"); a.set_title("F  SENSOR (solid) and BOTH (dashed) per seed vs the floor", loc="left", fontweight="bold"); style(a); logx(a); a.legend(frameon=False, fontsize=6.5, ncol=2)
fig.suptitle("The floor of an element on the memory axis: derived ε = 1/mem against the sweep (0069; arm, 3 seeds)", fontsize=12, color=INK, x=0.05, ha="left", y=0.965)
fig.savefig(S + "fig_floor.png", dpi=150); print("saved")
