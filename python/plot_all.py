"""
plot_all.py
===========
Publication-quality figures for uncertainty-aware active reconstruction.
Loads RL, Random, Sequential metrics CSVs from hardcoded paths.
Saves all figures to OUTPUT_DIR as PNG and PDF.

USAGE:
    python plot_all.py

EDIT THE THREE PATHS BELOW to match your machine.
"""

import os
import csv
import statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ═══════════════════════════════════════════════════════════════════════════
# ── CONFIGURE THESE PATHS ──────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

RL_CSV         = r"D:\Downloads_IDM\ActiveReconstruction_Final\Assets\Metrics\RL_metrics.csv"
RANDOM_CSV     = r"D:\Downloads_IDM\ActiveReconstruction_Final\Assets\Metrics\Random_metrics.csv"
SEQUENTIAL_CSV = r"D:\Downloads_IDM\ActiveReconstruction_Final\Assets\Metrics\Sequential_metrics.csv"

# All figures saved here (created automatically)
OUTPUT_DIR     = r"D:\Downloads_IDM\ActiveReconstruction_Final\Assets\Metrics\Figures"

# ═══════════════════════════════════════════════════════════════════════════
# ── STYLE CONSTANTS ────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

C_RL  = "#00c8ff"   # cyan
C_RND = "#ff4444"   # red
C_SEQ = "#ffcc00"   # yellow
C_BG  = "#0d0d0d"
C_AX  = "#1a1a1a"
C_GRID= "#333333"
C_TXT = "white"

LABELS = ["Random", "Sequential", "RL (PPO)"]
COLORS = [C_RND, C_SEQ, C_RL]

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         150,
    "savefig.dpi":        200,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  C_BG,
})

# ═══════════════════════════════════════════════════════════════════════════
# ── HELPERS ────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def load_csv(path):
    """Load metrics CSV. Returns dict of lists keyed by short name."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}\nEdit the path at the top of plot_all.py")
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    return {
        "episode": [int(r["Episode"])            for r in rows],
        "reward":  [float(r["TotalReward"])       for r in rows],
        "cov":     [float(r["Coverage"])           for r in rows],
        "uncert":  [float(r["MeanUncertainty"])    for r in rows],
        "acc":     [float(r["ReconstructionAccuracy"]) for r in rows],
        "images":  [int(r["ImageCount"])           for r in rows],
        "runtime": [float(r["RuntimeSeconds"])     for r in rows],
    }


def rolling(data, w=5):
    """Centred rolling mean, clamped at edges."""
    out = []
    n = len(data)
    for i in range(n):
        lo = max(0, i - w // 2)
        hi = min(n, i + w // 2 + 1)
        out.append(sum(data[lo:hi]) / (hi - lo))
    return out


def style_ax(ax):
    """Apply dark theme to axes."""
    ax.set_facecolor(C_AX)
    ax.tick_params(colors=C_TXT, which="both")
    ax.xaxis.label.set_color(C_TXT)
    ax.yaxis.label.set_color(C_TXT)
    ax.title.set_color(C_TXT)
    ax.grid(color=C_GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")


def save(fig, name):
    """Save figure as PNG and PDF."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    png = os.path.join(OUTPUT_DIR, name + ".png")
    pdf = os.path.join(OUTPUT_DIR, name + ".pdf")
    fig.savefig(png, facecolor=C_BG)
    fig.savefig(pdf, facecolor=C_BG)
    print(f"  Saved: {png}")
    print(f"  Saved: {pdf}")
    plt.close(fig)


def legend_handles():
    return [
        Line2D([0], [0], color=C_RL,  lw=2,   label="RL (PPO)"),
        Line2D([0], [0], color=C_RND, lw=1.5, label="Random",     ls="--"),
        Line2D([0], [0], color=C_SEQ, lw=1.5, label="Sequential", ls=":"),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# ── LOAD DATA ──────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

print("Loading CSVs...")
rl  = load_csv(RL_CSV)
rnd = load_csv(RANDOM_CSV)
seq = load_csv(SEQUENTIAL_CSV)
print(f"  RL: {len(rl['reward'])} episodes")
print(f"  Random: {len(rnd['reward'])} episodes")
print(f"  Sequential: {len(seq['reward'])} episodes")

eps_rl  = rl["episode"]
eps_rnd = rnd["episode"]
eps_seq = seq["episode"]

# ═══════════════════════════════════════════════════════════════════════════
# FIG 1 — 2x2 per-episode metric curves
# ═══════════════════════════════════════════════════════════════════════════

print("\n[Fig 1] Per-episode metric curves...")
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.patch.set_facecolor(C_BG)
fig.suptitle(
    "Reconstruction Performance: RL vs Baselines (64 viewpoints, 32 steps/episode)",
    color=C_TXT, fontsize=13, y=1.01, fontweight="bold"
)

metrics_2x2 = [
    ("reward", "Cumulative Reward",        "Reward",       True),
    ("cov",    "Voxel Coverage",           "Coverage",     True),
    ("uncert", "Mean Uncertainty",         "Uncertainty",  False),
    ("acc",    "Reconstruction Accuracy",  "Accuracy",     True),
]

for ax, (key, title, ylabel, higher_better) in zip(axes.flat, metrics_2x2):
    style_ax(ax)
    arrow = "↑ better" if higher_better else "↓ better"

    ax.plot(eps_rl,  rolling(rl[key]),  color=C_RL,  lw=2.2, label="RL (PPO)",   zorder=3)
    ax.plot(eps_rnd, rolling(rnd[key]), color=C_RND, lw=1.5, label="Random",     ls="--", zorder=2)
    ax.plot(eps_seq, rolling(seq[key]), color=C_SEQ, lw=1.5, label="Sequential", ls=":",  zorder=2)

    # Confidence band for RL (±0.5 std, static)
    rl_mean = np.mean(rl[key])
    rl_std  = np.std(rl[key])
    ax.axhspan(rl_mean - rl_std * 0.5, rl_mean + rl_std * 0.5,
               color=C_RL, alpha=0.08, zorder=1)

    ax.set_title(f"{title}  ({arrow})", pad=6)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.legend(handles=legend_handles(), facecolor="#222",
              labelcolor=C_TXT, framealpha=0.85, loc="best")

plt.tight_layout()
save(fig, "fig1_metric_curves")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 2 — Bar chart: mean ± std for all 4 metrics
# ═══════════════════════════════════════════════════════════════════════════

print("[Fig 2] Bar comparison chart...")
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
fig.patch.set_facecolor(C_BG)
fig.suptitle(
    "Mean Performance ± Std (50 Evaluation Episodes)",
    color=C_TXT, fontsize=13, fontweight="bold", y=1.02
)

bar_metrics = [
    ("reward", "Avg Cumulative Reward",      True),
    ("cov",    "Avg Voxel Coverage",         True),
    ("uncert", "Avg Mean Uncertainty",       False),
    ("acc",    "Avg Reconstruction Accuracy",True),
]

for ax, (key, title, higher_better) in zip(axes, bar_metrics):
    style_ax(ax)
    vals = [np.mean(rnd[key]), np.mean(seq[key]), np.mean(rl[key])]
    errs = [np.std(rnd[key]),  np.std(seq[key]),  np.std(rl[key])]
    x = np.arange(3)

    bars = ax.bar(x, vals, color=COLORS, alpha=0.88,
                  yerr=errs, capsize=6,
                  error_kw=dict(ecolor=C_TXT, lw=1.5, capthick=1.5),
                  zorder=3)

    # Value labels
    for bar, v, e in zip(bars, vals, errs):
        ypos = v + e + abs(v) * 0.02 + 0.01
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{v:.3f}", ha="center", va="bottom",
                color=C_TXT, fontsize=8.5, fontweight="bold")

    arrow = "↑" if higher_better else "↓"
    ax.set_title(f"{title}\n({arrow} better)", pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=12, ha="right")
    ax.yaxis.grid(color=C_GRID, alpha=0.7)
    ax.set_axisbelow(True)

plt.tight_layout()
save(fig, "fig2_bar_comparison")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 3 — Efficiency: scatter + box plots
# ═══════════════════════════════════════════════════════════════════════════

print("[Fig 3] Efficiency and consistency plots...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(C_BG)
fig.suptitle("Efficiency and Consistency Analysis",
             color=C_TXT, fontsize=13, fontweight="bold", y=1.02)

# Left: Coverage vs Reward scatter
ax = axes[0]
style_ax(ax)
ax.scatter(rnd["reward"], rnd["cov"], color=C_RND, alpha=0.65, s=30,
           label="Random",     zorder=3, edgecolors="none")
ax.scatter(seq["reward"], seq["cov"], color=C_SEQ, alpha=0.65, s=30,
           label="Sequential", zorder=3, edgecolors="none", marker="s")
ax.scatter(rl["reward"],  rl["cov"],  color=C_RL,  alpha=0.80, s=45,
           label="RL (PPO)",  zorder=4, edgecolors="white", linewidths=0.4)

# Mean markers
for d, col, mk in [(rnd, C_RND, "o"), (seq, C_SEQ, "s"), (rl, C_RL, "D")]:
    ax.scatter(np.mean(d["reward"]), np.mean(d["cov"]),
               color=col, s=160, marker=mk, edgecolors="white",
               linewidths=1.5, zorder=5)

ax.set_xlabel("Cumulative Reward")
ax.set_ylabel("Voxel Coverage")
ax.set_title("Coverage vs Reward per Episode\n(large markers = mean)")
ax.legend(facecolor="#222", labelcolor=C_TXT, framealpha=0.85)

# Annotate RL cluster
ax.annotate("RL cluster\n(high reward\n+ high coverage)",
            xy=(np.mean(rl["reward"]), np.mean(rl["cov"])),
            xytext=(np.mean(rl["reward"]) - 1.5, np.mean(rl["cov"]) - 0.04),
            color=C_RL, fontsize=8,
            arrowprops=dict(arrowstyle="->", color=C_RL, lw=1.2))

# Right: Accuracy box plots
ax = axes[1]
style_ax(ax)
bp = ax.boxplot(
    [rnd["acc"], seq["acc"], rl["acc"]],
    tick_labels=LABELS,
    patch_artist=True,
    widths=0.5,
    medianprops=dict(color="white", lw=2.5),
    whiskerprops=dict(color="#aaaaaa", lw=1.2),
    capprops=dict(color="#aaaaaa", lw=1.2),
    flierprops=dict(markerfacecolor="#888", marker="o",
                    markersize=4, linestyle="none"),
    boxprops=dict(lw=1.5),
)
for patch, col in zip(bp["boxes"], COLORS):
    patch.set_facecolor(col)
    patch.set_alpha(0.82)

ax.set_ylabel("Reconstruction Accuracy")
ax.set_title("Accuracy Distribution (50 episodes)\n(↑ better)")
ax.yaxis.grid(color=C_GRID, alpha=0.7)
ax.set_axisbelow(True)

# Individual points (jittered)
for i, (d, col) in enumerate([(rnd, C_RND), (seq, C_SEQ), (rl, C_RL)], 1):
    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(d["acc"]))
    ax.scatter(np.full(len(d["acc"]), i) + jitter, d["acc"],
               color=col, alpha=0.4, s=15, zorder=3, edgecolors="none")

plt.tight_layout()
save(fig, "fig3_efficiency_consistency")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4 — Uncertainty reduction curve (mean per episode step proxy)
# ═══════════════════════════════════════════════════════════════════════════

print("[Fig 4] Uncertainty reduction over episodes...")
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(C_BG)
style_ax(ax)

ax.plot(eps_rl,  rolling(rl["uncert"],  7), color=C_RL,  lw=2.2, label="RL (PPO)")
ax.plot(eps_rnd, rolling(rnd["uncert"], 7), color=C_RND, lw=1.5, label="Random",     ls="--")
ax.plot(eps_seq, rolling(seq["uncert"], 7), color=C_SEQ, lw=1.5, label="Sequential", ls=":")

# Geometric ceiling annotation
ax.axhline(y=0.63, color="#888888", lw=1.0, ls="-.", alpha=0.7)
ax.text(len(eps_rl) * 0.02, 0.64,
        "~63% uncertainty ceiling\n(geometric self-occlusion limit)",
        color="#aaaaaa", fontsize=8.5, va="bottom")

ax.set_xlabel("Episode")
ax.set_ylabel("Mean Uncertainty (over occupied voxels)")
ax.set_title("Uncertainty Reduction Over Evaluation Episodes  (↓ better, 7-ep rolling avg)",
             pad=8)
ax.legend(handles=legend_handles(), facecolor="#222",
          labelcolor=C_TXT, framealpha=0.85)
ax.set_ylim(0.30, 0.75)

plt.tight_layout()
save(fig, "fig4_uncertainty_reduction")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 5 — Reward distribution violin plot
# ═══════════════════════════════════════════════════════════════════════════

print("[Fig 5] Reward distribution violins...")
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(C_BG)
style_ax(ax)

data_violin = [rnd["reward"], seq["reward"], rl["reward"]]
parts = ax.violinplot(data_violin, positions=[1, 2, 3],
                      showmeans=True, showmedians=True, showextrema=True)

for i, (pc, col) in enumerate(zip(parts["bodies"], COLORS)):
    pc.set_facecolor(col)
    pc.set_alpha(0.75)
    pc.set_edgecolor("white")

parts["cmeans"].set_color("white")
parts["cmeans"].set_linewidth(2)
parts["cmedians"].set_color("#dddddd")
parts["cbars"].set_color("#888888")
parts["cmins"].set_color("#888888")
parts["cmaxes"].set_color("#888888")

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(LABELS)
ax.set_ylabel("Cumulative Reward per Episode")
ax.set_title("Reward Distribution Across 50 Episodes\n(white line = mean, grey = median)",
             pad=8)

# Mean value labels
for i, d in enumerate([rnd, seq, rl], 1):
    ax.text(i, np.mean(d["reward"]) + 0.15,
            f"μ={np.mean(d['reward']):.3f}",
            ha="center", va="bottom", color=C_TXT, fontsize=9, fontweight="bold")

plt.tight_layout()
save(fig, "fig5_reward_violin")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 6 — Coverage vs Images captured (efficiency)
# ═══════════════════════════════════════════════════════════════════════════

print("[Fig 6] Coverage efficiency (coverage per image)...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(C_BG)
fig.suptitle("Observation Efficiency Analysis",
             color=C_TXT, fontsize=13, fontweight="bold", y=1.02)

# Left: coverage efficiency = coverage / image_count
ax = axes[0]
style_ax(ax)

for d, col, lbl, ls in [
    (rnd, C_RND, "Random",     "--"),
    (seq, C_SEQ, "Sequential", ":"),
    (rl,  C_RL,  "RL (PPO)",   "-"),
]:
    eff = [c / max(im, 1) for c, im in zip(d["cov"], d["images"])]
    ax.plot(d["episode"], rolling(eff, 5), color=col, lw=1.8 if lbl != "RL (PPO)" else 2.2,
            label=lbl, ls=ls)

ax.set_xlabel("Episode")
ax.set_ylabel("Coverage / Image Count")
ax.set_title("Coverage Efficiency\n(↑ better: more coverage per captured image)")
ax.legend(handles=legend_handles(), facecolor="#222", labelcolor=C_TXT, framealpha=0.85)

# Right: cumulative reward per image
ax = axes[1]
style_ax(ax)

for d, col, lbl, ls in [
    (rnd, C_RND, "Random",     "--"),
    (seq, C_SEQ, "Sequential", ":"),
    (rl,  C_RL,  "RL (PPO)",   "-"),
]:
    eff = [r / max(im, 1) for r, im in zip(d["reward"], d["images"])]
    ax.plot(d["episode"], rolling(eff, 5), color=col, lw=1.8 if lbl != "RL (PPO)" else 2.2,
            label=lbl, ls=ls)

ax.set_xlabel("Episode")
ax.set_ylabel("Reward / Image Count")
ax.set_title("Reward Efficiency\n(↑ better: more reward per image captured)")
ax.legend(handles=legend_handles(), facecolor="#222", labelcolor=C_TXT, framealpha=0.85)

plt.tight_layout()
save(fig, "fig6_observation_efficiency")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 7 — Statistical summary table figure
# ═══════════════════════════════════════════════════════════════════════════

print("[Fig 7] Summary statistics table...")
fig, ax = plt.subplots(figsize=(12, 4))
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)
ax.axis("off")

col_labels = ["Metric", "Random", "Sequential", "RL (PPO)", "RL vs Random", "RL vs Sequential"]
rows_data = []

for key, label, better in [
    ("reward", "Avg Reward",       "higher"),
    ("cov",    "Avg Coverage",     "higher"),
    ("uncert", "Avg Uncertainty",  "lower"),
    ("acc",    "Avg Accuracy",     "higher"),
]:
    rm = np.mean(rnd[key]); rs = np.std(rnd[key])
    sm = np.mean(seq[key]); ss = np.std(seq[key])
    lm = np.mean(rl[key]);  ls_ = np.std(rl[key])
    d_rnd = lm - rm
    d_seq = lm - sm
    sign  = "+" if d_rnd > 0 else ""
    rows_data.append([
        label,
        f"{rm:.4f} ±{rs:.4f}",
        f"{sm:.4f} ±{ss:.4f}",
        f"{lm:.4f} ±{ls_:.4f}",
        f"{sign}{d_rnd:.4f}",
        f"{'+' if d_seq>0 else ''}{d_seq:.4f}",
    ])

table = ax.table(
    cellText=rows_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1, 2.2)

# Style header
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#2a2a4a")
    table[0, j].set_text_props(color=C_TXT, fontweight="bold")

# Style data rows
row_colors = ["#1a1a2a", "#1e1e2e"]
for i in range(1, len(rows_data) + 1):
    for j in range(len(col_labels)):
        cell = table[i, j]
        cell.set_facecolor(row_colors[i % 2])
        # Colour RL column cyan
        txt_col = C_TXT
        if j == 3:  txt_col = C_RL
        if j == 4:
            v = float(rows_data[i-1][4])
            txt_col = C_RL if v > 0 else C_RND
        if j == 5:
            v = float(rows_data[i-1][5])
            txt_col = C_RL if v > 0 else C_RND
        cell.set_text_props(color=txt_col)

ax.set_title("Summary Statistics Table — 50 Evaluation Episodes",
             color=C_TXT, fontsize=12, pad=15, fontweight="bold")

plt.tight_layout()
save(fig, "fig7_summary_table")


# ═══════════════════════════════════════════════════════════════════════════
# PRINT CONSOLE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"{'Metric':<22} {'Random':>14} {'Sequential':>14} {'RL (PPO)':>14}")
print("-"*60)
for key, label in [("reward","Reward"),("cov","Coverage"),
                   ("uncert","Uncertainty"),("acc","Accuracy")]:
    rm = np.mean(rnd[key]); sm = np.mean(seq[key]); lm = np.mean(rl[key])
    print(f"{label:<22} {rm:>14.4f} {sm:>14.4f} {lm:>14.4f}")
print("-"*60)
print(f"{'RL vs Random (reward)':<22} {np.mean(rl['reward'])-np.mean(rnd['reward']):>+14.4f}")
print(f"{'RL vs Seq   (reward)':<22} {np.mean(rl['reward'])-np.mean(seq['reward']):>+14.4f}")
print(f"{'RL vs Random (acc)':<22} {np.mean(rl['acc'])-np.mean(rnd['acc']):>+14.4f}")
print(f"{'RL vs Seq   (acc)':<22} {np.mean(rl['acc'])-np.mean(seq['acc']):>+14.4f}")
print("="*60)
print(f"\nAll figures saved to:\n  {OUTPUT_DIR}")
