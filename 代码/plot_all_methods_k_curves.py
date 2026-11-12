# -*- coding: utf-8 -*-
"""
All-method K curves, one panel per metric.

Data: 数据/multiseed_results.json, rec.<method>, 5 seeds.
Outputs:
  - 数据/all_methods_k_curves.csv
  - 图片/all_methods_k_curves.pdf / .png   (2x2 compound)
  - 图片/kcurve_<metric>.pdf / .png        (per-metric single panels)
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "数据")
OUT = os.path.join(ROOT, "图片")
os.makedirs(OUT, exist_ok=True)

METHODS = [
    ("unified_dc", "DA (ours)", "#2F5D8A", "o", 2.4, 5.2),
    ("han", "HAN", "#8C6D9E", "P", 1.15, 3.6),
    ("gin", "GIN", "#B08A5A", "D", 1.15, 3.4),
    ("gat", "GAT", "#6E8FAE", "^", 1.15, 3.6),
    ("film", "FiLM", "#B87C6E", "X", 1.15, 3.6),
    ("gatv2", "GATv2", "#9AB3C9", "v", 1.15, 3.6),
    ("graphsage", "GraphSAGE", "#8A96A3", "s", 1.15, 3.4),
    ("graphconv", "GraphConv", "#6F7B7D", "*", 1.15, 4.0),
]
SEEDS = ["42", "1", "2", "3", "4"]
KS = [5, 10, 20, 50, 100]
METRICS = [
    ("MRR", "MRR@K", "(a)", (0.010, 0.045), [0.01, 0.02, 0.03, 0.04], "{:.2f}"),
    ("NDCG", "NDCG@K", "(b)", (0.46, 0.70), [0.50, 0.55, 0.60, 0.65, 0.70], "{:.2f}"),
    ("HitRate", "HitRate@K", "(c)", (0.0, 0.56), [0.0, 0.2, 0.4, 0.6], "{:.2f}"),
    ("Precision", "Precision@K", "(d)", (0.005, 0.017), [0.006, 0.009, 0.012, 0.015], "{:.3f}"),
]

d = json.load(open(os.path.join(DATA, "multiseed_results.json"), encoding="utf-8"))
rec = d["rec"]

rows = []
for mk, name, _, _, _, _ in METHODS:
    for metric, _, _, _, _, _ in METRICS:
        for k in KS:
            vals = np.array([rec[mk][s][f"{metric}@{k}"] for s in SEEDS], dtype=float)
            rows.append(dict(metric=metric, method=mk, method_name=name, K=k,
                             mean=vals.mean(), std=vals.std(ddof=0)))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(DATA, "all_methods_k_curves.csv"), index=False, encoding="utf-8-sig")

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 8,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#C9D2DB",
    "xtick.color": "#4A5563",
    "ytick.color": "#4A5563",
})


def draw_panel(ax, metric, title, letter, ylim, yticks, fmt, show_legend=False):
    for mk, name, color, marker, lw, ms in METHODS:
        g = df[(df.metric == metric) & (df.method == mk)].sort_values("K")
        ax.plot(g["K"], g["mean"], color=color, marker=marker, markersize=ms,
                linewidth=lw, label=name, zorder=10 if mk == "unified_dc" else 3,
                markerfacecolor="white" if mk == "unified_dc" else color,
                markeredgecolor=color, markeredgewidth=0.9)
    ax.set_title(f"{letter} {title}", loc="left", fontsize=9.2, fontweight="bold", pad=6)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels([fmt.format(y) for y in yticks], fontsize=7.2)
    ax.set_xlim(2, 108)
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS], fontsize=7.4)
    ax.grid(axis="y", color="#E8EDF2", linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="both", length=0, pad=4)


# ---------------- compound 2x2 ----------------
fig, axes = plt.subplots(2, 2, figsize=(7.9, 5.9), sharex=True, dpi=300)
fig.patch.set_facecolor("white")
for ax, (metric, title, letter, ylim, yticks, fmt) in zip(axes.flat, METRICS):
    draw_panel(ax, metric, title, letter, ylim, yticks, fmt)

handles, labels = axes.flat[0].get_legend_handles_labels()
leg = fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.008),
                 frameon=False, fontsize=7.1, handlelength=2.0, columnspacing=1.2,
                 handletextpad=0.5)
leg.get_texts()[0].set_fontweight("bold")

fig.text(0.5, 0.115, "K  (number of recommended crew)", fontsize=8.0, color="#1F2933", ha="center")
plt.subplots_adjust(left=0.085, right=0.985, top=0.95, bottom=0.21, hspace=0.34, wspace=0.24)
fig.savefig(os.path.join(OUT, "all_methods_k_curves.pdf"), bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(OUT, "all_methods_k_curves.png"), dpi=300, bbox_inches="tight", facecolor="white")

# ---------------- per-metric single panels ----------------
for metric, title, letter, ylim, yticks, fmt in METRICS:
    fig1, ax1 = plt.subplots(figsize=(6.6, 3.9), dpi=300)
    fig1.patch.set_facecolor("white")
    draw_panel(ax1, metric, title, letter, ylim, yticks, fmt)
    handles, labels = ax1.get_legend_handles_labels()
    leg1 = ax1.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.40),
                      ncol=4, frameon=False, fontsize=6.8, handlelength=2.0,
                      columnspacing=1.0, handletextpad=0.4)
    leg1.get_texts()[0].set_fontweight("bold")
    ax1.set_xlabel("K  (number of recommended crew)", fontsize=7.8, labelpad=6)
    fig1.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.30)
    fig1.savefig(os.path.join(OUT, f"kcurve_{metric}.pdf"), bbox_inches="tight", facecolor="white")
    fig1.savefig(os.path.join(OUT, f"kcurve_{metric}.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig1)

print(df.groupby(["metric", "method_name"])["mean"].agg(["first", "last"]).to_string())
print("saved", os.path.join(OUT, "all_methods_k_curves.pdf"))
