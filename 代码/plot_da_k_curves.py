# -*- coding: utf-8 -*-
"""
DA (UnifiedDC) recommendation metrics as a function of K.

Data: 数据/multiseed_results.json, rec.unified_dc, 5 seeds.
Outputs:
  - 数据/da_k_curves.csv
  - 图片/da_k_curves.pdf / .png
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "数据")
OUT = os.path.join(ROOT, "图片")
os.makedirs(OUT, exist_ok=True)

NAVY = "#2F5D8A"
INK = "#1F2933"
MID = "#8A96A3"
GRID = "#E8EDF2"
BAND = "#2F5D8A"

SEEDS = ["42", "1", "2", "3", "4"]
KS = [5, 10, 20, 50, 100]
METRICS = [
    ("MRR", "MRR@K", "(a)"),
    ("NDCG", "NDCG@K", "(b)"),
    ("HitRate", "HitRate@K", "(c)"),
    ("Precision", "Precision@K", "(d)"),
]

d = json.load(open(os.path.join(DATA, "multiseed_results.json"), encoding="utf-8"))
rec = d["rec"]["unified_dc"]

rows = []
for k in KS:
    row = {"K": k}
    for key, _, _ in METRICS:
        vals = np.array([rec[s][f"{key}@{k}"] for s in SEEDS], dtype=float)
        row[f"{key}_mean"] = vals.mean()
        row[f"{key}_std"] = vals.std(ddof=0)
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(DATA, "da_k_curves.csv"), index=False, encoding="utf-8-sig")

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 8,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#C9D2DB",
    "xtick.color": "#4A5563",
    "ytick.color": "#4A5563",
    "mathtext.fontset": "stix",
})

fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.25), sharex=True, dpi=300)
fig.patch.set_facecolor("white")

ylims = {
    "MRR": (0.012, 0.056),
    "NDCG": (0.575, 0.735),
    "HitRate": (0.0, 0.65),
    "Precision": (0.0065, 0.0215),
}
ytickfmt = {
    "MRR": lambda x, _: f"{x:.2f}",
    "NDCG": lambda x, _: f"{x:.2f}",
    "HitRate": lambda x, _: f"{x:.2f}",
    "Precision": lambda x, _: f"{x:.3f}",
}
label_fmt = {"MRR": "{:.3f}", "NDCG": "{:.3f}", "HitRate": "{:.3f}", "Precision": "{:.4f}"}

for ax, (key, title, letter) in zip(axes.flat, METRICS):
    mean = df[f"{key}_mean"].values
    std = df[f"{key}_std"].values

    ax.fill_between(KS, mean - std, mean + std, color=BAND, alpha=0.12, linewidth=0)
    ax.plot(KS, mean, color=NAVY, linewidth=1.6, marker="o", markersize=4.2,
            markerfacecolor="white", markeredgecolor=NAVY, markeredgewidth=1.1, zorder=3)

    ax.axvline(10, color="#C9D2DB", linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)

    for x, y in zip(KS, mean):
        if x == 100:
            ax.annotate(label_fmt[key].format(y), (x, y), textcoords="offset points",
                        xytext=(-2, 7), ha="center", va="bottom", fontsize=6.3, color=INK)

    # ratio K=100 / K=5 in the lower-right corner
    ratio = mean[-1] / mean[0]
    ax.text(0.985, 0.045, f"$\\times${ratio:.2f}  (K=100 vs 5)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2, color=MID)

    ax.set_title(f"{letter} {title}", loc="left", fontsize=9.2, fontweight="bold",
                 color=INK, pad=6)
    ax.set_ylim(*ylims[key])
    ax.yaxis.set_major_formatter(FuncFormatter(ytickfmt[key]))
    ax.set_xlim(2, 108)
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS], fontsize=7.6)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="both", length=0, pad=4)

fig.text(0.012, 0.975, "DA (UnifiedDC)", fontsize=9.2, fontweight="bold", color=NAVY, ha="left")
fig.text(0.145, 0.975, "·  5-seed mean ± std", fontsize=7.6, color=MID, ha="left")
fig.text(0.5, 0.055, "K  (number of recommended crew)", fontsize=8.2, color=INK, ha="center")
fig.text(0.5, 0.018,
         "dashed line: K = 10 (reported setting)   ·   band: ±1 std over 5 seeds",
         fontsize=6.6, color=MID, ha="center")

plt.subplots_adjust(left=0.095, right=0.985, top=0.90, bottom=0.135, hspace=0.34, wspace=0.24)
fig.savefig(os.path.join(OUT, "da_k_curves.pdf"), bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(OUT, "da_k_curves.png"), dpi=300, bbox_inches="tight", facecolor="white")
print(df.to_string(index=False))
print("saved", os.path.join(OUT, "da_k_curves.pdf"))
