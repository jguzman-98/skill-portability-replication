"""
Generate a histogram of raw portability scores across all 525 occupations.
Annotates notable occupations and marks mean/median with vertical dashed lines.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).parent.parent
DATA_DIR = PROJECT / "data"
OUTPUT_DIR = PROJECT / "output"
FIGURES_DIR = PROJECT / "figures"

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(OUTPUT_DIR / "portability_by_occupation.csv")
scores = df["portability"]

print(f"Loaded {len(df)} occupations")
print(f"  Mean portability:   {scores.mean():.4f}")
print(f"  Median portability: {scores.median():.4f}")
print(f"  Min:  {scores.min():.6f}")
print(f"  Max:  {scores.max():.4f}")

# ── Notable occupations to annotate ──────────────────────────────────────────
# Each entry: search string, (x_text, y_text) in data coords for the label,
# with the arrow pointing from the label to the actual data x-value on the x-axis.
annotations = [
    {"label": "Petroleum\nengineers",  "search": "Petroleum engineers",           "x_text": 0.8,   "y_frac": 0.50},
    {"label": "Software\ndevelopers",  "search": "Software developers",           "x_text": 1.8,   "y_frac": 0.75},
    {"label": "Registered\nnurses",    "search": "Registered nurses",             "x_text": 2.8,   "y_frac": 0.55},
    {"label": "Receptionists",         "search": "Receptionists and information", "x_text": 5.2,   "y_frac": 0.55},
    {"label": "Cashiers",              "search": "Cashiers",                      "x_text": 8.0,   "y_frac": 0.55},
]

# ── Plot ─────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(11, 5.5))

# Histogram
n_bins = 35
counts, bin_edges, patches = ax.hist(
    scores,
    bins=n_bins,
    color="#6baed6",
    edgecolor="white",
    linewidth=0.6,
    alpha=0.85,
    zorder=2,
)

# Mean and median lines
mean_val = scores.mean()
median_val = scores.median()

ax.axvline(mean_val, color="#d62728", linestyle="--", linewidth=1.3, zorder=3,
           label=f"Mean = {mean_val:.2f}")
ax.axvline(median_val, color="#2ca02c", linestyle="--", linewidth=1.3, zorder=3,
           label=f"Median = {median_val:.2f}")

# ── Annotate notable occupations ────────────────────────────────────────────
y_max = counts.max()

for ann in annotations:
    row = df[df["occ_title"].str.contains(ann["search"], case=False, na=False)]
    if row.empty:
        print(f"  WARNING: could not find '{ann['search']}' in occ_title")
        continue
    x_val = row["portability"].values[0]
    x_text = ann["x_text"]
    y_text = y_max * ann["y_frac"]

    ax.annotate(
        ann["label"],
        xy=(x_val, 0),
        xytext=(x_text, y_text),
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="bottom",
        arrowprops=dict(
            arrowstyle="-|>",
            color="#333333",
            lw=1.0,
            connectionstyle="arc3,rad=0.1",
        ),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#aaaaaa", alpha=0.9),
        zorder=5,
    )

# ── Labels and formatting ───────────────────────────────────────────────────
ax.set_xlabel("Raw portability score (employment-weighted predicted switching rate)",
              fontsize=11)
ax.set_ylabel("Number of occupations", fontsize=11)
ax.set_title("Distribution of Portability Scores (N = 525)", fontsize=13,
             fontweight="bold")

ax.legend(loc="center right", fontsize=9, framealpha=0.9,
          bbox_to_anchor=(0.98, 0.40))
ax.set_xlim(left=-0.15)
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Remove top/right spines for cleaner look
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

# ── Save ─────────────────────────────────────────────────────────────────────
fig.savefig(FIGURES_DIR / "fig_portability_distribution.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGURES_DIR / "fig_portability_distribution.pdf", bbox_inches="tight")
print("\nSaved to figures/fig_portability_distribution.png and .pdf")
plt.close()
