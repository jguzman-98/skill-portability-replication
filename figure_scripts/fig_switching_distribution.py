"""
Figure: Switching count distribution.

Two-panel figure showing (a) the full distribution including the 95% zero mass,
and (b) the conditional distribution among nonzero pairs (heavy right tail).
Justifies PPML over OLS at a glance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path(__file__).parent.parent
DATA = PROJECT / "data"
OUTPUT = PROJECT / "output"
FIGURES_DIR = PROJECT / "figures"

# ── load ────────────────────────────────────────────────────────────
pw = pd.read_csv(DATA / "pairwise_dataset.csv", usecols=["switches"])
s = pw["switches"]

n_total = len(s)
n_zero = (s == 0).sum()
n_nonzero = (s > 0).sum()
pct_zero = n_zero / n_total * 100

print(f"Total: {n_total:,}, zeros: {n_zero:,} ({pct_zero:.1f}%), nonzero: {n_nonzero:,}")

nz = s[s > 0]

# ── build figure ────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                gridspec_kw={"width_ratios": [1, 1.3]})

# --- Panel A: zero vs nonzero ---
bars = ax1.bar(
    ["Zero\n(no switches)", "Nonzero\n(≥1 switch)"],
    [n_zero, n_nonzero],
    color=["#A8C4E0", "#2874A6"],
    edgecolor="white",
    width=0.55,
)

# Annotate counts and percentages
ax1.text(0, n_zero + 3000, f"{n_zero:,}\n({pct_zero:.1f}%)",
         ha="center", va="bottom", fontsize=10, fontweight="bold", color="#333")
ax1.text(1, n_nonzero + 3000, f"{n_nonzero:,}\n({100-pct_zero:.1f}%)",
         ha="center", va="bottom", fontsize=10, fontweight="bold", color="#333")

ax1.set_ylabel("Number of directed pairs", fontsize=10.5)
ax1.set_title("(a) Extensive margin: 95% of pairs\nhave zero observed switches",
              fontsize=11, fontweight="bold", pad=10)
ax1.set_ylim(0, n_zero * 1.12)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.tick_params(labelsize=9.5)
ax1.yaxis.grid(True, linestyle="--", alpha=0.3)
ax1.set_axisbelow(True)

# --- Panel B: distribution of nonzero counts (log-scale x) ---
# Use custom bins on log scale for the histogram
bins = [0.5, 1.5, 2.5, 3.5, 5.5, 10.5, 25.5, 50.5, 100.5, 250.5]
counts_hist, edges = np.histogram(nz, bins=bins)

# Bar positions: use midpoints on a log-ish scale for readability
labels = ["1", "2", "3", "4–5", "6–10", "11–25", "26–50", "51–100", "101+"]
x_pos = np.arange(len(labels))

ax2.bar(x_pos, counts_hist, color="#2874A6", edgecolor="white", width=0.7, alpha=0.85)

# Annotate each bar
for i, c in enumerate(counts_hist):
    if c > 0:
        ax2.text(i, c + 80, f"{c:,}", ha="center", va="bottom", fontsize=7.5, color="#333")

ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_xlabel("Switches per pair", fontsize=10.5)
ax2.set_ylabel("Number of pairs", fontsize=10.5)
ax2.set_title(f"(b) Intensive margin: conditional on switching\n"
              f"(median = {nz.median():.0f}, mean = {nz.mean():.1f}, max = {nz.max()})",
              fontsize=11, fontweight="bold", pad=10)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.tick_params(labelsize=9)
ax2.yaxis.grid(True, linestyle="--", alpha=0.3)
ax2.set_axisbelow(True)

fig.tight_layout(w_pad=3)
fig.savefig(FIGURES_DIR / "fig_switching_distribution.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGURES_DIR / "fig_switching_distribution.pdf", bbox_inches="tight")
print("\nSaved to figures/fig_switching_distribution.png and .pdf")
plt.close()
