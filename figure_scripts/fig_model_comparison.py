"""
Figure: Baseline model fit comparison (grouped bar chart).

Grouped bars showing R² across all 6 skill distance variants × 2 estimators
(OLS and PPML), using the baseline specification from model_comparison.csv.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).parent.parent
DATA_DIR = PROJECT / "data"
OUTPUT_DIR = PROJECT / "output"
FIGURES_DIR = PROJECT / "figures"

# ── load and filter to baseline specification ───────────────────────
df = pd.read_csv(OUTPUT_DIR / "model_comparison.csv")
df = df[df["specification"] == "baseline"].copy()

# ── readable labels, ordered by PPML R² (descending) ───────────────
label_map = {
    "ml_lasso": "LASSO",
    "ml_random_forest": "Random Forest",
    "factor_analysis": "Factor Analysis",
    "euclidean": "Euclidean",
    "ml_xgboost": "XGBoost",
    "angular_separation": "Angular Separation",
}

# Get PPML R² to sort variants
ppml = df[df["estimator"] == "PPML"].set_index("skill_distance")["R2"]
variant_order = ppml.sort_values(ascending=False).index.tolist()

labels = [label_map[v] for v in variant_order]
ols_r2 = [df[(df["skill_distance"] == v) & (df["estimator"] == "OLS_log")]["R2"].values[0]
           for v in variant_order]
ppml_r2 = [df[(df["skill_distance"] == v) & (df["estimator"] == "PPML")]["R2"].values[0]
            for v in variant_order]

# ── build figure ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))

x = np.arange(len(labels))
width = 0.35

bars_ols = ax.bar(x - width / 2, ols_r2, width, label="OLS (log)",
                  color="#A8C4E0", edgecolor="white", linewidth=0.6)
bars_ppml = ax.bar(x + width / 2, ppml_r2, width, label="PPML",
                   color="#2874A6", edgecolor="white", linewidth=0.6)

# Value labels on top of each bar
for bar in bars_ols:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
            f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, color="#555555")

for bar in bars_ppml:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
            f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, color="#1a1a1a",
            fontweight="medium")

# Axes
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("$R^2$", fontsize=11)
ax.set_title("Baseline Model Fit: $R^2$ by Skill Distance Variant and Estimator",
             fontsize=12, fontweight="bold", pad=12)
ax.set_ylim(0, 0.56)
ax.legend(fontsize=9.5, loc="upper right", framealpha=0.9)

# Light horizontal grid
ax.yaxis.grid(True, linestyle="--", alpha=0.3)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_model_comparison.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGURES_DIR / "fig_model_comparison.pdf", bbox_inches="tight")
print("Saved to figures/fig_model_comparison.png and .pdf")
plt.close()
