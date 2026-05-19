"""
Figure: LASSO coefficient plot.

Re-fits LassoCV on the pairwise diff_* features (same setup as 05_estimate_models.py)
and plots the top 40 nonzero coefficients as a horizontal dot plot, grouped and
colored by O*NET category (Skill, Ability, Knowledge, Work Activity).

Signed coefficients are shown: negative means larger skill gap → fewer switches
(barrier to mobility); positive means larger gap → more switches.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from pathlib import Path

PROJECT = Path(__file__).parent.parent
DATA_DIR = PROJECT / "data"
FIGURES_DIR = PROJECT / "figures"

# ── load pairwise dataset and re-fit LASSO ──────────────────────────
print("Loading pairwise dataset...")
pw = pd.read_csv(DATA_DIR / "pairwise_dataset.csv")

diff_cols = [c for c in pw.columns if c.startswith("diff_")]
X = pw[diff_cols].values
y = pw["switches"].values

print(f"  {len(diff_cols)} features, {len(y):,} pairs")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Fitting LassoCV (5-fold)...")
lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
lasso.fit(X_scaled, y)

nonzero = np.sum(lasso.coef_ != 0)
print(f"  alpha={lasso.alpha_:.6f}, nonzero: {nonzero}/{len(lasso.coef_)}")

# ── build coefficient dataframe ─────────────────────────────────────
coef_df = pd.DataFrame({
    "feature": diff_cols,
    "coef": lasso.coef_,
    "abs_coef": np.abs(lasso.coef_),
})
coef_df = coef_df[coef_df["coef"] != 0].copy()

# Assign O*NET category from feature prefix
def get_category(feat):
    name = feat.replace("diff_", "")
    if name.startswith("skill_"):
        return "Skill"
    elif name.startswith("ability_"):
        return "Ability"
    elif name.startswith("knowledge_"):
        return "Knowledge"
    elif name.startswith("activity_"):
        return "Work Activity"
    return "Other"

coef_df["category"] = coef_df["feature"].apply(get_category)

# Clean up feature names for display
def clean_name(feat):
    name = feat.replace("diff_", "")
    # Strip category prefix
    for prefix in ["skill_", "ability_", "knowledge_",
                    "activity_lv_", "activity_im_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    # Title case, replace underscores
    name = name.replace("_", " ").title()
    # Add scale suffix for activities
    if "activity_lv_" in feat:
        name += " (level)"
    elif "activity_im_" in feat:
        name += " (imp.)"
    return name

coef_df["label"] = coef_df["feature"].apply(clean_name)

# ── select top 40 by absolute value ─────────────────────────────────
top = coef_df.nlargest(40, "abs_coef").sort_values("coef", ascending=True).reset_index(drop=True)

print(f"\nTop 40 by |coef| — category breakdown:")
print(top["category"].value_counts().to_string())

# ── build figure ────────────────────────────────────────────────────
category_colors = {
    "Skill": "#2874A6",
    "Ability": "#D4AC0D",
    "Knowledge": "#28B463",
    "Work Activity": "#CB4335",
}

fig, ax = plt.subplots(figsize=(8, 11))

y_pos = np.arange(len(top))
colors = [category_colors[c] for c in top["category"]]

ax.barh(y_pos, top["coef"], height=0.7, color=colors, edgecolor="white",
        linewidth=0.4, alpha=0.85)

# Zero line
ax.axvline(0, color="black", linewidth=0.8, zorder=0)

# Labels
ax.set_yticks(y_pos)
ax.set_yticklabels(top["label"], fontsize=8)
ax.set_xlabel("LASSO Coefficient (standardized)", fontsize=10)
ax.set_title(
    f"Top 40 LASSO Coefficients by Magnitude ({nonzero} nonzero of {len(diff_cols)})",
    fontsize=12, fontweight="bold", pad=12,
)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=v, label=k, alpha=0.85)
                   for k, v in category_colors.items()
                   if k in top["category"].values]
ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9)

# Styling
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.grid(True, linestyle="--", alpha=0.3)
ax.set_axisbelow(True)
ax.invert_yaxis()

fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_lasso_coefficients.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGURES_DIR / "fig_lasso_coefficients.pdf", bbox_inches="tight")
print("\nSaved to figures/fig_lasso_coefficients.png and .pdf")
plt.close()
