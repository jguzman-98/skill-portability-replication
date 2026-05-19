"""
Generate a radar/spider chart comparing skill profiles of three occupations:
Cashiers, Retail salespersons, and Petroleum engineers.

Uses the top 10 LASSO features from feature_importance.csv as dimensions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).parent.parent
DATA_DIR = PROJECT / "data"
OUTPUT_DIR = PROJECT / "output"
FIGURES_DIR = PROJECT / "figures"

SKILL_VECTORS_PATH = DATA_DIR / "skill_vectors_by_census2018.csv"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"
PORTABILITY_PATH = OUTPUT_DIR / "portability_by_occupation.csv"

# ── Target occupations ─────────────────────────────────────────────────────────
TARGET_TITLES = ["Cashiers", "Retail salespersons", "Petroleum engineers"]

# ── Step 1: Load portability file and find census codes ────────────────────────
port_df = pd.read_csv(PORTABILITY_PATH)

occ_lookup = {}
for title in TARGET_TITLES:
    match = port_df[port_df["occ_title"].str.contains(title, case=False, na=False)]
    if match.empty:
        raise ValueError(f"Could not find occupation matching '{title}'")
    occ_lookup[title] = int(match.iloc[0]["occ"])

print("Occupation census codes:")
for title, code in occ_lookup.items():
    print(f"  {title}: {code}")

# ── Step 2: Load feature importance, filter to LASSO top 10 ───────────────────
feat_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
feat_lasso = (
    feat_df[feat_df["model"] == "lasso"]
    .sort_values("importance", ascending=False)
    .head(10)
)

# Strip "diff_" prefix to get skill dimension column names
top_features = feat_lasso["feature"].str.replace("^diff_", "", regex=True).tolist()
print(f"\nTop 10 LASSO features:")
for i, f in enumerate(top_features, 1):
    print(f"  {i}. {f}")

# ── Step 3: Load skill vectors, filter to 3 occupations and top 10 dims ───────
skill_df = pd.read_csv(SKILL_VECTORS_PATH)
codes = list(occ_lookup.values())
skill_sub = skill_df[skill_df["census_code"].isin(codes)].copy()

if len(skill_sub) != 3:
    raise ValueError(
        f"Expected 3 rows, got {len(skill_sub)}. "
        f"Codes found: {skill_sub['census_code'].tolist()}"
    )

# Verify all top features exist as columns
missing = [f for f in top_features if f not in skill_sub.columns]
if missing:
    raise ValueError(f"Missing columns in skill_vectors: {missing}")

skill_sub = skill_sub.set_index("census_code")[top_features]

# ── Step 4: Build radar chart ─────────────────────────────────────────────────

def clean_label(col_name, max_len=22):
    """Convert column name to a cleaned-up label for the chart."""
    # Remove category prefixes (skill_, ability_, knowledge_, activity_im_, activity_lv_)
    label = col_name
    for prefix in [
        "activity_lv_", "activity_im_", "skill_", "ability_", "knowledge_"
    ]:
        if label.startswith(prefix):
            label = label[len(prefix):]
            break
    # Replace underscores with spaces, title case
    label = label.replace("_", " ").title()
    # Truncate if too long
    if len(label) > max_len:
        label = label[:max_len - 1] + "."
    return label

labels = [clean_label(f) for f in top_features]
N = len(labels)

# Compute angles for each axis
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close the polygon

# Colorblind-friendly palette (Wong, 2011 — Nature Methods)
colors = [
    "#0072B2",  # blue  — Cashiers
    "#D55E00",  # vermillion — Retail salespersons
    "#009E73",  # bluish green — Petroleum engineers
]

# Style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.linewidth": 0.5,
    "figure.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for idx, (title, code) in enumerate(occ_lookup.items()):
    values = skill_sub.loc[code, top_features].values.tolist()
    values += values[:1]  # close polygon
    ax.plot(
        angles,
        values,
        linewidth=2,
        linestyle="-",
        label=title,
        color=colors[idx],
    )
    ax.fill(angles, values, alpha=0.12, color=colors[idx])

# Configure axes
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=8.5)

# Adjust label alignment for readability
for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
    if angle in (0, np.pi):
        label.set_horizontalalignment("center")
    elif 0 < angle < np.pi:
        label.set_horizontalalignment("left")
    else:
        label.set_horizontalalignment("right")

# Radial grid
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=7, color="grey")
ax.set_rlabel_position(30)

# Grid styling
ax.grid(color="grey", linewidth=0.3, linestyle="-", alpha=0.5)
ax.spines["polar"].set_visible(False)

# Title
ax.set_title(
    "Skill Profiles: Top 10 Portability-Predictive Dimensions",
    fontsize=13,
    fontweight="bold",
    pad=25,
)

# Legend
legend = ax.legend(
    loc="lower right",
    bbox_to_anchor=(1.25, -0.05),
    fontsize=9,
    frameon=True,
    framealpha=0.9,
    edgecolor="grey",
)

plt.tight_layout()

# ── Step 5: Save ──────────────────────────────────────────────────────────────
out_png = FIGURES_DIR / "fig_skill_radar.png"
out_pdf = FIGURES_DIR / "fig_skill_radar.pdf"

fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")

print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")
plt.close(fig)
