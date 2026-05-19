"""
Figure: Portability by Industry Group.

Categorizes 525 Census 2018 occupations into ~9 broad industry/sector groups
and shows horizontal box plots (with individual occupation dots) of LASSO
portability rank, ordered by group median.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path(__file__).parent.parent
DATA_DIR = PROJECT / "data"
OUTPUT_DIR = PROJECT / "output"
FIGURES_DIR = PROJECT / "figures"

# ── load portability indices ────────────────────────────────────────
lasso = pd.read_csv(OUTPUT_DIR / "portability_index_fixed_delta1.csv")
factor = pd.read_csv(OUTPUT_DIR / "portability_index_fixed_delta1_factor.csv")

lasso = lasso.rename(columns={"rank": "rank_lasso"})
factor = factor.rename(columns={"rank": "rank_factor"})

df = lasso[["occ", "rank_lasso", "occ_title"]].merge(
    factor[["occ", "rank_factor"]], on="occ", how="inner"
)
df["occ_int"] = df["occ"].astype(int)


# ── map Census 2018 codes to industry groups ────────────────────────
def assign_industry_group(code):
    """Assign a Census 2018 occupation code to a broad industry group."""

    # Tech: Computer and Mathematical (1005–1240)
    if 1005 <= code <= 1240:
        return "Tech"

    # STEM: Architecture/Engineering (1300–1560) + Life/Physical/Social Science (1600–1980)
    if 1300 <= code <= 1560:
        return "STEM"
    if 1600 <= code <= 1980:
        return "STEM"

    # Healthcare: Practitioners (3000–3550) + Support (3600–3655)
    if 3000 <= code <= 3550:
        return "Healthcare"
    if 3600 <= code <= 3655:
        return "Healthcare"

    # Education: Education, Training, Library (2200–2555)
    if 2200 <= code <= 2555:
        return "Education"

    # Skilled Trades: Construction/Extraction (6200–6950) + Install/Repair (7000–7640)
    if 6200 <= code <= 6950:
        return "Skilled Trades"
    if 7000 <= code <= 7640:
        return "Skilled Trades"

    # Service: Food Prep (3960–4160) + Cleaning (4200–4255) +
    #          Personal Care (4330–4655) + Protective Service (3700–3960)
    if 3700 <= code <= 3960:
        return "Service"
    if 3961 <= code <= 4160:
        return "Service"
    if 4200 <= code <= 4255:
        return "Service"
    if 4330 <= code <= 4655:
        return "Service"

    # Business/Office: Management (10–440) + Business/Finance (500–960) +
    #                  Sales (4700–4965) + Office/Admin (5000–5940) + Legal (2100–2180)
    if 10 <= code <= 440:
        return "Business/Office"
    if 500 <= code <= 960:
        return "Business/Office"
    if 2100 <= code <= 2180:
        return "Business/Office"
    if 4700 <= code <= 4965:
        return "Business/Office"
    if 5000 <= code <= 5940:
        return "Business/Office"

    # Production/Transport: Production (7700–8990) + Transportation (9000–9760) +
    #                       Farming (6005–6130)
    if 6005 <= code <= 6130:
        return "Production/Transport"
    if 7700 <= code <= 8990:
        return "Production/Transport"
    if 9000 <= code <= 9760:
        return "Production/Transport"

    # Arts/Social: Arts/Design/Entertainment (2600–2970) +
    #              Community/Social Service (2000–2060)
    if 2000 <= code <= 2060:
        return "Arts/Social"
    if 2600 <= code <= 2970:
        return "Arts/Social"

    return "Other"


df["industry"] = df["occ_int"].apply(assign_industry_group)

# Drop any "Other" (should be very few, if any)
n_other = (df["industry"] == "Other").sum()
if n_other > 0:
    print(f"Warning: {n_other} occupations not classified — dropping from plot")
    df = df[df["industry"] != "Other"].copy()

# ── compute group stats and order by median LASSO rank ──────────────
group_stats = (
    df.groupby("industry")["rank_lasso"]
    .agg(["median", "count"])
    .sort_values("median")
)
group_order = group_stats.index.tolist()  # most portable (low rank) first

print("\nIndustry group sizes and median LASSO rank:")
for grp in group_order:
    med = group_stats.loc[grp, "median"]
    cnt = int(group_stats.loc[grp, "count"])
    print(f"  {grp:<22s}  n={cnt:>3d}  median rank={med:.0f}")

# ── build figure ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

# Prepare data for each group in order
group_data = [df.loc[df["industry"] == grp, "rank_lasso"].values for grp in group_order]
y_positions = np.arange(len(group_order))

# Box plots
bp = ax.boxplot(
    group_data,
    positions=y_positions,
    vert=False,
    widths=0.5,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color="black", linewidth=1.5),
    boxprops=dict(facecolor="#D4E6F1", edgecolor="#2C3E50", linewidth=0.8),
    whiskerprops=dict(color="#2C3E50", linewidth=0.8),
    capprops=dict(color="#2C3E50", linewidth=0.8),
)

# Overlay individual occupation dots (jittered vertically)
rng = np.random.default_rng(42)
for i, grp in enumerate(group_order):
    vals = df.loc[df["industry"] == grp, "rank_lasso"].values
    jitter = rng.uniform(-0.18, 0.18, size=len(vals))
    ax.scatter(
        vals,
        np.full_like(vals, i, dtype=float) + jitter,
        s=10,
        alpha=0.45,
        color="#2874A6",
        edgecolors="none",
        zorder=3,
    )

# Labels
labels = [f"{grp}  (n={int(group_stats.loc[grp, 'count'])})" for grp in group_order]
ax.set_yticks(y_positions)
ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlabel("LASSO portability rank (1 = most portable)", fontsize=11)
ax.set_title("Portability rank by industry group", fontsize=13, fontweight="bold")
ax.set_xlim(0, 540)
ax.invert_yaxis()  # most portable group at top
ax.tick_params(axis="x", labelsize=9)

# Light grid on x-axis
ax.grid(axis="x", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_portability_by_industry.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGURES_DIR / "fig_portability_by_industry.pdf", bbox_inches="tight")
print("\nSaved to figures/fig_portability_by_industry.png and .pdf")
plt.close()
