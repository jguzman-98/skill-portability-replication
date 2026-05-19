"""
Visualize the geographic concentration of 3 occupations across US states.

Creates a 1x3 panel of horizontal bar charts showing the top 15 states by
employment share for Cashiers, Registered Nurses, and Petroleum Engineers.
Illustrates how some occupations (cashiers, nurses) are geographically dispersed
while others (petroleum engineers) are concentrated in a few states -- the
pattern captured by the Duncan index of geographic distance.
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

# ── FIPS code to state abbreviation mapping ──────────────────────────────────
FIPS_TO_STATE = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT",
    10: "DE", 11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL",
    18: "IN", 19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD",
    25: "MA", 26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE",
    32: "NV", 33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND",
    39: "OH", 40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD",
    47: "TN", 48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV",
    55: "WI", 56: "WY",
}

# ── Occupations to plot ──────────────────────────────────────────────────────
TARGET_OCCUPATIONS = ["Cashiers", "Registered nurses", "Petroleum engineers"]

# ── Load occupation titles and find census codes ─────────────────────────────
port_df = pd.read_csv(OUTPUT_DIR / "portability_by_occupation.csv")

occ_map = {}  # display_name -> census code
for name in TARGET_OCCUPATIONS:
    match = port_df[port_df["occ_title"].str.contains(name, case=False, na=False)]
    if match.empty:
        raise ValueError(f"Could not find occupation matching '{name}'")
    occ_code = int(match.iloc[0]["occ"])
    occ_title = match.iloc[0]["occ_title"]
    occ_map[occ_title] = occ_code
    print(f"  {occ_title}: census code {occ_code}")

# ── Load state employment and filter to target occupations ───────────────────
state_df = pd.read_csv(DATA_DIR / "state_employment.csv")
target_codes = list(occ_map.values())
state_df = state_df[state_df["occ"].isin(target_codes)].copy()

# Map FIPS to state abbreviation
state_df["state_abbr"] = state_df["statefip"].map(FIPS_TO_STATE)
state_df = state_df.dropna(subset=["state_abbr"])

# ── Compute employment shares per occupation ─────────────────────────────────
shares = []
for title, code in occ_map.items():
    occ_data = state_df[state_df["occ"] == code].copy()
    total_emp = occ_data["weighted_employment"].sum()
    occ_data["emp_share"] = occ_data["weighted_employment"] / total_emp
    occ_data["occ_title"] = title
    shares.append(occ_data)
    print(f"  {title}: {len(occ_data)} states, total employment = {total_emp:,.0f}")

shares_df = pd.concat(shares, ignore_index=True)

# ── Determine consistent x-axis scale ───────────────────────────────────────
# Find the maximum share across all occupations (top-15 subset) for alignment
x_max = 0
for title in occ_map:
    occ_data = shares_df[shares_df["occ_title"] == title].nlargest(15, "emp_share")
    if occ_data["emp_share"].max() > x_max:
        x_max = occ_data["emp_share"].max()
x_max = np.ceil(x_max * 100) / 100 + 0.02  # Add small buffer

# ── Color palette ────────────────────────────────────────────────────────────
COLORS = {
    "Cashiers": "#6baed6",
    "Registered nurses": "#74c476",
    "Petroleum engineers": "#fd8d3c",
}

# ── Plot: 1x3 panel of horizontal bar charts ────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=False)

for idx, (title, code) in enumerate(occ_map.items()):
    ax = axes[idx]

    # Get top 15 states by employment share
    occ_data = shares_df[shares_df["occ_title"] == title].copy()
    occ_data = occ_data.nlargest(15, "emp_share").sort_values("emp_share")

    # Draw horizontal bars
    color = COLORS.get(title, "#6baed6")
    bars = ax.barh(
        occ_data["state_abbr"],
        occ_data["emp_share"],
        color=color,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
        zorder=2,
    )

    # Add share labels at end of each bar
    for bar, share in zip(bars, occ_data["emp_share"]):
        ax.text(
            bar.get_width() + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{share:.1%}",
            va="center",
            ha="left",
            fontsize=7.5,
            color="#333333",
        )

    # Formatting
    ax.set_xlim(0, x_max)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Share of national employment", fontsize=9)

    # Compute a simple concentration stat: share in top 5 states
    top5_share = occ_data.nlargest(5, "emp_share")["emp_share"].sum()
    ax.text(
        0.97, 0.03,
        f"Top 5 states: {top5_share:.0%}",
        transform=ax.transAxes,
        fontsize=8,
        ha="right",
        va="bottom",
        color="#555555",
        fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.8),
    )

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=8)

fig.suptitle(
    "Geographic Concentration of Employment",
    fontsize=15,
    fontweight="bold",
    y=0.98,
)

plt.tight_layout(rect=[0, 0, 1, 0.94])

# ── Save ─────────────────────────────────────────────────────────────────────
fig.savefig(FIGURES_DIR / "fig_geographic_concentration.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGURES_DIR / "fig_geographic_concentration.pdf", bbox_inches="tight")
print("\nSaved to figures/fig_geographic_concentration.png and .pdf")
plt.close()
