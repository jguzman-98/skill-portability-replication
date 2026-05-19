"""
Generate a heatmap of predicted switching rates between ~20 recognizable occupations
spanning the full portability spectrum.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path

PROJECT = Path(__file__).parent.parent
DATA_DIR = PROJECT / "data"
OUTPUT_DIR = PROJECT / "output"
FIGURES_DIR = PROJECT / "figures"

# ── 1. Load portability data and select ~20 recognizable occupations ─────────

port = pd.read_csv(OUTPUT_DIR / "portability_by_occupation.csv")

# Target occupations by keyword (case-insensitive partial match)
targets = [
    "Cashiers",
    "Retail salespersons",
    "Customer service representatives",
    "Receptionists and information clerks",
    "Office clerks, general",
    "Driver/sales workers and truck drivers",
    "Janitors and building cleaners",
    "Elementary and middle school teachers",
    "Secondary school teachers",
    "Accountants and auditors",
    "Registered nurses",
    "Software developers",
    "Electricians",
    "Aircraft pilots and flight engineers",
    "Aerospace engineers",
    "Petroleum engineers",
    "Actors",
    "Dancers and choreographers",
    "Waiters and waitresses",
    "First-Line supervisors of retail sales workers",
]

# Match exactly by occ_title
selected = port[port["occ_title"].isin(targets)].copy()

# Verify we got them all
missing = set(targets) - set(selected["occ_title"])
if missing:
    # Try case-insensitive fallback
    for t in list(missing):
        mask = port["occ_title"].str.lower() == t.lower()
        if mask.any():
            selected = pd.concat([selected, port[mask]])
            missing.discard(t)
    if missing:
        print(f"WARNING: Could not find occupations: {missing}")

selected = selected.drop_duplicates(subset="occ")

# Sort by portability rank (rank=1 is most portable)
selected = selected.sort_values("rank")

print(f"Selected {len(selected)} occupations (ranks {selected['rank'].min()}–{selected['rank'].max()}):")
for _, row in selected.iterrows():
    print(f"  Rank {int(row['rank']):>3d}: [{int(row['occ'])}] {row['occ_title']}")

# Census codes for filtering
occ_codes = set(selected["occ"].astype(int))

# Build short labels for readability
short_labels = {
    "Cashiers": "Cashiers",
    "Retail salespersons": "Retail Salespersons",
    "Customer service representatives": "Customer Service Reps",
    "Receptionists and information clerks": "Receptionists",
    "Office clerks, general": "Office Clerks",
    "Driver/sales workers and truck drivers": "Truck Drivers",
    "Janitors and building cleaners": "Janitors",
    "Elementary and middle school teachers": "Elementary Teachers",
    "Secondary school teachers": "Secondary Teachers",
    "Accountants and auditors": "Accountants",
    "Registered nurses": "Registered Nurses",
    "Software developers": "Software Developers",
    "Electricians": "Electricians",
    "Aircraft pilots and flight engineers": "Aircraft Pilots",
    "Aerospace engineers": "Aerospace Engineers",
    "Petroleum engineers": "Petroleum Engineers",
    "Actors": "Actors",
    "Dancers and choreographers": "Dancers",
    "Waiters and waitresses": "Waiters/Waitresses",
    "First-Line supervisors of retail sales workers": "Retail Supervisors",
}

# Map occ code -> short label, ordered by rank
code_to_label = {}
ordered_codes = []
for _, row in selected.iterrows():
    code = int(row["occ"])
    title = row["occ_title"]
    label = short_labels.get(title, title[:25])
    code_to_label[code] = label
    ordered_codes.append(code)

# ── 2. Load predictions and filter to selected occupations ───────────────────

print("\nLoading predictions...")
preds = pd.read_csv(OUTPUT_DIR / "skill_portability_predictions.csv",
                     usecols=["occ_origin", "occ_dest", "predicted_switches"])

preds = preds[
    preds["occ_origin"].isin(occ_codes) & preds["occ_dest"].isin(occ_codes)
].copy()

print(f"Filtered to {len(preds)} rows (expected {len(occ_codes) * (len(occ_codes) - 1)})")

# ── 3. Pivot into matrix ────────────────────────────────────────────────────

pivot = preds.pivot(index="occ_origin", columns="occ_dest", values="predicted_switches")

# Reindex to match our rank ordering
pivot = pivot.reindex(index=ordered_codes, columns=ordered_codes)

# Apply log10 transform with small epsilon
epsilon = 1e-6
log_matrix = np.log10(pivot.values + epsilon)

# Mask the diagonal (no self-switches)
n = len(ordered_codes)
mask = np.eye(n, dtype=bool)

# ── 4. Create heatmap ───────────────────────────────────────────────────────

labels = [code_to_label[c] for c in ordered_codes]

fig, ax = plt.subplots(figsize=(14, 12))

# Use seaborn heatmap with mask for diagonal
sns.heatmap(
    log_matrix,
    mask=mask,
    xticklabels=labels,
    yticklabels=labels,
    cmap="YlOrRd",
    cbar_kws={"label": "log₁₀(Predicted Switches)", "shrink": 0.8},
    linewidths=0.3,
    linecolor="white",
    ax=ax,
    square=True,
)

ax.set_title("Predicted Switching Rates Between Selected Occupations",
             fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("Destination Occupation", fontsize=12, labelpad=10)
ax.set_ylabel("Origin Occupation", fontsize=12, labelpad=10)

# Rotate labels for readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

plt.tight_layout()

# ── 5. Save ─────────────────────────────────────────────────────────────────

fig.savefig(FIGURES_DIR / "fig_switching_heatmap.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGURES_DIR / "fig_switching_heatmap.pdf", bbox_inches="tight")
print("\nSaved: figures/fig_switching_heatmap.png")
print("Saved: figures/fig_switching_heatmap.pdf")

plt.close(fig)
