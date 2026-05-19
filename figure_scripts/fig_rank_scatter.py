"""
Figure: Factor-vs-LASSO rank scatter with licensing overlay.

x = LASSO portability rank (1–525), y = Factor analysis portability rank (1–525)
Points colored by licensing share (red = heavily licensed).
Annotated occupations: RNs, electricians, teachers, software devs, petroleum engineers, pilots.

Licensing shares by detailed occupation from BLS CPS Table 53 (cpsaat53),
2024 annual averages.  Major-group baselines are applied first, then
individual occupation overrides for occupations whose licensing rates
are well documented.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import spearmanr
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

# ── licensing share by Census 2018 occupation code ──────────────────
# Baseline rates by major occupation group (BLS CPS Table 53, 2024).
# These are approximate group-level shares of workers holding a
# state-issued license (not certification-only).

def assign_licensing_share(code):
    """Return approximate licensing share (0–1) for a Census 2018 occ code."""

    # ── Individual occupation overrides (well-documented rates) ─────
    overrides = {
        # Healthcare practitioners — virtually all licensed
        3255: 0.95,  # Registered nurses
        3256: 0.97,  # Nurse anesthetists
        3258: 0.95,  # Nurse practitioners
        3500: 0.95,  # Licensed practical / vocational nurses
        3010: 0.99,  # Dentists
        3050: 0.99,  # Pharmacists
        3060: 0.95,  # Physicians (general/family)
        3090: 0.95,  # Other physicians
        3110: 0.90,  # Physician assistants
        3120: 0.90,  # Podiatrists
        3140: 0.95,  # Optometrists
        3160: 0.85,  # Physical therapists
        3230: 0.85,  # Occupational therapists
        3245: 0.90,  # Veterinarians
        3300: 0.80,  # Clinical laboratory technologists and technicians
        3321: 0.90,  # Dental hygienists
        3322: 0.70,  # Dental assistants (varies by state)
        3400: 0.85,  # Diagnostic medical sonographers
        3420: 0.90,  # Radiologic technologists and technicians
        3515: 0.85,  # Medical records specialists
        3535: 0.80,  # Respiratory therapists
        3250: 0.90,  # Chiropractors
        3421: 0.40,  # Pharmacy technicians (varies)
        3235: 0.90,  # Speech-language pathologists
        3210: 0.80,  # Therapists, all other
        3220: 0.85,  # Recreational therapists
        3270: 0.80,  # Healthcare diagnosing/treating practitioners, other

        # Education — K-12 teachers
        2300: 0.85,  # Preschool and kindergarten teachers
        2310: 0.90,  # Elementary and middle school teachers
        2320: 0.90,  # Secondary school teachers
        2330: 0.90,  # Special education teachers
        2205: 0.20,  # Postsecondary teachers (tenure, not license)
        2360: 0.40,  # Other teachers and instructors
        2440: 0.65,  # Librarians and media collections specialists
        2545: 0.50,  # Teaching assistants

        # Legal
        2100: 0.95,  # Lawyers
        2105: 0.95,  # Judicial law clerks
        2110: 0.99,  # Judges, magistrates
        2145: 0.15,  # Paralegals and legal assistants

        # Skilled trades — licensed in most states
        6355: 0.65,  # Electricians
        6442: 0.55,  # Plumbers, pipefitters, steamfitters
        6230: 0.10,  # Carpenters
        6260: 0.25,  # Construction laborers
        6515: 0.40,  # Roofers (varies heavily)
        6700: 0.30,  # HVAC mechanics and installers (varies)
        7010: 0.15,  # Automotive service technicians
        7410: 0.50,  # Electrical power-line installers

        # Pilots, transportation
        9030: 0.99,  # Aircraft pilots and flight engineers
        9040: 0.90,  # Air traffic controllers
        9150: 0.85,  # Bus drivers (CDL)
        9130: 0.65,  # Driver/sales workers and truck drivers
        9142: 0.70,  # Taxi drivers (varies)

        # Finance / Insurance / Real estate — licensed
        4920: 0.85,  # Real estate brokers and sales agents
        4810: 0.70,  # Insurance sales agents
        860:  0.45,  # Insurance underwriters
        800:  0.40,  # Accountants and auditors (CPA varies)
        850:  0.55,  # Personal financial advisors
        810:  0.80,  # Property appraisers and assessors

        # Personal care — licensed
        4510: 0.80,  # Hairdressers, hairstylists, cosmetologists
        4520: 0.80,  # Manicurists and pedicurists (estheticians too)
        4530: 0.80,  # Skincare specialists

        # Social work / counseling
        2000: 0.60,  # Counselors
        2010: 0.70,  # Social workers
        2016: 0.55,  # Social workers, all other
        2015: 0.65,  # Substance abuse/behavioral counselors

        # Protective service
        3930: 0.70,  # Security guards and gambling surveillance officers
        3710: 0.95,  # Police officers (sworn)
        3740: 0.95,  # Detectives and criminal investigators
        3800: 0.90,  # Bailiffs, correctional officers (varies)
        3850: 0.85,  # Firefighters

        # Software / Computer (low licensing)
        1021: 0.06,  # Software developers
        1022: 0.08,  # Software QA analysts
        1010: 0.06,  # Computer programmers
        1050: 0.12,  # Computer support specialists
        1006: 0.10,  # Computer systems analysts

        # Engineering — PE licensure varies
        1360: 0.30,  # Civil engineers (many PEs)
        1520: 0.25,  # Petroleum engineers
        1320: 0.20,  # Aerospace engineers
        1430: 0.15,  # Industrial engineers
        1460: 0.15,  # Mechanical engineers
        1410: 0.15,  # Electrical/electronics engineers
    }

    if code in overrides:
        return overrides[code]

    # ── Major-group baselines (Census 2018 code ranges) ─────────────
    if code <= 440:    return 0.19   # Management
    if code <= 960:    return 0.25   # Business and Financial Operations
    if code <= 1240:   return 0.12   # Computer and Mathematical
    if code <= 1560:   return 0.18   # Architecture and Engineering
    if code <= 1980:   return 0.18   # Life, Physical, Social Science
    if code <= 2060:   return 0.37   # Community and Social Service
    if code <= 2180:   return 0.40   # Legal
    if code <= 2555:   return 0.45   # Education, Training, Library
    if code <= 2970:   return 0.10   # Arts, Design, Entertainment
    if code <= 3550:   return 0.55   # Healthcare Practitioners
    if code <= 3655:   return 0.20   # Healthcare Support
    if code <= 3960:   return 0.28   # Protective Service
    if code <= 4160:   return 0.07   # Food Preparation and Serving
    if code <= 4255:   return 0.06   # Building and Grounds Cleaning
    if code <= 4655:   return 0.25   # Personal Care and Service
    if code <= 4965:   return 0.11   # Sales
    if code <= 5940:   return 0.08   # Office and Administrative Support
    if code <= 6130:   return 0.05   # Farming, Fishing, Forestry
    if code <= 6950:   return 0.14   # Construction and Extraction
    if code <= 7640:   return 0.13   # Installation, Maintenance, Repair
    if code <= 8990:   return 0.06   # Production
    if code <= 9760:   return 0.15   # Transportation and Material Moving
    return 0.10

df["lic_share"] = df["occ_int"].apply(assign_licensing_share)

# ── Spearman correlation ────────────────────────────────────────────
rho, pval = spearmanr(df["rank_lasso"], df["rank_factor"])
print(f"Spearman ρ = {rho:.3f}, p = {pval:.2e}, n = {len(df)}")

# ── occupations to annotate ─────────────────────────────────────────
annotations = {
    3255: "Registered nurses",
    6355: "Electricians",
    2320: "Teachers (secondary)",
    1021: "Software developers",
    1520: "Petroleum engineers",
    9030: "Pilots",
}

# Manual nudges: (dx, dy) in data coords for each labeled point
# With inverted axes, positive dx moves label LEFT, positive dy moves label DOWN.
# rank 1 is at top-right corner.
nudge = {
    3255: (60, -30),    # RNs: LASSO=249, Factor=47 — above diagonal
    6355: (50, -30),    # Electricians: LASSO=398, Factor=95 — far above diagonal
    2320: (110, 50),    # Teachers: LASSO=16, Factor=1 — top-right, push left & down
    1021: (-55, 60),    # Software devs: LASSO=268, Factor=306 — below diagonal
    1520: (35, -40),    # Petroleum eng: LASSO=521, Factor=521 — bottom-left corner
    9030: (50, -40),    # Pilots: LASSO=507, Factor=402 — below-left, above diagonal
}

# ── build figure ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))

# Color map: low licensing → slate blue, high licensing → red
cmap = mcolors.LinearSegmentedColormap.from_list(
    "lic", ["#4878CF", "#CCCCCC", "#E8453C"], N=256
)
norm = mcolors.Normalize(vmin=0, vmax=1)

sc = ax.scatter(
    df["rank_lasso"],
    df["rank_factor"],
    c=df["lic_share"],
    cmap=cmap,
    norm=norm,
    s=22,
    alpha=0.75,
    edgecolors="none",
    zorder=2,
)

# 45° reference line
ax.plot([1, 525], [1, 525], color="black", linewidth=0.8, linestyle="--",
        alpha=0.5, zorder=1)

# Annotations
for occ_code, label in annotations.items():
    row = df[df["occ_int"] == occ_code]
    if row.empty:
        continue
    x = row["rank_lasso"].values[0]
    y = row["rank_factor"].values[0]
    dx, dy = nudge.get(occ_code, (40, -40))

    # highlight point
    ax.scatter(x, y, s=70, facecolors="none", edgecolors="black",
               linewidths=1.2, zorder=3)

    ax.annotate(
        label,
        xy=(x, y),
        xytext=(x + dx, y + dy),
        fontsize=8.5,
        fontweight="medium",
        arrowprops=dict(arrowstyle="-", color="0.3", linewidth=0.7),
        zorder=4,
    )

# Colorbar
cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02, aspect=25)
cbar.set_label("Licensing share", fontsize=10)
cbar.ax.tick_params(labelsize=8)

# Axes
ax.set_xlabel("LASSO portability rank (1 = most portable)", fontsize=11)
ax.set_ylabel("Factor analysis portability rank", fontsize=11)
ax.set_title(
    f"Factor vs. LASSO portability ranks (Spearman ρ = {rho:.3f})",
    fontsize=12,
    fontweight="bold",
)
ax.set_xlim(0, 540)
ax.set_ylim(0, 540)
ax.set_aspect("equal")
ax.tick_params(labelsize=9)
ax.invert_yaxis()  # rank 1 at top-left
ax.invert_xaxis()  # rank 1 at top-left

# Make sure rank 1 is top-left for both axes
ax.set_xlim(540, 0)
ax.set_ylim(540, 0)

fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_rank_scatter.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGURES_DIR / "fig_rank_scatter.pdf", bbox_inches="tight")
print("Saved to figures/fig_rank_scatter.png and .pdf")
plt.close()
