"""
Figure: Portability vs. Long-Term Unemployment scatter.

Scatter plot of the fixed-δ₁ portability index (x) vs. LTU share (y) for 524
occupations, with the OLS fitted line from Equation 7 (controlling for
employment trend — plotted as a partial regression / added-variable line).
Points sized by labor force and notable occupations annotated.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

PROJECT = Path(__file__).parent.parent
DATA = PROJECT / "data"
OUTPUT = PROJECT / "output"
FIGURES_DIR = PROJECT / "figures"

# ── load and merge ──────────────────────────────────────────────────
port = pd.read_csv(OUTPUT / "portability_index_fixed_delta1.csv")
ltu = pd.read_csv(DATA / "long_term_unemployment.csv")
trend = pd.read_csv(DATA / "employment_trends.csv")

for d in [port, ltu, trend]:
    d["occ"] = d["occ"].astype(float).astype(int).astype(str)

df = port[["occ", "portability_index", "occ_title"]].merge(
    ltu[["occ", "ltu_share", "labor_force", "n_ltu"]], on="occ"
).merge(
    trend[["occ", "emp_trend"]], on="occ"
)
df = df.dropna()
print(f"Sample: {len(df)} occupations")

# ── z-score for regression (matches 07_sectoral_downturn.py) ────────
df["port_z"] = (df["portability_index"] - df["portability_index"].mean()) / df["portability_index"].std()
df["trend_z"] = (df["emp_trend"] - df["emp_trend"].mean()) / df["emp_trend"].std()

# ── Equation 7 regression ──────────────────────────────────────────
X = sm.add_constant(df[["trend_z", "port_z"]])
model = sm.OLS(df["ltu_share"], X).fit(cov_type="HC1")
a2 = model.params["port_z"]
p2 = model.pvalues["port_z"]
r2 = model.rsquared
ci = model.conf_int().loc["port_z"]
print(f"  α₂ = {a2:+.4f}, p = {p2:.4f}, R² = {r2:.3f}")
print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

# ── partial regression (added-variable plot) ────────────────────────
# Residualize both LTU and portability on employment trend
X_trend = sm.add_constant(df[["trend_z"]])
resid_ltu = sm.OLS(df["ltu_share"], X_trend).fit().resid
resid_port = sm.OLS(df["port_z"], X_trend).fit().resid

# Fitted line on residualized data
slope_partial = sm.OLS(resid_ltu, sm.add_constant(resid_port)).fit()

# ── point sizes (proportional to log labor force) ───────────────────
log_lf = np.log1p(df["labor_force"])
sizes = 8 + 40 * (log_lf - log_lf.min()) / (log_lf.max() - log_lf.min())

# ── annotations ─────────────────────────────────────────────────────
annotations = {
    "1021": "Software developers",
    "3255": "Registered nurses",
    "6355": "Electricians",
    "4720": "Cashiers",
    "2320": "Secondary teachers",
    "9030": "Pilots",
}

# Manual nudge offsets: (dx, dy) in data-coordinate units
nudge = {
    "1021": (-0.07, 0.0030),
    "3255": (-0.07, -0.0030),
    "6355": (0.06, 0.0030),
    "4720": (-0.10, 0.0030),
    "2320": (-0.10, -0.0025),
    "9030": (0.06, 0.0030),
}

# ── build figure ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6.5))

# Scatter: residualized portability (x) vs. residualized LTU (y)
ax.scatter(
    resid_port, resid_ltu,
    s=sizes, alpha=0.45, color="#2874A6", edgecolors="none", zorder=2,
)

# Fitted line
x_line = np.linspace(resid_port.min(), resid_port.max(), 200)
y_line = slope_partial.params.iloc[0] + slope_partial.params.iloc[1] * x_line
ax.plot(x_line, y_line, color="#CB4335", linewidth=2, zorder=3,
        label=f"$\\hat{{\\alpha}}_2 = {a2:+.4f}$, $p = {p2:.3f}$")

# Zero lines
ax.axhline(0, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
ax.axvline(0, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)

# Annotate notable occupations
for occ_code, label in annotations.items():
    idx = df.index[df["occ"] == occ_code]
    if len(idx) == 0:
        continue
    i = idx[0]
    x = resid_port.loc[i]
    y = resid_ltu.loc[i]
    dx, dy = nudge.get(occ_code, (0.05, 0.002))

    # highlight point
    ax.scatter(x, y, s=60, facecolors="none", edgecolors="black",
               linewidths=1.0, zorder=4)

    ax.annotate(
        label,
        xy=(x, y),
        xytext=(x + dx, y + dy),
        fontsize=8, fontweight="medium",
        arrowprops=dict(arrowstyle="-", color="0.3", linewidth=0.6),
        zorder=5,
    )

# Labels and title
ax.set_xlabel("Portability index (residualized on employment trend)", fontsize=10.5)
ax.set_ylabel("LTU share (residualized on employment trend)", fontsize=10.5)
ax.set_title(
    f"Portability and Long-Term Unemployment ($N = {len(df)}$, $R^2 = {r2:.3f}$)",
    fontsize=12, fontweight="bold", pad=12,
)
ax.legend(fontsize=10, loc="upper right", framealpha=0.9)

# Styling
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=9)

fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_portability_ltu.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGURES_DIR / "fig_portability_ltu.pdf", bbox_inches="tight")
print("\nSaved to figures/fig_portability_ltu.png and .pdf")
plt.close()
