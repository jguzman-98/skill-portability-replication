"""
Step 5d: Full coefficient tables with two-way clustered standard errors.

Re-estimates the PPML and OLS gravity models for 3 key variants (LASSO,
Factor Analysis, Euclidean) under baseline and fixed-delta1 specifications.
Computes two-way clustered SEs (Cameron, Gelbach & Miller 2011) on
occ_origin x occ_dest (525 x 525 clusters).

This script loads precomputed data and ML distances -- no ML pipeline re-run.

Outputs:
  output/full_coefficients.csv       -- All coefficients with HC1 and two-way SEs
  output/model_comparison.csv        -- Updated with se_beta1_twoway, p_beta1_twoway
  output/tab_full_coefficients.tex   -- Baseline full coefficient table (LaTeX)
  output/tab_fixed_coefficients.tex  -- Fixed-delta1 full coefficient table (LaTeX)
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Poisson

from utils_clustering import twoway_cluster_se, pvalues_from_se

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT = Path(__file__).parent.parent
DATA_DIR = PROJECT / "data"
OUTPUT = PROJECT / "output"

# Three key variants for the full coefficient table
VARIANTS = {
    "ml_lasso": "ml_dist_lasso",
    "factor_analysis": "factor_dist",
    "euclidean": "euclidean_dist",
}

VARIANT_LABELS = {
    "ml_lasso": "LASSO",
    "factor_analysis": "Factor Analysis",
    "euclidean": "Euclidean",
}

VARIABLE_ORDER_BASELINE = [
    "skill_distance",
    "geographic_distance",
    "total_switches_out",
    "openings_share_dest",
    "const",
]

VARIABLE_ORDER_FIXED = [
    "skill_distance",
    "geographic_distance",
    "openings_share_dest",
    "const",
]

VARIABLE_LABELS = {
    "skill_distance": r"Skill distance ($\hat{\beta}_1$)",
    "geographic_distance": r"Geographic distance ($\hat{\beta}_2$)",
    "total_switches_out": r"Total switches out ($\hat{\delta}_1$)",
    "openings_share_dest": r"Openings share ($\hat{\delta}_2$)",
    "const": "Constant",
}


# ====================================================================
# Estimation helpers
# ====================================================================

def _extract_coefficients(model, X_df, specification, label, estimator,
                          se_tw, p_tw, r2, results):
    """Extract all coefficients from a fitted model into results list."""
    X_cols = list(sm.add_constant(X_df).columns)
    for i, var_name in enumerate(X_cols):
        results.append({
            "specification": specification,
            "skill_distance": label,
            "estimator": estimator,
            "variable": var_name,
            "coefficient": model.params[var_name],
            "se_hc1": model.bse[var_name],
            "se_twoway": se_tw[i],
            "p_hc1": model.pvalues[var_name],
            "p_twoway": p_tw[i],
            "r2": r2,
            "n_obs": int(model.nobs),
        })


def _compute_twoway_se(model, y, X, cluster_origin, cluster_dest, is_ols=False):
    """Compute two-way clustered SEs for a fitted model."""
    X_arr = np.asarray(X)
    if is_ols:
        resid = np.asarray(model.resid)
        bread_weights = np.ones(len(y))
    else:
        resid = y - np.asarray(model.mu)
        bread_weights = np.asarray(model.mu)

    se_tw, _ = twoway_cluster_se(X_arr, resid, bread_weights,
                                  cluster_origin, cluster_dest)
    p_tw = pvalues_from_se(np.asarray(model.params), se_tw)
    return se_tw, p_tw


# ====================================================================
# LaTeX generation
# ====================================================================

def _stars(p):
    if np.isnan(p):
        return ""
    if p < 0.01:
        return "^{***}"
    elif p < 0.05:
        return "^{**}"
    elif p < 0.10:
        return "^{*}"
    return ""


def _fmt_coef(coef, p_twoway):
    """Format coefficient with significance stars."""
    s = _stars(p_twoway)
    if abs(coef) >= 10:
        return f"${coef:.3f}{s}$"
    elif abs(coef) >= 0.01:
        return f"${coef:.4f}{s}$"
    else:
        return f"${coef:.6f}{s}$"


def _fmt_se(se, bracket="()"):
    """Format SE in parentheses (HC1) or brackets (two-way)."""
    o, c = ("(", ")") if bracket == "()" else ("[", "]")
    if abs(se) >= 10:
        return f"{o}{se:.3f}{c}"
    elif abs(se) >= 0.01:
        return f"{o}{se:.4f}{c}"
    else:
        return f"{o}{se:.6f}{c}"


def _write_latex_table(coef_df, specification, var_order, path, caption, label,
                       delta1_info=None):
    """Generate a LaTeX table with 3 PPML variants side-by-side."""
    variant_keys = list(VARIANTS.keys())
    spec_data = coef_df[(coef_df["specification"] == specification) &
                        (coef_df["estimator"] == "PPML")]

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")

    # Header
    lines.append(" & ".join([""] + [f"({i+1})" for i in range(3)]) + r" \\")
    lines.append(" & ".join(
        [""] + [VARIANT_LABELS[v] for v in variant_keys]
    ) + r" \\")
    lines.append(r"\midrule")

    # Coefficient rows
    for vi, var in enumerate(var_order):
        # Coefficient + stars
        cells = [VARIABLE_LABELS[var]]
        for vk in variant_keys:
            row = spec_data[(spec_data["skill_distance"] == vk) &
                            (spec_data["variable"] == var)]
            if len(row) == 1:
                r = row.iloc[0]
                cells.append(_fmt_coef(r["coefficient"], r["p_twoway"]))
            else:
                cells.append("---")
        lines.append(" & ".join(cells) + r" \\")

        # HC1 SE row (parentheses)
        cells = [""]
        for vk in variant_keys:
            row = spec_data[(spec_data["skill_distance"] == vk) &
                            (spec_data["variable"] == var)]
            if len(row) == 1:
                cells.append(_fmt_se(row.iloc[0]["se_hc1"], "()"))
            else:
                cells.append("")
        lines.append(" & ".join(cells) + r" \\")

        # Two-way SE row (brackets)
        cells = [""]
        for vk in variant_keys:
            row = spec_data[(spec_data["skill_distance"] == vk) &
                            (spec_data["variable"] == var)]
            if len(row) == 1:
                cells.append(_fmt_se(row.iloc[0]["se_twoway"], "[]"))
            else:
                cells.append("")
        tail = r"\\[0.5em]" if vi < len(var_order) - 1 else r" \\"
        lines.append(" & ".join(cells) + tail)

    lines.append(r"\midrule")

    # N row
    n_cells = ["$N$"]
    for vk in variant_keys:
        vk_data = spec_data[spec_data["skill_distance"] == vk]
        if len(vk_data) > 0:
            n_cells.append(f"{int(vk_data.iloc[0]['n_obs']):,}")
        else:
            n_cells.append("")
    lines.append(" & ".join(n_cells) + r" \\")

    # R2 row
    r2_cells = [r"Pseudo-$R^2$"]
    for vk in variant_keys:
        vk_data = spec_data[spec_data["skill_distance"] == vk]
        if len(vk_data) > 0:
            r2_cells.append(f"{vk_data.iloc[0]['r2']:.3f}")
        else:
            r2_cells.append("")
    lines.append(" & ".join(r2_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # Notes
    lines.append(r"\begin{minipage}{0.9\textwidth}")
    lines.append(r"\vspace{0.5em}")
    if specification == "baseline":
        note = (
            r"\footnotesize \textit{Notes:} PPML estimates of Equation "
            r"(\ref{eq:switching}). HC1-robust standard errors in parentheses; "
            r"two-way clustered standard errors (origin $\times$ destination "
            r"occupation) in brackets \citep{cameron2011}. "
            r"Significance stars based on two-way clustered $p$-values: "
            r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$. "
            r"$N = 275{,}100$ directed occupation pairs, clustered on 525 "
            r"origins and 525 destinations."
        )
        if delta1_info:
            note += (
                " The freely estimated $\\hat{\\delta}_1$ on total switches out "
                "ranges from " + delta1_info + " across variants, "
                "supporting the fixed-$\\delta_1 = 1$ exposure specification "
                "in Table \\ref{tab:fixed_coefficients}."
            )
    else:
        note = (
            r"\footnotesize \textit{Notes:} PPML estimates with $\delta_1 = 1$ "
            r"imposed via exposure term (total switches out of origin). "
            r"HC1-robust standard errors in parentheses; two-way clustered "
            r"standard errors in brackets \citep{cameron2011}. "
            r"Significance stars based on two-way clustered $p$-values: "
            r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$."
        )
    lines.append(note)
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    path.write_text(tex)
    print(f"  Saved: {path}")


# ====================================================================
# MAIN
# ====================================================================

def main():
    # -- 1. Load data --
    print("Loading pairwise dataset...")
    df = pd.read_csv(DATA_DIR / "pairwise_dataset.csv",
                     dtype={"occ_origin": str, "occ_dest": str})
    key_cols = ["switches", "total_switches_out", "emp_origin", "emp_dest",
                "openings_share_dest", "geographic_distance"]
    df = df.dropna(subset=key_cols).reset_index(drop=True)
    print(f"  {len(df):,} pairs after dropping NaN")

    # Merge ML distances from predictions file
    pred = pd.read_csv(OUTPUT / "skill_portability_predictions.csv",
                       dtype={"occ_origin": str, "occ_dest": str})
    ml_cols = [c for c in pred.columns if c.startswith("ml_dist_")]
    df = df.merge(pred[["occ_origin", "occ_dest"] + ml_cols],
                  on=["occ_origin", "occ_dest"], how="left")
    print(f"  Merged ML distances: {ml_cols}")

    # Cluster identifiers
    cluster_origin = df["occ_origin"].values
    cluster_dest = df["occ_dest"].values
    n_origin = len(np.unique(cluster_origin))
    n_dest = len(np.unique(cluster_dest))
    print(f"  Clusters: {n_origin} origins x {n_dest} destinations")

    y = df["switches"].values
    results = []
    delta1_values = {}  # Track freely-estimated delta1

    # -- 2. Baseline estimation --
    print("\n" + "=" * 70)
    print("BASELINE -- Full Coefficients with Two-Way Clustered SEs")
    print("=" * 70)

    for label, col in VARIANTS.items():
        print(f"\n  Variant: {VARIANT_LABELS[label]}")

        # Build RHS matching 05_estimate_models.py column order
        X_df = pd.DataFrame({
            "skill_distance": df[col].values,
            "total_switches_out": df["total_switches_out"].values,
            "openings_share_dest": df["openings_share_dest"].values,
        })
        X_df["geographic_distance"] = df["geographic_distance"].values
        X = sm.add_constant(X_df)

        # --- PPML ---
        print("    PPML:")
        model_ppml = GLM(y, X, family=Poisson()).fit(
            cov_type="HC1", maxiter=100)
        r2_ppml = 1 - model_ppml.deviance / model_ppml.null_deviance

        se_tw, p_tw = _compute_twoway_se(
            model_ppml, y, X, cluster_origin, cluster_dest, is_ols=False)
        _extract_coefficients(
            model_ppml, X_df, "baseline", label, "PPML",
            se_tw, p_tw, r2_ppml, results)

        # Report key results
        b1 = model_ppml.params["skill_distance"]
        se1_hc1 = model_ppml.bse["skill_distance"]
        idx_sd = list(X.columns).index("skill_distance")
        se1_tw = se_tw[idx_sd]
        d1 = model_ppml.params["total_switches_out"]
        delta1_values[label] = d1
        print(f"      R2={r2_ppml:.4f}  b1={b1:+.6f}  "
              f"SE_HC1={se1_hc1:.6f}  SE_2way={se1_tw:.6f}")
        print(f"      d1={d1:.6f} (freely estimated)")

        # --- OLS log ---
        print("    OLS_log:")
        y_log = np.log1p(y)
        model_ols = sm.OLS(y_log, X).fit(cov_type="HC1")
        r2_ols = model_ols.rsquared

        se_tw_ols, p_tw_ols = _compute_twoway_se(
            model_ols, y_log, X, cluster_origin, cluster_dest, is_ols=True)
        _extract_coefficients(
            model_ols, X_df, "baseline", label, "OLS_log",
            se_tw_ols, p_tw_ols, r2_ols, results)

        b1_ols = model_ols.params["skill_distance"]
        se1_hc1_ols = model_ols.bse["skill_distance"]
        se1_tw_ols = se_tw_ols[idx_sd]
        print(f"      R2={r2_ols:.4f}  b1={b1_ols:+.6f}  "
              f"SE_HC1={se1_hc1_ols:.6f}  SE_2way={se1_tw_ols:.6f}")

    # -- 3. Fixed-d1 estimation --
    print("\n" + "=" * 70)
    print("FIXED d1 = 1 -- Full Coefficients with Two-Way Clustered SEs")
    print("=" * 70)

    total_out = df["total_switches_out"].values
    exposure = np.maximum(total_out, 1).astype(float)

    for label, col in VARIANTS.items():
        print(f"\n  Variant: {VARIANT_LABELS[label]}")

        # Build RHS (no total_switches_out)
        X_df = pd.DataFrame({
            "skill_distance": df[col].values,
            "openings_share_dest": df["openings_share_dest"].values,
        })
        X_df["geographic_distance"] = df["geographic_distance"].values
        X = sm.add_constant(X_df)

        # --- PPML with exposure ---
        print("    PPML (exposure):")
        model_ppml = GLM(y, X, family=Poisson(), exposure=exposure).fit(
            cov_type="HC1", maxiter=100)
        r2_ppml = 1 - model_ppml.deviance / model_ppml.null_deviance

        se_tw, p_tw = _compute_twoway_se(
            model_ppml, y, X, cluster_origin, cluster_dest, is_ols=False)
        _extract_coefficients(
            model_ppml, X_df, "fixed_delta1", label, "PPML",
            se_tw, p_tw, r2_ppml, results)

        b1 = model_ppml.params["skill_distance"]
        se1_hc1 = model_ppml.bse["skill_distance"]
        idx_sd = list(X.columns).index("skill_distance")
        se1_tw = se_tw[idx_sd]
        print(f"      R2={r2_ppml:.4f}  b1={b1:+.6f}  "
              f"SE_HC1={se1_hc1:.6f}  SE_2way={se1_tw:.6f}")

        # --- OLS fixed-d1 ---
        print("    OLS_log (adjusted y):")
        y_adj = np.log1p(y) - np.log1p(total_out)
        model_ols = sm.OLS(y_adj, X).fit(cov_type="HC1")
        r2_ols = model_ols.rsquared

        se_tw_ols, p_tw_ols = _compute_twoway_se(
            model_ols, y_adj, X, cluster_origin, cluster_dest, is_ols=True)
        _extract_coefficients(
            model_ols, X_df, "fixed_delta1", label, "OLS_log",
            se_tw_ols, p_tw_ols, r2_ols, results)

        b1_ols = model_ols.params["skill_distance"]
        se1_hc1_ols = model_ols.bse["skill_distance"]
        se1_tw_ols = se_tw_ols[idx_sd]
        print(f"      R2={r2_ols:.4f}  b1={b1_ols:+.6f}  "
              f"SE_HC1={se1_hc1_ols:.6f}  SE_2way={se1_tw_ols:.6f}")

    # -- 4. Save outputs --
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    coef_df = pd.DataFrame(results)
    coef_df.to_csv(OUTPUT / "full_coefficients.csv", index=False)
    print(f"  Saved: output/full_coefficients.csv ({len(coef_df)} rows)")

    # Update model_comparison.csv with two-way SE columns
    mc_path = OUTPUT / "model_comparison.csv"
    mc = pd.read_csv(mc_path)
    if "se_beta1_twoway" not in mc.columns:
        mc["se_beta1_twoway"] = np.nan
        mc["p_beta1_twoway"] = np.nan

    beta1_rows = coef_df[coef_df["variable"] == "skill_distance"]
    for _, row in beta1_rows.iterrows():
        mask = ((mc["specification"] == row["specification"]) &
                (mc["skill_distance"] == row["skill_distance"]) &
                (mc["estimator"] == row["estimator"]))
        if mask.any():
            mc.loc[mask, "se_beta1_twoway"] = row["se_twoway"]
            mc.loc[mask, "p_beta1_twoway"] = row["p_twoway"]

    mc.to_csv(mc_path, index=False)
    print(f"  Updated: output/model_comparison.csv")

    # -- 5. Verification --
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    # Check that two-way SEs >= HC1 SEs (expected)
    ppml_rows = coef_df[coef_df["estimator"] == "PPML"]
    se_ratio = ppml_rows["se_twoway"] / ppml_rows["se_hc1"]
    n_larger = (se_ratio >= 1.0).sum()
    n_total = len(se_ratio)
    print(f"\n  Two-way SE >= HC1 SE: {n_larger}/{n_total} coefficients")
    print(f"  SE ratio (two-way / HC1):  "
          f"mean={se_ratio.mean():.3f}  min={se_ratio.min():.3f}  "
          f"max={se_ratio.max():.3f}")

    # Check beta1 still significant with two-way SEs
    beta1_ppml = coef_df[(coef_df["variable"] == "skill_distance") &
                         (coef_df["estimator"] == "PPML")]
    print(f"\n  b1 significance (PPML, two-way clustered):")
    for _, row in beta1_ppml.iterrows():
        print(f"    {row['specification']:15s} {row['skill_distance']:18s}  "
              f"b1={row['coefficient']:+.4f}  p_twoway={row['p_twoway']:.2e}")

    # Report freely-estimated d1
    print(f"\n  Freely-estimated d1 (PPML baseline):")
    for label, d1 in delta1_values.items():
        print(f"    {VARIANT_LABELS[label]:20s}  d1 = {d1:.6f}")

    # -- 6. Generate LaTeX tables --
    print("\n" + "=" * 70)
    print("GENERATING LATEX TABLES")
    print("=" * 70)

    # Format delta1 info for table note
    d1_vals = list(delta1_values.values())
    d1_info = f"{min(d1_vals):.4f} to {max(d1_vals):.4f}"

    _write_latex_table(
        coef_df, "baseline", VARIABLE_ORDER_BASELINE,
        OUTPUT / "tab_full_coefficients.tex",
        r"PPML Coefficient Estimates: Baseline Specification",
        "tab:full_coefficients",
        delta1_info=d1_info,
    )

    _write_latex_table(
        coef_df, "fixed_delta1", VARIABLE_ORDER_FIXED,
        OUTPUT / "tab_fixed_coefficients.tex",
        r"PPML Coefficient Estimates: Fixed $\delta_1 = 1$ Specification",
        "tab:fixed_coefficients",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
