"""
Step 5b: Build presentation/summary outputs for the progress report.

Fast, additive script (runs in a few minutes) that produces two outputs
for the Equations 1-5 writeup:

  1. output/geographic_contribution.csv
       Re-fits Equation 1 WITHOUT the geographic distance regressor for
       each skill distance variant x estimator, then reports delta-R2 vs. the
       baseline (which has geographic distance) in model_comparison.csv.

  2. output/portability_by_occupation.csv
       Aggregates PPML fitted values (predicted_switches) from
       skill_portability_predictions.csv to one portability score per
       origin occupation per Equations 5-6, merged with Census 2018 titles.

This script does NOT re-run Step 5 and does NOT modify any of its outputs.
It reads model_comparison.csv and skill_portability_predictions.csv as they
are, and fits a small number of auxiliary models for the geographic
contribution comparison.
"""

import pandas as pd
import numpy as np
import re
import warnings
from pathlib import Path
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Poisson

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT = Path(__file__).parent.parent
DATA_DIR = PROJECT / "data"
OUTPUT = PROJECT / "output"
RAW_DIR = PROJECT / "raw"

CROSSWALK_PATH = RAW_DIR / "census_2018_occ_crosswalk.csv"

DIRECT_MEASURES = {
    "euclidean": "euclidean_dist",
    "angular_separation": "angular_separation",
    "factor_analysis": "factor_dist",
}
ML_MEASURES = {
    "ml_lasso": "ml_dist_lasso",
    "ml_random_forest": "ml_dist_random_forest",
    "ml_xgboost": "ml_dist_xgboost",
}
ALL_MEASURES = {**DIRECT_MEASURES, **ML_MEASURES}


def norm_code(s) -> str:
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return str(s).strip()


# ====================================================================
# PART 1: Geographic Distance Contribution
# ====================================================================

def compute_geographic_contribution():
    """Re-fit Equation 1 without the geographic distance regressor and
    compute delta-R2 vs. the with-geo baseline already in model_comparison.csv.
    """
    print("=" * 70)
    print("PART 1: Geographic Distance Contribution")
    print("=" * 70)

    df = pd.read_csv(
        DATA_DIR / "pairwise_dataset.csv",
        dtype={"occ_origin": str, "occ_dest": str},
    )
    key_cols = ["switches", "total_switches_out", "emp_origin", "emp_dest"]
    df = df.dropna(subset=key_cols).reset_index(drop=True)
    print(f"  Loaded {len(df):,} pairs from pairwise_dataset.csv")

    # ML skill distances live in the predictions file produced by Step 5
    pred_df = pd.read_csv(
        OUTPUT / "skill_portability_predictions.csv",
        dtype={"occ_origin": str, "occ_dest": str},
    )
    ml_cols_present = [c for c in pred_df.columns if c.startswith("ml_dist_")]
    df = df.merge(
        pred_df[["occ_origin", "occ_dest"] + ml_cols_present],
        on=["occ_origin", "occ_dest"], how="left",
    )
    missing = df[ml_cols_present].isna().sum().sum()
    if missing > 0:
        print(f"  WARNING: {missing} missing ML distance values after merge")

    # Baseline R2 from Step 5 (with geographic distance)
    comp = pd.read_csv(OUTPUT / "model_comparison.csv")
    baseline_with = (
        comp[comp["specification"] == "baseline"]
        .set_index(["skill_distance", "estimator"])["R2"]
        .to_dict()
    )

    y = df["switches"].values
    results = []

    for label, col in ALL_MEASURES.items():
        if col not in df.columns:
            print(f"\n  {label}: column '{col}' not found, skipping")
            continue
        print(f"\n  {label}")

        X_df = pd.DataFrame({
            "skill_distance": df[col].values,
            "total_switches_out": df["total_switches_out"].values,
            "openings_share_dest": df["openings_share_dest"].values,
        })
        X = sm.add_constant(X_df)

        # OLS on log(1+switches)
        try:
            m_ols = sm.OLS(np.log1p(y), X).fit(cov_type="HC1")
            r2_no = m_ols.rsquared
            r2_with = baseline_with.get((label, "OLS_log"), np.nan)
            delta = r2_with - r2_no
            print(f"    OLS_log   no_geo R2={r2_no:.4f}  with_geo R2={r2_with:.4f}  delta={delta:+.4f}")
            results.append({
                "skill_distance": label,
                "estimator": "OLS_log",
                "r2_no_geo": r2_no,
                "r2_with_geo": r2_with,
                "delta_r2": delta,
            })
        except Exception as e:
            print(f"    OLS_log   FAILED: {e}")

        # PPML
        try:
            m_ppml = GLM(y, X, family=Poisson()).fit(cov_type="HC1", maxiter=100)
            r2_no = 1 - m_ppml.deviance / m_ppml.null_deviance
            r2_with = baseline_with.get((label, "PPML"), np.nan)
            delta = r2_with - r2_no
            print(f"    PPML      no_geo R2={r2_no:.4f}  with_geo R2={r2_with:.4f}  delta={delta:+.4f}")
            results.append({
                "skill_distance": label,
                "estimator": "PPML",
                "r2_no_geo": r2_no,
                "r2_with_geo": r2_with,
                "delta_r2": delta,
            })
        except Exception as e:
            print(f"    PPML      FAILED: {e}")

    out = pd.DataFrame(results)
    out_path = OUTPUT / "geographic_contribution.csv"
    out.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}  ({len(out)} rows)")
    return out


# ====================================================================
# PART 2: Portability by Occupation
# ====================================================================

def load_occ_titles() -> pd.DataFrame:
    """Load (census_code, title) pairs from the Census 2018 crosswalk."""
    cw = pd.read_csv(CROSSWALK_PATH, header=None, dtype=str)
    records = []
    for _, row in cw.iterrows():
        code = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ""
        title = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
        if not re.match(r"^\d{4}$", code):
            continue
        records.append({"occ": str(int(code)), "occ_title": title})
    return pd.DataFrame(records).drop_duplicates("occ")


def compute_portability_by_occupation():
    """Aggregate PPML fitted values into one portability score per origin
    occupation, per Equations 5-6:
        Portability_o = sum_d  w_d x y-hat_{o,d}, where w_d = EmpShare_d
    """
    print("\n" + "=" * 70)
    print("PART 2: Portability by Occupation (Equation 5)")
    print("=" * 70)

    pairs = pd.read_csv(
        OUTPUT / "skill_portability_predictions.csv",
        dtype={"occ_origin": str, "occ_dest": str},
    )
    pairs["occ_origin"] = pairs["occ_origin"].apply(norm_code)
    pairs["occ_dest"] = pairs["occ_dest"].apply(norm_code)

    if "predicted_switches" not in pairs.columns:
        print("  ERROR: 'predicted_switches' not found in predictions file.")
        print("         Re-run Step 5 to generate it.")
        return None

    # Employment-share weights w_d, pooled across years
    emp_w = pd.read_csv(DATA_DIR / "employment_counts_weighted.csv")
    emp_w["occ"] = emp_w["occ"].apply(norm_code)
    emp_pooled = emp_w.groupby("occ")["weighted_employment"].sum().reset_index()
    total = emp_pooled["weighted_employment"].sum()
    emp_pooled["emp_share"] = emp_pooled["weighted_employment"] / total

    pairs = pairs.merge(
        emp_pooled[["occ", "emp_share"]].rename(columns={"occ": "occ_dest"}),
        on="occ_dest", how="left",
    )
    pairs["emp_share"] = pairs["emp_share"].fillna(0)
    pairs["weighted_pred"] = pairs["emp_share"] * pairs["predicted_switches"]

    portability = (
        pairs.groupby("occ_origin")["weighted_pred"]
             .sum()
             .reset_index()
             .rename(columns={"occ_origin": "occ", "weighted_pred": "portability"})
    )

    # Origin employment for context
    portability = portability.merge(
        emp_pooled[["occ", "weighted_employment"]], on="occ", how="left"
    )

    # Per-worker normalized version
    portability["portability_per_million"] = 1e6 * (
        portability["portability"] / portability["weighted_employment"]
    )

    # Titles
    titles = load_occ_titles()
    portability = portability.merge(titles, on="occ", how="left")

    # Rank and percentile for BOTH versions (1 = most portable)
    portability = (
        portability.sort_values("portability", ascending=False)
                   .reset_index(drop=True)
    )
    portability["rank"] = portability.index + 1
    n = len(portability)
    portability["percentile"] = 100 * (1 - (portability["rank"] - 1) / n)

    # Rank on per-worker version
    pw_order = (
        portability["portability_per_million"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    portability["rank_per_worker"] = pw_order
    portability["percentile_per_worker"] = 100 * (1 - (pw_order - 1) / n)

    out_path = OUTPUT / "portability_by_occupation.csv"
    portability.to_csv(out_path, index=False)
    print(f"  Aggregated {n} occupations")
    print(f"  Saved: {out_path}")

    # Summary stats -- raw (spec-faithful)
    p = portability["portability"]
    print(f"\n  Distribution of Portability_o (raw, spec-faithful):")
    print(f"    mean   = {p.mean():.2f}    median = {p.median():.2f}")
    print(f"    sd     = {p.std():.2f}    IQR = [{p.quantile(0.25):.2f}, {p.quantile(0.75):.2f}]")
    print(f"    min    = {p.min():.2f}    max = {p.max():.2f}")

    pw = portability["portability_per_million"]
    print(f"\n  Distribution of Portability per million origin obs:")
    print(f"    mean   = {pw.mean():.2f}    median = {pw.median():.2f}")
    print(f"    sd     = {pw.std():.2f}    IQR = [{pw.quantile(0.25):.2f}, {pw.quantile(0.75):.2f}]")
    print(f"    min    = {pw.min():.2f}    max = {pw.max():.2f}")

    # Spearman correlation between the two rankings
    rho = portability[["portability", "portability_per_million"]].corr(method="spearman").iloc[0, 1]
    print(f"\n  Spearman rank correlation (raw vs per-worker): rho = {rho:.3f}")

    print("\n  TOP 10 by RAW portability (spec-faithful):")
    for _, row in portability.head(10).iterrows():
        title = str(row.get("occ_title") or "")[:50]
        print(f"    {row['occ']:>6s}  {title:<50s}  {row['portability']:7.2f}")

    print("\n  BOTTOM 10 by RAW portability:")
    for _, row in portability.tail(10).iterrows():
        title = str(row.get("occ_title") or "")[:50]
        print(f"    {row['occ']:>6s}  {title:<50s}  {row['portability']:7.2f}")

    pw_sorted = portability.sort_values("portability_per_million", ascending=False)
    print("\n  TOP 10 by PER-WORKER portability rate (per million obs):")
    for _, row in pw_sorted.head(10).iterrows():
        title = str(row.get("occ_title") or "")[:50]
        print(f"    {row['occ']:>6s}  {title:<50s}  {row['portability_per_million']:8.2f}")

    print("\n  BOTTOM 10 by PER-WORKER portability rate:")
    for _, row in pw_sorted.tail(10).iterrows():
        title = str(row.get("occ_title") or "")[:50]
        print(f"    {row['occ']:>6s}  {title:<50s}  {row['portability_per_million']:8.2f}")

    return portability


# ====================================================================
# MAIN
# ====================================================================

def main():
    compute_geographic_contribution()
    compute_portability_by_occupation()
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
