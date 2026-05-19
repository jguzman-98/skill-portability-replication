"""
Step 5c: Fixed-d1 PPML across all 6 skill distance variants + Option C portability index.

1. Run fixed-d1 = 1 PPML (exposure = total_switches_out) for all 6 variants.
2. Report R2 table.
3. Construct the Option C index from the best ML model:
     m-hat_{o,d} = exp(X*b-hat) = y-hat / exposure   (predicted rate per switcher)
     PortRate_o = sum_d w_d * m-hat_{o,d}
     Portability_o = rank(PortRate_o) / N   in [0, 1]
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
RAW_DIR = PROJECT / "raw"
DATA_DIR = PROJECT / "data"
OUTPUT = PROJECT / "output"

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


def load_occ_titles() -> pd.DataFrame:
    cw = pd.read_csv(CROSSWALK_PATH, header=None, dtype=str)
    records = []
    for _, row in cw.iterrows():
        code = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ""
        title = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
        if not re.match(r"^\d{4}$", code):
            continue
        records.append({"occ": str(int(code)), "occ_title": title})
    return pd.DataFrame(records).drop_duplicates("occ")


def main():
    # -- Load data --
    print("Loading pairwise dataset...")
    df = pd.read_csv(
        DATA_DIR / "pairwise_dataset.csv",
        dtype={"occ_origin": str, "occ_dest": str},
    )
    key_cols = ["switches", "total_switches_out", "emp_origin", "emp_dest"]
    df = df.dropna(subset=key_cols).reset_index(drop=True)
    print(f"  {len(df):,} pairs after dropping NaN")

    # Merge ML skill distances from predictions file
    pred = pd.read_csv(
        OUTPUT / "skill_portability_predictions.csv",
        dtype={"occ_origin": str, "occ_dest": str},
    )
    ml_cols = [c for c in pred.columns if c.startswith("ml_dist_")]
    df = df.merge(
        pred[["occ_origin", "occ_dest"] + ml_cols],
        on=["occ_origin", "occ_dest"],
        how="left",
    )
    print(f"  Merged ML distance columns: {ml_cols}")

    # -- Run fixed-d1 PPML for all 6 variants --
    print("\n" + "=" * 70)
    print("Fixed d1 = 1 PPML (exposure = total_switches_out)")
    print("=" * 70)

    y = df["switches"].values
    total_out = df["total_switches_out"].values
    exposure = np.maximum(total_out, 1).astype(float)

    results = []
    fitted_models = {}

    for label, col in ALL_MEASURES.items():
        if col not in df.columns:
            print(f"\n  {label}: column '{col}' not found, skipping")
            continue

        X_df = pd.DataFrame({
            "skill_distance": df[col].values,
            "openings_share_dest": df["openings_share_dest"].values,
        })
        if "geographic_distance" in df.columns:
            X_df["geographic_distance"] = df["geographic_distance"].values

        X = sm.add_constant(X_df)

        try:
            model = GLM(y, X, family=Poisson(), exposure=exposure).fit(
                cov_type="HC1", maxiter=100
            )
            r2 = 1 - model.deviance / model.null_deviance
            beta1 = model.params["skill_distance"]
            se1 = model.bse["skill_distance"]
            p1 = model.pvalues["skill_distance"]

            results.append({
                "skill_distance": label,
                "R2": r2,
                "beta1": beta1,
                "se_beta1": se1,
                "p_beta1": p1,
                "n_obs": int(model.nobs),
            })
            fitted_models[label] = model

            print(f"  {label:<25s}  R2={r2:.4f}  b1={beta1:+.6f}  "
                  f"(SE={se1:.6f}, p={p1:.2e})")

        except Exception as e:
            print(f"  {label:<25s}  FAILED: {e}")

    # -- Results table --
    res_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    print("\n" + "=" * 70)
    print("Fixed d1 = 1 PPML -- R2 Comparison")
    print("=" * 70)
    print(f"\n{'Skill Distance':<25s} {'R2':>8s} {'b1':>12s} {'p-value':>12s}")
    print("-" * 60)
    for _, row in res_df.iterrows():
        print(f"{row['skill_distance']:<25s} {row['R2']:>8.4f} "
              f"{row['beta1']:>+12.6f} {row['p_beta1']:>12.2e}")

    # Identify best ML model
    ml_results = res_df[res_df["skill_distance"].str.startswith("ml_")]
    if ml_results.empty:
        print("\n  ERROR: No ML models estimated successfully.")
        return
    best_ml = ml_results.iloc[0]
    best_label = best_ml["skill_distance"]
    print(f"\n  Best ML model (fixed d1): {best_label} (R2={best_ml['R2']:.4f})")

    # -- Construct Option C index --
    print("\n" + "=" * 70)
    print(f"Option C Portability Index (using {best_label})")
    print("=" * 70)

    best_model = fitted_models[best_label]

    # m-hat_{o,d} = y-hat / exposure = exp(X*b-hat), the predicted rate per switcher
    fitted_values = np.asarray(best_model.mu)  # y-hat = exposure * exp(X*b-hat)
    rate_per_switcher = fitted_values / exposure  # exp(X*b-hat)

    df["fitted_fixed_delta1"] = fitted_values
    df["rate_per_switcher"] = rate_per_switcher

    # Employment-share weights w_d
    emp_w = pd.read_csv(DATA_DIR / "employment_counts_weighted.csv")
    emp_w["occ"] = emp_w["occ"].apply(norm_code)
    emp_pooled = emp_w.groupby("occ")["weighted_employment"].sum().reset_index()
    total_emp = emp_pooled["weighted_employment"].sum()
    emp_pooled["emp_share"] = emp_pooled["weighted_employment"] / total_emp

    df["occ_dest_norm"] = df["occ_dest"].apply(norm_code)
    df = df.merge(
        emp_pooled[["occ", "emp_share"]].rename(
            columns={"occ": "occ_dest_norm"}
        ),
        on="occ_dest_norm",
        how="left",
    )
    df["emp_share"] = df["emp_share"].fillna(0)

    # PortRate_o = sum_d w_d * m-hat_{o,d}
    df["weighted_rate"] = df["emp_share"] * df["rate_per_switcher"]

    portability = (
        df.groupby("occ_origin")["weighted_rate"]
          .sum()
          .reset_index()
          .rename(columns={"occ_origin": "occ", "weighted_rate": "port_rate"})
    )
    portability["occ"] = portability["occ"].apply(norm_code)

    # Rank-normalize to [0, 1]
    n = len(portability)
    portability["rank"] = portability["port_rate"].rank(ascending=False, method="first").astype(int)
    portability["portability_index"] = 1 - (portability["rank"] - 1) / (n - 1)

    # Also compute min-max version
    pmin = portability["port_rate"].min()
    pmax = portability["port_rate"].max()
    portability["portability_minmax"] = (portability["port_rate"] - pmin) / (pmax - pmin)

    # Merge titles and employment
    titles = load_occ_titles()
    portability = portability.merge(titles, on="occ", how="left")
    portability = portability.merge(
        emp_pooled[["occ", "weighted_employment"]], on="occ", how="left"
    )

    # Sort by index
    portability = portability.sort_values("portability_index", ascending=False).reset_index(drop=True)

    # -- Save --
    out_path = OUTPUT / "portability_index_fixed_delta1.csv"
    portability.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}  ({n} occupations)")

    # -- Summary stats --
    pr = portability["port_rate"]
    print(f"\n  PortRate_o (raw, before normalization):")
    print(f"    mean   = {pr.mean():.6f}    median = {pr.median():.6f}")
    print(f"    sd     = {pr.std():.6f}    IQR = [{pr.quantile(0.25):.6f}, {pr.quantile(0.75):.6f}]")
    print(f"    min    = {pr.min():.6f}    max = {pr.max():.6f}")

    # -- Top 10 / Bottom 10 --
    print(f"\n  TOP 10 by Portability Index (Option C, rank-normalized):")
    print(f"  {'Rank':>4s}  {'Code':>6s}  {'Occupation':<50s}  {'Index':>6s}  {'PortRate':>10s}")
    print(f"  {'-'*80}")
    for _, row in portability.head(10).iterrows():
        title = str(row.get("occ_title") or "")[:50]
        print(f"  {row['rank']:>4d}  {row['occ']:>6s}  {title:<50s}  "
              f"{row['portability_index']:>6.3f}  {row['port_rate']:>10.6f}")

    print(f"\n  BOTTOM 10 by Portability Index:")
    print(f"  {'Rank':>4s}  {'Code':>6s}  {'Occupation':<50s}  {'Index':>6s}  {'PortRate':>10s}")
    print(f"  {'-'*80}")
    for _, row in portability.tail(10).iterrows():
        title = str(row.get("occ_title") or "")[:50]
        print(f"  {row['rank']:>4d}  {row['occ']:>6s}  {title:<50s}  "
              f"{row['portability_index']:>6.3f}  {row['port_rate']:>10.6f}")

    # -- Spearman correlations with existing measures --
    existing = pd.read_csv(OUTPUT / "portability_by_occupation.csv")
    existing["occ"] = existing["occ"].apply(norm_code)
    merged = portability.merge(existing[["occ", "portability", "portability_per_million"]],
                               on="occ", how="inner")
    rho_raw = merged[["portability_index", "portability"]].corr(method="spearman").iloc[0, 1]
    rho_pw = merged[["portability_index", "portability_per_million"]].corr(method="spearman").iloc[0, 1]
    print(f"\n  Spearman correlations with existing measures:")
    print(f"    vs. raw Portability_o:       rho = {rho_raw:.3f}")
    print(f"    vs. per-worker rate:         rho = {rho_pw:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
