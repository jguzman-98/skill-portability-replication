"""
Step 5: Estimate switching models per spec equations (1)-(2).

Part A -- Construct skill distance variants:
  Direct metrics (from Step 4): Euclidean, Angular separation, Factor Analysis
  ML-based (trained here): LASSO, Random Forest, XGBoost
    Each ML model predicts Switches_{o,d} from per-dimension skill differences.
    The out-of-fold predicted value = that model's Skill Distance measure.

Part B -- For each skill distance variant, estimate the main switching model:
  Switches_{o,d} = b0 + b1 SkillDistance_{o,d} + b2 GeographicDistance_{o,d}
                 + d1 Switches_{o,d*} + d2 OpeningsShare_d + v_{o,d}

  Estimators: OLS (log outcome), Pseudo Poisson ML (PPML)

Part C -- Compare R-squared and b1 across all variants to select preferred metric.

Outputs:
  output/model_comparison.csv              -- R-squared, b1, SE for each variant x estimator
  output/skill_portability_predictions.csv -- predicted switches from best specification
  output/feature_importance.csv            -- variable importance from ML models
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Poisson

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed. Skipping XGBRegressor.")

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT = Path(__file__).parent.parent
DATA_DIR = PROJECT / "data"
OUT_DIR = PROJECT / "output"
OUT_DIR.mkdir(exist_ok=True)


# ====================================================================
# PART A: Construct ML-based skill distances
# ====================================================================

def build_ml_skill_distances(df):
    """Train ML models on diff_* features to predict Switches_{o,d}.

    Returns out-of-fold predicted values for each model, which serve as
    ML-based skill distance measures. Also returns feature importances.
    """
    print("=" * 70)
    print("PART A: Construct ML-based Skill Distances")
    print("=" * 70)

    diff_cols = [c for c in df.columns if c.startswith("diff_")]
    X = df[diff_cols].values
    y = df["switches"].values

    print(f"  Features: {len(diff_cols)} diff_* columns")
    print(f"  Sample: {len(y):,} pairs, {(y > 0).sum():,} nonzero ({(y > 0).mean()*100:.1f}%)")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    ml_distances = {}
    importances = {}

    # Scale features for LASSO
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. LASSO
    print("\n  [1/3] LassoCV...")
    lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
    lasso_preds = cross_val_predict(lasso, X_scaled, y, cv=kf)
    lasso.fit(X_scaled, y)
    nonzero = np.sum(lasso.coef_ != 0)
    print(f"    alpha={lasso.alpha_:.6f}, nonzero coefs: {nonzero}/{len(lasso.coef_)}")
    ml_distances["lasso"] = lasso_preds
    importances["lasso"] = pd.Series(np.abs(lasso.coef_), index=diff_cols)

    # 2. Random Forest
    print("\n  [2/3] Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=50,
                               random_state=42, n_jobs=-1)
    rf_preds = cross_val_predict(rf, X, y, cv=kf)
    rf.fit(X, y)
    print(f"    OOF R-squared: {1 - np.sum((y - rf_preds)**2) / np.sum((y - y.mean())**2):.4f}")
    ml_distances["random_forest"] = rf_preds
    importances["random_forest"] = pd.Series(rf.feature_importances_, index=diff_cols)

    # 3. XGBoost
    if HAS_XGB:
        print("\n  [3/3] XGBoost...")
        xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           random_state=42, n_jobs=-1, verbosity=0)
        xgb_preds = cross_val_predict(xgb, X, y, cv=kf)
        xgb.fit(X, y)
        print(f"    OOF R-squared: {1 - np.sum((y - xgb_preds)**2) / np.sum((y - y.mean())**2):.4f}")
        ml_distances["xgboost"] = xgb_preds
        importances["xgboost"] = pd.Series(xgb.feature_importances_, index=diff_cols)

    return ml_distances, importances


# ====================================================================
# PART B: Estimate equation (1) for each skill distance variant
# ====================================================================

def estimate_ols(y, X_df, label):
    """OLS on log(1 + Switches) with HC1 robust SEs."""
    y_log = np.log1p(y)
    X = sm.add_constant(X_df)
    model = sm.OLS(y_log, X).fit(cov_type="HC1")
    return model


def estimate_ppml(y, X_df, label):
    """Pseudo Poisson Maximum Likelihood (Santos Silva & Tenreyro 2006)."""
    X = sm.add_constant(X_df)
    model = GLM(y, X, family=Poisson()).fit(cov_type="HC1", maxiter=100)
    return model


def _build_rhs(df, skill_dist_col):
    """Build the RHS dataframe for equation (1)."""
    X_df = pd.DataFrame({
        "skill_distance": df[skill_dist_col].values,
        "total_switches_out": df["total_switches_out"].values,
        "openings_share_dest": df["openings_share_dest"].values,
    })
    if "geographic_distance" in df.columns:
        X_df["geographic_distance"] = df["geographic_distance"].values
    # Include year dummies if present (for year FE specification)
    for col in df.columns:
        if col.startswith("year_"):
            X_df[col] = df[col].values
    return X_df


def run_equation1(df, skill_dist_col, skill_dist_label, results_list,
                  specification="baseline"):
    """Run equation (1) for one skill distance variant, both OLS and PPML.

    Returns dict of {estimator_name: fitted_model} for successfully estimated models.
    """

    X_df = _build_rhs(df, skill_dist_col)
    y = df["switches"].values
    fitted = {}

    for estimator_name, estimator_fn in [("OLS_log", estimate_ols), ("PPML", estimate_ppml)]:
        try:
            model = estimator_fn(y, X_df, f"{skill_dist_label}_{estimator_name}")
            fitted[estimator_name] = model

            # Extract skill_distance coefficient
            beta1 = model.params["skill_distance"]
            se1 = model.bse["skill_distance"]
            p1 = model.pvalues["skill_distance"]

            if estimator_name == "OLS_log":
                r2 = model.rsquared
            else:
                # Pseudo R-squared for PPML (deviance-based)
                r2 = 1 - model.deviance / model.null_deviance

            results_list.append({
                "specification": specification,
                "skill_distance": skill_dist_label,
                "estimator": estimator_name,
                "R2": r2,
                "beta1_skill_dist": beta1,
                "se_beta1": se1,
                "p_beta1": p1,
                "n_obs": int(model.nobs),
            })

            print(f"    {estimator_name:8s}  R2={r2:.4f}  b1={beta1:+.6f}  (SE={se1:.6f}, p={p1:.4f})")

        except Exception as e:
            print(f"    {estimator_name:8s}  FAILED: {e}")
            results_list.append({
                "specification": specification,
                "skill_distance": skill_dist_label,
                "estimator": estimator_name,
                "R2": np.nan,
                "beta1_skill_dist": np.nan,
                "se_beta1": np.nan,
                "p_beta1": np.nan,
                "n_obs": len(y),
            })

    return fitted


# ====================================================================
# ADDITIONAL CHECKS (spec "Additional Checks" section)
# ====================================================================

def run_fixed_delta1(df, skill_dist_col, skill_dist_label, results_list):
    """Check 3a: Fix d1 = 1 by moving total_switches_out to LHS.

    OLS: regress log(1+switches) - log(1+total_switches_out) on remaining RHS
    PPML: use exposure=total_switches_out in GLM Poisson
    """
    X_df = pd.DataFrame({
        "skill_distance": df[skill_dist_col].values,
        "openings_share_dest": df["openings_share_dest"].values,
    })
    if "geographic_distance" in df.columns:
        X_df["geographic_distance"] = df["geographic_distance"].values

    y = df["switches"].values
    total_out = df["total_switches_out"].values

    spec_name = "fixed_delta1"

    # OLS: log(1+switches) - log(1+total_switches_out)
    try:
        y_adj = np.log1p(y) - np.log1p(total_out)
        X = sm.add_constant(X_df)
        model = sm.OLS(y_adj, X).fit(cov_type="HC1")
        beta1 = model.params["skill_distance"]
        se1 = model.bse["skill_distance"]
        p1 = model.pvalues["skill_distance"]
        results_list.append({
            "specification": spec_name,
            "skill_distance": skill_dist_label,
            "estimator": "OLS_log",
            "R2": model.rsquared,
            "beta1_skill_dist": beta1,
            "se_beta1": se1,
            "p_beta1": p1,
            "n_obs": int(model.nobs),
        })
        print(f"    OLS_log   R2={model.rsquared:.4f}  b1={beta1:+.6f}  (SE={se1:.6f}, p={p1:.4f})")
    except Exception as e:
        print(f"    OLS_log   FAILED: {e}")

    # PPML: with exposure = total_switches_out
    try:
        # exposure must be > 0 for log link; use max(1, total_out)
        exposure = np.maximum(total_out, 1).astype(float)
        X = sm.add_constant(X_df)
        model = GLM(y, X, family=Poisson(), exposure=exposure).fit(
            cov_type="HC1", maxiter=100
        )
        beta1 = model.params["skill_distance"]
        se1 = model.bse["skill_distance"]
        p1 = model.pvalues["skill_distance"]
        r2 = 1 - model.deviance / model.null_deviance
        results_list.append({
            "specification": spec_name,
            "skill_distance": skill_dist_label,
            "estimator": "PPML",
            "R2": r2,
            "beta1_skill_dist": beta1,
            "se_beta1": se1,
            "p_beta1": p1,
            "n_obs": int(model.nobs),
        })
        print(f"    PPML      R2={r2:.4f}  b1={beta1:+.6f}  (SE={se1:.6f}, p={p1:.4f})")
    except Exception as e:
        print(f"    PPML      FAILED: {e}")


def run_no_small_occs(df, skill_dist_col, skill_dist_label, results_list,
                      threshold):
    """Check 3b: Remove pairs where either origin or dest has < threshold employment."""
    mask = (df["emp_origin"] >= threshold) & (df["emp_dest"] >= threshold)
    df_filtered = df[mask]
    spec_name = f"no_small_occ_{threshold}"
    print(f"    (filtered to {len(df_filtered):,} pairs, threshold={threshold})")
    run_equation1(df_filtered, skill_dist_col, skill_dist_label, results_list,
                  specification=spec_name)


def run_year_fe(df_year, all_measures, results_list):
    """Check: Year fixed effects -- absorb aggregate time trends.

    df_year is the pre-built year-level expanded dataset with year dummies already
    attached. Runs equation (1) with year dummies for each skill distance variant.
    """
    for label, col in all_measures.items():
        print(f"\n  Skill Distance: {label}")
        run_equation1(df_year, col, label, results_list, specification="year_fe")


def run_zero_inflated(df, skill_dist_col, skill_dist_label, results_list):
    """Check 3c: Two-part model for zero-inflation.

    Part 1: Logit for P(switches > 0)
    Part 2: OLS on log(switches) for positive subsample
    """
    from statsmodels.discrete.discrete_model import Logit

    X_df = _build_rhs(df, skill_dist_col)
    y = df["switches"].values
    spec_name = "two_part"

    # Part 1: Logit on any switching
    try:
        y_binary = (y > 0).astype(int)
        X = sm.add_constant(X_df)
        logit_model = Logit(y_binary, X).fit(disp=0, cov_type="HC1")
        beta1_logit = logit_model.params["skill_distance"]
        se1_logit = logit_model.bse["skill_distance"]
        p1_logit = logit_model.pvalues["skill_distance"]
        r2_logit = logit_model.prsquared  # McFadden pseudo-R-squared

        results_list.append({
            "specification": f"{spec_name}_logit",
            "skill_distance": skill_dist_label,
            "estimator": "Logit",
            "R2": r2_logit,
            "beta1_skill_dist": beta1_logit,
            "se_beta1": se1_logit,
            "p_beta1": p1_logit,
            "n_obs": int(logit_model.nobs),
        })
        print(f"    Logit     R2={r2_logit:.4f}  b1={beta1_logit:+.6f}  (SE={se1_logit:.6f}, p={p1_logit:.4f})")
    except Exception as e:
        print(f"    Logit     FAILED: {e}")

    # Part 2: OLS on log(switches) for positive subsample
    try:
        pos_mask = y > 0
        y_pos = np.log(y[pos_mask])
        X_pos = sm.add_constant(X_df[pos_mask])
        ols_model = sm.OLS(y_pos, X_pos).fit(cov_type="HC1")
        beta1_ols = ols_model.params["skill_distance"]
        se1_ols = ols_model.bse["skill_distance"]
        p1_ols = ols_model.pvalues["skill_distance"]

        results_list.append({
            "specification": f"{spec_name}_positive",
            "skill_distance": skill_dist_label,
            "estimator": "OLS_log_positive",
            "R2": ols_model.rsquared,
            "beta1_skill_dist": beta1_ols,
            "se_beta1": se1_ols,
            "p_beta1": p1_ols,
            "n_obs": int(ols_model.nobs),
        })
        print(f"    OLS_pos   R2={ols_model.rsquared:.4f}  b1={beta1_ols:+.6f}  "
              f"(SE={se1_ols:.6f}, p={p1_ols:.4f}, n={int(ols_model.nobs):,})")
    except Exception as e:
        print(f"    OLS_pos   FAILED: {e}")


# ====================================================================
# MAIN
# ====================================================================

def main():
    print("Loading pairwise dataset...")
    df = pd.read_csv(DATA_DIR / "pairwise_dataset.csv",
                     dtype={"occ_origin": str, "occ_dest": str})
    print(f"  {len(df):,} pairs, {len(df.columns)} columns")

    # Drop rows with NaN in key columns
    key_cols = ["switches", "total_switches_out", "emp_origin", "emp_dest"]
    df = df.dropna(subset=key_cols)
    print(f"  After dropping NaN: {len(df):,}")

    # --- Part A: ML skill distances ---
    ml_distances, ml_importances = build_ml_skill_distances(df)

    # Add ML predictions as columns
    for name, preds in ml_distances.items():
        df[f"ml_dist_{name}"] = preds

    # --- Part B: Estimate equation (1) for each skill distance variant ---
    print("\n" + "=" * 70)
    print("PART B: Estimate Equation (1) -- Main Switching Model")
    print("=" * 70)

    results_list = []

    # Direct skill distance measures (computed in Step 4)
    direct_measures = {
        "euclidean": "euclidean_dist",
        "angular_separation": "angular_separation",
        "factor_analysis": "factor_dist",
    }

    # ML-based skill distance measures
    ml_measures = {f"ml_{name}": f"ml_dist_{name}" for name in ml_distances}

    all_measures = {**direct_measures, **ml_measures}

    baseline_models = {}
    for label, col in all_measures.items():
        print(f"\n  Skill Distance: {label}")
        baseline_models[label] = run_equation1(df, col, label, results_list)

    # --- Part C: Compare baseline results ---
    print("\n" + "=" * 70)
    print("PART C: Model Comparison (Baseline)")
    print("=" * 70)

    baseline = pd.DataFrame(results_list)

    print(f"\n{'Skill Distance':<25} {'Estimator':<10} {'R2':>8} {'b1':>12} {'p-value':>10}")
    print("-" * 70)
    for _, row in baseline.iterrows():
        print(f"{row['skill_distance']:<25} {row['estimator']:<10} "
              f"{row['R2']:>8.4f} {row['beta1_skill_dist']:>12.6f} {row['p_beta1']:>10.4f}")

    # Best specification by R-squared
    best_ols = baseline[baseline["estimator"] == "OLS_log"].sort_values("R2", ascending=False).iloc[0]
    best_ppml = baseline[baseline["estimator"] == "PPML"].sort_values("R2", ascending=False).iloc[0]
    print(f"\n  Best OLS:  {best_ols['skill_distance']} (R2={best_ols['R2']:.4f})")
    print(f"  Best PPML: {best_ppml['skill_distance']} (R2={best_ppml['R2']:.4f})")

    # --- Part D: Additional Checks (all 6 skill distance variants) ---
    print("\n" + "=" * 70)
    print("PART D: Additional Checks (all 6 skill distance variants)")
    print("=" * 70)

    for label, col in all_measures.items():
        print(f"\n  {'─' * 60}")
        print(f"  Skill Distance: {label}")
        print(f"  {'─' * 60}")

        # Check 3a: Fix d1 = 1
        print(f"\n  Check 3a: Fix d1 = 1")
        run_fixed_delta1(df, col, label, results_list)

        # Check 3b: Remove small occupations
        for threshold in [100, 500]:
            print(f"\n  Check 3b: No small occs (threshold={threshold})")
            run_no_small_occs(df, col, label, results_list, threshold)

        # Check 3c: Two-part model for zero-inflation
        print(f"\n  Check 3c: Two-part model (logit + OLS positives)")
        run_zero_inflated(df, col, label, results_list)

    # --- Part E: Year Fixed Effects ---
    print("\n" + "=" * 70)
    print("PART E: Year Fixed Effects")
    print("=" * 70)

    # Load year-level switching data (normalize occ codes to match pairwise dataset)
    sw_by_year = pd.read_csv(DATA_DIR / "switching_matrix_by_year.csv")
    for col in ["occ_origin", "occ_dest"]:
        sw_by_year[col] = sw_by_year[col].astype(float).astype(int).astype(str)
    out_by_year = pd.read_csv(DATA_DIR / "total_switchers_out_by_year.csv")
    out_by_year["occ"] = out_by_year["occ"].astype(float).astype(int).astype(str)

    years = sorted(sw_by_year["year"].unique())
    print(f"  Years: {years}")

    # Expand pooled pairwise df to year-level: cross-join pairs x years
    years_df = pd.DataFrame({"year": years})
    pair_cols = [c for c in df.columns if c != "switches" and c != "total_switches_out"]
    df_pairs = df[pair_cols].copy()
    df_pairs["_key"] = 1
    years_df["_key"] = 1
    df_year = df_pairs.merge(years_df, on="_key").drop(columns="_key")
    print(f"  Expanded to {len(df_year):,} pair-year rows")

    # Merge year-level switches
    df_year = df_year.merge(
        sw_by_year, on=["occ_origin", "occ_dest", "year"], how="left"
    )
    df_year["switches"] = df_year["switches"].fillna(0).astype(int)

    # Merge year-level total_switches_out (match occ -> occ_origin)
    df_year = df_year.merge(
        out_by_year.rename(columns={"occ": "occ_origin"}),
        on=["occ_origin", "year"], how="left"
    )
    df_year["total_switches_out"] = df_year["total_switches_out"].fillna(0)

    # Add year dummies (omit lowest year as base)
    base_year = min(years)
    for yr in years:
        if yr != base_year:
            df_year[f"year_{yr}"] = (df_year["year"] == yr).astype(int)
    print(f"  Year dummies: {[f'year_{yr}' for yr in years if yr != base_year]}")
    print(f"  Base year: {base_year}")

    run_year_fe(df_year, all_measures, results_list)

    # --- Save all results ---
    comparison = pd.DataFrame(results_list)
    comparison.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    n_baseline = len(baseline)
    n_checks = len(comparison) - n_baseline
    print(f"\n  Total results: {len(comparison)} ({n_baseline} baseline + {n_checks} additional checks)")

    # --- Save predictions from best ML model ---
    best_row = baseline.sort_values("R2", ascending=False).iloc[0]
    best_dist = best_row["skill_distance"]
    print(f"\n  Best overall: {best_dist} / {best_row['estimator']} (R2={best_row['R2']:.4f})")

    # Save all skill distance measures for each pair
    pred_cols = ["occ_origin", "occ_dest", "switches"]
    for label, col in all_measures.items():
        pred_cols.append(col)
    pred_df = df[pred_cols].copy()

    # Add Equation 1 fitted values (y-hat) from best PPML model for Equation 5 aggregation.
    # Use np.asarray(...) to force positional alignment: .mu may be a pandas Series
    # whose index wouldn't match pred_df's (post-dropna) index.
    best_ppml_label = best_ppml['skill_distance']
    if best_ppml_label in baseline_models and "PPML" in baseline_models[best_ppml_label]:
        best_ppml_model = baseline_models[best_ppml_label]["PPML"]
        pred_df["predicted_switches"] = np.asarray(best_ppml_model.mu)
        print(f"  Equation 1 fitted values from {best_ppml_label}/PPML saved as 'predicted_switches'")
    else:
        print(f"  WARNING: Could not extract PPML fitted values for {best_ppml_label}")

    pred_df.to_csv(OUT_DIR / "skill_portability_predictions.csv", index=False)
    print(f"  Predictions saved: {len(pred_df):,} pairs")

    # --- Save feature importance from ML models ---
    imp_records = []
    for model_name, imp in ml_importances.items():
        top = imp.sort_values(ascending=False).head(20)
        for feat, val in top.items():
            imp_records.append({"model": model_name, "feature": feat, "importance": val})
    imp_df = pd.DataFrame(imp_records)
    imp_df.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    print(f"\n  Feature importance saved (top 20 per ML model)")

    print(f"\nAll outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
