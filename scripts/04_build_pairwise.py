"""
Step 4: Build pairwise dataset with skill distances and switching outcomes.

Creates all directed occupation pairs, merges switching counts (raw, unweighted),
total switchers out of each origin, employment counts, and skill vectors.

Computes multiple skill distance metrics per the spec:
  - Euclidean distance (overall and per skill group)
  - Angular separation (arccos of cosine similarity)
  - Factor analysis distance (top 4 factors)
  - Per-dimension absolute differences (for ML models in Step 5)

Output: data/pairwise_dataset.csv
"""

import pandas as pd
import numpy as np
from itertools import product
from pathlib import Path
from sklearn.decomposition import FactorAnalysis

PROJECT = Path(__file__).parent.parent
DATA_DIR = PROJECT / "data"


def to_occ_str(s):
    """Normalize occupation codes to string-of-int (no floats like '10.0')."""
    return s.astype(float).astype(int).astype(str)


def main():
    print("Loading input data...")
    skill_vectors = pd.read_csv(DATA_DIR / "skill_vectors_by_census2018.csv", index_col=0)
    switching = pd.read_csv(DATA_DIR / "switching_matrix.csv")
    total_out = pd.read_csv(DATA_DIR / "total_switchers_out.csv")
    stayers = pd.read_csv(DATA_DIR / "stayer_counts.csv")
    emp_counts = pd.read_csv(DATA_DIR / "employment_counts.csv")

    # Normalize all occupation codes
    switching["occ_origin"] = to_occ_str(switching["occ_origin"])
    switching["occ_dest"] = to_occ_str(switching["occ_dest"])
    total_out["occ"] = to_occ_str(total_out["occ"])
    stayers["occ"] = to_occ_str(stayers["occ"])
    emp_counts["occ"] = to_occ_str(emp_counts["occ"])
    skill_vectors.index = skill_vectors.index.astype(float).astype(int).astype(str)

    # Pool employment across years (raw person counts)
    emp_pooled = emp_counts.groupby("occ")["employment"].sum().reset_index()

    skill_cols = list(skill_vectors.columns)

    # Identify skill groups by prefix
    skill_groups = {}
    for col in skill_cols:
        group = col.split("_")[0]  # "skill", "ability", "knowledge", "activity"
        skill_groups.setdefault(group, []).append(col)
    print(f"Skill groups: {', '.join(f'{g}: {len(c)}' for g, c in skill_groups.items())}")
    print(f"Total dimensions: {len(skill_cols)}")

    # --- Factor Analysis (top 4 factors) on occupation skill vectors ---
    print("\nFitting Factor Analysis (4 factors) on skill vectors...")
    fa = FactorAnalysis(n_components=4, random_state=42)
    fa_scores = fa.fit_transform(skill_vectors.values)
    explained_var = fa.noise_variance_
    print(f"  Factor Analysis fit on {skill_vectors.shape[0]} occupations x {skill_vectors.shape[1]} dimensions")

    # Store factor scores indexed by occupation
    fa_df = pd.DataFrame(fa_scores, index=skill_vectors.index,
                         columns=[f"factor_{i+1}" for i in range(4)])

    # Get occupations with both CPS data and skill vectors
    cps_occs = set(switching["occ_origin"]) | set(switching["occ_dest"]) | set(stayers["occ"])
    skill_occs = set(skill_vectors.index)
    valid_occs = sorted(cps_occs & skill_occs, key=int)
    print(f"\nOccupations: {len(cps_occs)} in CPS, {len(skill_occs)} with skills, {len(valid_occs)} in both")

    # Create all directed pairs
    print(f"Creating directed pairs for {len(valid_occs)} occupations...")
    pairs = [(o, d) for o, d in product(valid_occs, valid_occs) if o != d]
    print(f"  Total pairs: {len(pairs):,}")

    pairs_df = pd.DataFrame(pairs, columns=["occ_origin", "occ_dest"])

    # Merge switching counts (raw person counts)
    pairs_df = pairs_df.merge(switching, on=["occ_origin", "occ_dest"], how="left")
    pairs_df["switches"] = pairs_df["switches"].fillna(0).astype(int)

    # Merge total switchers out of origin (Switches_{o,d*})
    pairs_df = pairs_df.merge(
        total_out.rename(columns={"occ": "occ_origin"}),
        on="occ_origin", how="left"
    )
    pairs_df["total_switches_out"] = pairs_df["total_switches_out"].fillna(0).astype(int)

    # Merge stayer counts for origin
    pairs_df = pairs_df.merge(
        stayers.rename(columns={"occ": "occ_origin", "stayers": "stayers_origin"}),
        on="occ_origin", how="left"
    )

    # Merge employment counts (pooled)
    pairs_df = pairs_df.merge(
        emp_pooled.rename(columns={"occ": "occ_origin", "employment": "emp_origin"}),
        on="occ_origin", how="left"
    )
    pairs_df = pairs_df.merge(
        emp_pooled.rename(columns={"occ": "occ_dest", "employment": "emp_dest"}),
        on="occ_dest", how="left"
    )

    # Openings share: use Lightcast data if available, else fall back to emp share
    openings_path = DATA_DIR / "openings_share_by_census2018.csv"
    if openings_path.exists():
        print("Loading real openings share from Lightcast data...")
        openings = pd.read_csv(openings_path)
        openings["census_code"] = to_occ_str(openings["census_code"])
        pairs_df = pairs_df.merge(
            openings[["census_code", "openings_share"]].rename(
                columns={"census_code": "occ_dest", "openings_share": "openings_share_dest"}
            ),
            on="occ_dest", how="left"
        )
        matched = pairs_df["openings_share_dest"].notna().sum()
        print(f"  Matched {matched:,}/{len(pairs_df):,} pairs ({matched/len(pairs_df)*100:.1f}%)")
        # Fill unmatched with emp-based fallback
        total_emp = emp_pooled["employment"].sum()
        fallback = pairs_df["emp_dest"] / total_emp
        pairs_df["openings_share_dest"] = pairs_df["openings_share_dest"].fillna(fallback)
    else:
        print("No Lightcast openings data found, using emp-share placeholder")
        total_emp = emp_pooled["employment"].sum()
        pairs_df["openings_share_dest"] = pairs_df["emp_dest"] / total_emp

    # Compute skill distances (vectorized)
    print("Computing skill distances...")
    skill_matrix = skill_vectors.loc[valid_occs].values
    skill_idx = {occ: i for i, occ in enumerate(valid_occs)}

    origin_indices = [skill_idx[o] for o in pairs_df["occ_origin"]]
    dest_indices = [skill_idx[d] for d in pairs_df["occ_dest"]]

    origin_skills = skill_matrix[origin_indices]
    dest_skills = skill_matrix[dest_indices]

    # Absolute differences (for ML models in Step 5)
    abs_diffs = np.abs(origin_skills - dest_skills)

    # 1. Euclidean distance (overall)
    pairs_df["euclidean_dist"] = np.sqrt(np.sum(abs_diffs**2, axis=1))

    # Per-group Euclidean distances
    for group, cols in skill_groups.items():
        col_indices = [skill_cols.index(c) for c in cols]
        group_diffs = abs_diffs[:, col_indices]
        pairs_df[f"euclidean_{group}"] = np.sqrt(np.sum(group_diffs**2, axis=1))

    # 2. Angular separation (arccos of cosine similarity)
    norms_o = np.linalg.norm(origin_skills, axis=1)
    norms_d = np.linalg.norm(dest_skills, axis=1)
    # Avoid division by zero
    safe_norms_o = np.where(norms_o == 0, 1, norms_o)
    safe_norms_d = np.where(norms_d == 0, 1, norms_d)
    cosine_sim = np.sum(origin_skills * dest_skills, axis=1) / (safe_norms_o * safe_norms_d)
    # Clamp to [-1, 1] to avoid floating point issues with arccos
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
    pairs_df["cosine_sim"] = cosine_sim
    pairs_df["angular_separation"] = np.arccos(cosine_sim)

    # 3. Factor analysis distance (Euclidean distance in top-4 factor space)
    fa_matrix = fa_df.loc[valid_occs].values
    origin_factors = fa_matrix[origin_indices]
    dest_factors = fa_matrix[dest_indices]
    pairs_df["factor_dist"] = np.sqrt(np.sum((origin_factors - dest_factors)**2, axis=1))

    # Add per-dimension absolute differences as columns (for ML models)
    diff_df = pd.DataFrame(abs_diffs, columns=[f"diff_{c}" for c in skill_cols], index=pairs_df.index)
    pairs_df = pd.concat([pairs_df, diff_df], axis=1)

    # Geographic distance: merge if available
    geo_path = DATA_DIR / "geographic_distance.csv"
    if geo_path.exists():
        print("\nLoading geographic distance (Duncan index)...")
        geo = pd.read_csv(geo_path)
        geo["occ_origin"] = to_occ_str(geo["occ_origin"])
        geo["occ_dest"] = to_occ_str(geo["occ_dest"])
        n_before = len(pairs_df)
        pairs_df = pairs_df.merge(geo, on=["occ_origin", "occ_dest"], how="left")
        matched = pairs_df["geographic_distance"].notna().sum()
        print(f"  Matched {matched:,}/{n_before:,} pairs ({matched/n_before*100:.1f}%)")
        pairs_df["geographic_distance"] = pairs_df["geographic_distance"].fillna(
            pairs_df["geographic_distance"].median()
        )
    else:
        print("\nNo geographic distance data found (run 03b_build_geographic_distance.py first)")

    # Summary stats
    nonzero = (pairs_df["switches"] > 0).sum()
    print(f"\nDataset summary:")
    print(f"  Total pairs: {len(pairs_df):,}")
    print(f"  Pairs with nonzero switches: {nonzero:,} ({nonzero/len(pairs_df)*100:.1f}%)")
    print(f"  euclidean_dist:     mean={pairs_df['euclidean_dist'].mean():.4f}, std={pairs_df['euclidean_dist'].std():.4f}")
    print(f"  angular_separation: mean={pairs_df['angular_separation'].mean():.4f}, std={pairs_df['angular_separation'].std():.4f}")
    print(f"  factor_dist:        mean={pairs_df['factor_dist'].mean():.4f}, std={pairs_df['factor_dist'].std():.4f}")
    print(f"  cosine_sim:         mean={pairs_df['cosine_sim'].mean():.4f}, std={pairs_df['cosine_sim'].std():.4f}")

    # Save
    out_path = DATA_DIR / "pairwise_dataset.csv"
    pairs_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path} ({len(pairs_df):,} rows x {len(pairs_df.columns)} columns)")


if __name__ == "__main__":
    main()
