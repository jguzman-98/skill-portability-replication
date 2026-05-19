"""
Step 3b: Build geographic distance between occupations using Duncan overlap index.

Inputs:
  - ACS extract from IPUMS USA (CSV, gzipped) -- must be ACS 2021 1-year
    with variables: STATEFIP, PUMA, OCC, PERWT, EMPSTAT, AGE
  - Dorn PUMA->CZ crosswalk: raw/cw_puma2010_czone.zip

Output: data/geographic_distance.csv (occ_origin, occ_dest, geographic_distance)

Logic:
  1. Load ACS, filter to employed age 16-64
  2. Create PUMA identifier: STATEFIP * 100000 + PUMA (matches Dorn's puma2010 format)
  3. Merge with Dorn crosswalk, weight each person by PERWT * afactor
  4. Compute CZ-level employment by occupation: emp_{o,cz}
  5. Compute shares: emp_share_{o,cz} = emp_{o,cz} / emp_o
  6. Duncan index: geographic_distance_{o,d} = 0.5 * sum_CZ |share_{o,cz} - share_{d,cz}|
  7. Save all directed pairs

Why ACS 2021: ACS 2022+ switched to 2020-vintage PUMAs which don't match Dorn's
cw_puma2010_czone.dta. ACS 2021 is the most recent year with 2010 PUMAs.

Usage: python 03b_build_geographic_distance.py <path_to_acs_extract.csv.gz>
"""

import pandas as pd
import numpy as np
import sys
import zipfile
import io
from itertools import product
from pathlib import Path

PROJECT = Path(__file__).parent.parent
RAW_DIR = PROJECT / "raw"
DATA_DIR = PROJECT / "data"
DATA_DIR.mkdir(exist_ok=True)

DORN_CW_PATH = RAW_DIR / "cw_puma2010_czone.zip"

# Military / invalid occupation codes
INVALID_OCC = {0, 9920} | set(range(9800, 9841))


def load_dorn_crosswalk() -> pd.DataFrame:
    """Load Dorn PUMA 2010 -> Commuting Zone crosswalk from zip."""
    with zipfile.ZipFile(DORN_CW_PATH, "r") as z:
        with z.open("cw_puma2010_czone.dta") as f:
            cw = pd.read_stata(io.BytesIO(f.read()))
    cw["puma2010"] = cw["puma2010"].astype(int)
    cw["czone"] = cw["czone"].astype(int)
    print(f"  Dorn crosswalk: {len(cw)} PUMA-CZ mappings, "
          f"{cw['puma2010'].nunique()} PUMAs, {cw['czone'].nunique()} CZs")
    return cw


def main():
    print("=" * 70)
    print("Step 3b: Build Geographic Distance (Duncan Overlap Index)")
    print("=" * 70)

    if len(sys.argv) < 2:
        print("\nUsage: python 03b_build_geographic_distance.py <path_to_acs_extract.csv.gz>")
        print("\nThe ACS extract must be ACS 2021 1-year from IPUMS USA with:")
        print("  Variables: STATEFIP, PUMA, OCC, PERWT, EMPSTAT, AGE")
        print("  Format: CSV (gzipped)")
        print("\nWhy ACS 2021: ACS 2022+ uses 2020-vintage PUMAs which don't match")
        print("Dorn's cw_puma2010_czone.dta. ACS 2021 is the last year with 2010 PUMAs.")
        sys.exit(1)

    acs_path = Path(sys.argv[1])

    # --- Load ACS ---
    print(f"\nLoading ACS extract from {acs_path}...")
    acs = pd.read_csv(acs_path)
    acs.columns = acs.columns.str.upper()
    print(f"  Raw rows: {len(acs):,}")

    # Verify required columns
    required = {"STATEFIP", "PUMA", "OCC", "PERWT", "EMPSTAT", "AGE"}
    missing = required - set(acs.columns)
    if missing:
        print(f"  ERROR: Missing columns: {missing}")
        print(f"  Available: {sorted(acs.columns.tolist())}")
        sys.exit(1)

    # Filter to employed, age 16-64
    acs = acs[(acs["AGE"] >= 16) & (acs["AGE"] <= 64)]
    print(f"  After AGE 16-64: {len(acs):,}")

    acs = acs[acs["EMPSTAT"] == 1]  # ACS EMPSTAT=1 is employed
    print(f"  After EMPSTAT employed: {len(acs):,}")

    # Remove invalid occupation codes
    acs = acs[~acs["OCC"].isin(INVALID_OCC)]
    acs = acs[acs["OCC"] > 0]
    print(f"  After valid OCC: {len(acs):,}")

    # --- Create PUMA identifier matching Dorn format ---
    acs["puma2010"] = acs["STATEFIP"] * 100000 + acs["PUMA"]
    print(f"  Unique PUMAs: {acs['puma2010'].nunique()}")

    # --- Load Dorn crosswalk ---
    print("\nLoading Dorn PUMA -> CZ crosswalk...")
    dorn_cw = load_dorn_crosswalk()

    # --- Merge ACS with Dorn crosswalk ---
    print("\nMerging ACS with CZ crosswalk...")
    n_before = len(acs)
    acs = acs.merge(dorn_cw, on="puma2010", how="inner")
    n_after = len(acs)
    print(f"  Matched: {n_after:,}/{n_before:,} person-rows "
          f"({n_after/n_before*100:.1f}%)")
    print(f"  Note: Some rows expand (PUMA spans multiple CZs)")

    # Weight each person by PERWT * afactor
    acs["weight"] = acs["PERWT"] * acs["afactor"]

    # Normalize occupation codes
    acs["occ"] = acs["OCC"].astype(int).astype(str)

    # --- Compute CZ-level employment by occupation ---
    print("\nComputing CZ-level employment by occupation...")
    emp_occ_cz = (
        acs.groupby(["occ", "czone"])["weight"]
        .sum()
        .reset_index()
        .rename(columns={"weight": "emp_occ_cz"})
    )
    print(f"  {len(emp_occ_cz):,} occ-CZ cells")
    print(f"  {emp_occ_cz['occ'].nunique()} occupations, "
          f"{emp_occ_cz['czone'].nunique()} commuting zones")

    # Total employment per occupation
    emp_occ = emp_occ_cz.groupby("occ")["emp_occ_cz"].sum().reset_index()
    emp_occ.columns = ["occ", "emp_total"]

    # Merge to get shares
    emp_occ_cz = emp_occ_cz.merge(emp_occ, on="occ")
    emp_occ_cz["share"] = emp_occ_cz["emp_occ_cz"] / emp_occ_cz["emp_total"]

    # --- Build shares matrix (vectorized Duncan index) ---
    print("\nBuilding occupation x CZ shares matrix...")
    occs = sorted(emp_occ_cz["occ"].unique(), key=lambda x: int(x))
    czs = sorted(emp_occ_cz["czone"].unique())
    n_occs = len(occs)
    n_czs = len(czs)
    print(f"  Matrix: {n_occs} occupations x {n_czs} commuting zones")

    # Create pivot table: occupations x CZs with shares
    occ_idx = {o: i for i, o in enumerate(occs)}
    cz_idx = {c: i for i, c in enumerate(czs)}

    shares_matrix = np.zeros((n_occs, n_czs))
    for _, row in emp_occ_cz.iterrows():
        i = occ_idx[row["occ"]]
        j = cz_idx[row["czone"]]
        shares_matrix[i, j] = row["share"]

    # Verify rows sum to ~1
    row_sums = shares_matrix.sum(axis=1)
    print(f"  Row sums: min={row_sums.min():.6f}, max={row_sums.max():.6f}, "
          f"mean={row_sums.mean():.6f}")

    # --- Compute Duncan index for all directed pairs ---
    print(f"\nComputing Duncan index for {n_occs * (n_occs - 1):,} directed pairs...")

    # Vectorized: for each pair (o, d), compute 0.5 * sum|share_o - share_d|
    # Use broadcasting: shape (n_occs, 1, n_czs) - (1, n_occs, n_czs) -> (n_occs, n_occs, n_czs)
    # This could be memory-intensive for large n_occs. With ~525 occs x ~700 CZs,
    # the full tensor is ~525^2 x 700 x 8 bytes ~ 1.5 GB. Process in chunks.

    records = []
    chunk_size = 50  # Process 50 origins at a time

    for start in range(0, n_occs, chunk_size):
        end = min(start + chunk_size, n_occs)
        chunk = shares_matrix[start:end]  # (chunk, n_czs)

        # Broadcast: (chunk, 1, n_czs) - (1, n_occs, n_czs)
        abs_diff = np.abs(chunk[:, np.newaxis, :] - shares_matrix[np.newaxis, :, :])
        duncan = 0.5 * abs_diff.sum(axis=2)  # (chunk, n_occs)

        for i_local in range(end - start):
            i_global = start + i_local
            occ_o = occs[i_global]
            for j in range(n_occs):
                if i_global == j:
                    continue
                records.append({
                    "occ_origin": occ_o,
                    "occ_dest": occs[j],
                    "geographic_distance": duncan[i_local, j],
                })

        if (end % 100 == 0) or end == n_occs:
            print(f"  Processed {end}/{n_occs} origins...")

    geo_dist = pd.DataFrame(records)

    # --- Summary stats ---
    print(f"\nGeographic distance summary:")
    print(f"  Pairs: {len(geo_dist):,}")
    print(f"  Mean: {geo_dist['geographic_distance'].mean():.4f}")
    print(f"  Std:  {geo_dist['geographic_distance'].std():.4f}")
    print(f"  Min:  {geo_dist['geographic_distance'].min():.4f}")
    print(f"  Max:  {geo_dist['geographic_distance'].max():.4f}")
    print(f"  Values in [0, 1]: {(geo_dist['geographic_distance'] >= 0).all() and (geo_dist['geographic_distance'] <= 1).all()}")

    # Top 5 most geographically distant pairs
    top5 = geo_dist.nlargest(5, "geographic_distance")
    print(f"\n  Top 5 most geographically distant pairs:")
    for _, row in top5.iterrows():
        print(f"    {row['occ_origin']} -> {row['occ_dest']}: {row['geographic_distance']:.4f}")

    # Save
    out_path = DATA_DIR / "geographic_distance.csv"
    geo_dist.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path} ({len(geo_dist):,} rows)")


if __name__ == "__main__":
    main()
