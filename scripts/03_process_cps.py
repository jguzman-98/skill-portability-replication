"""
Step 3: Process CPS switching data from IPUMS extract.

Loads the CPS extract (with OCC variable), filters to 2020+ ASEC respondents,
identifies switchers vs stayers using exact occupation code comparison
(no adjacent-code filtering), and builds directional switching counts
using raw (unweighted) person counts.

Weighted versions of employment and state employment are also produced
for use in the portability aggregation (equations 6-7 in the spec).

Outputs:
  data/switching_matrix.csv              -- (occ_origin, occ_dest, switches) raw counts
  data/total_switchers_out.csv           -- (occ, total_switches_out) raw counts per origin
  data/switching_matrix_by_year.csv      -- (occ_origin, occ_dest, year, switches) year-level
  data/total_switchers_out_by_year.csv   -- (occ, year, total_switches_out) year-level
  data/stayer_counts.csv                 -- (occ, stayers) raw counts
  data/employment_counts.csv             -- (occ, year, employment) raw counts
  data/employment_counts_weighted.csv    -- (occ, year, weighted_employment) for eq 6-7
  data/state_employment.csv              -- (statefip, occ, weighted_employment) for geographic distance

Usage: python 03_process_cps.py <path_to_cps_extract.csv.gz>
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
DATA_DIR = PROJECT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Military / invalid occupation codes to exclude
# 9800-9840 covers all armed forces codes; 9920 = unemployment
INVALID_OCC = {0, 9920} | set(range(9800, 9841))


def main():
    # Accept CPS extract path as command-line argument
    if len(sys.argv) < 2:
        print("Usage: python 03_process_cps.py <path_to_cps_extract.csv.gz>")
        sys.exit(1)

    cps_path = Path(sys.argv[1])
    print(f"Loading CPS extract from {cps_path}...")
    df = pd.read_csv(cps_path)
    print(f"  Raw rows: {len(df):,}")

    # Standardize column names to uppercase
    df.columns = df.columns.str.upper()

    # Filter to YEAR >= 2020
    df = df[df["YEAR"] >= 2020]
    print(f"  After YEAR >= 2020: {len(df):,}")

    # Filter to ASEC supplement respondents
    df = df[df["ASECFLAG"] > 0]
    print(f"  After ASECFLAG > 0: {len(df):,}")

    # Filter to age 16-64
    df = df[(df["AGE"] >= 16) & (df["AGE"] <= 64)]
    print(f"  After AGE 16-64: {len(df):,}")

    # Filter to currently employed (EMPSTAT 10=employed at work, 12=has job not at work)
    df = df[df["EMPSTAT"].isin([10, 12])]
    print(f"  After EMPSTAT employed: {len(df):,}")

    # Filter to valid OCC and OCCLY
    df = df[~df["OCC"].isin(INVALID_OCC) & ~df["OCCLY"].isin(INVALID_OCC)]
    df = df[(df["OCC"] != 0) & (df["OCCLY"] != 0)]
    print(f"  After valid OCC/OCCLY: {len(df):,}")

    # --- Classify switchers vs stayers (exact code comparison, no adjacent-code filter) ---
    df["is_switcher"] = df["OCC"] != df["OCCLY"]

    switchers = df[df["is_switcher"]].copy()
    stayers = df[~df["is_switcher"]].copy()

    # --- Switching matrix (raw person counts, unweighted) ---
    switching_matrix = (
        switchers.groupby(["OCCLY", "OCC"])
        .size()
        .reset_index(name="switches")
        .rename(columns={"OCCLY": "occ_origin", "OCC": "occ_dest"})
    )
    print(f"\nSwitching matrix: {len(switching_matrix):,} directed pairs")
    print(f"  Total switchers: {switching_matrix['switches'].sum():,}")

    # --- Total switchers out of each origin (Switches_{o,d*} in the spec) ---
    total_switches_out = (
        switching_matrix.groupby("occ_origin")["switches"]
        .sum()
        .reset_index()
        .rename(columns={"occ_origin": "occ", "switches": "total_switches_out"})
    )
    print(f"\nTotal switchers out: {len(total_switches_out):,} occupations")

    # --- Switching matrix by year (for year fixed effects) ---
    switching_matrix_by_year = (
        switchers.groupby(["OCCLY", "OCC", "YEAR"])
        .size()
        .reset_index(name="switches")
        .rename(columns={"OCCLY": "occ_origin", "OCC": "occ_dest", "YEAR": "year"})
    )
    print(f"\nSwitching matrix by year: {len(switching_matrix_by_year):,} directed pair-year cells")

    total_out_by_year = (
        switching_matrix_by_year.groupby(["occ_origin", "year"])["switches"]
        .sum()
        .reset_index()
        .rename(columns={"occ_origin": "occ", "switches": "total_switches_out"})
    )
    print(f"Total switchers out by year: {len(total_out_by_year):,} occ-year cells")

    # --- Stayer counts (raw person counts) ---
    stayer_counts = (
        stayers.groupby("OCC")
        .size()
        .reset_index(name="stayers")
        .rename(columns={"OCC": "occ"})
    )
    print(f"\nStayer counts: {len(stayer_counts):,} occupations")
    print(f"  Total stayers: {stayer_counts['stayers'].sum():,}")

    # --- Employment counts by occ x year (raw person counts) ---
    emp_counts = (
        df.groupby(["OCC", "YEAR"])
        .size()
        .reset_index(name="employment")
        .rename(columns={"OCC": "occ", "YEAR": "year"})
    )
    print(f"\nEmployment counts (unweighted): {len(emp_counts):,} occ-year cells")

    # --- Employment counts by occ x year (weighted, for portability aggregation eqs 6-7) ---
    emp_counts_weighted = (
        df.groupby(["OCC", "YEAR"])["ASECWT"]
        .sum()
        .reset_index()
        .rename(columns={"OCC": "occ", "YEAR": "year", "ASECWT": "weighted_employment"})
    )
    print(f"Employment counts (weighted): {len(emp_counts_weighted):,} occ-year cells")

    # --- State employment counts (weighted, for geographic distance) ---
    state_emp = (
        df.groupby(["STATEFIP", "OCC"])["ASECWT"]
        .sum()
        .reset_index()
        .rename(columns={"STATEFIP": "statefip", "OCC": "occ", "ASECWT": "weighted_employment"})
    )
    print(f"State employment: {len(state_emp):,} state-occ cells")

    # --- Sanity checks ---
    total_switchers = switching_matrix["switches"].sum()
    total_stayers = stayer_counts["stayers"].sum()
    switch_rate = total_switchers / (total_switchers + total_stayers) * 100

    print(f"\nSanity checks:")
    print(f"  Total persons: {len(df):,}")
    print(f"  Switcher share: {switch_rate:.1f}%")
    print(f"  Unique occupations (OCC): {df['OCC'].nunique()}")
    print(f"  Unique occupations (OCCLY): {df['OCCLY'].nunique()}")

    # --- Save ---
    switching_matrix.to_csv(DATA_DIR / "switching_matrix.csv", index=False)
    total_switches_out.to_csv(DATA_DIR / "total_switchers_out.csv", index=False)
    switching_matrix_by_year.to_csv(DATA_DIR / "switching_matrix_by_year.csv", index=False)
    total_out_by_year.to_csv(DATA_DIR / "total_switchers_out_by_year.csv", index=False)
    stayer_counts.to_csv(DATA_DIR / "stayer_counts.csv", index=False)
    emp_counts.to_csv(DATA_DIR / "employment_counts.csv", index=False)
    emp_counts_weighted.to_csv(DATA_DIR / "employment_counts_weighted.csv", index=False)
    state_emp.to_csv(DATA_DIR / "state_employment.csv", index=False)
    print(f"\nAll outputs saved to {DATA_DIR}/")


if __name__ == "__main__":
    main()
