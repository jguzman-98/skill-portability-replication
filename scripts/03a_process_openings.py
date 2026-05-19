"""
Step 3a: Process Lightcast job postings -> openings share by Census 2018 code.

Input:  raw/lightcast_time_series.csv (Lightcast data, SOC 2021 5-digit)
Output: data/openings_share_by_census2018.csv (census_code, total_postings, openings_share)

Logic:
  1. Filter to year=2023, sum total_postings by soc_2021_5 across months
  2. Map SOC 2021 -> Census 2018 using inverted crosswalk (SOC -> Census)
     SOC 2021 ~ SOC 2018 for most codes (minor revision)
  3. Compute openings_share = occ_postings / grand_total_postings
  4. Save with match rate diagnostics
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

PROJECT = Path(__file__).parent.parent
RAW_DIR = PROJECT / "raw"
DATA_DIR = PROJECT / "data"
DATA_DIR.mkdir(exist_ok=True)

LIGHTCAST_PATH = RAW_DIR / "lightcast_time_series.csv"
CROSSWALK_PATH = RAW_DIR / "census_2018_occ_crosswalk.csv"


def parse_crosswalk(path: Path) -> pd.DataFrame:
    """Parse the messy Census crosswalk CSV (same as 02_build_crosswalk.py)."""
    df = pd.read_csv(path, header=None, dtype=str)
    records = []
    for _, row in df.iterrows():
        census_code = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ""
        soc_code = str(row.iloc[3]).strip() if pd.notna(row.iloc[3]) else ""
        title = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
        if not re.match(r"^\d{4}$", census_code):
            continue
        if not soc_code or soc_code.lower() == "none":
            continue
        records.append({
            "census_code": census_code,
            "census_title": title,
            "soc_code": soc_code,
        })
    return pd.DataFrame(records)


def build_soc_to_census_map(crosswalk: pd.DataFrame) -> dict:
    """Build inverted map: SOC 6-digit prefix -> list of Census codes.

    The crosswalk maps Census -> SOC. We invert it: for each SOC code
    (including wildcards like 13-20XX), expand to 6-digit prefixes and
    map back to the Census code.
    """
    soc_to_census = {}
    for _, row in crosswalk.iterrows():
        census_code = row["census_code"]
        soc_code = row["soc_code"]
        # Store the exact SOC -> Census mapping
        # Strip wildcards for the map key (use first 7 chars = XX-XXXX)
        key = soc_code.replace("X", "")
        # Use the full SOC code as key
        soc_to_census.setdefault(soc_code, []).append(census_code)
    return soc_to_census


def match_soc_to_census(soc_5digit: str, crosswalk: pd.DataFrame) -> list:
    """Match a Lightcast SOC 2021 5-digit code to Census 2018 codes.

    Strategy:
      1. Exact match on SOC code in crosswalk
      2. Prefix match: SOC 2021 XX-XXXX -> try matching XX-XXX0, XX-XXXX
      3. Broader prefix: first 5 chars (XX-XX)
    """
    soc = soc_5digit.strip()

    # Try exact match
    exact = crosswalk[crosswalk["soc_code"] == soc]["census_code"].tolist()
    if exact:
        return exact

    # Try matching with trailing 0 (some crosswalk codes end in 0)
    soc_with_0 = soc + "0"
    exact0 = crosswalk[crosswalk["soc_code"] == soc_with_0]["census_code"].tolist()
    if exact0:
        return exact0

    # Try prefix match (drop last digit)
    prefix = soc[:-1]
    matches = crosswalk[crosswalk["soc_code"].str.startswith(prefix)]["census_code"].tolist()
    if matches:
        return matches

    # Try broader prefix (first 5 chars, e.g., "13-20")
    broad = soc[:5]
    matches = crosswalk[crosswalk["soc_code"].str.startswith(broad)]["census_code"].tolist()
    if matches:
        return matches

    # Try wildcard match: some crosswalk entries have XX like "13-20XX"
    wildcard_matches = []
    for _, row in crosswalk.iterrows():
        if "X" in row["soc_code"]:
            pattern = row["soc_code"].replace("X", ".")
            if re.match(f"^{pattern[:len(soc)]}",  soc):
                wildcard_matches.append(row["census_code"])
    if wildcard_matches:
        return wildcard_matches

    return []


def main():
    print("=" * 70)
    print("Step 3a: Process Lightcast Job Postings -> Openings Share")
    print("=" * 70)

    # Load Lightcast data
    print(f"\nLoading Lightcast data from {LIGHTCAST_PATH}...")
    lc = pd.read_csv(LIGHTCAST_PATH)
    print(f"  {len(lc):,} rows, {lc['soc_2021_5'].nunique()} SOC codes, "
          f"years {sorted(lc['year'].unique())}")

    # Filter to year 2023
    lc_2023 = lc[lc["year"] == 2023].copy()
    print(f"  Year 2023: {len(lc_2023):,} rows, {lc_2023['soc_2021_5'].nunique()} SOC codes")

    # Sum total_postings by SOC across months
    postings_by_soc = (
        lc_2023.groupby("soc_2021_5")["total_postings"]
        .sum()
        .reset_index()
        .rename(columns={"soc_2021_5": "soc_code"})
    )
    print(f"  Aggregated to {len(postings_by_soc)} SOC codes, "
          f"{postings_by_soc['total_postings'].sum():,.0f} total postings")

    # Parse crosswalk
    crosswalk = parse_crosswalk(CROSSWALK_PATH)
    print(f"\nCrosswalk: {len(crosswalk)} Census -> SOC mappings")

    # Match each Lightcast SOC to Census codes
    records = []
    matched_socs = 0
    unmatched_socs = []

    for _, row in postings_by_soc.iterrows():
        soc = row["soc_code"]
        postings = row["total_postings"]

        census_codes = match_soc_to_census(soc, crosswalk)

        if not census_codes:
            unmatched_socs.append(soc)
            continue

        matched_socs += 1
        # Split postings evenly across matched Census codes
        per_census = postings / len(census_codes)
        for cc in census_codes:
            records.append({
                "census_code": cc,
                "total_postings": per_census,
            })

    # Aggregate by Census code (multiple SOCs may map to same Census code)
    result = pd.DataFrame(records)
    result = result.groupby("census_code")["total_postings"].sum().reset_index()

    # Compute openings share
    grand_total = result["total_postings"].sum()
    result["openings_share"] = result["total_postings"] / grand_total

    # Match rate diagnostics
    total_socs = len(postings_by_soc)
    matched_postings = postings_by_soc[~postings_by_soc["soc_code"].isin(unmatched_socs)]["total_postings"].sum()
    total_postings = postings_by_soc["total_postings"].sum()

    print(f"\nMatch results:")
    print(f"  SOC codes matched: {matched_socs}/{total_socs} ({matched_socs/total_socs*100:.1f}%)")
    print(f"  Postings matched: {matched_postings:,.0f}/{total_postings:,.0f} "
          f"({matched_postings/total_postings*100:.1f}%)")
    print(f"  Census codes with postings: {len(result)}")

    if unmatched_socs:
        print(f"\n  Unmatched SOC codes ({len(unmatched_socs)}):")
        for s in unmatched_socs[:10]:
            name = lc_2023[lc_2023["soc_2021_5"] == s]["soc_2021_5_name"].iloc[0] if len(lc_2023[lc_2023["soc_2021_5"] == s]) > 0 else "?"
            print(f"    {s}: {name}")
        if len(unmatched_socs) > 10:
            print(f"    ... and {len(unmatched_socs) - 10} more")

    # Summary stats
    print(f"\nOpenings share summary:")
    print(f"  Mean: {result['openings_share'].mean():.6f}")
    print(f"  Std:  {result['openings_share'].std():.6f}")
    print(f"  Max:  {result['openings_share'].max():.6f} "
          f"(Census {result.loc[result['openings_share'].idxmax(), 'census_code']})")
    print(f"  Sum:  {result['openings_share'].sum():.6f} (should be ~1.0)")

    # Save
    out_path = DATA_DIR / "openings_share_by_census2018.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path} ({len(result)} rows)")


if __name__ == "__main__":
    main()
