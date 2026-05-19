"""
Step 2: Build Census 2018 -> O*NET crosswalk with skill vectors.

Parses the messy Census crosswalk CSV, maps Census 2018 codes to SOC 2018,
handles SOC wildcards (e.g., 13-20XX -> matches all 13-20** O*NET codes),
averages skill vectors for Census codes mapping to multiple SOCs,
and produces a final skill-vector file keyed by Census 2018 code.

Output: data/skill_vectors_by_census2018.csv
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

PROJECT = Path(__file__).parent.parent
RAW_DIR = PROJECT / "raw"
DATA_DIR = PROJECT / "data"
DATA_DIR.mkdir(exist_ok=True)

CROSSWALK_PATH = RAW_DIR / "census_2018_occ_crosswalk.csv"
SKILL_MATRIX_PATH = DATA_DIR / "onet_skill_matrix.csv"


def parse_crosswalk(path: Path) -> pd.DataFrame:
    """Parse the messy crosswalk CSV, extracting Census code -> SOC code mappings."""
    df = pd.read_csv(path, header=None, dtype=str)
    # Columns: 0=empty/category, 1=title, 2=census_code, 3=soc_code
    # Actual data rows have a 4-digit census code in column 2

    records = []
    for _, row in df.iterrows():
        census_code = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ""
        soc_code = str(row.iloc[3]).strip() if pd.notna(row.iloc[3]) else ""
        title = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""

        # Valid census codes are 4-digit numbers
        if not re.match(r"^\d{4}$", census_code):
            continue
        # Skip rows with no SOC code or "none"
        if not soc_code or soc_code.lower() == "none":
            continue

        records.append({
            "census_code": census_code,
            "census_title": title,
            "soc_code": soc_code,
        })

    result = pd.DataFrame(records)
    print(f"Parsed {len(result)} Census -> SOC mappings")
    return result


def resolve_soc_to_onet(soc_code: str, onet_socs: list[str]) -> list[str]:
    """Resolve a SOC code (possibly with wildcards) to matching O*NET SOC codes."""
    # Replace X with wildcard pattern for regex matching
    if "X" in soc_code:
        # e.g., "13-20XX" -> matches "13-20.."
        pattern = soc_code.replace("X", ".")
        matches = [s for s in onet_socs if re.match(f"^{pattern}$", s)]
        return matches

    # Exact match first
    if soc_code in onet_socs:
        return [soc_code]

    # SOC codes like "13-2070" may map to O*NET "13-2071", "13-2072"
    # Try prefix matching (drop last digit and match)
    prefix = soc_code[:-1]  # e.g., "13-2070" -> "13-207"
    matches = [s for s in onet_socs if s.startswith(prefix)]
    if matches:
        return matches

    # Try broader prefix (first 5 chars, e.g., "13-20")
    prefix = soc_code[:5]
    matches = [s for s in onet_socs if s.startswith(prefix)]
    if matches:
        return matches

    return []


def main():
    # Load skill matrix
    skill_matrix = pd.read_csv(SKILL_MATRIX_PATH, index_col=0)
    onet_socs = list(skill_matrix.index)
    print(f"Loaded O*NET skill matrix: {skill_matrix.shape[0]} SOCs x {skill_matrix.shape[1]} dimensions")

    # Parse crosswalk
    crosswalk = parse_crosswalk(CROSSWALK_PATH)

    # Resolve each Census code to O*NET SOC codes and average skill vectors
    results = []
    matched = 0
    unmatched = []

    for _, row in crosswalk.iterrows():
        census_code = row["census_code"]
        soc_code = row["soc_code"]

        matches = resolve_soc_to_onet(soc_code, onet_socs)

        if not matches:
            unmatched.append((census_code, row["census_title"], soc_code))
            continue

        # Average skill vectors across matching O*NET SOCs
        skill_vec = skill_matrix.loc[matches].mean()
        skill_vec.name = census_code
        results.append(skill_vec)
        matched += 1

    # Build output dataframe
    skill_by_census = pd.DataFrame(results)
    skill_by_census.index.name = "census_code"

    # Report
    total = len(crosswalk)
    match_rate = matched / total * 100
    print(f"\nMatch results:")
    print(f"  Matched: {matched}/{total} ({match_rate:.1f}%)")
    print(f"  Unmatched: {len(unmatched)}")
    if unmatched:
        print("\n  Unmatched occupations:")
        for code, title, soc in unmatched:
            print(f"    Census {code}: {title} (SOC: {soc})")

    # Save
    out_path = DATA_DIR / "skill_vectors_by_census2018.csv"
    skill_by_census.to_csv(out_path)
    print(f"\nSaved to {out_path}")
    print(f"Final shape: {skill_by_census.shape[0]} Census codes x {skill_by_census.shape[1]} dimensions")
    print(f"NaN count: {skill_by_census.isna().sum().sum()}")


if __name__ == "__main__":
    main()
