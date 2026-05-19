"""
Step 6: Map AI Exposure (Eloundou et al. 2023) to Census 2018 codes.

Maps AI exposure scores from SOC codes to Census 2018 codes using the
same crosswalk resolution strategy as 02_build_crosswalk.py.

Input:
  - raw/ai_exposure.csv (Eloundou et al. 2023 AI exposure by O*NET SOC)
  - raw/census_2018_occ_crosswalk.csv

Output:
  - data/ai_exposure_by_census2018.csv
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

PROJECT = Path(__file__).parent.parent
RAW_DIR = PROJECT / "raw"
DATA_DIR = PROJECT / "data"
DATA_DIR.mkdir(exist_ok=True)

AI_EXPOSURE_PATH = RAW_DIR / "ai_exposure.csv"
CROSSWALK_PATH = RAW_DIR / "census_2018_occ_crosswalk.csv"

EXPOSURE_COLS = ["gpt4_beta", "automation", "human_beta"]


def norm_code(s) -> str:
    """Normalize a Census code to str(int(float(x))) -- project convention."""
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return str(s).strip()


def parse_crosswalk(path: Path) -> pd.DataFrame:
    """Parse the messy Census crosswalk CSV (reused from 02_build_crosswalk.py)."""
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


def resolve_soc_to_ai(soc_code: str, ai_socs: list[str]) -> list[str]:
    """Resolve a crosswalk SOC (possibly wildcarded) to matching AI-exposure SOC codes.

    Same three-tier strategy as 02_build_crosswalk.py:
      1. Wildcard (X) -> regex match
      2. Exact match
      3. Prefix match (drop last digit, then first 5 chars)
    """
    if "X" in soc_code:
        pattern = soc_code.replace("X", ".")
        return [s for s in ai_socs if re.match(f"^{pattern}$", s)]
    if soc_code in ai_socs:
        return [soc_code]
    prefix = soc_code[:-1]
    matches = [s for s in ai_socs if s.startswith(prefix)]
    if matches:
        return matches
    prefix = soc_code[:5]
    matches = [s for s in ai_socs if s.startswith(prefix)]
    if matches:
        return matches
    return []


def build_ai_exposure_by_census():
    """Map AI exposure from SOC -> Census 2018 and save."""
    print("=" * 70)
    print("Map AI Exposure -> Census 2018 codes")
    print("=" * 70)

    # Load AI exposure and collapse O*NET detail -> 6-digit SOC
    ai_raw = pd.read_csv(AI_EXPOSURE_PATH)
    print(f"AI exposure raw: {len(ai_raw)} O*NET rows, "
          f"{ai_raw['OCC_CODE'].nunique()} unique SOC codes")

    ai_soc = ai_raw.groupby("OCC_CODE")[EXPOSURE_COLS].mean().reset_index()
    ai_soc_list = list(ai_soc["OCC_CODE"])
    print(f"After SOC-level collapse: {len(ai_soc)} SOC codes")

    # Parse crosswalk
    crosswalk = parse_crosswalk(CROSSWALK_PATH)
    print(f"Crosswalk: {len(crosswalk)} Census -> SOC mappings")

    # Resolve each Census code -> matching SOC codes -> average AI exposure
    records = []
    matched = 0
    unmatched = []

    for _, row in crosswalk.iterrows():
        census_code = row["census_code"]
        soc_code = row["soc_code"]
        matches = resolve_soc_to_ai(soc_code, ai_soc_list)
        if not matches:
            unmatched.append((census_code, row["census_title"], soc_code))
            continue
        exposure_vals = ai_soc[ai_soc["OCC_CODE"].isin(matches)][EXPOSURE_COLS].mean()
        rec = {"census_code": norm_code(census_code)}
        for col in EXPOSURE_COLS:
            rec[col] = exposure_vals[col]
        records.append(rec)
        matched += 1

    result = pd.DataFrame(records)
    print(f"\nMatch results: {matched}/{len(crosswalk)} "
          f"({matched / len(crosswalk) * 100:.1f}%)")
    if unmatched:
        print(f"Unmatched ({len(unmatched)}):")
        for code, title, soc in unmatched:
            print(f"  Census {code}: {title} (SOC: {soc})")

    out_path = DATA_DIR / "ai_exposure_by_census2018.csv"
    result.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(result)} rows)")
    return result


def main():
    build_ai_exposure_by_census()
    print("\nDone.")


if __name__ == "__main__":
    main()
