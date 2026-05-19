"""
Step 1: Process O*NET data into a unified skill matrix.

Loads Skills, Abilities, and Knowledge from O*NET 30.1 (Level scale only),
plus Work Activities (both Level and Importance scales), aggregates
detailed SOC codes to 6-digit, pivots wide, and min-max normalizes
each dimension.

Output: data/onet_skill_matrix.csv (~500 SOC codes x ~202 dimensions)
  - 35 skills, 52 abilities, 33 knowledge (LV scale) = 120
  - 41 work activities x 2 scales (LV + IM)           =  82
  - Total                                              = 202
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).parent.parent
RAW_DIR = PROJECT / "raw"
DATA_DIR = PROJECT / "data"
DATA_DIR.mkdir(exist_ok=True)

ONET_DIR = RAW_DIR / "onet"

# Files using Level (LV) scale only
FILES_LV = {
    "skill": ONET_DIR / "Skills.txt",
    "ability": ONET_DIR / "Abilities.txt",
    "knowledge": ONET_DIR / "Knowledge.txt",
}

# Work Activities uses both Level (LV) and Importance (IM) scales
WORK_ACTIVITIES_PATH = ONET_DIR / "Work Activities.txt"


def load_onet_file(path: Path, prefix: str, scales: list[str] = None) -> pd.DataFrame:
    """Load one O*NET file, filter to specified scales, clean SOC codes.

    When multiple scales are given (e.g. LV + IM for Work Activities),
    the scale abbreviation is embedded in the dimension name to keep
    them distinct.
    """
    if scales is None:
        scales = ["LV"]
    df = pd.read_csv(path, sep="\t")
    df = df[(df["Scale ID"].isin(scales)) & (df["Recommend Suppress"] == "N")].copy()
    # Strip .XX suffix to get 6-digit SOC
    df["soc6"] = df["O*NET-SOC Code"].str.replace(r"\.\d+$", "", regex=True)
    # Build dimension name: include scale suffix when using multiple scales
    element_clean = df["Element Name"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    if len(scales) > 1:
        df["dimension"] = prefix + "_" + df["Scale ID"].str.lower() + "_" + element_clean
    else:
        df["dimension"] = prefix + "_" + element_clean
    df["value"] = df["Data Value"].astype(float)
    return df[["soc6", "dimension", "value"]]


def main():
    # Load Skills, Abilities, Knowledge (LV scale only)
    frames = []
    for prefix, path in FILES_LV.items():
        print(f"Loading {prefix} from {path.name}...")
        df = load_onet_file(path, prefix, scales=["LV"])
        print(f"  {len(df)} rows, {df['dimension'].nunique()} dimensions, {df['soc6'].nunique()} SOC codes")
        frames.append(df)

    # Load Work Activities (both LV and IM scales -> 41 x 2 = 82 dimensions)
    print(f"Loading work activities from {WORK_ACTIVITIES_PATH.name} (LV + IM)...")
    df_wa = load_onet_file(WORK_ACTIVITIES_PATH, "activity", scales=["LV", "IM"])
    print(f"  {len(df_wa)} rows, {df_wa['dimension'].nunique()} dimensions, {df_wa['soc6'].nunique()} SOC codes")
    frames.append(df_wa)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined: {len(combined)} rows, {combined['dimension'].nunique()} dimensions")

    # Average across detailed occupations within same 6-digit SOC
    averaged = combined.groupby(["soc6", "dimension"])["value"].mean().reset_index()

    # Pivot wide
    skill_matrix = averaged.pivot(index="soc6", columns="dimension", values="value")
    print(f"Pivoted: {skill_matrix.shape[0]} SOC codes x {skill_matrix.shape[1]} dimensions")

    # Check for NaNs
    nan_count = skill_matrix.isna().sum().sum()
    if nan_count > 0:
        print(f"WARNING: {nan_count} NaN values found. Filling with 0.")
        skill_matrix = skill_matrix.fillna(0)

    # Min-max normalize each dimension to [0, 1]
    mins = skill_matrix.min()
    maxs = skill_matrix.max()
    ranges = maxs - mins
    # Avoid division by zero for constant columns
    ranges = ranges.replace(0, 1)
    skill_matrix = (skill_matrix - mins) / ranges

    # Save
    out_path = DATA_DIR / "onet_skill_matrix.csv"
    skill_matrix.to_csv(out_path)
    print(f"\nSaved to {out_path}")
    print(f"Final shape: {skill_matrix.shape[0]} SOC codes x {skill_matrix.shape[1]} dimensions")
    print(f"Value range: [{skill_matrix.min().min():.4f}, {skill_matrix.max().max():.4f}]")
    print(f"NaN count: {skill_matrix.isna().sum().sum()}")


if __name__ == "__main__":
    main()
