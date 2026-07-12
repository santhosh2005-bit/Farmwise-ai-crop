
"""
preprocess.py — Data cleaning and preprocessing pipeline for all 5 CSVs.

Discovered issues (from audit):
─────────────────────────────────────────────────────────────────────────
  File            │ Issue
─────────────────────────────────────────────────────────────────────────
  rainfall.csv    │ Column name has leading space (' Area')
                  │ 'average_rain_fall_mm_per_year' is dtype object — 
                  │   6 rows contain '..' instead of a number
                  │ 774 null values (11.5%)
  temp.csv        │ 6 958 exact duplicate rows
                  │ 2 547 null avg_temp values (3.6%)
                  │ Encoding issue: "Côte D'Ivoire" stored as "C�te D'Ivoire"
  yield.csv       │ Redundant metadata columns (Domain Code, Element Code, etc.)
                  │ 8 rows with yield Value == 0 (invalid for yield data)
  pesticides.csv  │ Redundant metadata columns (Domain, Element, Unit, Item)
  yield_df.csv    │ Spurious 'Unnamed: 0' index column
─────────────────────────────────────────────────────────────────────────

Run:
    python data/preprocess.py

Produces cleaned files in data/cleaned/ and a final re-merged yield_df.csv.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import numpy as np


# ─── Paths ───────────────────────────────────────────────────
DATA_DIR: Path = Path(__file__).resolve().parent
CLEAN_DIR: Path = DATA_DIR / "cleaned"
CLEAN_DIR.mkdir(exist_ok=True)


# ─── 1. Pesticides ──────────────────────────────────────────

def clean_pesticides(path: Path) -> pd.DataFrame:
    """Clean pesticides.csv.

    Steps:
    - Drop redundant columns (Domain, Element, Item, Unit).
    - Rename columns for consistency with the merged dataset.
    - Sort by Area and Year.
    """
    print("\n[1/5] Cleaning pesticides.csv …")
    df = pd.read_csv(path)
    print(f"  Raw shape: {df.shape}")

    # Drop metadata columns — every row is "Pesticides Use / Use / Pesticides (total) / tonnes …"
    df = df.drop(columns=["Domain", "Element", "Item", "Unit"])

    # Rename for merge consistency
    df = df.rename(columns={"Value": "pesticides_tonnes"})

    # Sort
    df = df.sort_values(["Area", "Year"]).reset_index(drop=True)

    print(f"  Cleaned shape: {df.shape}")
    print(f"  Nulls remaining: {df.isnull().sum().sum()}")
    return df


# ─── 2. Rainfall ────────────────────────────────────────────

def clean_rainfall(path: Path) -> pd.DataFrame:
    """Clean rainfall.csv.

    Steps:
    - Fix leading space in column name (' Area' → 'Area').
    - Replace placeholder '..' with NaN, then convert to float.
    - Drop rows where rainfall is null (no meaningful imputation possible
      for country-level annual rainfall).
    - Sort by Area and Year.
    """
    print("\n[2/5] Cleaning rainfall.csv …")
    df = pd.read_csv(path)
    print(f"  Raw shape: {df.shape}")

    # Fix column name
    df.columns = df.columns.str.strip()

    # Replace '..' placeholder with NaN, coerce to numeric
    df["average_rain_fall_mm_per_year"] = pd.to_numeric(
        df["average_rain_fall_mm_per_year"], errors="coerce"
    )
    nulls_before = df["average_rain_fall_mm_per_year"].isnull().sum()
    print(f"  Null rainfall values: {nulls_before}")

    # Drop null rainfall rows
    df = df.dropna(subset=["average_rain_fall_mm_per_year"])

    # Ensure integer-like floats are clean
    df["average_rain_fall_mm_per_year"] = df["average_rain_fall_mm_per_year"].astype(float)

    df = df.sort_values(["Area", "Year"]).reset_index(drop=True)

    print(f"  Cleaned shape: {df.shape}")
    print(f"  Nulls remaining: {df.isnull().sum().sum()}")
    return df


# ─── 3. Temperature ─────────────────────────────────────────

def clean_temp(path: Path) -> pd.DataFrame:
    """Clean temp.csv.

    Steps:
    - Remove exact duplicate rows (6 958 duplicates).
    - Fix encoding: replace garbled 'Côte D'Ivoire' with the correct name.
    - Rename columns for merge consistency (year→Year, country→Area).
    - Drop rows where avg_temp is null.
    - Sort by Area and Year.
    """
    print("\n[3/5] Cleaning temp.csv …")
    df = pd.read_csv(path)
    print(f"  Raw shape: {df.shape}")

    # Remove exact duplicates
    dupes = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"  Dropped {dupes} duplicate rows")

    # Fix encoding — the garbled string for Côte D'Ivoire
    df["country"] = df["country"].replace(
        {c: "Côte D'Ivoire" for c in df["country"].unique() if "te D" in str(c) and "Ivoire" in str(c)}
    )

    # Rename for consistency
    df = df.rename(columns={"year": "Year", "country": "Area"})

    # Drop null temperature rows
    nulls_before = df["avg_temp"].isnull().sum()
    df = df.dropna(subset=["avg_temp"])
    print(f"  Dropped {nulls_before} rows with null avg_temp")

    df = df.sort_values(["Area", "Year"]).reset_index(drop=True)

    print(f"  Cleaned shape: {df.shape}")
    print(f"  Nulls remaining: {df.isnull().sum().sum()}")
    return df


# ─── 4. Yield ────────────────────────────────────────────────

def clean_yield(path: Path) -> pd.DataFrame:
    """Clean yield.csv.

    Steps:
    - Drop redundant metadata columns.
    - Remove rows where yield Value == 0 (invalid for crop yield).
    - Rename 'Value' → 'hg/ha_yield' for merge consistency.
    - Sort by Area, Item, Year.
    """
    print("\n[4/5] Cleaning yield.csv …")
    df = pd.read_csv(path)
    print(f"  Raw shape: {df.shape}")

    # Drop metadata columns
    drop_cols = ["Domain Code", "Domain", "Area Code", "Element Code",
                 "Element", "Item Code", "Year Code", "Unit"]
    df = df.drop(columns=drop_cols)

    # Remove zero-yield rows (invalid — yield can't be zero)
    zero_count = (df["Value"] == 0).sum()
    df = df[df["Value"] > 0]
    print(f"  Dropped {zero_count} rows with zero yield")

    # Rename
    df = df.rename(columns={"Value": "hg/ha_yield"})

    df = df.sort_values(["Area", "Item", "Year"]).reset_index(drop=True)

    print(f"  Cleaned shape: {df.shape}")
    print(f"  Nulls remaining: {df.isnull().sum().sum()}")
    return df


# ─── 5. Yield DF (merged) ───────────────────────────────────

def clean_yield_df(path: Path) -> pd.DataFrame:
    """Clean yield_df.csv.

    Steps:
    - Drop the spurious 'Unnamed: 0' index column.
    - Verify no nulls exist (the original merge already dropped them).
    - Sort by Area, Item, Year.
    """
    print("\n[5/5] Cleaning yield_df.csv …")
    df = pd.read_csv(path)
    print(f"  Raw shape: {df.shape}")

    # Drop index column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        print("  Dropped 'Unnamed: 0' column")

    df = df.sort_values(["Area", "Item", "Year"]).reset_index(drop=True)

    print(f"  Cleaned shape: {df.shape}")
    print(f"  Nulls remaining: {df.isnull().sum().sum()}")
    return df


# ─── Re-merge from cleaned sources ──────────────────────────

def rebuild_merged_dataset(
    yield_df: pd.DataFrame,
    rainfall_df: pd.DataFrame,
    temp_df: pd.DataFrame,
    pesticides_df: pd.DataFrame,
) -> pd.DataFrame:
    """Re-build the merged dataset from the individually-cleaned sources.

    Merge order:
    1. Start with yield (base — one row per Area × Item × Year).
    2. Left-join rainfall on (Area, Year).
    3. Left-join temperature on (Area, Year).
    4. Left-join pesticides on (Area, Year).
    5. Drop any rows that ended up with nulls after the join.

    Returns the final, analysis-ready DataFrame.
    """
    print("\n[*] Rebuilding merged dataset from cleaned sources …")

    merged = yield_df.copy()

    # Rainfall
    merged = merged.merge(rainfall_df, on=["Area", "Year"], how="left")

    # Temperature
    merged = merged.merge(temp_df, on=["Area", "Year"], how="left")

    # Pesticides
    merged = merged.merge(pesticides_df, on=["Area", "Year"], how="left")

    before = len(merged)
    merged = merged.dropna()
    after = len(merged)
    print(f"  Dropped {before - after} rows with nulls after merge")

    merged = merged.sort_values(["Area", "Item", "Year"]).reset_index(drop=True)
    print(f"  Final merged shape: {merged.shape}")
    return merged


# ─── Summary report ─────────────────────────────────────────

def print_summary(name: str, df: pd.DataFrame) -> None:
    """Print a compact quality-check summary for a cleaned DataFrame."""
    print(f"\n{'-'*50}")
    print(f"  {name}")
    print(f"  Shape:      {df.shape}")
    print(f"  Nulls:      {df.isnull().sum().sum()}")
    print(f"  Duplicates: {df.duplicated().sum()}")
    print(f"  Columns:    {list(df.columns)}")
    if df.select_dtypes(include=[np.number]).shape[1] > 0:
        print(f"  Dtypes:     {dict(df.dtypes)}")
    print(f"{'-'*50}")


# ─── Main pipeline ──────────────────────────────────────────

def main() -> None:
    """Run the full preprocessing pipeline."""
    print("=" * 60)
    print("  FarmWise Data Preprocessing Pipeline")
    print("=" * 60)

    # Clean each file
    pesticides = clean_pesticides(DATA_DIR / "pesticides.csv")
    rainfall   = clean_rainfall(DATA_DIR / "rainfall.csv")
    temp       = clean_temp(DATA_DIR / "temp.csv")
    yield_raw  = clean_yield(DATA_DIR / "yield.csv")
    yield_df   = clean_yield_df(DATA_DIR / "yield_df.csv")

    # Save individually-cleaned files
    pesticides.to_csv(CLEAN_DIR / "pesticides_clean.csv", index=False)
    rainfall.to_csv(CLEAN_DIR / "rainfall_clean.csv", index=False)
    temp.to_csv(CLEAN_DIR / "temp_clean.csv", index=False)
    yield_raw.to_csv(CLEAN_DIR / "yield_clean.csv", index=False)
    yield_df.to_csv(CLEAN_DIR / "yield_df_clean.csv", index=False)

    # Re-merge from cleaned components
    merged = rebuild_merged_dataset(yield_raw, rainfall, temp, pesticides)
    merged.to_csv(CLEAN_DIR / "yield_df_merged.csv", index=False)

    # Also overwrite the top-level yield_df.csv so the backend picks it up
    merged.to_csv(DATA_DIR / "yield_df.csv", index=False)

    # Final summary
    print("\n" + "=" * 60)
    print("  Cleaning Complete — Summary")
    print("=" * 60)

    for name, df in [
        ("pesticides_clean.csv", pesticides),
        ("rainfall_clean.csv", rainfall),
        ("temp_clean.csv", temp),
        ("yield_clean.csv", yield_raw),
        ("yield_df_clean.csv", yield_df),
        ("yield_df_merged.csv (NEW)", merged),
    ]:
        print_summary(name, df)

    print(f"\n[OK] All cleaned files saved to: {CLEAN_DIR}")
    print(f"[OK] yield_df.csv updated in-place at: {DATA_DIR / 'yield_df.csv'}")


if __name__ == "__main__":
    main()
