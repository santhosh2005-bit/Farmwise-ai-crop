"""
dataset.py — Dataset loading & inspection utilities.

Loads the merged agricultural dataset (yield_df.csv) once on startup and
exposes lightweight helpers that the tool-calling layer can use without
importing pandas directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from backend.config import DATASET_PATH


# ─── Module-level cache ─────────────────────────────────────
_df: pd.DataFrame | None = None


def load_dataset(path: str | None = None) -> pd.DataFrame:
    """Load the CSV dataset into a pandas DataFrame.

    Parameters
    ----------
    path : str | None
        Filesystem path to the CSV file.  Falls back to
        ``config.DATASET_PATH`` when *None*.

    Returns
    -------
    pd.DataFrame
        The loaded (and cached) dataset.

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist on disk.
    """
    global _df
    resolved: str = path or DATASET_PATH

    if _df is None or path is not None:
        file = Path(resolved)
        if not file.exists():
            raise FileNotFoundError(f"Dataset not found at {file}")
        _df = pd.read_csv(file)

    return _df


def get_dataframe() -> pd.DataFrame:
    """Return the cached DataFrame, loading it if necessary."""
    return load_dataset()


def get_column_names() -> list[str]:
    """Return the list of column names in the dataset."""
    return list(get_dataframe().columns)


def get_schema_summary() -> dict[str, str]:
    """Return a mapping of column names to their pandas dtype strings.

    Useful for injecting dataset context into the LLM system prompt.
    """
    df: pd.DataFrame = get_dataframe()
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


def get_unique_values(column: str, limit: int = 50) -> list[Any]:
    """Return up to *limit* unique values for a given column.

    Parameters
    ----------
    column : str
        The column to inspect.
    limit : int
        Maximum number of unique values to return (default 50).
    """
    df: pd.DataFrame = get_dataframe()
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataset.")
    return df[column].dropna().unique()[:limit].tolist()


def get_sample_rows(n: int = 5) -> list[dict[str, Any]]:
    """Return *n* sample rows as a list of dicts (records orientation)."""
    return get_dataframe().head(n).to_dict(orient="records")
