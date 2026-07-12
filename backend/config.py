"""
config.py — Centralised configuration for the FarmWise AI Copilot backend.

Reads all settings from environment variables (via a .env file) so that
secrets are never hard-coded.  Every other module imports from here.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ─── Load .env from the backend/ directory ──────────────────
_ENV_PATH: Path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


# ─── Groq API ───────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ─── Dataset ────────────────────────────────────────────────
# Default to the pre-merged yield_df.csv that ships with the repo.
_DEFAULT_DATASET: str = str(
    Path(__file__).resolve().parent.parent / "data" / "yield_df.csv"
)
DATASET_PATH: str = os.getenv("DATASET_PATH", _DEFAULT_DATASET)

# ─── Server ─────────────────────────────────────────────────
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
