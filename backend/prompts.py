"""
prompts.py — System prompt templates for the FarmWise AI Copilot.

All prompt engineering lives here so that the rest of the codebase stays
free of long string literals.  Templates accept runtime context (schema,
sample rows, available tools) via standard Python string formatting.
"""

from __future__ import annotations


import os

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
try:
    with open(_PROMPT_PATH, "r", encoding="utf-8") as _f:
        SYSTEM_PROMPT_TEMPLATE: str = _f.read().strip()
except Exception:
    SYSTEM_PROMPT_TEMPLATE = ""


def build_system_prompt(
    schema: dict[str, str],
    sample_rows: list[dict],
    tools_description: str,
    countries: list[str],
    crops: list[str],
    year_range: tuple[int, int],
) -> str:
    """Render the system prompt with runtime dataset context."""
    return SYSTEM_PROMPT_TEMPLATE.format(
        schema=schema,
        year_range=f"{year_range[0]} - {year_range[1]}",
    )
