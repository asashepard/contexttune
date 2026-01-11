"""Run ID generation utilities."""
from __future__ import annotations

import secrets
from datetime import datetime


def make_run_id(prefix: str) -> str:
    """Generate a filesystem-safe run ID with timestamp and random suffix.

    Format: <prefix>_<YYYYmmdd_HHMMSS>_<4-char-hex>

    Args:
        prefix: A descriptive prefix for the run (e.g., "sanity", "eval").

    Returns:
        A unique run ID string safe for use in file paths.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(2)  # 4 hex chars
    return f"{prefix}_{timestamp}_{suffix}"
