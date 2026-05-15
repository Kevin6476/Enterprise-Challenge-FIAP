"""
Derived constants for the ingestion layer.

Lists are computed from PAYMENTS_SCHEMA so that schema.py remains the single
source of truth — adding or changing a column type there is enough.
"""

from config.schema import PAYMENTS_SCHEMA

DATE_COLUMNS_PAYMENTS: list[str] = [
    col for col, spec in PAYMENTS_SCHEMA.items()
    if spec.dtype == "datetime64[ns]"
]

NUMERIC_COLUMNS_PAYMENTS: list[str] = [
    col for col, spec in PAYMENTS_SCHEMA.items()
    if spec.dtype == "float64"
]