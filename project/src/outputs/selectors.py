import pandas as pd

from config.feature_catalog import (
    LEAKY_COLS,
    SAFE_CATEGORICAL_FEATURES,
    SAFE_NUMERIC_FEATURES,
    TARGET_COL,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

ALL_SAFE_FEATURES: list[str] = SAFE_NUMERIC_FEATURES + SAFE_CATEGORICAL_FEATURES


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in ALL_SAFE_FEATURES if c in df.columns]
    missing = [c for c in ALL_SAFE_FEATURES if c not in df.columns]
    if missing:
        logger.warning(f"Expected features not found in dataframe: {missing}")
    return df[available]


def select_target(df: pd.DataFrame) -> pd.Series:
    return df[TARGET_COL]


def audit_leakage(df: pd.DataFrame) -> list[str]:
    present = sorted(c for c in LEAKY_COLS if c in df.columns)
    absent = sorted(c for c in LEAKY_COLS if c not in df.columns)
    if absent:
        logger.warning(f"Leaky columns already absent from dataframe: {absent}")
    return present
