"""Risk scoring — applies a trained model to the full features dataset.

Outputs per boleto:
    default_probability  float [0, 1]
    risk_score           float [0, 100]
    risk_category        str   low | medium | high
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config.feature_catalog import (
    DATE_COLS,
    IDENTIFIER_COLS,
    SAFE_CATEGORICAL_FEATURES,
    SAFE_NUMERIC_FEATURES,
    TARGET_COL,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_LOW_THRESHOLD = 0.3
_HIGH_THRESHOLD = 0.6

_BI_CONTEXT_COLS = [
    "payer_uf",
    "beneficiary_uf",
    "payer_cd_cnae_prin",
    "beneficiary_cd_cnae_prin",
    "vlr_nominal",
]


def _classify_risk(prob: float) -> str:
    if prob < _LOW_THRESHOLD:
        return "low"
    if prob < _HIGH_THRESHOLD:
        return "medium"
    return "high"


def _apply_preprocessing(
    df: pd.DataFrame,
    preprocessors: dict,
    feature_cols: list[str],
) -> pd.DataFrame:
    num_cols = [c for c in SAFE_NUMERIC_FEATURES if c in feature_cols]
    cat_cols = [c for c in SAFE_CATEGORICAL_FEATURES if c in feature_cols]

    X = df[feature_cols].copy()

    for col in cat_cols:
        X[col] = X[col].astype("object").fillna("UNKNOWN").astype(str)

    X[num_cols] = preprocessors["imputer"].transform(X[num_cols])
    X[cat_cols] = preprocessors["encoder"].transform(X[cat_cols])

    return X


def score_dataset(
    df_features: pd.DataFrame,
    model,
    preprocessors: dict,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Return a BI-ready scored dataframe for all boletos."""
    X = _apply_preprocessing(df_features, preprocessors, feature_cols)
    probabilities = model.predict_proba(X)[:, 1]

    keep = (
        IDENTIFIER_COLS
        + DATE_COLS
        + [TARGET_COL]
        + [c for c in _BI_CONTEXT_COLS if c in df_features.columns]
    )
    df_scored = df_features[[c for c in keep if c in df_features.columns]].copy()

    df_scored["default_probability"] = np.round(probabilities, 4)
    df_scored["risk_score"] = (probabilities * 100).round(1)
    df_scored["risk_category"] = df_scored["default_probability"].apply(_classify_risk)

    counts = df_scored["risk_category"].value_counts()
    logger.info(
        f"Scored {len(df_scored)} boletos — "
        f"high: {counts.get('high', 0)} | "
        f"medium: {counts.get('medium', 0)} | "
        f"low: {counts.get('low', 0)}"
    )

    return df_scored
