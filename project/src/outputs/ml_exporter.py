"""
ML dataset preparation pipeline.

Applies feature selection, categorical encoding, numeric imputation and
stratified train/test splitting to the features dataset.

Outputs (saved to data/outputs/ml/):
    X_train.parquet, X_test.parquet  — feature matrices
    y_train.parquet, y_test.parquet  — target vectors
    preprocessors.pkl                — fitted encoder + imputer for reproducibility
"""

import pickle
from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from config.feature_catalog import SAFE_CATEGORICAL_FEATURES, SAFE_NUMERIC_FEATURES
from src.outputs.selectors import audit_leakage, select_features, select_target
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TEST_SIZE = 0.2
_RANDOM_STATE = 42


def _to_string_safe(series: pd.Series) -> pd.Series:
    """Convert any series to string, mapping all null variants to 'UNKNOWN'."""
    return series.astype("object").fillna("UNKNOWN").astype(str)


def _encode_categoricals(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, OrdinalEncoder]:
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in cat_cols:
        X_train[col] = _to_string_safe(X_train[col])
        X_test[col] = _to_string_safe(X_test[col])

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = encoder.transform(X_test[cat_cols])

    return X_train, X_test, encoder


def _impute_numerics(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    num_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, SimpleImputer]:
    X_train = X_train.copy()
    X_test = X_test.copy()

    imputer = SimpleImputer(strategy="median")
    X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
    X_test[num_cols] = imputer.transform(X_test[num_cols])

    return X_train, X_test, imputer


# Audit


def _log_audit(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    leaky_cols: list[str],
    present_num: list[str],
    present_cat: list[str],
) -> None:
    logger.info("=== ML Preparation Audit ===")

    logger.info(f"Safe features: {X_train.shape[1]} total")
    logger.info(f"  Numeric     : {len(present_num)}")
    logger.info(f"  Categorical : {len(present_cat)}")

    logger.info(f"Leaky columns excluded: {len(leaky_cols)}")
    for col in leaky_cols:
        logger.info(f"  ✗ {col}")

    logger.info(
        f"Dataset split: {len(X_train)} train / {len(X_test)} test "
        f"(stratified, test_size={_TEST_SIZE})"
    )

    train_default_rate = y_train.mean()
    test_default_rate = y_test.mean()
    logger.info(
        f"Default rate — train: {train_default_rate:.1%} | test: {test_default_rate:.1%}"
    )

    minority_rate = y_train.mean()
    majority_rate = 1 - minority_rate
    ratio = majority_rate / minority_rate
    logger.info(
        f"Class imbalance: {majority_rate:.1%} non-default : {minority_rate:.1%} default "
        f"({ratio:.1f}:1) — use class_weight='balanced' in models"
    )

    zero_var = [c for c in X_train.columns if X_train[c].nunique() <= 1]
    if zero_var:
        logger.warning(
            f"Zero-variance features detected (will not contribute to model): {zero_var}"
        )

    remaining_nulls = X_train.columns[X_train.isna().any()].tolist()
    if remaining_nulls:
        logger.warning(f"Null values remain after imputation: {remaining_nulls}")
    else:
        logger.info("No null values remain after imputation.")

    logger.info("=== Audit complete ===")


# Public API


def export_ml_dataset(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Audit leakage
    leaky = audit_leakage(df)
    logger.info(f"Leakage audit: {len(leaky)} column(s) identified and excluded.")

    # Select features and target
    X = select_features(df)
    y = select_target(df)

    present_num = [c for c in SAFE_NUMERIC_FEATURES if c in X.columns]
    present_cat = [c for c in SAFE_CATEGORICAL_FEATURES if c in X.columns]
    logger.info(
        f"Feature selection: {len(present_num)} numeric + {len(present_cat)} categorical "
        f"= {X.shape[1]} features."
    )

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=_TEST_SIZE,
        random_state=_RANDOM_STATE,
        stratify=y,
    )
    logger.info(
        f"Stratified split: {len(X_train)} train rows / {len(X_test)} test rows."
    )

    # Impute numerics (fit on train only)
    X_train, X_test, imputer = _impute_numerics(X_train, X_test, present_num)
    logger.info(f"Numeric imputation: median strategy applied to {len(present_num)} columns.")

    # Encode categoricals (fit on train only)
    X_train, X_test, encoder = _encode_categoricals(X_train, X_test, present_cat)
    logger.info(f"Categorical encoding: OrdinalEncoder applied to {len(present_cat)} columns.")

    # Audit
    _log_audit(X_train, X_test, y_train, y_test, leaky, present_num, present_cat)

    # Save Parquet datasets
    X_train.to_parquet(output_dir / "X_train.parquet", index=False)
    X_test.to_parquet(output_dir / "X_test.parquet", index=False)
    y_train.to_frame().to_parquet(output_dir / "y_train.parquet", index=False)
    y_test.to_frame().to_parquet(output_dir / "y_test.parquet", index=False)
    logger.info(f"Saved ML datasets → {output_dir}")

    # Save fitted preprocessors for reproducibility
    with open(output_dir / "preprocessors.pkl", "wb") as f:
        pickle.dump({"encoder": encoder, "imputer": imputer}, f)
    logger.info(f"Saved preprocessors.pkl → {output_dir}")
