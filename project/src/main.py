import pickle
import pandas as pd
from config.paths import (
    BI_OUTPUT_DIR,
    EVALUATION_DIR,
    FEATURES_DIR,
    ML_OUTPUT_DIR,
    MODEL_DIR,
    PAYMENTS_ENRICHED_FILE,
    PAYMENTS_FEATURES_FILE,
)
from src.analytics.bi_exporter import export_bi_datasets
from src.etl.cleaner import clean_auxiliary, clean_payments
from src.etl.deduplicator import remove_duplicates
from src.etl.enricher import enrich_payments
from src.features.aggregations import add_beneficiary_features
from src.features.behavioral import add_payer_features
from src.features.temporal import add_temporal_features
from src.ingestion.loader import load_auxiliary, load_payment
from src.ingestion.validator import validate_auxiliary, validate_payments
from src.models.evaluator import evaluate_models
from src.models.explainer import export_feature_importance
from src.models.trainer import train_models
from src.outputs.ml_exporter import export_ml_dataset
from src.scoring.scorer import score_dataset
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    logger.info("=== FIDC Analytics Pipeline ===")

    # Ingestion
    df_payments = load_payment()
    df_auxiliary = load_auxiliary()

    # Validation
    validate_payments(df_payments)
    validate_auxiliary(df_auxiliary)

    # ETL
    df_payments = clean_payments(df_payments)
    df_payments = remove_duplicates(df_payments, key_cols=["id_boleto"])

    df_auxiliary = clean_auxiliary(df_auxiliary)
    df_auxiliary = remove_duplicates(df_auxiliary, key_cols=["id_cnpj"])

    df_enriched = enrich_payments(df_payments, df_auxiliary)

    PAYMENTS_ENRICHED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_enriched.to_parquet(PAYMENTS_ENRICHED_FILE, index=False)
    logger.info(f"Saved enriched dataset → {PAYMENTS_ENRICHED_FILE}")

    # Feature Engineering
    df_features = add_temporal_features(df_enriched)
    df_features = add_payer_features(df_features)
    df_features = add_beneficiary_features(df_features)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(PAYMENTS_FEATURES_FILE, index=False)
    logger.info(f"Saved features dataset → {PAYMENTS_FEATURES_FILE}")

    # ML Preparation
    export_ml_dataset(df_features, ML_OUTPUT_DIR)

    X_train = pd.read_parquet(ML_OUTPUT_DIR / "X_train.parquet")
    X_test = pd.read_parquet(ML_OUTPUT_DIR / "X_test.parquet")
    y_train = pd.read_parquet(ML_OUTPUT_DIR / "y_train.parquet").squeeze()
    y_test = pd.read_parquet(ML_OUTPUT_DIR / "y_test.parquet").squeeze()

    with open(ML_OUTPUT_DIR / "preprocessors.pkl", "rb") as f:
        preprocessors = pickle.load(f)

    # Model Training
    logger.info("=== Model Training ===")
    models = train_models(X_train, y_train, MODEL_DIR)

    # Model Evaluation
    logger.info("=== Model Evaluation ===")
    metrics = evaluate_models(models, X_test, y_test, EVALUATION_DIR)

    best_name = max(metrics, key=lambda k: metrics[k]["auc_roc"])
    best_model = models[best_name]
    logger.info(
        f"Best model: {best_name} (AUC-ROC={metrics[best_name]['auc_roc']:.4f})"
    )

    # Feature Importance
    export_feature_importance(models, X_train.columns.tolist(), BI_OUTPUT_DIR)

    # Risk Scoring (full dataset)
    logger.info("=== Risk Scoring ===")
    feature_cols = X_train.columns.tolist()
    df_scored = score_dataset(df_features, best_model, preprocessors, feature_cols)

    BI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_scored.to_parquet(BI_OUTPUT_DIR / "boleto_scores.parquet", index=False)
    logger.info(f"Saved boleto_scores.parquet -> {BI_OUTPUT_DIR}")

    # BI Analytics Export
    logger.info("=== BI Analytics Export ===")
    export_bi_datasets(df_scored, BI_OUTPUT_DIR)

    logger.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
