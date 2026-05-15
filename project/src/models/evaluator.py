"""Model evaluation — AUC-ROC, AUC-PR, F1, confusion matrix."""

import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_models(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    eval_dir: Path,
) -> dict:
    eval_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "auc_roc": round(float(roc_auc_score(y_test, y_prob)), 4),
            "auc_pr": round(float(average_precision_score(y_test, y_prob)), 4),
            "f1": round(float(f1_score(y_test, y_pred)), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        results[name] = metrics
        logger.info(
            f"{name}: AUC-ROC={metrics['auc_roc']:.4f} | "
            f"AUC-PR={metrics['auc_pr']:.4f} | F1={metrics['f1']:.4f}"
        )

    with open(eval_dir / "model_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)

    summary = [
        {
            "model": name,
            "auc_roc": m["auc_roc"],
            "auc_pr": m["auc_pr"],
            "f1": m["f1"],
        }
        for name, m in results.items()
    ]
    pd.DataFrame(summary).to_parquet(eval_dir / "model_comparison.parquet", index=False)
    logger.info(f"Saved evaluation results → {eval_dir}")

    return results
