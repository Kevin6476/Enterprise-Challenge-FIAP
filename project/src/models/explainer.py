"""Feature importance extraction and export for BI dashboards."""

from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def export_feature_importance(
    models: dict,
    feature_names: list[str],
    bi_dir: Path,
) -> None:
    bi_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for model_name, model in models.items():
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            logger.warning(f"Cannot extract importances from {model_name}, skipping.")
            continue

        for fname, score in zip(feature_names, importances):
            records.append({"model": model_name, "feature": fname, "importance": float(score)})

    if not records:
        logger.warning("No feature importance data to export.")
        return

    df = pd.DataFrame(records).sort_values(["model", "importance"], ascending=[True, False])
    df.to_parquet(bi_dir / "feature_importance.parquet", index=False)
    logger.info(f"Saved feature_importance.parquet → {bi_dir}")
