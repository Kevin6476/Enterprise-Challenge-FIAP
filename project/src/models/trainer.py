"""Model training - fits Logistic Regression, Random Forest, and XGBoost."""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.utils.logger import get_logger

logger = get_logger(__name__)

_RANDOM_STATE = 42


def _save_model(model: object, output_path: Path) -> None:
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    backup_path = output_path.with_name(f".{output_path.name}.bak")
    if temp_path.exists():
        temp_path.unlink()
    if backup_path.exists():
        backup_path.unlink()
    joblib.dump(model, temp_path)
    try:
        temp_path.replace(output_path)
    except OSError:
        if output_path.exists():
            output_path.replace(backup_path)
        try:
            temp_path.replace(output_path)
        except OSError:
            if backup_path.exists():
                backup_path.replace(output_path)
            raise
        else:
            if backup_path.exists():
                backup_path.unlink()


def train_models(X_train: pd.DataFrame, y_train: pd.Series, model_dir: Path) -> dict:
    model_dir.mkdir(parents=True, exist_ok=True)

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=5000,
            solver="saga",
            class_weight="balanced",
            C=0.1,
            random_state=_RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            n_jobs=-1,
            random_state=_RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            verbosity=0,
            random_state=_RANDOM_STATE,
        ),
    }

    trained: dict = {}
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        _save_model(model, model_dir / f"{name}.pkl")
        trained[name] = model
        logger.info(f"  Saved {name}.pkl -> {model_dir}")

    return trained
