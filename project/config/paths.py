from pathlib import Path

# root of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# data
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# raw files
AUXILIARY_FILE = RAW_DATA_DIR / "auxiliary.csv"
PAYMENTS_FILE = RAW_DATA_DIR / "payments.csv"

# processed outputs
PAYMENTS_ENRICHED_FILE = PROCESSED_DATA_DIR / "payments_enriched.parquet"

# feature outputs
FEATURES_DIR = DATA_DIR / "features"
PAYMENTS_FEATURES_FILE = FEATURES_DIR / "payments_features.parquet"

# ml outputs
ML_OUTPUT_DIR = DATA_DIR / "outputs" / "ml"
MODEL_DIR = ML_OUTPUT_DIR / "models"
EVALUATION_DIR = ML_OUTPUT_DIR / "evaluation"

# bi outputs
BI_OUTPUT_DIR = DATA_DIR / "outputs" / "bi"