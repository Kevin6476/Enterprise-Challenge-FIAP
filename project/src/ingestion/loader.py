import pandas as pd
from config.paths import AUXILIARY_FILE, PAYMENTS_FILE
from config.constants import DATE_COLUMNS_PAYMENTS, NUMERIC_COLUMNS_PAYMENTS


def load_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


def parse_dates(df: pd.DataFrame, date_columns: list) -> pd.DataFrame:
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def parse_numeric(df: pd.DataFrame, numeric_columns: list) -> pd.DataFrame:
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_payment() -> pd.DataFrame:
    df = load_csv(PAYMENTS_FILE)
    df = parse_dates(df, DATE_COLUMNS_PAYMENTS)
    df = parse_numeric(df, NUMERIC_COLUMNS_PAYMENTS)
    return df


def load_auxiliary() -> pd.DataFrame:
    df = load_csv(AUXILIARY_FILE)
    return df