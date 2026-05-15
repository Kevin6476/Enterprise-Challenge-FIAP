import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def remove_duplicates(
    df: pd.DataFrame, key_cols: list[str], keep: str = "last"
) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=key_cols, keep=keep)
    removed = before - len(df)

    if removed:
        logger.warning(f"Removed {removed} duplicate row(s) keyed by {key_cols}.")
    else:
        logger.info(f"No duplicates found by {key_cols}.")

    return df
