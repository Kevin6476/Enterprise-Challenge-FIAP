import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def clean_payments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["tipo_baixa"] = df["tipo_baixa"].str.strip()
    df["tipo_especie"] = df["tipo_especie"].str.strip()

    # Extract the numeric code from labels like "5 - Baixa integral por solicitacao do cedente"
    df["settlement_code"] = (
        df["tipo_baixa"].str.extract(r"^(\d+)").astype("Int64")
    )

    logger.info(f"clean_payments: {len(df)} rows processed, settlement_code extracted.")
    return df


def clean_auxiliary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["uf"] = df["uf"].str.strip().str.upper()

    # CNAE codes are integers but land as float64 due to pandas null handling
    df["cd_cnae_prin"] = df["cd_cnae_prin"].astype("Int64")

    logger.info(f"clean_auxiliary: {len(df)} rows processed.")
    return df
