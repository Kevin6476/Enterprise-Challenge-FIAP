import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def add_beneficiary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates per beneficiary/assignor (id_beneficiario) and joins back.

    beneficiary_default_rate shares the same leakage caveat as payer_default_rate:
    safe in a time-based split, leaks in a random split. See ML preparation phase.
    """
    agg = (
        df.groupby("id_beneficiario")
        .agg(
            beneficiary_boleto_count=("id_boleto", "count"),
            beneficiary_default_rate=("is_defaulted", "mean"),
            beneficiary_avg_nominal_value=("vlr_nominal", "mean"),
            beneficiary_total_portfolio_value=("vlr_nominal", "sum"),
        )
        .reset_index()
    )

    df = df.merge(agg, on="id_beneficiario", how="left")
    logger.info(
        f"add_beneficiary_features: {agg.shape[0]} unique beneficiaries | "
        f"avg boletos per beneficiary: {agg['beneficiary_boleto_count'].mean():.1f}."
    )
    return df
