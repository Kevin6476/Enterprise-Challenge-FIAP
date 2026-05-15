import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def add_payer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates per payer (id_pagador) and joins back to the main dataframe.

    Note: payer_default_rate uses is_defaulted (the target). In a time-based
    split this represents historical behavior; in a random split it leaks signal
    from the test set. The ML preparation phase handles the correct split strategy.
    """
    agg = (
        df.groupby("id_pagador")
        .agg(
            payer_boleto_count=("id_boleto", "count"),
            payer_default_rate=("is_defaulted", "mean"),
            payer_avg_nominal_value=("vlr_nominal", "mean"),
            payer_avg_delay_days=("payment_delay_days", "mean"),
            payer_max_delay_days=("payment_delay_days", "max"),
            payer_avg_term_days=("boleto_term_days", "mean"),
        )
        .reset_index()
    )

    df = df.merge(agg, on="id_pagador", how="left")
    logger.info(
        f"add_payer_features: {agg.shape[0]} unique payers | "
        f"avg boletos per payer: {agg['payer_boleto_count'].mean():.1f}."
    )
    return df
