import pandas as pd
from config.schema import AUXILIARY_SCHEMA, JOIN_KEYS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Auxiliary columns to carry into each join (excludes the join key id_cnpj)
_AUX_COLS = [col for col in AUXILIARY_SCHEMA if col != "id_cnpj"]


def _join_auxiliary(
    df: pd.DataFrame,
    df_aux: pd.DataFrame,
    payments_col: str,
    auxiliary_col: str,
    prefix: str,
    how: str,
    expected_match_rate: float,
) -> pd.DataFrame:
    aux_subset = df_aux[[auxiliary_col] + _AUX_COLS].rename(
        columns={col: f"{prefix}_{col}" for col in _AUX_COLS}
    )

    merged = df.merge(
        aux_subset,
        left_on=payments_col,
        right_on=auxiliary_col,
        how=how,
        validate="many_to_one",
        indicator=True,
    )

    match_rate = (merged["_merge"] == "both").mean()
    merged = merged.drop(columns=["_merge", auxiliary_col])

    logger.info(
        f"Join [{prefix}] {payments_col} → {auxiliary_col}: "
        f"match rate {match_rate:.1%} (expected {expected_match_rate:.1%})."
    )
    if match_rate < expected_match_rate - 0.01:
        logger.warning(
            f"[{prefix}] Match rate {match_rate:.1%} is below expected {expected_match_rate:.1%}."
        )

    return merged


def enrich_payments(
    df_payments: pd.DataFrame, df_auxiliary: pd.DataFrame
) -> pd.DataFrame:
    df = _join_auxiliary(
        df_payments,
        df_auxiliary,
        payments_col=JOIN_KEYS["payer"]["payments_col"],
        auxiliary_col=JOIN_KEYS["payer"]["auxiliary_col"],
        prefix="payer",
        how=JOIN_KEYS["payer"]["how"],
        expected_match_rate=JOIN_KEYS["payer"]["expected_match_rate"],
    )

    df = _join_auxiliary(
        df,
        df_auxiliary,
        payments_col=JOIN_KEYS["beneficiary"]["payments_col"],
        auxiliary_col=JOIN_KEYS["beneficiary"]["auxiliary_col"],
        prefix="beneficiary",
        how=JOIN_KEYS["beneficiary"]["how"],
        expected_match_rate=JOIN_KEYS["beneficiary"]["expected_match_rate"],
    )

    logger.info(f"enrich_payments: output shape {df.shape}.")
    return df
