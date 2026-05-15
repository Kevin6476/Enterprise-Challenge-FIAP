"""CNAE analysis — default rates and risk metrics by economic sector."""

import pandas as pd


def _agg_by_cnae(df: pd.DataFrame, cnae_col: str, perspective: str) -> pd.DataFrame:
    agg = (
        df.groupby(cnae_col)
        .agg(
            total_boletos=("id_boleto", "count"),
            total_value=("vlr_nominal", "sum"),
            defaulted_boletos=("is_defaulted", "sum"),
            default_rate=("is_defaulted", "mean"),
            avg_risk_score=("risk_score", "mean"),
            avg_default_probability=("default_probability", "mean"),
        )
        .reset_index()
        .rename(columns={cnae_col: "cd_cnae"})
    )
    agg["perspective"] = perspective
    return agg


def build_cnae_analysis(df: pd.DataFrame) -> pd.DataFrame:
    payer = _agg_by_cnae(df, "payer_cd_cnae_prin", "payer")
    beneficiary = _agg_by_cnae(df, "beneficiary_cd_cnae_prin", "beneficiary")

    combined = (
        pd.concat([payer, beneficiary], ignore_index=True)
        .sort_values(["perspective", "total_boletos"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return combined.round({
        "total_value": 2,
        "default_rate": 4,
        "avg_risk_score": 2,
        "avg_default_probability": 4,
    })
