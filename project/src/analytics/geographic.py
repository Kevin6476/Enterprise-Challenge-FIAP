"""UF analysis — default rates and risk metrics by Brazilian state."""

import pandas as pd


def _agg_by_uf(df: pd.DataFrame, uf_col: str, perspective: str) -> pd.DataFrame:
    agg = (
        df.groupby(uf_col)
        .agg(
            total_boletos=("id_boleto", "count"),
            total_value=("vlr_nominal", "sum"),
            defaulted_boletos=("is_defaulted", "sum"),
            default_rate=("is_defaulted", "mean"),
            avg_risk_score=("risk_score", "mean"),
            avg_default_probability=("default_probability", "mean"),
        )
        .reset_index()
        .rename(columns={uf_col: "uf"})
    )
    agg["perspective"] = perspective
    return agg


def build_uf_analysis(df: pd.DataFrame) -> pd.DataFrame:
    payer = _agg_by_uf(df, "payer_uf", "payer")
    beneficiary = _agg_by_uf(df, "beneficiary_uf", "beneficiary")

    combined = (
        pd.concat([payer, beneficiary], ignore_index=True)
        .sort_values(["perspective", "default_rate"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return combined.round({
        "total_value": 2,
        "default_rate": 4,
        "avg_risk_score": 2,
        "avg_default_probability": 4,
    })
