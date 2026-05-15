"""Beneficiary risk ranking — per-beneficiary aggregated risk metrics."""

import pandas as pd


def _risk_category(p: float) -> str:
    if p < 0.3:
        return "low"
    if p < 0.6:
        return "medium"
    return "high"


def build_beneficiary_risk(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("id_beneficiario")
        .agg(
            total_boletos=("id_boleto", "count"),
            total_value=("vlr_nominal", "sum"),
            avg_boleto_value=("vlr_nominal", "mean"),
            defaulted_boletos=("is_defaulted", "sum"),
            default_rate=("is_defaulted", "mean"),
            avg_default_probability=("default_probability", "mean"),
            avg_risk_score=("risk_score", "mean"),
            max_risk_score=("risk_score", "max"),
            uf=("beneficiary_uf", "first"),
            cd_cnae_prin=("beneficiary_cd_cnae_prin", "first"),
        )
        .reset_index()
    )

    agg["risk_category"] = agg["avg_default_probability"].apply(_risk_category)
    agg = agg.sort_values("avg_risk_score", ascending=False).reset_index(drop=True)

    return agg.round({
        "total_value": 2,
        "avg_boleto_value": 2,
        "default_rate": 4,
        "avg_default_probability": 4,
        "avg_risk_score": 2,
        "max_risk_score": 2,
    })
