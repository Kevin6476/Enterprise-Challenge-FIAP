"""Temporal analysis — default and risk trends by vencimento month."""

import pandas as pd


def build_temporal_analysis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["due_year_month"] = pd.to_datetime(df["dt_vencimento"]).dt.to_period("M").astype(str)

    agg = (
        df.groupby("due_year_month")
        .agg(
            total_boletos=("id_boleto", "count"),
            total_value=("vlr_nominal", "sum"),
            defaulted_boletos=("is_defaulted", "sum"),
            default_rate=("is_defaulted", "mean"),
            avg_risk_score=("risk_score", "mean"),
            avg_default_probability=("default_probability", "mean"),
            high_risk_count=(
                "risk_category",
                lambda s: (s == "high").sum(),
            ),
        )
        .reset_index()
        .sort_values("due_year_month")
    )

    return agg.round({
        "total_value": 2,
        "default_rate": 4,
        "avg_risk_score": 2,
        "avg_default_probability": 4,
    })
