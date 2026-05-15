"""Portfolio overview — aggregate KPIs for executive dashboard."""

import pandas as pd


def build_portfolio_overview(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    defaulted = int(df["is_defaulted"].sum())
    total_value = df["vlr_nominal"].sum()
    defaulted_value = df.loc[df["is_defaulted"] == 1, "vlr_nominal"].sum()

    return pd.DataFrame([{
        "total_boletos": total,
        "defaulted_boletos": defaulted,
        "non_defaulted_boletos": total - defaulted,
        "default_rate": round(defaulted / total, 4),
        "total_portfolio_value": round(float(total_value), 2),
        "defaulted_portfolio_value": round(float(defaulted_value), 2),
        "non_defaulted_value": round(float(total_value - defaulted_value), 2),
        "avg_boleto_value": round(float(df["vlr_nominal"].mean()), 2),
        "avg_risk_score": round(float(df["risk_score"].mean()), 2),
        "high_risk_count": int((df["risk_category"] == "high").sum()),
        "medium_risk_count": int((df["risk_category"] == "medium").sum()),
        "low_risk_count": int((df["risk_category"] == "low").sum()),
        "high_risk_rate": round(float((df["risk_category"] == "high").mean()), 4),
        "medium_risk_rate": round(float((df["risk_category"] == "medium").mean()), 4),
        "low_risk_rate": round(float((df["risk_category"] == "low").mean()), 4),
    }])
