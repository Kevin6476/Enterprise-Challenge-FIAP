"""Score distribution — probability buckets for histogram visualization."""

import numpy as np
import pandas as pd


def build_score_distribution(df: pd.DataFrame) -> pd.DataFrame:
    bins = np.arange(0, 1.1, 0.1)
    labels = [f"{int(b * 100)}-{int(b * 100 + 10)}%" for b in bins[:-1]]

    df = df.copy()
    df["probability_bucket"] = pd.cut(
        df["default_probability"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    agg = (
        df.groupby("probability_bucket", observed=True)
        .agg(
            boleto_count=("id_boleto", "count"),
            actual_default_count=("is_defaulted", "sum"),
            actual_default_rate=("is_defaulted", "mean"),
            total_value=("vlr_nominal", "sum"),
        )
        .reset_index()
    )

    agg["pct_of_portfolio"] = (agg["boleto_count"] / len(df)).round(4)

    return agg.round({"actual_default_rate": 4, "total_value": 2})
