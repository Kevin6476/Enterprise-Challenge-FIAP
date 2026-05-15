import pandas as pd
from config.business_rules import DEFAULT_THRESHOLD_DAYS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    today = pd.Timestamp.today().normalize()

    # Credit term: days between issuance and due date
    df["boleto_term_days"] = (df["dt_vencimento"] - df["dt_emissao"]).dt.days

    # Delay relative to due date — negative means paid early, NaN means not yet paid
    df["payment_delay_days"] = (df["dt_pagamento"] - df["dt_vencimento"]).dt.days

    # Days since due date for still-unpaid boletos; 0 for settled ones
    df["days_overdue"] = (today - df["dt_vencimento"]).dt.days.where(
        df["dt_pagamento"].isna(), other=0
    )

    # Target variable: 1 if unpaid or paid after DEFAULT_THRESHOLD_DAYS (from business_rules)
    df["is_defaulted"] = (
        df["dt_pagamento"].isna()
        | (df["payment_delay_days"] > DEFAULT_THRESHOLD_DAYS)
    ).astype(int)

    df["due_month"] = df["dt_vencimento"].dt.month
    df["due_day_of_week"] = df["dt_vencimento"].dt.dayofweek  # 0 = Monday

    logger.info(
        f"add_temporal_features: {len(df)} rows | "
        f"default rate {df['is_defaulted'].mean():.1%} "
        f"(rule: unpaid OR payment_delay_days > {DEFAULT_THRESHOLD_DAYS})."
    )
    return df
