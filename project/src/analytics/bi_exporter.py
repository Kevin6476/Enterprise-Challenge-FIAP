from pathlib import Path
import pandas as pd
from src.analytics.beneficiary import build_beneficiary_risk
from src.analytics.geographic import build_uf_analysis
from src.analytics.payer import build_payer_risk
from src.analytics.portfolio import build_portfolio_overview
from src.analytics.score_distribution import build_score_distribution
from src.analytics.sector import build_cnae_analysis
from src.analytics.temporal import build_temporal_analysis
from src.utils.logger import get_logger

logger = get_logger(__name__)


def export_bi_datasets(df_scored: pd.DataFrame, bi_dir: Path) -> None:
    bi_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "portfolio_overview": build_portfolio_overview(df_scored),
        "payer_risk": build_payer_risk(df_scored),
        "beneficiary_risk": build_beneficiary_risk(df_scored),
        "temporal_analysis": build_temporal_analysis(df_scored),
        "uf_analysis": build_uf_analysis(df_scored),
        "cnae_analysis": build_cnae_analysis(df_scored),
        "score_distribution": build_score_distribution(df_scored),
    }

    for name, df in datasets.items():
        path = bi_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.info(f"  {name}.parquet → {len(df)} rows")

    logger.info(f"BI export complete: {len(datasets)} datasets → {bi_dir}")
