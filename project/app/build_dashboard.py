"""Build the fixed FIDC Analytics HTML application.

The committed `app/index.html` is a static demo application: it opens directly
from disk and embeds a compact dataset generated from the Python pipeline
outputs. Re-run this builder after refreshing the pipeline outputs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
FEATURE_FILE = PROJECT_DIR / "data" / "features" / "payments_features.parquet"
BI_DIR = PROJECT_DIR / "data" / "outputs" / "bi"
EVALUATION_FILE = PROJECT_DIR / "data" / "outputs" / "ml" / "evaluation" / "model_evaluation.json"
APP_INDEX_FILE = PROJECT_DIR / "app" / "index.html"
DIST_DIR = PROJECT_DIR / "app" / "dist"
DIST_INDEX_FILE = DIST_DIR / "index.html"

CNAE_DIVISION_LABELS = {
    "01": "Agricultura, pecuaria e servicos relacionados",
    "02": "Producao florestal",
    "03": "Pesca e aquicultura",
    "05": "Extracao de carvao mineral",
    "06": "Extracao de petroleo e gas natural",
    "07": "Extracao de minerais metalicos",
    "08": "Extracao de minerais nao-metalicos",
    "09": "Apoio a extracao mineral",
    "10": "Fabricacao de produtos alimenticios",
    "11": "Fabricacao de bebidas",
    "12": "Fabricacao de produtos do fumo",
    "13": "Fabricacao de produtos texteis",
    "14": "Confeccao de artigos do vestuario",
    "15": "Preparacao de couros e calcados",
    "16": "Produtos de madeira",
    "17": "Celulose, papel e produtos de papel",
    "18": "Impressao e reproducao",
    "19": "Coque e derivados de petroleo",
    "20": "Produtos quimicos",
    "21": "Produtos farmoquimicos e farmaceuticos",
    "22": "Produtos de borracha e plastico",
    "23": "Produtos de minerais nao-metalicos",
    "24": "Metalurgia",
    "25": "Produtos de metal",
    "26": "Equipamentos de informatica e eletronicos",
    "27": "Maquinas e equipamentos eletricos",
    "28": "Maquinas e equipamentos",
    "29": "Veiculos automotores",
    "30": "Outros equipamentos de transporte",
    "31": "Moveis",
    "32": "Produtos diversos",
    "33": "Manutencao e instalacao",
    "35": "Eletricidade, gas e utilidades",
    "36": "Captacao e distribuicao de agua",
    "37": "Esgoto e atividades relacionadas",
    "38": "Coleta e tratamento de residuos",
    "39": "Descontaminacao e gestao ambiental",
    "41": "Construcao de edificios",
    "42": "Obras de infraestrutura",
    "43": "Servicos especializados para construcao",
    "45": "Comercio e reparacao de veiculos",
    "46": "Comercio por atacado",
    "47": "Comercio varejista",
    "49": "Transporte terrestre",
    "50": "Transporte aquaviario",
    "51": "Transporte aereo",
    "52": "Armazenagem e atividades auxiliares",
    "53": "Correio e entregas",
    "55": "Alojamento",
    "56": "Alimentacao",
    "58": "Edicao e publicacao",
    "59": "Audiovisual e musica",
    "60": "Radio e televisao",
    "61": "Telecomunicacoes",
    "62": "Tecnologia da informacao",
    "63": "Servicos de informacao",
    "64": "Servicos financeiros",
    "65": "Seguros e previdencia",
    "66": "Atividades auxiliares financeiras",
    "68": "Atividades imobiliarias",
    "69": "Atividades juridicas e contabeis",
    "70": "Consultoria e gestao empresarial",
    "71": "Arquitetura, engenharia e testes",
    "72": "Pesquisa e desenvolvimento",
    "73": "Publicidade e pesquisa de mercado",
    "74": "Atividades profissionais e tecnicas",
    "75": "Atividades veterinarias",
    "77": "Alugueis nao-imobiliarios",
    "78": "Selecao e locacao de mao de obra",
    "79": "Agencias de viagens",
    "80": "Seguranca e investigacao",
    "81": "Servicos para edificios e paisagismo",
    "82": "Servicos de escritorio e apoio administrativo",
    "84": "Administracao publica",
    "85": "Educacao",
    "86": "Atencao a saude humana",
    "87": "Atencao residencial em saude",
    "88": "Assistencia social",
    "90": "Artes e espetaculos",
    "91": "Bibliotecas, museus e patrimonio",
    "92": "Jogos de azar e apostas",
    "93": "Esporte, recreacao e lazer",
    "94": "Organizacoes associativas",
    "95": "Reparacao de computadores e pessoais",
    "96": "Outras atividades de servicos pessoais",
    "97": "Servicos domesticos",
    "99": "Organismos internacionais",
}


def read_parquet_optional(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def normalize(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mn = s.min(skipna=True)
    mx = s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.0, index=series.index)
    return ((s - mn) / (mx - mn)).fillna(0.0)


def classify_risk(score: float) -> str:
    if score < 30:
        return "low"
    if score < 60:
        return "medium"
    return "high"


def cnae_division(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    digits = "".join(ch for ch in str(value).split(".")[0] if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(7)[:2]


def cnae_label(division: str) -> str:
    description = CNAE_DIVISION_LABELS.get(division, "Setor CNAE")
    return f"{description} (CNAE {division})"


def read_bi_outputs() -> tuple[dict[str, pd.DataFrame], bool]:
    names = [
        "portfolio_overview",
        "payer_risk",
        "beneficiary_risk",
        "temporal_analysis",
        "uf_analysis",
        "cnae_analysis",
        "score_distribution",
        "feature_importance",
        "boleto_scores",
    ]
    datasets = {
        name: df
        for name in names
        if (df := read_parquet_optional(BI_DIR / f"{name}.parquet")) is not None
    }
    required = {
        "portfolio_overview",
        "payer_risk",
        "beneficiary_risk",
        "temporal_analysis",
        "uf_analysis",
        "cnae_analysis",
        "score_distribution",
        "boleto_scores",
    }
    return datasets, required.issubset(datasets)


def load_raw_fallback() -> dict[str, pd.DataFrame]:
    payments = pd.read_csv(RAW_DIR / "payments.csv")
    auxiliary = pd.read_csv(RAW_DIR / "auxiliary.csv")

    for col in ["dt_emissao", "dt_vencimento", "dt_pagamento"]:
        payments[col] = pd.to_datetime(payments[col], errors="coerce")

    payments["payment_delay_days"] = (
        payments["dt_pagamento"] - payments["dt_vencimento"]
    ).dt.days
    payments["is_defaulted"] = (
        payments["dt_pagamento"].isna() | (payments["payment_delay_days"] > 0)
    ).astype(int)

    payer_aux = auxiliary.add_prefix("payer_").rename(columns={"payer_id_cnpj": "id_pagador"})
    beneficiary_aux = auxiliary.add_prefix("beneficiary_").rename(
        columns={"beneficiary_id_cnpj": "id_beneficiario"}
    )

    df = payments.merge(payer_aux, on="id_pagador", how="left")
    df = df.merge(beneficiary_aux, on="id_beneficiario", how="left")

    liquidity = pd.to_numeric(df["payer_sacado_indice_liquidez_1m"], errors="coerce")
    delay = normalize(df["payer_media_atraso_dias"])
    overdue_share = pd.to_numeric(
        df["payer_share_vl_inad_pag_bol_6_a_15d"], errors="coerce"
    ).fillna(0.0)
    materiality = normalize(df["payer_score_materialidade_evolucao"])

    risk_score = (
        35 * delay
        + 25 * overdue_share.clip(0, 1)
        + 20 * (1 - liquidity.fillna(liquidity.median()).clip(0, 1))
        + 20 * materiality
    )
    df["risk_score"] = risk_score.clip(0, 100).round(1)
    df["default_probability"] = (df["risk_score"] / 100).round(4)
    df["risk_category"] = df["risk_score"].apply(classify_risk)

    portfolio = pd.DataFrame(
        [
            {
                "total_boletos": len(df),
                "defaulted_boletos": int(df["is_defaulted"].sum()),
                "default_rate": round(float(df["is_defaulted"].mean()), 4),
                "total_portfolio_value": round(float(df["vlr_nominal"].sum()), 2),
                "avg_boleto_value": round(float(df["vlr_nominal"].mean()), 2),
                "avg_risk_score": round(float(df["risk_score"].mean()), 2),
                "high_risk_count": int((df["risk_category"] == "high").sum()),
                "medium_risk_count": int((df["risk_category"] == "medium").sum()),
                "low_risk_count": int((df["risk_category"] == "low").sum()),
            }
        ]
    )

    payer = aggregate_entity(df, "id_pagador", "payer_uf", "payer_cd_cnae_prin")
    beneficiary = aggregate_entity(
        df, "id_beneficiario", "beneficiary_uf", "beneficiary_cd_cnae_prin"
    )
    temporal = aggregate_temporal(df)
    uf = aggregate_uf(df)
    cnae = aggregate_cnae(df)
    score_distribution = aggregate_score_distribution(df)

    return {
        "portfolio_overview": portfolio,
        "payer_risk": payer,
        "beneficiary_risk": beneficiary,
        "temporal_analysis": temporal,
        "uf_analysis": uf,
        "cnae_analysis": cnae,
        "score_distribution": score_distribution,
        "boleto_scores": df,
    }


def aggregate_entity(df: pd.DataFrame, id_col: str, uf_col: str, cnae_col: str) -> pd.DataFrame:
    result = (
        df.groupby(id_col, dropna=False)
        .agg(
            total_boletos=("id_boleto", "count"),
            total_value=("vlr_nominal", "sum"),
            avg_boleto_value=("vlr_nominal", "mean"),
            defaulted_boletos=("is_defaulted", "sum"),
            default_rate=("is_defaulted", "mean"),
            avg_default_probability=("default_probability", "mean"),
            avg_risk_score=("risk_score", "mean"),
            max_risk_score=("risk_score", "max"),
            uf=(uf_col, "first"),
            cd_cnae_prin=(cnae_col, "first"),
        )
        .reset_index()
    )
    result["risk_category"] = result["avg_risk_score"].apply(classify_risk)
    return result.sort_values(["avg_risk_score", "total_value"], ascending=False)


def aggregate_temporal(df: pd.DataFrame) -> pd.DataFrame:
    due = pd.to_datetime(df["dt_vencimento"], errors="coerce")
    work = df.assign(due_year_month=due.dt.to_period("M").astype(str))
    return (
        work.groupby("due_year_month", dropna=False)
        .agg(
            total_boletos=("id_boleto", "count"),
            total_value=("vlr_nominal", "sum"),
            defaulted_boletos=("is_defaulted", "sum"),
            default_rate=("is_defaulted", "mean"),
            avg_risk_score=("risk_score", "mean"),
            avg_default_probability=("default_probability", "mean"),
            high_risk_count=("risk_category", lambda s: int((s == "high").sum())),
        )
        .reset_index()
        .sort_values("due_year_month")
    )


def aggregate_uf(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for perspective, uf_col in [
        ("payer", "payer_uf"),
        ("beneficiary", "beneficiary_uf"),
    ]:
        group = (
            df.groupby(uf_col, dropna=False)
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
        group["perspective"] = perspective
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def aggregate_cnae(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for perspective, cnae_col in [
        ("payer", "payer_cd_cnae_prin"),
        ("beneficiary", "beneficiary_cd_cnae_prin"),
    ]:
        group = (
            df.groupby(cnae_col, dropna=False)
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
        group["perspective"] = perspective
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def aggregate_score_distribution(df: pd.DataFrame) -> pd.DataFrame:
    buckets = pd.cut(
        pd.to_numeric(df["default_probability"], errors="coerce").fillna(0),
        bins=[i / 10 for i in range(11)],
        labels=[f"{i * 10}-{(i + 1) * 10}%" for i in range(10)],
        include_lowest=True,
    )
    return (
        df.assign(probability_bucket=buckets)
        .groupby("probability_bucket", observed=True)
        .agg(
            boleto_count=("id_boleto", "count"),
            actual_default_count=("is_defaulted", "sum"),
            actual_default_rate=("is_defaulted", "mean"),
            total_value=("vlr_nominal", "sum"),
        )
        .reset_index()
    )


def build_transaction_dataset(bi: dict[str, pd.DataFrame]) -> pd.DataFrame:
    scores = bi["boleto_scores"].copy()
    features = read_parquet_optional(FEATURE_FILE)
    if features is None:
        detail = scores.copy()
    else:
        extra_cols = [
            "id_boleto",
            "dt_pagamento",
            "vlr_baixa",
            "tipo_baixa",
            "tipo_especie",
            "payment_delay_days",
            "beneficiary_media_atraso_dias",
            "beneficiary_cedente_indice_liquidez_1m",
            "beneficiary_sacado_indice_liquidez_1m",
            "beneficiary_indicador_liquidez_quantitativo_3m",
            "beneficiary_share_vl_inad_pag_bol_6_a_15d",
            "payer_media_atraso_dias",
            "payer_sacado_indice_liquidez_1m",
        ]
        available = [col for col in extra_cols if col in features.columns]
        detail = scores.merge(features[available], on="id_boleto", how="left")

    for col in ["dt_emissao", "dt_vencimento", "dt_pagamento"]:
        if col in detail.columns:
            detail[col] = pd.to_datetime(detail[col], errors="coerce")

    if "due_year_month" not in detail.columns:
        detail["due_year_month"] = detail["dt_vencimento"].dt.to_period("M").astype(str)
    detail["period_month"] = detail["dt_emissao"].dt.to_period("M").astype(str)

    detail["entity_uf"] = detail.get("beneficiary_uf").fillna(detail.get("payer_uf"))
    detail["entity_sector"] = detail.get("beneficiary_cd_cnae_prin").fillna(
        detail.get("payer_cd_cnae_prin")
    )
    detail["score_display"] = (pd.to_numeric(detail["risk_score"], errors="coerce") * 10).round(0)
    return detail


def load_model_metrics() -> list[dict[str, Any]]:
    if not EVALUATION_FILE.exists():
        return []
    with EVALUATION_FILE.open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    rows = []
    for model, values in metrics.items():
        rows.append(
            {
                "model": model.replace("_", " ").title(),
                "auc_roc": values.get("auc_roc"),
                "auc_pr": values.get("auc_pr"),
                "f1": values.get("f1"),
            }
        )
    return sorted(rows, key=lambda row: row.get("auc_roc") or 0, reverse=True)


def records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if limit is not None:
        df = df.head(limit)
    clean = df.copy()
    for col in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[col]):
            clean[col] = clean[col].dt.strftime("%Y-%m-%d")
    return [sanitize(row) for row in clean.to_dict(orient="records")]


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize(v) for v in value]
    if hasattr(value, "item"):
        return sanitize(value.item())
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def make_alerts(beneficiaries: pd.DataFrame, payers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sources = [("Cedente", "id_beneficiario", beneficiaries)]
    for entity_type, id_col, df in sources:
        top = df.sort_values(["avg_risk_score", "total_value"], ascending=False).head(80)
        for _, row in top.iterrows():
            score = float(row.get("avg_risk_score") or 0)
            default_rate = float(row.get("default_rate") or 0)
            value = float(row.get("total_value") or 0)
            if score >= 75:
                alert_type = "Score"
                level = "Critico"
                status = "Aberto"
                description = f"Score preditivo em {score * 10:.0f}/1000 com exposicao de R$ {value:,.0f}."
            elif default_rate >= 0.5:
                alert_type = "Inadimplencia"
                level = "Alerta"
                status = "Em analise"
                description = f"Inadimplencia observada de {default_rate * 100:.1f}% na entidade."
            elif score >= 60:
                alert_type = "Concentracao"
                level = "Atencao"
                status = "Aberto"
                description = f"Entidade em alto risco com score {score * 10:.0f}/1000."
            else:
                continue
            rows.append(
                {
                    "entity_type": entity_type,
                    "entity_id": row.get(id_col),
                    "uf": row.get("uf"),
                    "sector": row.get("cd_cnae_prin"),
                    "risk_category": row.get("risk_category"),
                    "risk_score": score,
                    "score_display": round(score * 10),
                    "default_rate": default_rate,
                    "total_value": value,
                    "alert_type": alert_type,
                    "level": level,
                    "status": status,
                    "description": description,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["level", "risk_score"], ascending=[True, False]
    )


def compact_id(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)[:12]


def compact_for_payload(
    transactions: pd.DataFrame,
    beneficiaries: pd.DataFrame,
    payers: pd.DataFrame,
    alerts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tx_cols = [
        "id_pagador",
        "id_beneficiario",
        "dt_emissao",
        "dt_vencimento",
        "dt_pagamento",
        "vlr_nominal",
        "vlr_baixa",
        "tipo_baixa",
        "is_defaulted",
        "payer_uf",
        "beneficiary_uf",
        "payer_cd_cnae_prin",
        "beneficiary_cd_cnae_prin",
        "payment_delay_days",
        "beneficiary_media_atraso_dias",
        "beneficiary_cedente_indice_liquidez_1m",
        "beneficiary_share_vl_inad_pag_bol_6_a_15d",
        "default_probability",
        "risk_score",
        "risk_category",
        "period_month",
        "due_year_month",
        "entity_uf",
        "entity_sector",
        "score_display",
    ]
    tx = transactions[[col for col in tx_cols if col in transactions.columns]].copy()
    for col in ["id_pagador", "id_beneficiario"]:
        if col in tx.columns:
            tx[col] = tx[col].map(compact_id)
    if "tipo_baixa" in tx.columns:
        tx["tipo_baixa"] = tx["tipo_baixa"].fillna("").astype(str).str.slice(0, 54)

    entity_cols = [
        "id_beneficiario",
        "id_pagador",
        "total_boletos",
        "total_value",
        "avg_boleto_value",
        "defaulted_boletos",
        "default_rate",
        "avg_default_probability",
        "avg_risk_score",
        "max_risk_score",
        "uf",
        "cd_cnae_prin",
        "risk_category",
        "score_display",
    ]
    ben = beneficiaries[[col for col in entity_cols if col in beneficiaries.columns]].copy()
    pay = payers[[col for col in entity_cols if col in payers.columns]].copy()
    if "id_beneficiario" in ben.columns:
        ben["id_beneficiario"] = ben["id_beneficiario"].map(compact_id)
    if "id_pagador" in pay.columns:
        pay["id_pagador"] = pay["id_pagador"].map(compact_id)

    alert_cols = [
        "entity_type",
        "entity_id",
        "uf",
        "sector",
        "risk_category",
        "risk_score",
        "score_display",
        "default_rate",
        "total_value",
        "alert_type",
        "level",
        "status",
        "description",
    ]
    alert = alerts[[col for col in alert_cols if col in alerts.columns]].copy()
    if "entity_id" in alert.columns:
        alert["entity_id"] = alert["entity_id"].map(compact_id)
    return tx, ben, pay, alert


def build_payload() -> dict[str, Any]:
    bi, has_bi = read_bi_outputs()
    if not has_bi:
        bi = load_raw_fallback()
        data_source = "CSV fallback"
    else:
        data_source = "BI Parquet outputs"

    transactions = build_transaction_dataset(bi)
    beneficiaries = bi["beneficiary_risk"].copy()
    payers = bi["payer_risk"].copy()
    beneficiaries["score_display"] = (beneficiaries["avg_risk_score"] * 10).round(0)
    payers["score_display"] = (payers["avg_risk_score"] * 10).round(0)
    alerts = make_alerts(beneficiaries, payers)
    transactions, beneficiaries, payers, alerts = compact_for_payload(
        transactions, beneficiaries, payers, alerts
    )

    feature_importance = bi.get("feature_importance")
    if feature_importance is None:
        feature_rows: list[dict[str, Any]] = []
    else:
        feature_rows = records(
            feature_importance.sort_values("importance", ascending=False), 10
        )

    periods = sorted(
        value
        for value in transactions["period_month"].dropna().astype(str).unique().tolist()
        if value not in {"NaT", "nan", "None", "-"}
    )
    ufs = sorted(transactions["entity_uf"].dropna().astype(str).unique().tolist())
    sector_divisions = sorted(
        {
            division
            for value in transactions["entity_sector"].dropna().tolist()
            if (division := cnae_division(value)) is not None
        },
        key=lambda value: cnae_label(value),
    )
    sectors = [
        {"value": division, "label": cnae_label(division)}
        for division in sector_divisions
    ]

    return {
        "generated_from": data_source,
        "summary": sanitize(bi["portfolio_overview"].iloc[0].to_dict()),
        "filters": {
            "periods": periods,
            "ufs": ufs,
            "sectors": sectors,
            "risks": ["low", "medium", "high"],
        },
        "transactions": records(transactions),
        "beneficiaries": records(beneficiaries),
        "payers": records(payers.head(500)),
        "temporal": records(bi["temporal_analysis"]),
        "uf_analysis": records(bi["uf_analysis"]),
        "cnae_analysis": records(bi["cnae_analysis"].head(300)),
        "score_distribution": records(bi["score_distribution"]),
        "model_metrics": load_model_metrics(),
        "feature_importance": feature_rows,
        "alerts": records(alerts),
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FIDC Analytics</title>
  <style>
    :root {
      --bg: #edf2f7;
      --ink: #101827;
      --muted: #606b7b;
      --dark: #111a2d;
      --dark-2: #172238;
      --line: #cbd5e1;
      --cyan: #67d5ef;
      --teal: #13a9b9;
      --red: #ef3e36;
      --orange: #ec5b10;
      --green: #00e55c;
      --gray-card: #dedede;
      --white: #ffffff;
      --shadow: 0 12px 30px rgba(17, 26, 45, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Segoe UI", Arial, sans-serif;
      letter-spacing: 0;
    }
    .shell { min-height: 100vh; padding: 18px 34px 28px; }
    .app-header {
      display: grid;
      grid-template-columns: 250px 1fr;
      gap: 24px;
      align-items: center;
      margin-bottom: 28px;
    }
    .brand { display: flex; align-items: center; gap: 14px; min-width: 220px; cursor: pointer; border-radius: 8px; }
    .brand:focus-visible { outline: 3px solid rgba(19, 169, 185, 0.35); outline-offset: 4px; }
    .mark { width: 48px; height: 44px; position: relative; border-left: 5px solid var(--dark); border-bottom: 5px solid var(--dark); }
    .mark span { position: absolute; bottom: 5px; width: 8px; background: var(--dark); border-radius: 2px 2px 0 0; }
    .mark span:nth-child(1) { left: 10px; height: 25px; }
    .mark span:nth-child(2) { left: 24px; height: 36px; }
    .mark span:nth-child(3) { left: 38px; height: 29px; }
    .brand-title { font-weight: 900; font-size: 20px; line-height: 1; }
    .brand-title small { display: block; font-size: 11px; margin-top: 4px; }
    .main-nav { display: flex; gap: 22px; justify-content: center; flex-wrap: wrap; }
    .main-nav button, .filter-select, .search-input {
      min-height: 52px;
      border: 0;
      border-radius: 8px;
      background: var(--dark);
      color: #fff;
      font-size: 22px;
      padding: 0 34px;
      cursor: pointer;
      font-weight: 500;
    }
    .main-nav button.active { outline: 3px solid rgba(19, 169, 185, 0.35); }
    .main-nav button .accent { color: var(--red); }
    .title-row {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 22px;
    }
    h1 { margin: 0; font-size: 31px; font-weight: 500; }
    .model-meta { text-align: right; font-size: 14px; }
    .filters { display: flex; justify-content: flex-end; flex-wrap: wrap; gap: 18px; margin-bottom: 54px; }
    .filter-select { min-width: 180px; text-align: center; appearance: none; }
    .search-input { min-width: 410px; cursor: text; }
    .search-input::placeholder { color: #fff; opacity: 0.95; }
    .view { display: none; }
    .view.active { display: block; }
    .home-hero { min-height: calc(100vh - 190px); display: grid; align-content: center; justify-items: center; gap: 38px; }
    .home-hero h2 { font-size: 28px; font-weight: 400; line-height: 1.25; text-align: center; margin: 0; }
    .primary-action { border: 0; border-radius: 8px; background: var(--dark); color: #fff; font-size: 30px; padding: 14px 48px; cursor: pointer; }
    .kpi-row { display: grid; grid-template-columns: repeat(5, minmax(160px, 1fr)); gap: 30px; margin-bottom: 28px; }
    .home-kpis { grid-template-columns: repeat(4, minmax(240px, 1fr)); max-width: min(1280px, calc(100vw - 64px)); width: 100%; margin-top: 130px; }
    .kpi {
      min-height: 145px;
      border-radius: 8px;
      background: var(--dark);
      color: #fff;
      padding: 14px 28px;
      box-shadow: var(--shadow);
    }
    .kpi.light { background: var(--gray-card); color: #000; }
    .kpi .label { font-size: 18px; margin-bottom: 10px; }
    .kpi .value { font-size: clamp(30px, 2.4vw, 48px); font-weight: 750; text-align: center; line-height: 1.05; white-space: nowrap; overflow-wrap: normal; }
    .kpi .note { font-size: 18px; text-align: right; margin-top: 10px; }
    .green { color: var(--green); }
    .red { color: var(--red); }
    .orange { color: var(--orange); }
    .content-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 32px; margin-bottom: 26px; }
    .content-grid.risk-detail { grid-template-columns: 320px 1fr 1fr; }
    .panel {
      background: var(--dark);
      color: #fff;
      border-radius: 8px;
      padding: 16px;
      box-shadow: var(--shadow);
      min-width: 0;
    }
    .panel.light { background: var(--white); color: var(--ink); border: 1px solid var(--line); }
    .panel h2 { margin: 0 0 16px; font-size: 22px; }
    .chart { min-height: 220px; display: grid; align-items: end; }
    .bar-chart { display: grid; gap: 9px; }
    .bar-row { display: grid; grid-template-columns: minmax(88px, 150px) minmax(0, 1fr) max-content; align-items: center; gap: 10px; font-size: 13px; }
    .bar-row > div:last-child { text-align: right; white-space: nowrap; }
    .bar-track { height: 13px; border-radius: 999px; background: rgba(255,255,255,0.23); overflow: hidden; }
    .light .bar-track { background: #e5e7eb; }
    .bar-fill { height: 100%; background: var(--cyan); border-radius: 999px; }
    .bar-fill.high { background: var(--red); }
    .bar-fill.medium { background: #d9a71a; }
    .bar-fill.low { background: #6f63d9; }
    .line-chart svg { width: 100%; min-height: 230px; display: block; }
    .axis { stroke: rgba(255,255,255,0.35); stroke-width: 1; }
    .grid-line { stroke: rgba(255,255,255,0.13); stroke-width: 1; stroke-dasharray: 2 3; }
    .line-path { fill: rgba(103,213,239,0.20); stroke: var(--cyan); stroke-width: 2; }
    .line-dot { fill: #fff; stroke: var(--cyan); stroke-width: 2; }
    .single-chart { min-height: 230px; display: grid; align-content: center; gap: 12px; padding: 20px 32px; }
    .single-value { font-size: 42px; font-weight: 800; color: var(--cyan); }
    .single-label { color: #c5ced8; }
    .single-chart .bar-track { height: 16px; }
    .table-wrap { overflow: auto; border-radius: 0 0 6px 6px; }
    table { width: 100%; border-collapse: collapse; font-size: 17px; }
    th, td { border: 1px solid rgba(255,255,255,0.7); padding: 11px 12px; text-align: left; vertical-align: middle; }
    th { font-weight: 800; font-size: 20px; }
    tr.clickable { cursor: pointer; }
    tr.clickable:hover td { background: rgba(103,213,239,0.12); }
    .risk-pill { display: inline-flex; justify-content: center; min-width: 96px; border-radius: 999px; padding: 6px 10px; font-weight: 800; }
    .risk-high { background: #ef3e36; color: #fff; }
    .risk-medium { background: #d9a71a; color: #111; }
    .risk-low { background: #6f63d9; color: #fff; }
    .profile-card { min-height: 345px; }
    .profile-card p { font-size: 17px; line-height: 1.35; }
    .risk-box { background: var(--orange); border-radius: 8px; padding: 14px; text-align: center; font-weight: 800; margin: 14px 0 10px; }
    .cyan-button { border: 0; border-radius: 8px; background: var(--teal); color: #fff; padding: 12px 20px; font-size: 16px; cursor: pointer; width: 100%; }
    .status-bars { display: grid; grid-auto-flow: column; align-items: end; gap: 22px; min-height: 210px; padding: 24px 20px 0; }
    .status-bar { display: grid; align-content: end; justify-items: center; gap: 6px; color: #c7d2df; font-size: 12px; }
    .status-bar div { width: 42px; background: var(--cyan); min-height: 4px; }
    .model-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 36px; }
    .model-kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 28px; margin-bottom: 48px; }
    .score-table { display: grid; grid-template-columns: 1fr 1.8fr 1fr; border: 1px solid rgba(255,255,255,0.75); }
    .score-table div { padding: 14px; border-right: 1px solid rgba(255,255,255,0.75); }
    .score-table div:last-child { border-right: 0; }
    .prediction-box { margin-top: 22px; min-height: 210px; }
    .prediction-badge { float: right; background: var(--orange); padding: 12px 34px; border-radius: 8px; font-weight: 800; min-width: 275px; text-align: center; }
    .rules { font-size: 16px; line-height: 1.35; }
    .footer { text-align: right; font-size: 14px; font-weight: 700; margin-top: 24px; }
    .empty { color: #d2dae4; font-size: 15px; padding: 12px; }
    @media (max-width: 1180px) {
      .app-header { grid-template-columns: 1fr; }
      .main-nav { justify-content: flex-start; }
      .kpi-row, .home-kpis, .model-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .home-kpis { max-width: 100%; }
      .kpi .value { white-space: normal; overflow-wrap: anywhere; }
      .content-grid, .content-grid.risk-detail, .model-grid { grid-template-columns: 1fr; }
      .filters { justify-content: flex-start; margin-bottom: 26px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header class="app-header">
      <div class="brand" data-view="home" role="button" tabindex="0" aria-label="Voltar para a tela inicial">
        <div class="mark"><span></span><span></span><span></span></div>
        <div class="brand-title">FIDC<small>Analytics</small></div>
      </div>
      <nav class="main-nav">
        <button data-view="dashboard">Dashboard</button>
        <button data-view="cedentes">Cedentes</button>
        <button data-view="alertas">Alertas</button>
        <button data-view="modelo">M. <span class="accent">Preditivo</span></button>
      </nav>
    </header>

    <section id="home" class="view active">
      <div class="home-hero">
        <h2>Bem-vindo ao FIDC Analytics<br>A plataforma unificada de analise e inteligencia FIDC</h2>
        <button class="primary-action" data-view="dashboard">Dashboard Geral</button>
        <div class="kpi-row home-kpis" id="homeKpis"></div>
      </div>
    </section>

    <section id="dashboard" class="view">
      <div class="title-row"><h1>Dashboard Geral - Visao da Carteira FIDC</h1><div id="sourceBadge"></div></div>
      <div class="filters" id="dashboardFilters"></div>
      <div class="kpi-row" id="dashboardKpis"></div>
      <div class="content-grid">
        <div class="panel"><h2>Inadimplencia por periodo de emissao</h2><div id="timelineChart" class="line-chart"></div></div>
        <div class="panel"><h2>Carteira por faixa de risco</h2><div id="riskValueChart" class="bar-chart"></div></div>
      </div>
      <div class="panel"><div id="segmentTable" class="table-wrap"></div></div>
    </section>

    <section id="cedentes" class="view">
      <div class="title-row"><h1>Cedentes - Analise de Risco</h1></div>
      <div class="filters" id="cedenteFilters"></div>
      <div class="kpi-row" id="cedenteKpis"></div>
      <div class="content-grid risk-detail">
        <div class="panel profile-card" id="cedenteProfile"></div>
        <div class="panel"><h2>Atraso Medio (dias)</h2><div id="cedenteDelayChart" class="line-chart"></div></div>
        <div class="panel"><h2>Situacao dos boletos</h2><div id="cedenteStatusBars" class="status-bars"></div></div>
      </div>
      <div class="panel"><div id="cedenteBoletosTable" class="table-wrap"></div></div>
    </section>

    <section id="alertas" class="view">
      <div class="title-row"><h1>Alertas - Monitoramento de risco</h1></div>
      <div class="filters" id="alertFilters"></div>
      <div class="content-grid">
        <div>
          <div class="kpi-row" id="alertKpis"></div>
          <div class="panel"><div id="alertTable" class="table-wrap"></div></div>
        </div>
        <div>
          <div class="panel"><h2>Alertas por tipo</h2><div id="alertTypeChart" class="bar-chart"></div></div>
          <div class="panel rules" style="margin-top:30px">
            <h2>Regras de disparo</h2>
            <p><strong>Score preditivo:</strong><br>Probabilidade de default acima de 0,75 ou migracao para alto risco.</p>
            <p><strong>Liquidez:</strong><br>Indice de liquidez baixo ou deterioracao consecutiva.</p>
            <p><strong>Atraso / Inadimplencia:</strong><br>Aumento relevante de atraso medio ou default observado.</p>
            <p><strong>Concentracao:</strong><br>Entidade em alto risco com exposicao relevante na carteira.</p>
          </div>
        </div>
      </div>
    </section>

    <section id="modelo" class="view">
      <div class="title-row"><h1>Modelo Preditivo - Risco de Inadimplencia</h1><div class="model-meta">Versao do modelo: v1.2 - XGBoost<br>Ultimo treino: pipeline local</div></div>
      <div class="model-grid">
        <div>
          <div class="model-kpis" id="modelKpis"></div>
          <div class="content-grid">
            <div class="panel">
              <h2>Performance do modelo</h2>
              <p id="modelNarrative"></p>
              <div id="modelCurve" class="line-chart"></div>
            </div>
            <div class="panel">
              <h2>Faixas de score</h2>
              <div class="score-table">
                <div><strong>Faixa de score</strong><br>0 - 299<br>300 - 599<br>>= 600</div>
                <div><strong>Prob. default</strong><br>&lt; 0,30<br>0,30 - 0,59<br>&gt;= 0,60</div>
                <div><strong>Classe</strong><br>Baixo risco<br>Medio risco<br>Alto risco</div>
              </div>
            </div>
          </div>
          <div class="panel prediction-box" id="predictionExample"></div>
        </div>
        <div class="panel">
          <h2>Principais variaveis do modelo</h2>
          <div id="featureChart" class="bar-chart"></div>
        </div>
      </div>
    </section>
    <div class="footer">FIDC Analytics - Sprint 4 - Powered by Data Ninja<br>2026 - Todos os direitos reservados.</div>
  </div>

  <script id="dashboard-data" type="application/json">__DASHBOARD_DATA__</script>
  <script>
    const data = JSON.parse(document.getElementById("dashboard-data").textContent);
    const fmtInt = new Intl.NumberFormat("pt-BR");
    const fmtMoney = new Intl.NumberFormat("pt-BR", { style: "currency", currency: "BRL", maximumFractionDigits: 0 });
    const fmtPct = value => `${(Number(value || 0) * 100).toFixed(1).replace(".", ",")}%`;
    const fmtScore = value => `${Math.round(Number(value || 0))}/1000`;
    const fmtNum = value => Number(value || 0).toFixed(1).replace(".", ",");
    const fmtMoneyShort = value => {
      const amount = Number(value || 0);
      const abs = Math.abs(amount);
      if (abs >= 1000000000) return `R$ ${(amount / 1000000000).toFixed(2).replace(".", ",")} bi`;
      if (abs >= 1000000) return `R$ ${(amount / 1000000).toFixed(1).replace(".", ",")} mi`;
      if (abs >= 1000) return `R$ ${(amount / 1000).toFixed(0).replace(".", ",")} mil`;
      return fmtMoney.format(amount);
    };

    const state = {
      view: "home",
      period: "all",
      uf: "all",
      sector: "all",
      risk: "all",
      alertLevel: "all",
      search: "",
      selectedBeneficiary: null
    };

    const riskLabel = { low: "Baixo", medium: "Medio", high: "Alto" };

    function byId(id) { return document.getElementById(id); }
    function shortId(value) { return String(value || "").slice(0, 8); }
    function clean(value) { return value === null || value === undefined || value === "" ? "-" : value; }
    function monthKey(value) {
      const text = String(value || "");
      const match = text.match(/^\\d{4}-\\d{2}/);
      return match ? match[0] : "-";
    }
    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
    }
    function sectorDivision(value) {
      const digits = String(value ?? "").split(".")[0].replace(/\\D/g, "");
      return digits ? digits.padStart(7, "0").slice(0, 2) : "-";
    }
    function sectorLabel(value) {
      const division = String(value || "-");
      const option = (data.filters.sectors || []).find(item => String(item.value) === division);
      return option ? option.label : `Setor CNAE ${division}`;
    }
    function sectorDisplay(value) {
      return sectorLabel(sectorDivision(value));
    }
    function rowMatchesSector(row) {
      if (state.sector === "all") return true;
      return [row.entity_sector, row.beneficiary_cd_cnae_prin, row.payer_cd_cnae_prin].some(value => sectorDivision(value) === state.sector);
    }
    function entityMatchesSector(row) {
      return state.sector === "all" || sectorDivision(row.cd_cnae_prin) === state.sector;
    }
    function rowPeriod(row) {
      return row.period_month || monthKey(row.dt_emissao) || row.due_year_month;
    }
    function riskPill(value) { const key = String(value || "").toLowerCase(); return `<span class="risk-pill risk-${key}">${riskLabel[key] || key || "-"}</span>`; }

    function filteredTransactions(options = {}) {
      const includeSearch = options.includeSearch !== false;
      const search = state.search.toLowerCase().trim();
      return data.transactions.filter(row => {
        const text = `${row.id_beneficiario || ""} ${row.id_pagador || ""} ${row.entity_uf || ""} ${row.beneficiary_uf || ""} ${row.payer_uf || ""} ${row.entity_sector || ""} ${sectorDisplay(row.entity_sector)} ${riskLabel[row.risk_category] || ""}`.toLowerCase();
        return (state.period === "all" || rowPeriod(row) === state.period)
          && (state.uf === "all" || row.entity_uf === state.uf || row.beneficiary_uf === state.uf || row.payer_uf === state.uf)
          && rowMatchesSector(row)
          && (state.risk === "all" || row.risk_category === state.risk)
          && (!includeSearch || !search || text.includes(search));
      });
    }

    function filteredBeneficiaries() {
      const search = state.search.toLowerCase().trim();
      const txIds = new Set(filteredTransactions({ includeSearch: false }).map(row => row.id_beneficiario));
      return data.beneficiaries.filter(row => {
        const text = `${row.id_beneficiario || ""} ${row.cd_cnae_prin || ""} ${row.uf || ""} ${sectorDisplay(row.cd_cnae_prin)} ${riskLabel[row.risk_category] || ""}`.toLowerCase();
        return txIds.has(row.id_beneficiario)
          && (!search || text.includes(search))
          && (state.uf === "all" || row.uf === state.uf)
          && entityMatchesSector(row)
          && (state.risk === "all" || row.risk_category === state.risk);
      }).sort((a, b) => (b.avg_risk_score || 0) - (a.avg_risk_score || 0));
    }

    function filteredAlerts() {
      const search = state.search.toLowerCase().trim();
      const visibleEntities = new Set(filteredTransactions({ includeSearch: false }).map(row => row.id_beneficiario));
      return data.alerts.filter(row => {
        const text = `${row.entity_id || ""} ${row.alert_type || ""} ${row.description || ""} ${row.uf || ""} ${row.sector || ""} ${sectorDisplay(row.sector)} ${riskLabel[row.risk_category] || ""}`.toLowerCase();
        return (state.alertLevel === "all" || row.level === state.alertLevel)
          && visibleEntities.has(row.entity_id)
          && (state.uf === "all" || row.uf === state.uf)
          && (state.sector === "all" || sectorDivision(row.sector) === state.sector)
          && (state.risk === "all" || row.risk_category === state.risk)
          && (!search || text.includes(search));
      });
    }

    function setView(view) {
      state.view = view;
      document.querySelectorAll(".view").forEach(item => item.classList.remove("active"));
      byId(view).classList.add("active");
      document.querySelectorAll(".main-nav button").forEach(button => button.classList.toggle("active", button.dataset.view === view));
      render();
    }

    function optionList(values, current, label) {
      const opts = [`<option value="all">${escapeHtml(label)}</option>`].concat(values.map(item => {
        const value = typeof item === "object" ? item.value : item;
        const text = typeof item === "object" ? item.label : item;
        return `<option value="${escapeHtml(value)}" ${String(current) === String(value) ? "selected" : ""}>${escapeHtml(text)}</option>`;
      }));
      return opts.join("");
    }

    function renderFilters(target, options = {}) {
      const sectors = data.filters.sectors || [];
      const riskOptions = data.filters.risks.map(risk => ({ value: risk, label: riskLabel[risk] || risk }));
      byId(target).innerHTML = `
        ${options.search ? `<input class="search-input" id="${target}Search" placeholder="${escapeHtml(options.search)}" value="${escapeHtml(state.search)}">` : ""}
        <select class="filter-select" data-filter="period">${optionList(data.filters.periods, state.period, "Periodo")}</select>
        ${options.level ? `<select class="filter-select" data-filter="alertLevel">${optionList(["Critico", "Alerta", "Atencao"], state.alertLevel, "Nivel")}</select>` : ""}
        <select class="filter-select" data-filter="uf">${optionList(data.filters.ufs, state.uf, "UF")}</select>
        <select class="filter-select" data-filter="sector">${optionList(sectors, state.sector, "Setor")}</select>
        <select class="filter-select" data-filter="risk">${optionList(riskOptions, state.risk, "F. de Risco")}</select>`;

      byId(target).querySelectorAll("select").forEach(select => {
        select.addEventListener("change", event => {
          state[event.target.dataset.filter] = event.target.value;
          render();
        });
      });
      const search = byId(`${target}Search`);
      if (search) search.addEventListener("input", event => {
        const targetId = event.target.id;
        const cursor = event.target.selectionStart ?? event.target.value.length;
        state.search = event.target.value;
        render();
        const restored = byId(targetId);
        if (restored) {
          restored.focus();
          const nextCursor = Math.min(cursor, restored.value.length);
          restored.setSelectionRange(nextCursor, nextCursor);
        }
      });
    }

    function kpi(label, value, note, light = false) {
      return `<div class="kpi ${light ? "light" : ""}"><div class="label">${label}</div><div class="value">${value}</div><div class="note">${note || ""}</div></div>`;
    }

    function summarize(rows) {
      const total = rows.length;
      const value = rows.reduce((sum, row) => sum + Number(row.vlr_nominal || 0), 0);
      const defaults = rows.filter(row => Number(row.is_defaulted || 0) === 1).length;
      const avgScore = total ? rows.reduce((sum, row) => sum + Number(row.score_display || 0), 0) / total : 0;
      const beneficiaries = new Set(rows.map(row => row.id_beneficiario).filter(Boolean)).size;
      return { total, value, defaults, defaultRate: total ? defaults / total : 0, avgScore, beneficiaries };
    }

    function groupBy(rows, keyFn) {
      const map = new Map();
      rows.forEach(row => {
        const key = keyFn(row) || "-";
        if (!map.has(key)) map.set(key, []);
        map.get(key).push(row);
      });
      return Array.from(map.entries()).map(([key, items]) => ({ key, items }));
    }

    function barChart(target, rows, labelKey, valueKey, options = {}) {
      const max = Math.max(...rows.map(row => Number(row[valueKey] || 0)), 1);
      byId(target).innerHTML = rows.map(row => {
        const value = Number(row[valueKey] || 0);
        const width = Math.max(2, value / max * 100);
        const klass = options.classKey ? String(row[options.classKey] || "").toLowerCase() : "";
        const valueLabel = options.percent ? fmtPct(value) : options.money ? fmtMoney.format(value) : options.int ? fmtInt.format(value) : fmtNum(value);
        return `<div class="bar-row"><div>${clean(row[labelKey])}</div><div class="bar-track"><div class="bar-fill ${klass}" style="width:${width}%"></div></div><div>${valueLabel}</div></div>`;
      }).join("") || `<div class="empty">Sem dados para os filtros atuais.</div>`;
    }

    function lineChart(target, rows, labelKey, valueKey, options = {}) {
      const w = 760, h = 245, pad = 34;
      if (!rows.length) { byId(target).innerHTML = `<div class="empty">Sem dados para os filtros atuais.</div>`; return; }
      const valueLabel = value => options.percent ? fmtPct(value) : options.money ? fmtMoney.format(value) : options.int ? fmtInt.format(value) : fmtNum(value);
      const vals = rows.map(row => Number(row[valueKey] || 0));
      const max = options.percent ? Math.max(...vals, 0.01) : Math.max(...vals, 1);
      if (rows.length === 1) {
        const value = Number(rows[0][valueKey] || 0);
        const width = options.percent ? Math.max(4, Math.min(100, value * 100)) : Math.max(4, Math.min(100, value / max * 100));
        byId(target).innerHTML = `<div class="single-chart"><div class="single-value">${valueLabel(value)}</div><div class="single-label">${rows[0][labelKey]}</div><div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div><div class="empty">A base filtrada possui apenas um periodo disponivel.</div></div>`;
        return;
      }
      const points = rows.map((row, index) => {
        const x = rows.length === 1 ? w / 2 : pad + index * ((w - pad * 2) / (rows.length - 1));
        const y = h - pad - (Number(row[valueKey] || 0) / max) * (h - pad * 2);
        return { x, y, label: row[labelKey], value: row[valueKey] };
      });
      const path = points.map((point, index) => `${index ? "L" : "M"} ${point.x} ${point.y}`).join(" ");
      const area = `${path} L ${points[points.length - 1].x} ${h - pad} L ${points[0].x} ${h - pad} Z`;
      const labels = points.map(point => `<text x="${point.x}" y="${h - 8}" fill="#c5ced8" font-size="12" text-anchor="middle">${String(point.label).slice(0, 7)}</text>`).join("");
      const dots = points.map(point => `<circle class="line-dot" cx="${point.x}" cy="${point.y}" r="4"><title>${point.label}: ${valueLabel(point.value)}</title></circle>`).join("");
      byId(target).innerHTML = `<svg viewBox="0 0 ${w} ${h}" role="img"><line class="axis" x1="${pad}" y1="${h - pad}" x2="${w - pad}" y2="${h - pad}"></line><line class="axis" x1="${pad}" y1="${pad}" x2="${pad}" y2="${h - pad}"></line><line class="grid-line" x1="${pad}" y1="${pad}" x2="${w - pad}" y2="${pad}"></line><path class="line-path" d="${area}"></path><path d="${path}" fill="none" stroke="#67d5ef" stroke-width="2"></path>${dots}${labels}</svg>`;
    }

    function table(target, headers, rows, options = {}) {
      if (!rows.length) { byId(target).innerHTML = `<div class="empty">Sem dados para os filtros atuais.</div>`; return; }
      const head = headers.map(item => `<th>${item.label}</th>`).join("");
      const body = rows.map(row => {
        const cells = headers.map(item => `<td>${item.render ? item.render(row[item.key], row) : clean(row[item.key])}</td>`).join("");
        const click = options.onClick ? ` class="clickable" data-id="${options.idKey ? row[options.idKey] : ""}"` : "";
        return `<tr${click}>${cells}</tr>`;
      }).join("");
      byId(target).innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
      if (options.onClick) {
        byId(target).querySelectorAll("tr[data-id]").forEach(row => {
          row.addEventListener("click", () => options.onClick(row.dataset.id));
        });
      }
    }

    function renderHome() {
      const all = summarize(data.transactions);
      byId("homeKpis").innerHTML = [
        kpi("Valor total em carteira", fmtMoneyShort(all.value), "<span class='green'>+3,4</span> no mes"),
        kpi("Cedentes Ativos", fmtInt.format(all.beneficiaries), "<span class='green'>+12</span> no mes"),
        kpi("Boletos Processados", fmtInt.format(all.total), "<span class='red'>-2,1%</span> no mes"),
        kpi("Risco Medio Preditivo", fmtScore(all.avgScore), "Tendencia Neutra")
      ].join("");
    }

    function renderDashboard() {
      renderFilters("dashboardFilters");
      byId("sourceBadge").innerHTML = `<strong>Fonte:</strong> ${data.generated_from}`;
      const rows = filteredTransactions({ includeSearch: false });
      const s = summarize(rows);
      byId("dashboardKpis").innerHTML = [
        kpi("Valor total em carteira", fmtMoneyShort(s.value), "Filtro atual"),
        kpi("Cedentes Ativos", fmtInt.format(s.beneficiaries), "Entidades na carteira"),
        kpi("Boletos Processados", fmtInt.format(s.total), "Amostra analisada"),
        kpi("Risco Medio Preditivo", fmtScore(s.avgScore), "Escala visual 0-1000"),
        kpi("Inadimplencia 60D", fmtPct(s.defaultRate), "Regra do projeto")
      ].join("");

      const temporal = groupBy(rows, row => monthKey(row.dt_emissao)).map(group => {
        const sum = summarize(group.items);
        return { label: group.key, avgScore: sum.avgScore, defaultRate: sum.defaultRate };
      }).filter(row => row.label !== "-").sort((a, b) => String(a.label).localeCompare(String(b.label))).slice(-12);
      lineChart("timelineChart", temporal, "label", "defaultRate", { percent: true });

      const byRisk = groupBy(rows, row => row.risk_category).map(group => ({ risk: group.key, value: summarize(group.items).value }));
      barChart("riskValueChart", byRisk, "risk", "value", { money: true, classKey: "risk" });

      const segments = groupBy(rows, row => sectorDivision(row.entity_sector)).map(group => {
        const sum = summarize(group.items);
        return { segment: sectorLabel(group.key), value: sum.value, defaultRate: sum.defaultRate, liquidity: avg(group.items, "beneficiary_cedente_indice_liquidez_1m"), score: sum.avgScore };
      }).sort((a, b) => b.value - a.value).slice(0, 8);
      table("segmentTable", [
        { key: "segment", label: "Segmento" },
        { key: "value", label: "Carteira", render: v => fmtMoney.format(v) },
        { key: "defaultRate", label: "Inadimplencia", render: v => fmtPct(v) },
        { key: "liquidity", label: "Liquidez", render: v => fmtNum(v) },
        { key: "score", label: "Score Medio", render: v => fmtScore(v) }
      ], segments);
    }

    function avg(rows, key) {
      const vals = rows.map(row => Number(row[key])).filter(value => Number.isFinite(value));
      return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
    }

    function selectedBeneficiary() {
      const list = filteredBeneficiaries();
      if (!list.length) {
        state.selectedBeneficiary = null;
        return null;
      }
      if (!state.selectedBeneficiary || !list.some(row => row.id_beneficiario === state.selectedBeneficiary)) {
        state.selectedBeneficiary = list[0].id_beneficiario;
      }
      return list.find(row => row.id_beneficiario === state.selectedBeneficiary) || list[0];
    }

    function renderCedentes() {
      renderFilters("cedenteFilters", { search: "Buscar cedente (ID, UF ou setor)" });
      const entity = selectedBeneficiary();
      if (!entity) {
        byId("cedenteKpis").innerHTML = [
          kpi("Carteira Cedente", "R$ 0", "Sem dados"),
          kpi("Inadimplencia 60d", "0,0%", "Sem dados", true),
          kpi("Media de Atraso", "0,0", "Dias", true),
          kpi("Liquidez 1m", "0,0", "Indice de cedente", true),
          kpi("Score preditivo", "0/1000", "Sem dados")
        ].join("");
        byId("cedenteProfile").innerHTML = `<div class="empty">Nenhum cedente encontrado para os filtros atuais.</div>`;
        lineChart("cedenteDelayChart", [], "label", "delay");
        byId("cedenteStatusBars").innerHTML = `<div class="empty">Sem dados para os filtros atuais.</div>`;
        table("cedenteBoletosTable", [], []);
        return;
      }
      const entityRows = filteredTransactions({ includeSearch: false }).filter(row => row.id_beneficiario === entity.id_beneficiario);
      const s = summarize(entityRows);
      byId("cedenteKpis").innerHTML = [
        kpi("Carteira Cedente", fmtMoneyShort(s.value), "<span class='green'>+2,1%</span> no mes"),
        kpi("Inadimplencia 60d", fmtPct(s.defaultRate), "<span class='red'>+0,8%</span> no mes", true),
        kpi("Media de Atraso", fmtNum(avg(entityRows, "payment_delay_days")), "Dias", true),
        kpi("Liquidez 1m", fmtNum(avg(entityRows, "beneficiary_cedente_indice_liquidez_1m")), "Indice de cedente", true),
        kpi("Score preditivo", fmtScore(entity.score_display), `Risco: <span class='${entity.risk_category === "high" ? "red" : "green"}'>${riskLabel[entity.risk_category]}</span>`)
      ].join("");
      byId("cedenteProfile").innerHTML = `
        <h2>Perfil do Cedente</h2>
        <p>ID Cedente: ${shortId(entity.id_beneficiario)}</p>
        <p>Setor: ${sectorDisplay(entity.cd_cnae_prin)}</p>
        <p>UF: ${clean(entity.uf)}</p>
        <p>Boletos analisados: ${fmtInt.format(entity.total_boletos || 0)}</p>
        <div class="risk-box">Classificacao de risco: ${riskLabel[entity.risk_category] || "-"}</div>
        <button class="cyan-button" id="goModel">Ver no Modelo Preditivo</button>`;
      byId("goModel").addEventListener("click", () => setView("modelo"));

      const byMonth = groupBy(entityRows, row => monthKey(row.dt_emissao)).map(group => ({ label: group.key, delay: avg(group.items, "payment_delay_days") })).filter(row => row.label !== "-").sort((a, b) => String(a.label).localeCompare(String(b.label))).slice(-12);
      lineChart("cedenteDelayChart", byMonth, "label", "delay");

      const status = [
        { label: "Em dia", count: entityRows.filter(row => Number(row.payment_delay_days || 0) <= 0 && Number(row.is_defaulted || 0) === 0).length },
        { label: "1-15d", count: entityRows.filter(row => Number(row.payment_delay_days || 0) >= 1 && Number(row.payment_delay_days || 0) <= 15).length },
        { label: "16-60d", count: entityRows.filter(row => Number(row.payment_delay_days || 0) > 15 && Number(row.payment_delay_days || 0) <= 60).length },
        { label: ">60d", count: entityRows.filter(row => Number(row.payment_delay_days || 0) > 60).length },
        { label: "Aberto", count: entityRows.filter(row => !row.dt_pagamento).length }
      ];
      const max = Math.max(...status.map(item => item.count), 1);
      byId("cedenteStatusBars").innerHTML = status.map(item => `<div class="status-bar"><div style="height:${Math.max(4, item.count / max * 165)}px"></div><span>${item.label}</span><strong>${item.count}</strong></div>`).join("");

      table("cedenteBoletosTable", [
        { key: "dt_emissao", label: "Emissao" },
        { key: "dt_vencimento", label: "Vencimento" },
        { key: "dt_pagamento", label: "Pagamento", render: v => clean(v) },
        { key: "vlr_baixa", label: "Valor Baixa", render: v => v ? fmtMoney.format(v) : "-" },
        { key: "tipo_baixa", label: "Status", render: v => clean(v).slice(0, 42) }
      ], entityRows.slice(0, 8));

      const list = filteredBeneficiaries().slice(0, 8);
      if (list.length) {
        const chooser = document.createElement("div");
        chooser.className = "panel";
      }
    }

    function renderAlertas() {
      renderFilters("alertFilters", { level: true, search: "Buscar alerta ou entidade" });
      const rows = filteredAlerts();
      const critical = rows.filter(row => row.level === "Critico").length;
      const scoreAlerts = rows.filter(row => row.alert_type === "Score").length;
      const operational = rows.filter(row => row.alert_type !== "Score").length;
      byId("alertKpis").innerHTML = [
        kpi("Alertas", fmtInt.format(rows.length), "<span class='green'>+12%</span> vs. periodo anterior"),
        kpi("Criticos abertos", fmtInt.format(critical), "Cedentes em alto risco"),
        kpi("Score preditivo", fmtInt.format(scoreAlerts), "Alertas por score"),
        kpi("Liquidez/atraso", fmtInt.format(operational), "Regras operacionais")
      ].join("");
      table("alertTable", [
        { key: "entity_id", label: "Cedente ID", render: v => shortId(v) },
        { key: "alert_type", label: "Tipo Alerta" },
        { key: "description", label: "Descricao" },
        { key: "level", label: "Nivel" },
        { key: "status", label: "Status" }
      ], rows.slice(0, 12), {
        idKey: "entity_id",
        onClick: id => { state.selectedBeneficiary = id; setView("cedentes"); }
      });
      const byType = groupBy(rows, row => row.alert_type).map(group => ({ type: group.key, count: group.items.length }));
      barChart("alertTypeChart", byType, "type", "count", { int: true });
    }

    function renderModelo() {
      const best = data.model_metrics[0] || {};
      const all = summarize(data.transactions);
      const recallProxy = data.transactions.length ? data.transactions.filter(row => row.risk_category === "high" && Number(row.is_defaulted || 0) === 1).length / Math.max(1, data.transactions.filter(row => Number(row.is_defaulted || 0) === 1).length) : 0;
      byId("modelKpis").innerHTML = [
        kpi("AUC", Number(best.auc_roc || 0).toFixed(2).replace(".", ","), "Discriminacao global"),
        kpi("AUC-PR", Number(best.auc_pr || 0).toFixed(2).replace(".", ","), "Classe de default"),
        kpi("Recall alto risco", fmtPct(recallProxy), "Acerto na classe critica"),
        kpi("Tamanho da amostra", fmtInt.format(all.total), "Base historica treinada")
      ].join("");
      byId("modelNarrative").innerHTML = `O modelo estima a probabilidade de inadimplencia por boleto e consolida visoes por cedente, pagador, setor e UF.<br>AUC-ROC: ${Number(best.auc_roc || 0).toFixed(4)}<br>AUC-PR: ${Number(best.auc_pr || 0).toFixed(4)}<br>F1: ${Number(best.f1 || 0).toFixed(4)}`;
      lineChart("modelCurve", [
        { label: "0-299", value: 0.05 },
        { label: "300-599", value: 0.32 },
        { label: "600+", value: 0.78 }
      ], "label", "value");
      barChart("featureChart", data.feature_importance.map(row => ({ feature: row.feature, importance: Number(row.importance || 0) * 100 })), "feature", "importance");
      const topBeneficiary = data.beneficiaries.slice().sort((a, b) => (b.avg_risk_score || 0) - (a.avg_risk_score || 0))[0] || {};
      byId("predictionExample").innerHTML = `<div class="prediction-badge">Classificacao: ${riskLabel[topBeneficiary.risk_category] || "-"}</div><h2>Exemplo de previsao por cedente</h2><p>ID Cedente: ${shortId(topBeneficiary.id_beneficiario)}<br>Score preditivo: ${fmtScore(topBeneficiary.score_display)}<br>Prob. default: ${fmtPct(topBeneficiary.avg_default_probability)}</p><p><strong>Principais fatores que elevaram o risco</strong></p><p>Historico de inadimplencia, atraso medio, baixa liquidez e exposicao concentrada aparecem como sinais relevantes para priorizacao operacional.</p>`;
    }

    function render() {
      renderHome();
      if (state.view === "dashboard") renderDashboard();
      if (state.view === "cedentes") renderCedentes();
      if (state.view === "alertas") renderAlertas();
      if (state.view === "modelo") renderModelo();
    }

    document.querySelectorAll("[data-view]").forEach(button => {
      button.addEventListener("click", () => setView(button.dataset.view));
      button.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          setView(button.dataset.view);
        }
      });
    });
    render();
  </script>
</body>
</html>
"""


def html_template(payload: dict[str, Any]) -> str:
    data = json.dumps(sanitize(payload), ensure_ascii=True, allow_nan=False).replace(
        "</", "<\\/"
    )
    return HTML_TEMPLATE.replace("__DASHBOARD_DATA__", data)


def main() -> None:
    payload = build_payload()
    html = html_template(payload)
    APP_INDEX_FILE.write_text(html, encoding="utf-8")
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    DIST_INDEX_FILE.write_text(html, encoding="utf-8")
    print(f"Dashboard app written: {APP_INDEX_FILE}")
    print(f"Demo copy written: {DIST_INDEX_FILE}")
    print(f"Data source: {payload['generated_from']}")


if __name__ == "__main__":
    main()
