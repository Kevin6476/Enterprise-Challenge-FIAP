"""
Single source of truth for dataset schemas derived from EDA (exploratory_data_analysis.ipynb)

Each ColumnSpec captures the contract expected at ingestion time:
  dtype            : expected pandas dtype string
  nullable         : whether null values are structurally acceptable
  unique           : whether all values must be distinct within the column
  min_value        : minimum accepted numeric value (inclusive)
  max_value        : maximum accepted numeric value (inclusive)
  allowed          : frozenset of accepted string values; None means any value is valid
  description      : business meaning of the column
  null_rate_eda    : observed null rate in EDA — used as reference for threshold alerts

Business rules that span multiple columns (conditional nullability):
  - vlr_baixa is null when tipo_baixa code is 5, 6, 7 or 8 (non-financial settlements)
  - tipo_baixa is null when dt_pagamento is null (open / unpaid boletos)
  These rules are enforced by the validator, not by individual ColumnSpecs.
"""

from dataclasses import dataclass
from typing import FrozenSet, Optional

@dataclass(frozen=True)
class ColumnSpec:
    dtype: str
    nullable: bool
    unique: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed: Optional[FrozenSet[str]] = None
    description: str = ""
    null_rate_eda: Optional[float] = None

# Human-readable labels for settlement type codes (tipo_baixa column)
SETTLEMENT_TYPE_CODES: dict[str, str] = {
    "0": "Full interbank settlement",
    "1": "Full intrabank settlement",
    "5": "Full settlement at assignor request",
    "6": "Full settlement via protest",
    "7": "Full settlement by term expiry",
    "8": "Full settlement at destination institution request",
    "9": "Full interbank settlement via STR",
}

# Settlement type codes for which vlr_baixa is structurally null (no financial transfer occurs)
NON_FINANCIAL_SETTLEMENT_CODES: frozenset[str] = frozenset({"5", "6", "7", "8"})

# Values must match the raw CSV strings exactly — do not translate
ALLOWED_SETTLEMENT_TYPES: frozenset[str] = frozenset({
    "0 - Baixa integral interbancaria",
    "1 - Baixa integral intrabancaria",
    "5 - Baixa integral por solicitacao do cedente",
    "6 - Baixa integral por envio para protesto",
    "7 - Baixa integral por decurso de prazo",
    "8 - Baixa integral por solicitacao da instituicao destinataria",
    "9 - Baixa integral interbancaria - Liquidacao via STR",
})

# Values must match the raw CSV strings exactly — do not translate
ALLOWED_INSTRUMENT_TYPES: frozenset[str] = frozenset({
    "DM DUPLICATA MERCANTIL",
    "DMI DUPLICATA MERCANTIL INDICACAO",
    "DS DUPLICATA DE SERVICO",
    "DSI DUPLICATA DE SERVICO INDICACAO",
    "ME MENSALIDADE ESCOLAR",
    "NF NOTA FISCAL",
    "NP NOTA PROMISSORIA",
    "CARTAO DE CREDITO",
    "OUTROS",
})

VALID_STATE_CODES: frozenset[str] = frozenset({
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES",
    "GO", "MA", "MG", "MS", "MT", "PA", "PB", "PE",
    "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC",
    "SE", "SP", "TO",
})

PAYMENTS_SCHEMA: dict[str, ColumnSpec] = {
    "id_boleto": ColumnSpec(
        dtype="object",
        nullable=False,
        unique=True,
        description="Anonymized unique identifier for the boleto (SHA-256 hash).",
        null_rate_eda=0.0,
    ),
    "id_pagador": ColumnSpec(
        dtype="object",
        nullable=False,
        description=(
            "Anonymized payer identifier (CNPJ hash). "
            "100% match rate with id_cnpj in the auxiliary dataset."
        ),
        null_rate_eda=0.0,
    ),
    "id_beneficiario": ColumnSpec(
        dtype="object",
        nullable=False,
        description=(
            "Anonymized beneficiary/assignor identifier (CNPJ hash). "
            "100% match rate with id_cnpj in the auxiliary dataset."
        ),
        null_rate_eda=0.0,
    ),
    "dt_emissao": ColumnSpec(
        dtype="datetime64[ns]",
        nullable=False,
        description="Boleto issuance date.",
        null_rate_eda=0.0,
    ),
    "dt_vencimento": ColumnSpec(
        dtype="datetime64[ns]",
        nullable=False,
        description="Boleto due date.",
        null_rate_eda=0.0,
    ),
    "dt_pagamento": ColumnSpec(
        dtype="datetime64[ns]",
        nullable=True,
        description=(
            "Effective payment date. "
            "Null indicates an open or defaulted boleto (~1% of records)."
        ),
        null_rate_eda=0.0098,
    ),
    "vlr_nominal": ColumnSpec(
        dtype="float64",
        nullable=False,
        min_value=0.01,
        description="Nominal boleto value in BRL. Always positive.",
        null_rate_eda=0.0,
    ),
    "vlr_baixa": ColumnSpec(
        dtype="float64",
        nullable=True,
        min_value=0.01,
        description=(
            "Effective settlement value in BRL. "
            "Null when tipo_baixa code is 5, 6, 7 or 8 — non-financial settlements "
            "(assignor request, protest, term expiry, destination institution). "
            "~11.5% null rate is structurally expected."
        ),
        null_rate_eda=0.1152,
    ),
    "tipo_baixa": ColumnSpec(
        dtype="object",
        nullable=True,
        allowed=ALLOWED_SETTLEMENT_TYPES,
        description=(
            "Settlement type of the boleto. "
            "Null coincides with null dt_pagamento (open boleto)."
        ),
        null_rate_eda=0.0098,
    ),
    "tipo_especie": ColumnSpec(
        dtype="object",
        nullable=False,
        allowed=ALLOWED_INSTRUMENT_TYPES,
        description="Title instrument type (DM, DMI, DS, etc.). Dominated by merchant duplicates (~94%).",
        null_rate_eda=0.0,
    ),
}

AUXILIARY_SCHEMA: dict[str, ColumnSpec] = {
    "id_cnpj": ColumnSpec(
        dtype="object",
        nullable=False,
        unique=True,
        description="Anonymized CNPJ identifier. Primary key.",
        null_rate_eda=0.0,
    ),
    "cd_cnae_prin": ColumnSpec(
        dtype="float64",
        nullable=True,
        min_value=100_000.0,
        max_value=9_999_999.0,
        description=(
            "Primary CNAE activity code (6 or 7 digits). "
            "Stored as float64 due to pandas null handling."
        ),
        null_rate_eda=0.0004,
    ),
    "uf": ColumnSpec(
        dtype="object",
        nullable=True,
        allowed=VALID_STATE_CODES,
        description="Brazilian state code (federative unit). ~7.8% null rate.",
        null_rate_eda=0.0778,
    ),
    "sacado_indice_liquidez_1m": ColumnSpec(
        dtype="float64",
        nullable=True,
        min_value=0.0,
        max_value=1.0,
        description="Payer liquidity index over the last 30 days. Range [0, 1].",
        null_rate_eda=0.0041,
    ),
    "cedente_indice_liquidez_1m": ColumnSpec(
        dtype="float64",
        nullable=True,
        min_value=0.0,
        max_value=1.0,
        description=(
            "Assignor/beneficiary liquidity index over the last 30 days. Range [0, 1]. "
            "High structural null rate (~46.6%) — not a data anomaly."
        ),
        null_rate_eda=0.4659,
    ),
    "score_materialidade_evolucao": ColumnSpec(
        dtype="float64",
        nullable=True,
        min_value=0.0,
        max_value=1000.0,
        description="Financial materiality evolution score. Scale [0, 1000].",
        null_rate_eda=0.0007,
    ),
    "media_atraso_dias": ColumnSpec(
        dtype="float64",
        nullable=True,
        min_value=0.0,
        description="Historical average payment delay in days.",
        null_rate_eda=0.0011,
    ),
    "indicador_liquidez_quantitativo_3m": ColumnSpec(
        dtype="float64",
        nullable=True,
        min_value=0.0,
        max_value=1.0,
        description="Quantitative liquidity indicator over the last 3 months. Range [0, 1].",
        null_rate_eda=0.0043,
    ),
    "share_vl_inad_pag_bol_6_a_15d": ColumnSpec(
        dtype="float64",
        nullable=True,
        min_value=0.0,
        max_value=1.0,
        description="Share of overdue value (6–15 days) over total paid. Range [0, 1].",
        null_rate_eda=0.0011,
    ),
    "score_quantidade_v2": ColumnSpec(
        dtype="float64",
        nullable=True,
        min_value=0.0,
        max_value=1000.0,
        description="Transaction volume score v2. Scale [0, 1000].",
        null_rate_eda=0.0013,
    ),
    "score_materialidade_v2": ColumnSpec(
        dtype="float64",
        nullable=True,
        min_value=0.0,
        max_value=1000.0,
        description="Financial materiality score v2. Scale [0, 1000].",
        null_rate_eda=0.0013,
    ),
}


# Join relationship between datasets
# Both id_pagador and id_beneficiario in payments map to id_cnpj in auxiliary.
# EDA confirmed 100% match rate for both keys — join is safe with how="left".
JOIN_KEYS: dict[str, dict] = {
    "payer": {
        "payments_col": "id_pagador",
        "auxiliary_col": "id_cnpj",
        "expected_match_rate": 1.0,
        "how": "left",
        "description": "Enriches each boleto record with payer data.",
    },
    "beneficiary": {
        "payments_col": "id_beneficiario",
        "auxiliary_col": "id_cnpj",
        "expected_match_rate": 1.0,
        "how": "left",
        "description": "Enriches each boleto record with beneficiary/assignor data.",
    },
}