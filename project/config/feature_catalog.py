"""
Feature catalog — single source of truth for ML column classification.

Every column in the features dataset belongs to exactly one category:
    IDENTIFIER_COLS         — row keys, never model inputs
    TARGET_COL              — prediction label
    DATE_COLS               — kept for analysis/BI, excluded from ML
    LEAKY_COLS              — excluded from ML (direct or indirect leakage)
    SAFE_NUMERIC_FEATURES   — ready-to-use numeric model inputs
    SAFE_CATEGORICAL_FEATURES — categorical inputs requiring encoding

Leakage taxonomy:
    Direct   — columns that encode the outcome itself (dt_pagamento, vlr_baixa, etc.)
    Indirect — batch aggregations whose computation uses outcome or future boletos
               from the same dataset (payer_default_rate, beneficiary_boleto_count, etc.)
               Safe in a time-based split; unsafe in a random split with this dataset.
"""

IDENTIFIER_COLS: list[str] = [
    "id_boleto",
    "id_pagador",
    "id_beneficiario",
]

TARGET_COL: str = "is_defaulted"

DATE_COLS: list[str] = [
    "dt_emissao",
    "dt_vencimento",
]

LEAKY_COLS: list[str] = [
    # direct leakage: encode the payment outcome
    "dt_pagamento",
    "payment_delay_days",
    "days_overdue",
    "vlr_baixa",
    "tipo_baixa",        # populated at settlement, not at issuance
    "settlement_code",   # derived from tipo_baixa
    # indirect leakage: batch aggregations that use the outcome
    "payer_default_rate",          # uses is_defaulted from same batch
    "payer_avg_delay_days",        # uses payment_delay_days (leaky)
    "payer_max_delay_days",        # uses payment_delay_days (leaky)
    "payer_boleto_count",          # aggregates future boletos from same batch
    "payer_avg_nominal_value",     # aggregates same batch
    "payer_avg_term_days",         # aggregates same batch
    "beneficiary_default_rate",    # uses is_defaulted from same batch
    "beneficiary_boleto_count",    # aggregates same batch
    "beneficiary_avg_nominal_value",       # aggregates same batch
    "beneficiary_total_portfolio_value",   # aggregates same batch
]

SAFE_NUMERIC_FEATURES: list[str] = [
    # boleto characteristics (known at issuance)
    "vlr_nominal",
    "boleto_term_days",
    "due_month",       # always 5 in current dataset; retained for production
    "due_day_of_week",
    # payer external scores (Nuclea historical data, available at issuance)
    "payer_sacado_indice_liquidez_1m",
    "payer_cedente_indice_liquidez_1m",
    "payer_score_materialidade_evolucao",
    "payer_media_atraso_dias",
    "payer_indicador_liquidez_quantitativo_3m",
    "payer_share_vl_inad_pag_bol_6_a_15d",
    "payer_score_quantidade_v2",
    "payer_score_materialidade_v2",
    # beneficiary external scores
    "beneficiary_sacado_indice_liquidez_1m",
    "beneficiary_cedente_indice_liquidez_1m",
    "beneficiary_score_materialidade_evolucao",
    "beneficiary_media_atraso_dias",
    "beneficiary_indicador_liquidez_quantitativo_3m",
    "beneficiary_share_vl_inad_pag_bol_6_a_15d",
    "beneficiary_score_quantidade_v2",
    "beneficiary_score_materialidade_v2",
]

SAFE_CATEGORICAL_FEATURES: list[str] = [
    "tipo_especie",              # 9 fixed categories, known at issuance
    "payer_uf",                  # 27 Brazilian states (7% null → encoded as UNKNOWN)
    "beneficiary_uf",            # 27 states (14.7% null → encoded as UNKNOWN)
    "payer_cd_cnae_prin",        # CNAE activity code — high cardinality
    "beneficiary_cd_cnae_prin",  # CNAE activity code — high cardinality
]
