# FIDC Analytics Platform

> A production-oriented credit risk analytics platform for Brazilian receivables funds (FIDC), featuring end-to-end ETL, machine learning-based default prediction, risk scoring, and BI-ready analytical datasets.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Business Problem](#2-business-problem)
3. [What is a FIDC?](#3-what-is-a-fidc)
4. [System Objectives](#4-system-objectives)
5. [Architecture Overview](#5-architecture-overview)
6. [Project Structure](#6-project-structure)
7. [ETL Pipeline](#7-etl-pipeline)
8. [Feature Engineering](#8-feature-engineering)
9. [Leakage Prevention](#9-leakage-prevention)
10. [Machine Learning Preparation](#10-machine-learning-preparation)
11. [Modeling Strategy](#11-modeling-strategy)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [Risk Scoring](#13-risk-scoring)
14. [Business Intelligence Layer](#14-business-intelligence-layer)
15. [Dashboard Objectives](#15-dashboard-objectives)
16. [Technologies](#16-technologies)
17. [Installation](#17-installation)
18. [Running the Pipeline](#18-running-the-pipeline)
19. [Output Datasets](#19-output-datasets)
20. [Future Improvements](#20-future-improvements)
21. [Academic Conclusions](#21-academic-conclusions)

---

## 1. Project Overview

The **FIDC Analytics Platform** is a modular, end-to-end data and machine learning pipeline designed to support credit risk management in the context of Brazilian boleto receivables portfolios. It ingests raw payment and counterparty data, applies rigorous ETL transformations, engineers predictive features, trains classification models to anticipate default events, and exports a suite of analytical datasets ready for consumption by BI dashboards.

The platform simulates the operational infrastructure of a real FIDC (Fundo de Investimento em Direitos Creditórios), covering the full lifecycle from raw data intake to risk-scored portfolio monitoring.

**Key capabilities at a glance:**

| Capability | Details |
|---|---|
| Data ingestion | CSV → validated, schema-compliant DataFrames |
| ETL | Cleaning, deduplication, payer/beneficiary enrichment |
| Feature engineering | Temporal, behavioral, and sector-level features |
| ML pipeline | Leakage-safe preparation, stratified split, preprocessing |
| Modeling | Logistic Regression, Random Forest, XGBoost |
| Risk scoring | Probability → 0–100 score → categorical risk tier |
| BI exports | 8 Parquet datasets covering portfolio, geography, sector, and time |

---

## 2. Business Problem

A FIDC acquires receivables — primarily boletos (Brazilian bank-issued payment slips) — from companies (cedentes/beneficiários) and assumes the credit risk associated with the underlying obligors (sacados/pagadores).

The core business risk is **default**: a boleto that is not paid on time or not paid at all. This risk must be quantified, monitored, and managed at scale.

**Key business questions the platform addresses:**

- What is the probability that a given boleto will default?
- Which payers (sacados) carry the highest credit risk?
- Which beneficiaries (cedentes) have the most risk-concentrated portfolios?
- How has portfolio default behavior evolved over time?
- Which geographic regions or economic sectors present elevated default rates?
- How is the risk score distributed across the portfolio?

The platform answers these questions through predictive modeling and structured analytical datasets.

**Default Definition**

For this platform, a boleto is classified as **defaulted** if either of the following conditions holds:

```
is_defaulted = 1  if  dt_pagamento IS NULL           (unpaid boleto)
                  OR  payment_delay_days > 0          (paid after due date)
```

This conservative threshold ensures that any late payment — regardless of how few days — is treated as a credit event, consistent with standard FIDC credit risk governance.

---

## 3. What is a FIDC?

A **Fundo de Investimento em Direitos Creditórios (FIDC)** is a type of Brazilian investment vehicle regulated by the Comissão de Valores Mobiliários (CVM) that pools capital from investors and deploys it by purchasing receivables from companies.

```
┌─────────────────────────────────────────────────────────────┐
│                       FIDC FLOW                             │
│                                                             │
│  Company (Cedente)         FIDC              Investors      │
│  ┌────────────────┐   ┌──────────────┐   ┌────────────────┐│
│  │ Has receivables│──▶│ Buys rights  │◀──│ Inject capital ││
│  │ (boletos)      │   │ at discount  │   │ in senior/sub  ││
│  └────────────────┘   └──────┬───────┘   │ quotas         ││
│                              │           └────────────────┘│
│  Payer (Sacado)              │                             │
│  ┌────────────────┐          │                             │
│  │ Owes payment   │──▶ Pays or defaults                   │
│  │ on the boleto  │          │                             │
│  └────────────────┘          ▼                             │
│                       Credit risk borne                     │
│                       by the FIDC                           │
└─────────────────────────────────────────────────────────────┘
```

**Key participants:**

| Role | Portuguese | Description |
|---|---|---|
| Fund | FIDC | Acquires receivables, bears credit risk |
| Beneficiary | Cedente / Beneficiário | Sells receivables to the FIDC |
| Payer | Sacado / Pagador | Obligor who owes payment on each boleto |
| Investors | Cotistas | Provide capital; receive returns from collected payments |

**Risk dynamics:**

- The FIDC purchases boletos at a discount from face value
- If the payer defaults, the FIDC absorbs the loss
- Accurate default prediction is therefore essential to portfolio profitability and investor protection

---

## 4. System Objectives

The platform is built to fulfill the following operational and analytical objectives:

```
┌─────────────────────────────────────────────────────────────────┐
│                     SYSTEM OBJECTIVES                           │
│                                                                 │
│  1. DEFAULT IDENTIFICATION                                      │
│     Detect and quantify non-performing assets in the portfolio  │
│                                                                 │
│  2. PREDICTIVE RISK SCORING                                     │
│     Assign a probabilistic default risk score to every boleto   │
│                                                                 │
│  3. PORTFOLIO QUALITY MONITORING                                │
│     Track default rates, overdue volumes, and risk distribution │
│                                                                 │
│  4. CREDIT / RISK DECISION SUPPORT                              │
│     Provide ranked risk views (payer, beneficiary, sector, UF)  │
│     to inform onboarding, limits, and concentration decisions   │
│                                                                 │
│  5. BI DATASET PRODUCTION                                       │
│     Export structured analytical datasets ready for dashboards  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Architecture Overview

The platform follows a **layered, modular pipeline architecture**. Each layer has a single responsibility and communicates through well-defined Parquet files or in-memory DataFrames.

```
┌──────────────────────────────────────────────────────────────────────┐
│                     PIPELINE ARCHITECTURE                            │
│                                                                      │
│  ┌─────────────┐                                                     │
│  │  Raw Data   │  payments.csv  +  auxiliary.csv                     │
│  └──────┬──────┘                                                     │
│         │                                                            │
│         ▼                                                            │
│  ┌─────────────────────────────────┐                                 │
│  │  Layer 1 — INGESTION            │  Loader + Schema Validator       │
│  │  • Parse dates & numerics       │                                 │
│  │  • Schema contract checks       │                                 │
│  │  • Business rule validation     │                                 │
│  └──────────────┬──────────────────┘                                 │
│                 │                                                    │
│                 ▼                                                    │
│  ┌─────────────────────────────────┐                                 │
│  │  Layer 2 — ETL                  │  Cleaner + Deduplicator +       │
│  │  • Whitespace normalization     │  Enricher                       │
│  │  • Settlement code extraction   │                                 │
│  │  • Deduplication by key         │                                 │
│  │  • Payer & beneficiary join     │                                 │
│  └──────────────┬──────────────────┘                                 │
│                 │                                                    │
│                 ▼                                                    │
│  ┌─────────────────────────────────┐                                 │
│  │  Layer 3 — FEATURE ENGINEERING  │  Temporal + Behavioral +        │
│  │  • Temporal features            │  Aggregations                   │
│  │  • Target variable creation     │                                 │
│  │  • Payer aggregations           │                                 │
│  │  • Beneficiary aggregations     │                                 │
│  └──────────────┬──────────────────┘                                 │
│                 │           payments_features.parquet                │
│                 ▼                                                    │
│  ┌─────────────────────────────────┐                                 │
│  │  Layer 4 — ML PREPARATION       │  Selectors + ML Exporter        │
│  │  • Leakage audit                │                                 │
│  │  • Feature selection (30 cols)  │                                 │
│  │  • Stratified train/test split  │                                 │
│  │  • Encoding + Imputation        │                                 │
│  └──────────────┬──────────────────┘                                 │
│                 │           X_train / X_test / preprocessors.pkl     │
│                 ▼                                                    │
│  ┌─────────────────────────────────┐                                 │
│  │  Layer 5 — MODELING             │  Trainer + Evaluator +          │
│  │  • Logistic Regression          │  Explainer                      │
│  │  • Random Forest                │                                 │
│  │  • XGBoost (selected)           │                                 │
│  │  • ROC-AUC / PR-AUC / F1        │                                 │
│  └──────────────┬──────────────────┘                                 │
│                 │           xgboost.pkl + model_evaluation.json      │
│                 ▼                                                    │
│  ┌─────────────────────────────────┐                                 │
│  │  Layer 6 — RISK SCORING         │  Scorer                         │
│  │  • Apply model to full dataset  │                                 │
│  │  • default_probability [0,1]    │                                 │
│  │  • risk_score [0,100]           │                                 │
│  │  • risk_category (low/med/high) │                                 │
│  └──────────────┬──────────────────┘                                 │
│                 │           boleto_scores.parquet                    │
│                 ▼                                                    │
│  ┌─────────────────────────────────┐                                 │
│  │  Layer 7 — BI ANALYTICS         │  8 analytical datasets          │
│  │  • Portfolio overview           │                                 │
│  │  • Payer & beneficiary risk     │                                 │
│  │  • Geographic (UF) analysis     │                                 │
│  │  • Sector (CNAE) analysis       │                                 │
│  │  • Temporal trends              │                                 │
│  │  • Score distribution           │                                 │
│  │  • Feature importance           │                                 │
│  └─────────────────────────────────┘                                 │
│                 │           data/outputs/bi/*.parquet                │
│                 ▼                                                    │
│         [BI Dashboard / Reporting Layer]                             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 6. Project Structure

```
project/
│
├── requirements.txt                    # Python dependencies
│
├── config/                             # Configuration layer (single source of truth)
│   ├── __init__.py
│   ├── schema.py                       # Data contract: column types, constraints, EDA baselines
│   ├── constants.py                    # Derived constants from schema (column lists)
│   ├── feature_catalog.py             # ML feature classification: safe vs. leaky
│   ├── business_rules.py              # Domain constants (e.g., DEFAULT_THRESHOLD_DAYS)
│   └── paths.py                        # Centralized file path configuration
│
├── src/
│   ├── main.py                         # Pipeline orchestrator (entry point)
│   │
│   ├── ingestion/                      # Layer 1 — Data loading & validation
│   │   ├── loader.py                   # CSV ingest with type coercion
│   │   └── validator.py                # Schema & business rule checks
│   │
│   ├── etl/                            # Layer 2 — Transform & enrich
│   │   ├── cleaner.py                  # Whitespace, code normalization
│   │   ├── deduplicator.py             # Key-based deduplication
│   │   └── enricher.py                 # Payer & beneficiary join
│   │
│   ├── features/                       # Layer 3 — Feature engineering
│   │   ├── temporal.py                 # Delay, term, target variable (is_defaulted)
│   │   ├── behavioral.py               # Per-payer behavioral aggregations
│   │   └── aggregations.py             # Per-beneficiary portfolio aggregations
│   │
│   ├── outputs/                        # Layer 4 — ML dataset preparation
│   │   ├── selectors.py                # Feature/target selection + leakage audit
│   │   └── ml_exporter.py              # Split, encode, impute, export
│   │
│   ├── models/                         # Layer 5 — Model training & evaluation
│   │   ├── trainer.py                  # Train LR, RF, XGBoost
│   │   ├── evaluator.py                # Compute AUC-ROC, AUC-PR, F1
│   │   └── explainer.py                # Extract feature importance
│   │
│   ├── scoring/                        # Layer 6 — Risk scoring
│   │   └── scorer.py                   # Probability → score → category
│   │
│   ├── analytics/                      # Layer 7 — BI analytical datasets
│   │   ├── bi_exporter.py              # Orchestrates all BI exports
│   │   ├── portfolio.py                # Executive KPI summary
│   │   ├── payer.py                    # Per-payer risk ranking
│   │   ├── beneficiary.py              # Per-beneficiary risk ranking
│   │   ├── temporal.py                 # Trends by due month
│   │   ├── geographic.py               # Default rate by Brazilian state (UF)
│   │   ├── sector.py                   # Default rate by economic sector (CNAE)
│   │   └── score_distribution.py       # Probability bucket histogram
│   │
│   └── utils/
│       └── logger.py                   # Centralized structured logging
│
├── data/
│   ├── raw/                            # Source data (CSV, read-only)
│   │   ├── payments.csv                # 7,118 boleto records
│   │   └── auxiliary.csv               # 4,612 company records (payers + beneficiaries)
│   │
│   ├── processed/                      # Enriched dataset (Parquet)
│   │   └── payments_enriched.parquet
│   │
│   ├── features/                       # Feature-engineered dataset (Parquet)
│   │   └── payments_features.parquet
│   │
│   └── outputs/
│       ├── ml/                         # ML artifacts
│       │   ├── X_train.parquet
│       │   ├── X_test.parquet
│       │   ├── y_train.parquet
│       │   ├── y_test.parquet
│       │   ├── preprocessors.pkl       # Fitted encoder + imputer
│       │   ├── models/
│       │   │   ├── logistic_regression.pkl
│       │   │   ├── random_forest.pkl
│       │   │   └── xgboost.pkl
│       │   └── evaluation/
│       │       ├── model_evaluation.json
│       │       └── model_comparison.parquet
│       │
│       └── bi/                         # BI-ready analytical datasets
│           ├── boleto_scores.parquet
│           ├── portfolio_overview.parquet
│           ├── payer_risk.parquet
│           ├── beneficiary_risk.parquet
│           ├── temporal_analysis.parquet
│           ├── uf_analysis.parquet
│           ├── cnae_analysis.parquet
│           ├── score_distribution.parquet
│           └── feature_importance.parquet
│
└── notebooks/
    └── exploratory_data_analysis.ipynb  # EDA reference notebook
```

### Configuration Layer

The `config/` directory is the **single source of truth** for all data contracts and domain logic. All other modules derive their knowledge from it — no column names or business thresholds are hardcoded outside `config/`.

| File | Purpose |
|---|---|
| `schema.py` | Column-level data contract: dtype, nullability, bounds, allowed values |
| `feature_catalog.py` | Classifies every column as identifier, target, safe feature, or leaky column |
| `business_rules.py` | Formalizes domain thresholds (e.g., `DEFAULT_THRESHOLD_DAYS = 0`) |
| `paths.py` | Absolute paths for all input/output files |
| `constants.py` | Derives column lists (date cols, numeric cols) from `schema.py` |

---

## 7. ETL Pipeline

The ETL pipeline transforms raw CSV inputs into an enriched, analysis-ready dataset through four sequential stages.

```
payments.csv ─────┐
                  ├──▶ LOAD ──▶ VALIDATE ──▶ CLEAN ──▶ DEDUPLICATE ──▶ ENRICH
auxiliary.csv ────┘                                                        │
                                                                           ▼
                                                              payments_enriched.parquet
```

### Stage 1 — Ingestion (`src/ingestion/loader.py`)

Loads CSV files and applies type coercions defined in `config/schema.py`:
- Date columns parsed with `dateutil`
- Numeric columns cast to appropriate dtypes
- Columns not in schema are dropped at load time

### Stage 2 — Validation (`src/ingestion/validator.py`)

Runs a **ValidationReport** with severity-graded checks across both datasets:

| Check | Type | Description |
|---|---|---|
| Column presence | Error | All schema columns must exist |
| Non-null constraints | Error | Columns with `nullable=False` must have zero nulls |
| Uniqueness constraints | Error | Columns with `unique=True` must have no duplicates |
| Range bounds | Error | Numeric values must fall within `[min_value, max_value]` |
| Allowed values | Error | String columns must use only values in `allowed` frozenset |
| Null rate anomaly | Warning | Null rate > 2× EDA baseline + 5% triggers investigation alert |
| Settlement alignment | Error | `tipo_baixa` must be null iff `dt_pagamento` is null |
| Date ordering | Error | `dt_emissao ≤ dt_vencimento` for all records |

### Stage 3 — ETL Transformation (`src/etl/`)

**Cleaning:**
- Strip leading/trailing whitespace from `tipo_baixa`, `tipo_especie`
- Extract numeric settlement codes from descriptive strings (e.g., `"5 - Baixa integral"` → `5`)
- Normalize `uf` to uppercase; cast `cd_cnae_prin` to nullable integer

**Deduplication:**
- Payments: deduplicated by `id_boleto` (keep last)
- Auxiliary: deduplicated by `id_cnpj` (keep last)

### Stage 4 — Enrichment (`src/etl/enricher.py`)

Joins the payments dataset with the auxiliary dataset **twice** — once for the payer perspective and once for the beneficiary perspective:

```
payments  ──(id_pagador → id_cnpj)──▶  payer columns   (prefix: payer_*)
payments  ──(id_beneficiario → id_cnpj)──▶  beneficiary columns (prefix: beneficiary_*)
```

Expected join match rates are 100% for both keys, as validated against the EDA reference notebook. Both joins are **left joins** to preserve all boleto records.

---

## 8. Feature Engineering

Feature engineering generates the predictive signals used by the machine learning models. All features are computed exclusively from information available **at the time the boleto is issued** — no future information is used.

```
payments_enriched.parquet
        │
        ├──▶ temporal.py       ──▶ boleto_term_days, due_month, is_defaulted, ...
        ├──▶ behavioral.py     ──▶ payer_boleto_count, payer_default_rate, ...
        └──▶ aggregations.py   ──▶ beneficiary_boleto_count, beneficiary_default_rate, ...
                │
                ▼
        payments_features.parquet
```

### Temporal Features (`src/features/temporal.py`)

| Feature | Type | Description |
|---|---|---|
| `boleto_term_days` | Numeric | Days between issuance and due date |
| `payment_delay_days` | Numeric | Days late (null if unpaid) — **excluded from ML** |
| `days_overdue` | Numeric | Days since due date for unpaid boletos — **excluded from ML** |
| `is_defaulted` | Binary | Target variable (1 = defaulted) |
| `due_month` | Integer | Month extracted from `dt_vencimento` |
| `due_day_of_week` | Integer | Day of week extracted from `dt_vencimento` |

### Behavioral Features — Payer (`src/features/behavioral.py`)

Aggregated per `id_pagador` across the full portfolio:

| Feature | Description |
|---|---|
| `payer_boleto_count` | Total number of boletos issued to this payer |
| `payer_default_rate` | Historical default rate for this payer |
| `payer_avg_nominal_value` | Average boleto value |
| `payer_avg_delay_days` | Average payment delay in days |
| `payer_max_delay_days` | Maximum observed delay |
| `payer_avg_term_days` | Average boleto term length |

> **Note on split strategy:** Payer aggregation features computed over the full dataset introduce indirect leakage under random train/test splits, since a boleto's own label contributes to its payer's aggregated default rate. These features are correctly safe under strictly time-based splits, where no future payment outcomes inform past boleto features.

### Behavioral Features — Beneficiary (`src/features/aggregations.py`)

Aggregated per `id_beneficiario`:

| Feature | Description |
|---|---|
| `beneficiary_boleto_count` | Total boletos assigned by this beneficiary |
| `beneficiary_default_rate` | Default rate within beneficiary's portfolio |
| `beneficiary_avg_nominal_value` | Average boleto face value |
| `beneficiary_total_portfolio_value` | Cumulative portfolio value |

---

## 9. Leakage Prevention

Data leakage is one of the most critical failure modes in credit risk modeling. A model trained on leaked features will produce unrealistically high metrics during development and will fail catastrophically in production when future information is not available.

This platform implements a **formal leakage governance framework** through `config/feature_catalog.py`.

### Feature Classification

Every column in the dataset is explicitly assigned to one of five categories:

```
IDENTIFIER_COLS         → id_boleto, id_pagador, id_beneficiario
TARGET_COL              → is_defaulted
DATE_COLS               → dt_emissao, dt_vencimento
LEAKY_COLS              → (see below)
SAFE_NUMERIC_FEATURES   → 25 model inputs
SAFE_CATEGORICAL_FEATURES → 5 model inputs
```

### Excluded Leaky Columns

| Column | Leakage Type | Reason |
|---|---|---|
| `dt_pagamento` | Direct | Contains the actual payment outcome |
| `payment_delay_days` | Direct | Derived from `dt_pagamento` |
| `days_overdue` | Direct | Derived from `dt_pagamento` |
| `vlr_baixa` | Direct | Settlement value — only exists post-payment |
| `tipo_baixa` | Direct | Settlement type — only available post-payment |
| `settlement_code` | Direct | Numeric code derived from `tipo_baixa` |
| `payer_default_rate` | Indirect | Aggregated from outcomes of sibling boletos |
| `payer_avg_delay_days` | Indirect | Aggregated from outcomes of sibling boletos |
| `beneficiary_default_rate` | Indirect | Same rationale |
| `beneficiary_boleto_count` | Indirect | Volume signal with indirect leakage risk |

### Leakage Audit

The `src/outputs/selectors.py` module runs an explicit **leakage audit** before every ML preparation run:

```python
audit_leakage(df)
# → Logs all LEAKY_COLS present in the dataset
# → Confirms they will be excluded from model inputs
```

This audit runs as a mandatory step in the pipeline, ensuring that no leaky column can accidentally enter the model training process.

---

## 10. Machine Learning Preparation

The ML preparation layer transforms the feature-engineered dataset into training-ready matrices with reproducible preprocessing.

```
payments_features.parquet
         │
         ├──▶ audit_leakage()
         ├──▶ select_features()   ──▶ 25 numeric + 5 categorical = 30 features
         ├──▶ select_target()     ──▶ is_defaulted (binary)
         ├──▶ train_test_split()  ──▶ 80% train / 20% test (stratified)
         ├──▶ OrdinalEncoder      ──▶ fit on train, transform both
         ├──▶ SimpleImputer       ──▶ fit on train (median), transform both
         └──▶ save artifacts
                  ├── X_train.parquet, X_test.parquet
                  ├── y_train.parquet, y_test.parquet
                  └── preprocessors.pkl
```

### Safe Feature Set

**25 Numeric Features:**

| Category | Features |
|---|---|
| Boleto characteristics | `vlr_nominal`, `boleto_term_days`, `due_month`, `due_day_of_week` |
| Payer creditworthiness | `sacado_indice_liquidez_1m`, `media_atraso_dias`, `indicador_liquidez_quantitativo_3m`, `share_vl_inad_pag_bol_6_a_15d`, `score_materialidade_evolucao`, `score_quantidade_v2`, `score_materialidade_v2`, `cedente_indice_liquidez_1m` |
| Beneficiary creditworthiness | (same 8 fields, prefixed with `beneficiary_`) |

**5 Categorical Features:**

| Feature | Description |
|---|---|
| `tipo_especie` | Boleto instrument type (9 categories) |
| `payer_uf` | Brazilian state of the payer |
| `beneficiary_uf` | Brazilian state of the beneficiary |
| `payer_cd_cnae_prin` | Primary CNAE economic sector of the payer |
| `beneficiary_cd_cnae_prin` | Primary CNAE economic sector of the beneficiary |

### Preprocessing Design

**Stratified split:** The 80/20 split is stratified on `is_defaulted` to preserve the observed default rate in both partitions, which is critical given the class imbalance.

**Fit-on-train principle:** All preprocessing transformers (`OrdinalEncoder`, `SimpleImputer`) are fitted exclusively on the training set and then applied to both train and test sets. This prevents test-set information from influencing the preprocessing decisions — a form of leakage.

**Categorical encoding:** `OrdinalEncoder` with `handle_unknown='use_encoded_value'` and `unknown_value=-1` handles unseen categories gracefully.

**Numeric imputation:** Median imputation handles missing auxiliary scores (e.g., `cedente_indice_liquidez_1m` has ~46.6% nulls). Median is preferred over mean for robustness to outliers in financial datasets.

### Preprocessing Audit Log

After each run, the ML exporter logs a comprehensive audit report including:

- Feature counts by type (numeric, categorical)
- Leakage column count
- Default rate comparison (train vs. test)
- Class imbalance ratio
- Zero-variance feature detection
- Remaining null counts post-imputation

---

## 11. Modeling Strategy

Three classification algorithms are trained and evaluated. All models are configured to handle class imbalance explicitly.

```
X_train, y_train
      │
      ├──▶ Logistic Regression ──▶ logistic_regression.pkl
      ├──▶ Random Forest       ──▶ random_forest.pkl
      └──▶ XGBoost             ──▶ xgboost.pkl (selected)
```

### Model Configurations

**Logistic Regression**

```python
LogisticRegression(
    max_iter=5000,
    solver='saga',           # Handles large datasets efficiently
    class_weight='balanced', # Compensates for class imbalance
    C=0.1                    # L2 regularization
)
```

**Random Forest**

```python
RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    n_jobs=-1                # Parallel training
)
```

**XGBoost**

```python
XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=<computed>  # Ratio of negative/positive class
)
```

### Model Selection

The **best model is selected automatically** based on the highest ROC-AUC score on the test set. XGBoost was selected as the final model with the following performance:

| Metric | Value |
|---|---|
| **ROC-AUC** | **0.8892** |
| **PR-AUC** | **0.8149** |
| **F1 Score** | **0.7301** |

The selected model is applied to the full feature dataset (not just the test split) to generate risk scores for all boletos in the portfolio.

---

## 12. Evaluation Metrics

The platform uses three complementary metrics for model evaluation. Each captures a different aspect of performance in the context of imbalanced credit risk classification.

### ROC-AUC (Area Under the ROC Curve)

Measures the model's ability to **rank** defaulted boletos above non-defaulted ones across all possible classification thresholds.

```
ROC Curve: TPR (Sensitivity) vs. FPR (1 - Specificity)
AUC = 1.0 → Perfect ranking
AUC = 0.5 → Random classifier
AUC = 0.89 → Strong discriminatory power
```

ROC-AUC is threshold-independent and robust to class imbalance, making it the primary model selection criterion.

### PR-AUC (Area Under the Precision-Recall Curve)

Measures the trade-off between **precision** (of flagged defaults, how many are real) and **recall** (of all real defaults, how many are caught).

PR-AUC is more sensitive to minority class performance than ROC-AUC, which makes it particularly relevant for credit risk where correctly identifying true defaults is critical.

```
PR-AUC = 0.81 → Strong minority class recall with controlled false positives
```

### F1 Score

The harmonic mean of precision and recall at a specific operating threshold (typically 0.5):

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
F1 = 0.73 → Balanced precision-recall performance
```

### Metric Interpretation for Credit Risk

| Metric | Business Implication |
|---|---|
| High ROC-AUC | Model reliably separates high-risk from low-risk boletos |
| High PR-AUC | Model can identify most defaults without excessive false positives |
| High F1 | Operational classification is balanced at the chosen threshold |

---

## 13. Risk Scoring

The risk scoring module translates model output probabilities into business-ready risk signals applied to every boleto in the portfolio.

```
payments_features.parquet
         │
         ├──▶ Apply preprocessors (encoder + imputer)
         ├──▶ model.predict_proba() → default_probability [0.0, 1.0]
         ├──▶ Scale → risk_score [0, 100]
         └──▶ Classify → risk_category
```

### Scoring Outputs

| Column | Range | Description |
|---|---|---|
| `default_probability` | [0.0, 1.0] | Raw model output: probability of default |
| `risk_score` | [0, 100] | Scaled score: 0 = safest, 100 = highest risk |
| `risk_category` | low / medium / high | Categorical tier for operational use |

### Risk Category Thresholds

| Category | Probability Range | Risk Score Range |
|---|---|---|
| `low` | < 0.30 | 0 – 30 |
| `medium` | 0.30 – 0.59 | 30 – 59 |
| `high` | ≥ 0.60 | 60 – 100 |

### Scoring Dataset Schema (`boleto_scores.parquet`)

The primary scoring output includes:

- Identifiers: `id_boleto`, `id_pagador`, `id_beneficiario`
- Dates: `dt_emissao`, `dt_vencimento`
- Target: `is_defaulted`
- Context: `payer_uf`, `beneficiary_uf`, `payer_cd_cnae_prin`, `vlr_nominal`
- Scores: `default_probability`, `risk_score`, `risk_category`

---

## 14. Business Intelligence Layer

The BI layer consumes `boleto_scores.parquet` and generates eight structured analytical datasets for dashboard consumption. All outputs are exported as Parquet files optimized for columnar querying.

### BI Dataset Overview

| Dataset | File | Grain | Primary Use |
|---|---|---|---|
| Portfolio Overview | `portfolio_overview.parquet` | 1 row (global) | Executive KPI card |
| Payer Risk | `payer_risk.parquet` | Per payer | Risk-ranked payer list |
| Beneficiary Risk | `beneficiary_risk.parquet` | Per beneficiary | Assignor concentration view |
| Temporal Analysis | `temporal_analysis.parquet` | Per due month | Trend chart |
| Geographic Analysis | `uf_analysis.parquet` | Per state × perspective | Map / regional breakdown |
| Sector Analysis | `cnae_analysis.parquet` | Per CNAE × perspective | Sector heatmap |
| Score Distribution | `score_distribution.parquet` | Per probability bucket | Histogram |
| Feature Importance | `feature_importance.parquet` | Per model × feature | Model explainability |

### Portfolio Overview (`portfolio_overview.parquet`)

Aggregated executive KPIs for the entire portfolio:

| Metric | Description |
|---|---|
| `total_boletos` | Total number of boletos in portfolio |
| `defaulted_boletos` | Count of defaulted boletos |
| `default_rate` | Proportion of defaulted boletos |
| `total_portfolio_value` | Sum of `vlr_nominal` across all boletos |
| `defaulted_portfolio_value` | Nominal value at risk (defaulted boletos) |
| `avg_boleto_value` | Average face value |
| `avg_risk_score` | Average portfolio risk score |
| `high_risk_count` | Boletos with `risk_category = high` |
| `high_risk_rate` | Proportion of high-risk boletos |

### Payer Risk (`payer_risk.parquet`)

One row per payer, ranked by descending `avg_risk_score`:

`total_boletos` · `total_value` · `default_rate` · `avg_default_probability` · `avg_risk_score` · `max_risk_score` · `payer_uf` · `payer_cd_cnae_prin`

### Temporal Analysis (`temporal_analysis.parquet`)

One row per due month (grouped by `dt_vencimento` year-month):

`total_boletos` · `total_value` · `defaulted_boletos` · `default_rate` · `avg_risk_score` · `avg_default_probability` · `high_risk_count`

### Geographic Analysis (`uf_analysis.parquet`)

One row per Brazilian state × perspective (payer or beneficiary):

`uf` · `perspective` · `total_boletos` · `total_value` · `default_rate` · `avg_risk_score`

### Sector Analysis (`cnae_analysis.parquet`)

One row per CNAE code × perspective (payer or beneficiary):

`cd_cnae_prin` · `perspective` · `total_boletos` · `total_value` · `default_rate` · `avg_risk_score`

### Score Distribution (`score_distribution.parquet`)

Boletos binned into 10 probability buckets (0–10%, 10–20%, ..., 90–100%):

`probability_bucket` · `boleto_count` · `actual_default_count` · `actual_default_rate` · `total_value` · `pct_of_portfolio`

This dataset is used to validate model calibration: a well-calibrated model will show actual default rates that closely match each bucket's predicted probability range.

---

## 15. Dashboard Objectives

The BI datasets are designed to power a multi-view analytics dashboard covering the following use cases:

```
┌─────────────────────────────────────────────────────────────┐
│                    DASHBOARD VIEWS                          │
│                                                             │
│  1. PORTFOLIO SUMMARY                                       │
│     • Total, defaulted, and at-risk volumes                 │
│     • Average risk score gauge                              │
│     • Risk tier distribution (pie/donut)                    │
│                                                             │
│  2. PAYER RISK RANKING                                      │
│     • Sortable table of payers by risk score                │
│     • Filter by UF, CNAE, risk tier                        │
│                                                             │
│  3. BENEFICIARY EXPOSURE                                    │
│     • Concentration analysis by assignor                    │
│     • Portfolio value at risk per beneficiary               │
│                                                             │
│  4. TEMPORAL TRENDS                                         │
│     • Default rate trend line by month                      │
│     • Average risk score evolution                          │
│                                                             │
│  5. GEOGRAPHIC HEATMAP                                      │
│     • Default rate by Brazilian state                       │
│     • Payer vs. beneficiary perspectives                    │
│                                                             │
│  6. SECTOR ANALYSIS                                         │
│     • Default rate by economic sector (CNAE)                │
│     • Comparison across payer and beneficiary sectors       │
│                                                             │
│  7. MODEL CALIBRATION                                       │
│     • Score distribution histogram                          │
│     • Actual vs. predicted default rate by bucket           │
│                                                             │
│  8. MODEL EXPLAINABILITY                                    │
│     • Feature importance by model                           │
│     • Top drivers of default prediction                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 16. Technologies

### Core Stack

| Technology | Version | Role |
|---|---|---|
| Python | 3.10+ | Runtime |
| pandas | 3.0.2 | Data manipulation and transformation |
| scikit-learn | 1.8.0 | ML preprocessing, models, evaluation |
| XGBoost | ≥ 2.0.0 | Gradient boosting classification |
| PyArrow | 24.0.0 | Parquet I/O |
| NumPy | 2.4.4 | Numerical computing |
| SciPy | 1.17.1 | Statistical utilities |
| joblib | 1.5.3 | Model serialization |

### Architecture Decisions

| Decision | Rationale |
|---|---|
| Parquet for intermediate files | Columnar format preserves dtypes and is efficient for analytical queries |
| Single source of truth (`schema.py`) | Prevents schema drift across ingestion, validation, and feature engineering |
| Explicit feature catalog | Makes leakage prevention auditable and version-controlled |
| Fit-on-train preprocessing | Prevents test-set contamination in preprocessing steps |
| `class_weight='balanced'` | Handles ~1% default rate without requiring resampling |
| Fitted preprocessors serialized to `.pkl` | Ensures scoring uses identical transformations as training |

---

## 17. Installation

For the complete clean-machine walkthrough, use `../README_EXECUCAO.md`. The shortest Windows path is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_full_demo.ps1
```

### Prerequisites

- Python 3.10 or higher
- `pip` package manager

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd project

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.\.venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import pandas, sklearn, xgboost, pyarrow; print('OK')"
```

---

## 18. Running the Pipeline

The entire pipeline is orchestrated from a single entry point.

### Full Pipeline Run

```bash
python -m src.main
```

Run the command from the `project` folder. Using `python src/main.py` may fail because it does not load the local packages in the same way.

This executes all 9 phases sequentially:

```
[INFO] Phase 1/9 - Ingesting data...
[INFO] Phase 2/9 - Validating schema and business rules...
[INFO] Phase 3/9 - Running ETL (clean, deduplicate, enrich)...
[INFO] Phase 4/9 - Engineering features...
[INFO] Phase 5/9 - Preparing ML datasets...
[INFO] Phase 6/9 - Training models...
[INFO] Phase 7/9 - Evaluating models...
[INFO] Phase 8/9 - Scoring portfolio...
[INFO] Phase 9/9 - Exporting BI datasets...
=== Pipeline complete ===
```

### Expected Outputs

After a successful run, the following files are created or updated:

```
data/processed/payments_enriched.parquet
data/features/payments_features.parquet
data/outputs/ml/X_train.parquet
data/outputs/ml/X_test.parquet
data/outputs/ml/y_train.parquet
data/outputs/ml/y_test.parquet
data/outputs/ml/preprocessors.pkl
data/outputs/ml/models/logistic_regression.pkl
data/outputs/ml/models/random_forest.pkl
data/outputs/ml/models/xgboost.pkl
data/outputs/ml/evaluation/model_evaluation.json
data/outputs/ml/evaluation/model_comparison.parquet
data/outputs/bi/boleto_scores.parquet
data/outputs/bi/portfolio_overview.parquet
data/outputs/bi/payer_risk.parquet
data/outputs/bi/beneficiary_risk.parquet
data/outputs/bi/temporal_analysis.parquet
data/outputs/bi/uf_analysis.parquet
data/outputs/bi/cnae_analysis.parquet
data/outputs/bi/score_distribution.parquet
data/outputs/bi/feature_importance.parquet
```

### Running Individual Stages

Each module can be imported and called independently for development or debugging:

```python
from src.ingestion.loader import load_payment, load_auxiliary
from src.etl.enricher import enrich
from src.features.temporal import add_temporal_features
from src.outputs.ml_exporter import prepare_ml_datasets
from src.models.trainer import train_models
from src.scoring.scorer import score_dataset
```

---

## 19. Output Datasets

### Model Evaluation (`data/outputs/ml/evaluation/model_evaluation.json`)

```json
{
  "logistic_regression": {
    "roc_auc": 0.xxxx,
    "pr_auc": 0.xxxx,
    "f1": 0.xxxx,
    "confusion_matrix": [[TN, FP], [FN, TP]]
  },
  "random_forest": { ... },
  "xgboost": {
    "roc_auc": 0.8892,
    "pr_auc": 0.8149,
    "f1": 0.7301,
    "confusion_matrix": [[...], [...]]
  }
}
```

### Primary Scoring Output (`data/outputs/bi/boleto_scores.parquet`)

| Column | Type | Description |
|---|---|---|
| `id_boleto` | string | Boleto unique identifier |
| `id_pagador` | string | Payer identifier |
| `id_beneficiario` | string | Beneficiary identifier |
| `dt_emissao` | date | Issuance date |
| `dt_vencimento` | date | Due date |
| `vlr_nominal` | float | Face value in BRL |
| `payer_uf` | string | Payer state |
| `beneficiary_uf` | string | Beneficiary state |
| `payer_cd_cnae_prin` | int | Payer primary sector |
| `is_defaulted` | int | Ground truth: 0 or 1 |
| `default_probability` | float | Model predicted probability [0, 1] |
| `risk_score` | float | Scaled risk score [0, 100] |
| `risk_category` | string | low / medium / high |

---

## 20. Future Improvements

### Near-Term

| Improvement | Priority | Description |
|---|---|---|
| Time-based split | High | Replace random split with chronological split to eliminate indirect leakage from aggregation features |
| Threshold optimization | High | Select classification threshold using F-beta or Youden index rather than defaulting to 0.5 |
| Hyperparameter tuning | Medium | Grid search or Bayesian optimization for XGBoost hyperparameters |
| Cross-validation | Medium | K-fold CV for more robust generalization estimates |
| Model calibration | Medium | Apply Platt scaling or isotonic regression to calibrate predicted probabilities |

### Medium-Term

| Improvement | Priority | Description |
|---|---|---|
| Online scoring API | Medium | Expose the trained model as a REST endpoint for real-time boleto scoring |
| Incremental data ingestion | Medium | Support append-only ingestion for operational pipelines (replace full-refresh) |
| Feature store | Low | Centralize feature computation for reuse across models and use cases |
| SHAP explainability | Low | Per-prediction SHAP values for individual boleto risk explanation |

### Long-Term

| Improvement | Priority | Description |
|---|---|---|
| Survival analysis | Low | Model time-to-default using Cox regression or accelerated failure time models |
| LTV modeling | Low | Estimate expected loss given default for portfolio pricing |
| Anomaly detection | Low | Flag statistically anomalous boletos for manual review |
| Real-time dashboard | Low | Stream BI datasets to a live dashboard with sub-minute refresh |

---

## 21. Academic Conclusions

### Summary of Contributions

This platform demonstrates a complete and production-faithful implementation of a credit risk analytics system for FIDC receivables portfolios, encompassing:

1. **Rigorous data governance:** A schema-driven data contract (`schema.py`) eliminates implicit assumptions and makes the data pipeline testable, auditable, and maintainable at scale.

2. **Formal leakage prevention:** The explicit feature catalog (`feature_catalog.py`) and mandatory leakage audit address one of the most common and consequential errors in applied machine learning: inadvertent use of future information during model training. This framework makes the leakage boundary version-controlled and reviewable.

3. **Class imbalance handling:** With a default rate of approximately 1%, the dataset presents a significant class imbalance problem. The platform addresses this through stratified splitting, balanced class weights in all classifiers, and PR-AUC as a complementary metric — reflecting best practices for imbalanced binary classification.

4. **Reproducible ML artifacts:** By serializing fitted preprocessors alongside trained models, the platform ensures that scoring applied to new data uses precisely the same transformations as training — a requirement for production ML systems.

5. **BI-ready analytical layer:** The separation of the scoring layer from the analytical layer enables the same risk scores to power multiple downstream views (executive, operational, geographic, temporal) without redundant computation.

### Model Performance in Context

The final XGBoost model achieved a **ROC-AUC of 0.8892**, indicating strong discriminatory power despite the small dataset size (7,118 records) and high class imbalance. The **PR-AUC of 0.8149** confirms that this discrimination is preserved for the minority (default) class — the operationally critical class.

These results are consistent with XGBoost's known advantages in tabular, imbalanced classification tasks: its gradient boosting framework naturally handles non-linear feature interactions and its `scale_pos_weight` parameter directly compensates for class imbalance in the loss function.

### Limitations and Boundary Conditions

- **Dataset size:** 7,118 boletos is sufficient for model development but may not capture the full distributional complexity of a production FIDC portfolio. Real deployments should target at minimum tens of thousands of observations.
- **Random split vs. temporal split:** The current pipeline uses a random 80/20 stratified split. In production, a strictly temporal split is required to prevent indirect leakage from aggregation features and to simulate realistic model deployment conditions.
- **Static scoring:** The current system scores the full historical portfolio in batch. Production credit risk systems require point-in-time scoring at origination, which implies feature computation using only information available at or before the boleto's issuance date.
- **Single-vintage training:** The model is trained on a single time window (mainly May 2024). Seasonal patterns, economic cycles, and portfolio composition shifts may cause model degradation over time, requiring periodic retraining and monitoring.

### Engineering Principles Applied

| Principle | Application |
|---|---|
| Single source of truth | `schema.py` is the authoritative data contract; all modules derive from it |
| Separation of concerns | Each pipeline layer has one responsibility and communicates via Parquet |
| Fail fast | Validation errors surface at ingestion; downstream stages receive clean data |
| Reproducibility | Preprocessing artifacts serialized for deterministic scoring |
| Explicitness over convention | Feature catalog makes all ML decisions auditable |

---

## License

This project is developed for academic and portfolio purposes. All data used is synthetic or anonymized.

---

*Built with Python · pandas · scikit-learn · XGBoost · PyArrow*
