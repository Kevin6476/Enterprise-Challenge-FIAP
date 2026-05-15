# MVP Validation Report

Generated at: 2026-05-14 22:16:12

## Summary

- [x] Required files present
- [x] Raw CSVs readable
- [x] Model metrics readable; best model is XGBoost
- [x] Full pipeline log confirms completion
- [x] Fixed dashboard app embeds BI Parquet outputs
- [x] Python source syntax check

## Required Files

- [x] `../README.md` (8.9 KB)
- [x] `../README_EXECUCAO.md` (6.6 KB)
- [x] `../README_DASHBOARD.md` (1.8 KB)
- [x] `data/raw/payments.csv` (2109.5 KB)
- [x] `data/raw/auxiliary.csv` (652.1 KB)
- [x] `src/main.py` (3.8 KB)
- [x] `requirements.txt` (0.2 KB)
- [x] `README.md` (53.0 KB)
- [x] `docs/product_and_model_decisions.md` (4.0 KB)
- [x] `notebooks/exploratory_data_analysis.ipynb` (93.8 KB)
- [x] `data/outputs/ml/evaluation/model_evaluation.json` (0.6 KB)
- [x] `app/build_dashboard.py` (61.8 KB)
- [x] `app/index.html` (6261.2 KB)
- [x] `scripts/build_and_validate.py` (0.7 KB)
- [x] `scripts/run_full_demo.ps1` (1.5 KB)
- [x] `scripts/setup_windows.ps1` (1.1 KB)
- [x] `scripts/validate_mvp.py` (8.0 KB)
- [x] `tests/test_mvp_contracts.py` (2.3 KB)
- [x] `reports/environment_check.txt` (0.2 KB)
- [x] `reports/pipeline_run.log` (18.1 KB)

## Raw Data

- payments.csv: 7,118 rows x 10 columns
- auxiliary.csv: 4,612 rows x 11 columns
- payments top null rates: `{'vlr_baixa': 0.1152, 'dt_pagamento': 0.0098, 'tipo_baixa': 0.0098, 'id_boleto': 0.0, 'id_pagador': 0.0}`
- auxiliary top null rates: `{'cedente_indice_liquidez_1m': 0.466, 'uf': 0.0778, 'indicador_liquidez_quantitativo_3m': 0.0043, 'sacado_indice_liquidez_1m': 0.0041, 'score_quantidade_v2': 0.0013}`

## Model Metrics

- Best model: `xgboost`
- AUC-ROC: 0.8892
- AUC-PR: 0.8149
- F1: 0.7301

## Dashboard

- [x] `app/index.html` exists
- [x] Embedded JSON is valid
- Views detected: 5
- Data source: `BI Parquet outputs`
- Total boletos in dashboard: 7118.0
- Embedded transactions: 7118
- Embedded beneficiaries: 1189
- Embedded alerts: 80

## Pipeline Run

- [x] `reports/pipeline_run.log` exists
- [x] Pipeline completed successfully

## Python Syntax

- [x] `app\build_dashboard.py`
- [x] `config\business_rules.py`
- [x] `config\constants.py`
- [x] `config\feature_catalog.py`
- [x] `config\paths.py`
- [x] `config\schema.py`
- [x] `scripts\build_and_validate.py`
- [x] `scripts\validate_mvp.py`
- [x] `src\analytics\__init__.py`
- [x] `src\analytics\beneficiary.py`
- [x] `src\analytics\bi_exporter.py`
- [x] `src\analytics\geographic.py`
- [x] `src\analytics\payer.py`
- [x] `src\analytics\portfolio.py`
- [x] `src\analytics\score_distribution.py`
- [x] `src\analytics\sector.py`
- [x] `src\analytics\temporal.py`
- [x] `src\etl\__init__.py`
- [x] `src\etl\cleaner.py`
- [x] `src\etl\deduplicator.py`
- [x] `src\etl\enricher.py`
- [x] `src\features\__init__.py`
- [x] `src\features\aggregations.py`
- [x] `src\features\behavioral.py`
- [x] `src\features\temporal.py`
- [x] `src\ingestion\__init__.py`
- [x] `src\ingestion\loader.py`
- [x] `src\ingestion\validator.py`
- [x] `src\main.py`
- [x] `src\models\__init__.py`
- [x] `src\models\evaluator.py`
- [x] `src\models\explainer.py`
- [x] `src\models\trainer.py`
- [x] `src\outputs\__init__.py`
- [x] `src\outputs\ml_exporter.py`
- [x] `src\outputs\selectors.py`
- [x] `src\scoring\__init__.py`
- [x] `src\scoring\scorer.py`
- [x] `src\utils\__init__.py`
- [x] `src\utils\logger.py`
- [x] `tests\test_mvp_contracts.py`
