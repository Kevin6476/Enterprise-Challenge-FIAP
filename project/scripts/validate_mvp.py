"""Validate local MVP artifacts and write a Markdown evidence report."""

from __future__ import annotations

import ast
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_DIR / "reports"
REPORT_FILE = REPORT_DIR / "mvp_validation_report.md"


def ok(condition: bool) -> str:
    return "[x]" if condition else "[ ]"


def file_info(path: Path) -> dict[str, Any]:
    return {
        "path": path,
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() else 0,
    }


def check_required_files() -> list[dict[str, Any]]:
    relative_paths = [
        "../README.md",
        "../README_EXECUCAO.md",
        "../README_DASHBOARD.md",
        "data/raw/payments.csv",
        "data/raw/auxiliary.csv",
        "src/main.py",
        "requirements.txt",
        "README.md",
        "docs/product_and_model_decisions.md",
        "notebooks/exploratory_data_analysis.ipynb",
        "data/outputs/ml/evaluation/model_evaluation.json",
        "app/build_dashboard.py",
        "app/index.html",
        "scripts/build_and_validate.py",
        "scripts/run_full_demo.ps1",
        "scripts/setup_windows.ps1",
        "scripts/validate_mvp.py",
        "tests/test_mvp_contracts.py",
        "reports/environment_check.txt",
        "reports/pipeline_run.log",
    ]
    return [file_info(PROJECT_DIR / rel) | {"relative": rel} for rel in relative_paths]


def check_csvs() -> dict[str, Any]:
    payments = pd.read_csv(PROJECT_DIR / "data" / "raw" / "payments.csv")
    auxiliary = pd.read_csv(PROJECT_DIR / "data" / "raw" / "auxiliary.csv")
    return {
        "payments_rows": len(payments),
        "payments_cols": len(payments.columns),
        "payments_columns": list(payments.columns),
        "auxiliary_rows": len(auxiliary),
        "auxiliary_cols": len(auxiliary.columns),
        "auxiliary_columns": list(auxiliary.columns),
        "payments_nulls_top": payments.isna()
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .round(4)
        .to_dict(),
        "auxiliary_nulls_top": auxiliary.isna()
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .round(4)
        .to_dict(),
    }


def check_model_metrics() -> dict[str, Any]:
    path = PROJECT_DIR / "data" / "outputs" / "ml" / "evaluation" / "model_evaluation.json"
    with path.open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    best_name = max(metrics, key=lambda name: metrics[name].get("auc_roc", 0))
    return {
        "metrics": metrics,
        "best_name": best_name,
        "best_metrics": metrics[best_name],
    }


def check_python_syntax() -> list[dict[str, Any]]:
    results = []
    ignored_parts = {".venv", "__pycache__", ".git"}
    for path in sorted(PROJECT_DIR.rglob("*.py")):
        if ignored_parts.intersection(path.relative_to(PROJECT_DIR).parts):
            continue
        rel = path.relative_to(PROJECT_DIR)
        try:
            ast.parse(path.read_text(encoding="utf-8"))
            results.append({"path": str(rel), "passed": True, "error": ""})
        except SyntaxError as exc:
            results.append({"path": str(rel), "passed": False, "error": str(exc)})
    return results


def check_dashboard() -> dict[str, Any]:
    path = PROJECT_DIR / "app" / "index.html"
    info = file_info(path)
    if not path.exists():
        return info | {"json_ok": False, "views": 0, "source": None, "transactions": 0}

    html = path.read_text(encoding="utf-8")
    match = re.search(
        r'<script id="dashboard-data" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    if not match:
        return info | {"json_ok": False, "views": 0, "source": None, "transactions": 0}

    payload = json.loads(match.group(1))
    return info | {
        "json_ok": True,
        "views": len(re.findall(r'<section id="[^"]+" class="view', html)),
        "source": payload.get("generated_from"),
        "total_boletos": payload.get("summary", {}).get("total_boletos"),
        "transactions": len(payload.get("transactions", [])),
        "beneficiaries": len(payload.get("beneficiaries", [])),
        "alerts": len(payload.get("alerts", [])),
    }


def check_pipeline_log() -> dict[str, Any]:
    path = PROJECT_DIR / "reports" / "pipeline_run.log"
    info = file_info(path)
    if not path.exists():
        return info | {"completed": False}
    content = path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
    return info | {"completed": "=== Pipeline complete ===" in content}


def write_report() -> None:
    required = check_required_files()
    csvs = check_csvs()
    model = check_model_metrics()
    syntax = check_python_syntax()
    dashboard = check_dashboard()
    pipeline = check_pipeline_log()

    syntax_passed = all(item["passed"] for item in syntax)
    required_passed = all(item["exists"] for item in required)
    dashboard_passed = (
        dashboard["exists"]
        and dashboard["json_ok"]
        and dashboard["views"] >= 5
        and dashboard["source"] == "BI Parquet outputs"
        and dashboard["transactions"] > 0
    )
    pipeline_passed = pipeline["exists"] and pipeline["completed"]

    lines = [
        "# MVP Validation Report",
        "",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- {ok(required_passed)} Required files present",
        f"- {ok(csvs['payments_rows'] > 0 and csvs['auxiliary_rows'] > 0)} Raw CSVs readable",
        f"- {ok(model['best_name'] == 'xgboost')} Model metrics readable; best model is XGBoost",
        f"- {ok(pipeline_passed)} Full pipeline log confirms completion",
        f"- {ok(dashboard_passed)} Fixed dashboard app embeds BI Parquet outputs",
        f"- {ok(syntax_passed)} Python source syntax check",
        "",
        "## Required Files",
        "",
    ]

    for item in required:
        size_kb = item["size"] / 1024
        lines.append(f"- {ok(item['exists'])} `{item['relative']}` ({size_kb:.1f} KB)")

    lines += [
        "",
        "## Raw Data",
        "",
        f"- payments.csv: {csvs['payments_rows']:,} rows x {csvs['payments_cols']} columns",
        f"- auxiliary.csv: {csvs['auxiliary_rows']:,} rows x {csvs['auxiliary_cols']} columns",
        f"- payments top null rates: `{csvs['payments_nulls_top']}`",
        f"- auxiliary top null rates: `{csvs['auxiliary_nulls_top']}`",
        "",
        "## Model Metrics",
        "",
        f"- Best model: `{model['best_name']}`",
        f"- AUC-ROC: {model['best_metrics'].get('auc_roc')}",
        f"- AUC-PR: {model['best_metrics'].get('auc_pr')}",
        f"- F1: {model['best_metrics'].get('f1')}",
        "",
        "## Dashboard",
        "",
        f"- {ok(dashboard['exists'])} `app/index.html` exists",
        f"- {ok(dashboard['json_ok'])} Embedded JSON is valid",
        f"- Views detected: {dashboard['views']}",
        f"- Data source: `{dashboard['source']}`",
        f"- Total boletos in dashboard: {dashboard['total_boletos']}",
        f"- Embedded transactions: {dashboard['transactions']}",
        f"- Embedded beneficiaries: {dashboard['beneficiaries']}",
        f"- Embedded alerts: {dashboard['alerts']}",
        "",
        "## Pipeline Run",
        "",
        f"- {ok(pipeline['exists'])} `reports/pipeline_run.log` exists",
        f"- {ok(pipeline['completed'])} Pipeline completed successfully",
        "",
        "## Python Syntax",
        "",
    ]

    for item in syntax:
        suffix = "" if item["passed"] else f" - {item['error']}"
        lines.append(f"- {ok(item['passed'])} `{item['path']}`{suffix}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Validation report written: {REPORT_FILE}")

    if not (required_passed and dashboard_passed and pipeline_passed and syntax_passed):
        raise SystemExit(1)


if __name__ == "__main__":
    write_report()
