"""
Schema and business rule validator for ingested datasets.

Public entry points:
    validate_payments(df)  — structural checks + payments business rules
    validate_auxiliary(df) — structural checks for auxiliary

Both return a ValidationReport. Callers decide whether to raise on errors.

Severity levels:
    error   — data contract violated; pipeline should not proceed
    warning — anomaly detected (e.g. null rate spike); pipeline may proceed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config.schema import (
    AUXILIARY_SCHEMA,
    NON_FINANCIAL_SETTLEMENT_CODES,
    PAYMENTS_SCHEMA,
    ColumnSpec,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CheckResult:
    check: str
    column: Optional[str]
    passed: bool
    severity: str  # "error" | "warning"
    message: str
    detail: Optional[dict] = None


@dataclass
class ValidationReport:
    dataset: str
    total_rows: int
    results: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(not r.passed and r.severity == "error" for r in self.results)

    @property
    def errors(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity == "error"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity == "warning"]

    def log_summary(self) -> None:
        status = "PASSED" if self.passed else "FAILED"
        logger.info(
            f"[{self.dataset}] Validation {status} — "
            f"{self.total_rows} rows | {len(self.errors)} errors | {len(self.warnings)} warnings"
        )
        for result in self.errors:
            col = f"[{result.column}] " if result.column else ""
            logger.error(f"  ERROR  {col}{result.message}")
        for result in self.warnings:
            col = f"[{result.column}] " if result.column else ""
            logger.warning(f"  WARN   {col}{result.message}")


def _check_columns(
    df: pd.DataFrame, schema: dict[str, ColumnSpec], report: ValidationReport
) -> None:
    missing = sorted(set(schema.keys()) - set(df.columns))
    passed = len(missing) == 0
    report.results.append(
        CheckResult(
            check="columns",
            column=None,
            passed=passed,
            severity="error",
            message=f"Missing columns: {missing}" if not passed else "All expected columns present.",
            detail={"missing": missing} if not passed else None,
        )
    )


def _check_nullable(
    df: pd.DataFrame, schema: dict[str, ColumnSpec], report: ValidationReport
) -> None:
    for col, spec in schema.items():
        if col not in df.columns or spec.nullable:
            continue
        null_count = int(df[col].isna().sum())
        passed = null_count == 0
        report.results.append(
            CheckResult(
                check="nullable",
                column=col,
                passed=passed,
                severity="error",
                message=(
                    f"{null_count} unexpected null(s) in required column."
                    if not passed
                    else "No nulls."
                ),
                detail={"null_count": null_count} if not passed else None,
            )
        )


def _check_unique(
    df: pd.DataFrame, schema: dict[str, ColumnSpec], report: ValidationReport
) -> None:
    for col, spec in schema.items():
        if col not in df.columns or not spec.unique:
            continue
        dup_count = int(df[col].duplicated().sum())
        passed = dup_count == 0
        report.results.append(
            CheckResult(
                check="unique",
                column=col,
                passed=passed,
                severity="error",
                message=(
                    f"{dup_count} duplicate value(s) found."
                    if not passed
                    else "All values unique."
                ),
                detail={"duplicate_count": dup_count} if not passed else None,
            )
        )


def _check_ranges(
    df: pd.DataFrame, schema: dict[str, ColumnSpec], report: ValidationReport
) -> None:
    for col, spec in schema.items():
        if col not in df.columns:
            continue
        if spec.min_value is None and spec.max_value is None:
            continue

        series = df[col].dropna()
        violations = 0
        if spec.min_value is not None:
            violations += int((series < spec.min_value).sum())
        if spec.max_value is not None:
            violations += int((series > spec.max_value).sum())

        bounds = f"[{spec.min_value}, {spec.max_value}]"
        passed = violations == 0
        report.results.append(
            CheckResult(
                check="range",
                column=col,
                passed=passed,
                severity="error",
                message=(
                    f"{violations} value(s) outside expected range {bounds}."
                    if not passed
                    else f"All values within {bounds}."
                ),
                detail={"violations": violations, "bounds": bounds} if not passed else None,
            )
        )


def _check_allowed_values(
    df: pd.DataFrame, schema: dict[str, ColumnSpec], report: ValidationReport
) -> None:
    for col, spec in schema.items():
        if col not in df.columns or spec.allowed is None:
            continue
        unexpected = sorted(set(df[col].dropna().unique()) - spec.allowed)
        passed = len(unexpected) == 0
        report.results.append(
            CheckResult(
                check="allowed_values",
                column=col,
                passed=passed,
                severity="error",
                message=(
                    f"Unexpected value(s): {unexpected}"
                    if not passed
                    else "All values within allowed set."
                ),
                detail={"unexpected_values": unexpected} if not passed else None,
            )
        )


def _check_null_rate(
    df: pd.DataFrame, schema: dict[str, ColumnSpec], report: ValidationReport
) -> None:
    """Warns when observed null rate exceeds 2× the EDA baseline (floor: +5%)."""
    for col, spec in schema.items():
        if col not in df.columns or not spec.nullable:
            continue
        if spec.null_rate_eda is None or spec.null_rate_eda == 0.0:
            continue

        observed = df[col].isna().mean()
        threshold = max(spec.null_rate_eda * 2, spec.null_rate_eda + 0.05)
        passed = observed <= threshold
        report.results.append(
            CheckResult(
                check="null_rate",
                column=col,
                passed=passed,
                severity="warning",
                message=(
                    f"Null rate {observed:.1%} exceeds threshold {threshold:.1%} "
                    f"(EDA baseline: {spec.null_rate_eda:.1%})."
                    if not passed
                    else f"Null rate {observed:.1%} within expected range."
                ),
                detail={
                    "observed": round(float(observed), 4),
                    "threshold": round(threshold, 4),
                    "eda_baseline": spec.null_rate_eda,
                }
                if not passed
                else None,
            )
        )


def _check_settlement_null_alignment(df: pd.DataFrame, report: ValidationReport) -> None:
    """tipo_baixa is null if and only if dt_pagamento is null (open boleto)."""
    mismatches = int((df["tipo_baixa"].isna() != df["dt_pagamento"].isna()).sum())
    passed = mismatches == 0
    report.results.append(
        CheckResult(
            check="business_rule",
            column="tipo_baixa / dt_pagamento",
            passed=passed,
            severity="error",
            message=(
                f"{mismatches} row(s) where tipo_baixa null status does not align with dt_pagamento."
                if not passed
                else "tipo_baixa / dt_pagamento null alignment is valid."
            ),
            detail={"mismatches": mismatches} if not passed else None,
        )
    )


def _check_settlement_value_alignment(df: pd.DataFrame, report: ValidationReport) -> None:
    """
    vlr_baixa is null when tipo_baixa is null (open boleto) or its code
    is in NON_FINANCIAL_SETTLEMENT_CODES {5, 6, 7, 8}.
    vlr_baixa must be present for financial codes {0, 1, 9}.
    """
    extracted_code = df["tipo_baixa"].str.extract(r"^(\d+)")[0]
    is_non_financial = extracted_code.isin(NON_FINANCIAL_SETTLEMENT_CODES)
    tipo_is_null = df["tipo_baixa"].isna()
    vlr_is_null = df["vlr_baixa"].isna()

    expected_present = ~tipo_is_null & ~is_non_financial
    expected_null = tipo_is_null | is_non_financial

    missing_value = int((vlr_is_null & expected_present).sum())
    unexpected_value = int((~vlr_is_null & expected_null).sum())
    total = missing_value + unexpected_value

    passed = total == 0
    report.results.append(
        CheckResult(
            check="business_rule",
            column="vlr_baixa / tipo_baixa",
            passed=passed,
            severity="error",
            message=(
                f"{total} row(s) violate vlr_baixa/tipo_baixa alignment "
                f"({missing_value} missing, {unexpected_value} unexpected)."
                if not passed
                else "vlr_baixa / tipo_baixa null alignment is valid."
            ),
            detail={
                "missing_vlr_baixa": missing_value,
                "unexpected_vlr_baixa": unexpected_value,
            }
            if not passed
            else None,
        )
    )


def _check_date_ordering(df: pd.DataFrame, report: ValidationReport) -> None:
    """dt_emissao must be <= dt_vencimento for every boleto."""
    invalid = int((df["dt_emissao"] > df["dt_vencimento"]).sum())
    passed = invalid == 0
    report.results.append(
        CheckResult(
            check="business_rule",
            column="dt_emissao / dt_vencimento",
            passed=passed,
            severity="error",
            message=(
                f"{invalid} row(s) where dt_emissao > dt_vencimento."
                if not passed
                else "Date ordering is valid (dt_emissao <= dt_vencimento)."
            ),
            detail={"invalid_count": invalid} if not passed else None,
        )
    )


def _run_structural_checks(
    df: pd.DataFrame, schema: dict[str, ColumnSpec], report: ValidationReport
) -> None:
    _check_columns(df, schema, report)
    _check_nullable(df, schema, report)
    _check_unique(df, schema, report)
    _check_ranges(df, schema, report)
    _check_allowed_values(df, schema, report)
    _check_null_rate(df, schema, report)


def _run_payments_business_rules(df: pd.DataFrame, report: ValidationReport) -> None:
    _check_settlement_null_alignment(df, report)
    _check_settlement_value_alignment(df, report)
    _check_date_ordering(df, report)


def validate_payments(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport(dataset="payments", total_rows=len(df))
    _run_structural_checks(df, PAYMENTS_SCHEMA, report)
    _run_payments_business_rules(df, report)
    report.log_summary()
    return report


def validate_auxiliary(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport(dataset="auxiliary", total_rows=len(df))
    _run_structural_checks(df, AUXILIARY_SCHEMA, report)
    report.log_summary()
    return report
